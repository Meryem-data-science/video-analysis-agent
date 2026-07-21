"""Niveau 12 — Rapport vidéo final (`final_video_report`).

Agrège les analyses VLM validées (par segment) + les métadonnées de fusion
(timestamps, parole, langue) en une synthèse globale de la vidéo : description par
scène avec timestamps, personnages/objets, actions, relations audio-visuel, ton,
langue, incertitudes, et un résumé texte lisible.

Agrégation **déterministe** (pas d'appel modèle ici) : ce niveau ne fait que
combiner les artefacts existants.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .vlm_result_validation import (
    VlmResultValidationError,
    compare_validation_summary,
    validate_results_dir_with_data,
)


class FinalReportError(Exception):
    """Erreur de construction du rapport vidéo final."""


# Version du contrat de données du rapport (structure JSON stable).
# 1.7 : ajout d'un bloc `audio_scene` par scène (musique + SFX analysés par Gemini),
#       attaché depuis l'artefact audio_events_<stem>.json avec provenance
#       ("source": "gemini-audio" + modèle). Dégradation gracieuse si l'artefact est
#       absent/invalide (bloc présent mais vide).
# 1.8 : croisement DÉTERMINISTE son↔visuel (aucun LLM) : bloc `audio_visual_coherence`
#       par scène + audit global `sound_design_audit`.
#       (révisé en 1.10) FALLBACK par polarité : le mood VISUEL (VLM) est le SOCLE
#       (toujours présent) ; l'audio (Gemini) ENRICHIT quand il est exploitable. Les deux
#       vocabulaires sont normalisés vers une polarité canonique (positif/négatif/tension/
#       neutre, tolérante : inconnu -> neutre). Verdicts : cohérent / contraste /
#       pas_de_musique / audio_non_évalué. La valeur « indéterminé » est SUPPRIMÉE, et
#       l'audit distingue désormais « pas_de_musique » de « audio_non_évalué » + couverture.
# 1.9 : sous-objet `audio_scene.measured_features` par scène — features audio MESURÉES
#       localement par librosa (tempo_bpm/energy_rms/beats/audio_activity), apparié par
#       index depuis audio_features_<stem>.json. Toujours présent (défaut "absent"),
#       provenance "source": "librosa". 100% local, aucun LLM, aucune invention.
# 1.10 : suppression de l'OCR (PaddleOCR). Le texte à l'écran (`texte_ecran` par scène +
#        agrégat `texte_ecran_detecte`) est désormais LU NATIVEMENT par le VLM, au format
#        {text, lang, location}. Retrait du bloc de corroboration OCR↔VLM et de
#        `provenance.ocr`. Aucune dépendance OCR, aucun appel modèle ici (agrégation pure).
# 1.11 : validation VLM stricte recalculée, statut complete/partial/failed et rapport
#        construit uniquement depuis les résultats valides.
SCHEMA_VERSION = "1.11"


def _as_str(x: Any) -> str:
    """Garantit une chaîne (jamais None/absent)."""
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _as_list(x: Any) -> List[Any]:
    """Garantit une liste (jamais None ni type inattendu)."""
    return [i for i in x] if isinstance(x, list) else []


def _as_num(x: Any) -> Optional[float]:
    """Garantit un nombre ou None (jamais une chaîne/bool ambigus)."""
    if isinstance(x, bool):
        return None
    return x if isinstance(x, (int, float)) else None


# Indices de ton détectés par mots-clés (FR) dans les descriptions/relations.
_TONE_KEYWORDS = [
    ("humoristique", ("humoris", "drôle", "comique", "farce", "rire", "émoji")),
    ("nostalgique", ("nostalg",)),
    ("tension/colère", ("colère", "réprimand", "menac", "mécontent", "fâch", "dispute")),
    ("joyeux", ("joie", "joyeu", "enjoué", "festif", "amus")),
    ("calme", ("calme", "paisible", "serein", "tranquille")),
    ("triste", ("triste", "mélancol")),
    ("dramatique", ("dramatique", "tendu", "tension")),
]


def load_json(path: str, label: str) -> Dict[str, Any]:
    """Charge un fichier JSON avec des erreurs claires (introuvable / invalide)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FinalReportError(f"Fichier {label} introuvable : {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalReportError(f"JSON {label} invalide ({p}) : {exc}") from exc


def _dedup_preserve(items: List[Any]) -> List[Any]:
    """Déduplique en conservant l'ordre de première apparition.

    Pour les chaînes, la clé de déduplication est insensible à la casse et à la
    ponctuation/espace de fin (évite les quasi-doublons), mais on conserve le
    premier texte d'origine (juste « strippé »).
    """
    seen = set()
    out: List[Any] = []
    for it in items:
        if isinstance(it, str):
            val = it.strip()
            if not val:
                continue
            key = val.casefold().rstrip(" .;,:!?")
        else:
            val = it
            key = it
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


# Énoncé honnête (audio actif, segment sans parole) — affirme un audio PRÉSENT non
# verbal ; ne JAMAIS le confondre avec une incertitude « absence d'audio ».
_NON_VERBAL_AUDIO_NOTE = (
    "Audio non verbal actif (nature non identifiée : musique/bruit/ambiance), "
    "aucune parole transcrite sur ce segment."
)

# Motifs d'incertitudes affirmant une ABSENCE d'audio / de relation audio-visuelle.
# Comparés à un texte normalisé (insensible casse/accents/apostrophes typographiques).
_NO_AUDIO_PATTERNS = (
    "no audio", "no speech", "without audio", "no sound",
    "aucun audio", "pas d'audio", "sans audio",
    "absence totale d'audio", "absence d'audio", "absence d'audio/parole",
    "aucune piste audio", "aucune parole", "pas de parole", "aucun son",
    "aucune relation audio-visuelle", "aucune relation audio visuelle",
)


def _normalize_loose(text: str) -> str:
    """Minuscule + suppression des accents + apostrophes typographiques (matcher robuste)."""
    s = (text or "").replace("’", "'").replace("ʼ", "'")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.casefold()


def _is_no_audio_uncertainty(text: str) -> bool:
    """True si l'incertitude AFFIRME une absence d'audio / de relation audio-visuelle.

    L'énoncé honnête « audio non verbal actif » affirme au contraire un audio présent :
    il contient « aucune parole » mais décrit un audio ACTIF → jamais traité en absence.
    """
    norm = _normalize_loose(text)
    if "non verbal actif" in norm:
        return False
    return any(pat in norm for pat in _NO_AUDIO_PATTERNS)


def _segment_audio_active(fusion_seg: Optional[Dict[str, Any]], global_active: bool) -> bool:
    """Audio actif pour CE segment : flag par-segment si présent, sinon repli global."""
    asum = (fusion_seg or {}).get("audio_summary")
    if isinstance(asum, dict) and asum.get("audio_activity_mode") is not None:
        return asum.get("audio_activity_mode") == "active_audio"
    return global_active


def _detect_tone_hints(texts: List[str]) -> List[str]:
    """Détecte des indices de ton par mots-clés (déterministe)."""
    blob = " ".join(t for t in texts if t).lower()
    hints: List[str] = []
    for label, keys in _TONE_KEYWORDS:
        if any(k in blob for k in keys):
            hints.append(label)
    return hints or ["indéterminé"]


def _truncate(text: str, n: int = 240) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _default_measured_features() -> Dict[str, Any]:
    """Sous-objet `measured_features` par défaut (aucune donnée librosa)."""
    return {
        "source": None,
        "status": "absent",
        "tempo_bpm": None,
        "tempo_reason": None,
        "tempo_note": None,
        "energy_rms": None,
        "beat_count": None,
        "beat_density_per_s": None,
        "audio_activity": None,
        "method_notes": None,
    }


def _build_scene(
    seg_id: int, vlm: Dict[str, Any], fusion_seg: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Construit une entrée de scène (analyse VLM enrichie des timestamps de fusion).

    Contrat de données : ordre et nommage des clés constants ; tous les champs
    toujours présents avec un type cohérent (listes → [], chaînes → "", nombres →
    nombre ou None), y compris `evidence.visual`/`evidence.audio`.
    """
    fseg = fusion_seg or {}
    evidence = vlm.get("evidence") if isinstance(vlm.get("evidence"), dict) else {}
    return {
        "segment_id": int(seg_id),
        "start_s": _as_num(fseg.get("start_s")),
        "end_s": _as_num(fseg.get("end_s")),
        "duration_s": _as_num(fseg.get("duration_s")),
        "scene_description": _as_str(vlm.get("scene_description")),
        "people_objects": _as_list(vlm.get("people_objects")),
        "actions": _as_list(vlm.get("actions")),
        "motion_description": _as_str(vlm.get("motion_description")),
        "audio_visual_relation": _as_str(vlm.get("audio_visual_relation")),
        "uses_speech": bool(vlm.get("uses_speech")),
        "mood": _as_list(vlm.get("mood")),
        "combined_text": _as_str(fseg.get("combined_text")),
        "confidence": _as_num(vlm.get("confidence")),
        "uncertainties": _as_list(vlm.get("uncertainties")),
        "texte_ecran": _normalize_screen_text(vlm.get("texte_ecran")),
        "evidence": {
            "visual": _as_list(evidence.get("visual")),
            "audio": _as_list(evidence.get("audio")),
        },
        # Bloc audio par scène (musique + SFX), rempli depuis scene_audio si fourni.
        # Toujours présent ; défaut « absent » (jamais d'invention). `measured_features`
        # (librosa, local) est un sous-objet toujours présent (défaut « absent »).
        "audio_scene": {
            "source": None, "model": None, "status": "absent", "events": {},
            "measured_features": _default_measured_features(),
        },
        # Croisement son↔visuel (étape 5b), dérivé déterministe. Toujours présent ;
        # le mood visuel est le SOCLE, l'audio l'enrichit. Défaut (recalculé ensuite) :
        # « audio_non_évalué » — jamais « indéterminé ».
        "audio_visual_coherence": {
            "visual_tone": None, "visual_tone_source": None, "visual_polarity": "neutre",
            "audio_mood": None, "audio_mood_confidence": None, "audio_polarity": None,
            "verdict": "audio_non_évalué", "confidence": 0.0, "explication": "",
        },
    }


def _fmt_ts(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def _build_readable_summary(report: Dict[str, Any]) -> str:
    """Construit un résumé texte lisible (français) du rapport."""
    agg = report["aggregate"]
    stem = Path(report.get("video_path", "video")).stem or "video"
    lines: List[str] = []
    lines.append(f"Rapport vidéo final — {stem}")
    lines.append(
        f"Langue: {report.get('language') or '?'} | "
        f"Durée: {_fmt_ts(report.get('total_duration_s'))} s | "
        f"Scènes: {report.get('scene_count')} | "
        f"Confiance moyenne: {agg.get('average_confidence')}"
    )
    lines.append(f"Ton global: {', '.join(agg.get('tone', []))}")
    lines.append("")
    lines.append("Scènes :")
    for sc in report["scenes"]:
        lines.append(
            f"  [{_fmt_ts(sc['start_s'])}–{_fmt_ts(sc['end_s'])}s] "
            f"(conf {sc.get('confidence')}) {_truncate(sc['scene_description'])}"
        )
    lines.append("")

    people = agg.get("people_objects", [])
    actions = agg.get("actions", [])
    uncert = agg.get("uncertainties", [])

    def _cap(items: List[str], k: int = 10) -> str:
        shown = items[:k]
        suffix = f" (+{len(items) - k} autres)" if len(items) > k else ""
        return "; ".join(shown) + suffix if shown else "—"

    lines.append(f"Personnages/objets récurrents : {_cap(people)}")
    lines.append(f"Actions principales : {_cap(actions)}")
    lang = report.get("language") or "inconnue"
    n = report.get("scene_count") or 0
    sws = agg.get("segments_with_speech", 0)
    audio_active = agg.get("audio_activity_mode") == "active_audio"
    if audio_active:
        audio_line = (
            f"Audio : actif. Parole présente sur {sws}/{n} segment(s) ; "
            f"langue détectée : {lang}."
        )
        if agg.get("non_speech_audio_present"):
            audio_line += (
                f" Audio non verbal actif (nature non identifiée) sur "
                f"{n - sws} segment(s) sans parole transcrite."
            )
        tc = agg.get("average_transcript_confidence")
        if isinstance(tc, (int, float)) and tc < 0.7:
            audio_line += " Transcription automatique possiblement approximative."
    elif agg.get("uses_speech_any"):
        audio_line = (
            f"Audio : parole présente ({agg.get('audio_activity_mode') or '?'}) ; "
            f"langue détectée : {lang}."
        )
        tc = agg.get("average_transcript_confidence")
        if isinstance(tc, (int, float)) and tc < 0.7:
            audio_line += " Transcription automatique possiblement approximative."
    else:
        audio_line = "Audio : aucune parole exploitable (vidéo sans audio ou sans parole détectée)."
    lines.append(audio_line)
    lines.append(f"Incertitudes notables : {_cap(uncert, 6)}")
    lines.append("")
    lines.append(report.get("global_summary", ""))
    return "\n".join(lines).rstrip() + "\n"


_ARC_SUBKEYS = (
    "situation_initiale", "element_declencheur", "developpement", "climax", "denouement",
)


def _as_int_list(x: Any) -> List[int]:
    """Liste d'entiers (exclut bool/str ; arrondit les floats entiers)."""
    out: List[int] = []
    for i in _as_list(x):
        if isinstance(i, bool):
            continue
        if isinstance(i, int):
            out.append(i)
        elif isinstance(i, float) and i.is_integer():
            out.append(int(i))
    return out


def _normalize_arc_fields(
    raw: Optional[Dict[str, Any]],
    valid_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Normalise l'arc en structure stable (5 champs toujours présents/typés).

    `arc_confidence` est clampée dans [0, 1] ; `arc_evidence` est filtrée pour ne
    conserver que les segment_id réellement présents dans les résultats VLM
    (anti-hallucination : pas de preuve inventée hors des scènes analysées).
    """
    src = raw if isinstance(raw, dict) else {}
    arc_src = src.get("arc_narratif") if isinstance(src.get("arc_narratif"), dict) else {}
    conf = _as_num(src.get("arc_confidence"))
    conf = 0.0 if conf is None else max(0.0, min(1.0, float(conf)))
    evidence = _as_int_list(src.get("arc_evidence"))
    if valid_ids is not None:
        evidence = [i for i in evidence if i in valid_ids]
    return {
        "arc_narratif": {k: _as_str(arc_src.get(k)) for k in _ARC_SUBKEYS},
        "arc_emotionnel": _as_str(src.get("arc_emotionnel")),
        "resume_humain": _as_str(src.get("resume_humain")),
        "arc_confidence": conf,
        "arc_evidence": evidence,
    }


def _as_int(x: Any) -> Optional[int]:
    """Entier ou None (exclut bool ; arrondit un float entier)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return None


def _as_str_or_none(x: Any) -> Optional[str]:
    """Chaîne non vide ou None (jamais "" — pour des métadonnées « absentes = null »)."""
    return x if isinstance(x, str) and x.strip() else None


def _infer_format_hint(
    width: Optional[int], height: Optional[int], duration_s: Optional[float]
) -> Optional[str]:
    """Description DÉTERMINISTE du format (résolution + durée), SANS affirmer d'origine.

    C'est une description du format observé, pas une affirmation de plateforme/source.
    Résolution inconnue -> None (aucune valeur inventée).
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return None
    # Carré : tolérance ~5% sur le plus grand côté.
    if abs(width - height) <= 0.05 * max(width, height):
        return "carré (format typique fil Instagram)"
    if height > width:
        if isinstance(duration_s, (int, float)) and not isinstance(duration_s, bool) and duration_s <= 60:
            return "vertical court (format typique des plateformes mobiles type TikTok/Reels/Shorts)"
        return "vertical long"
    return "paysage (format standard/web)"


def _build_provenance(
    *,
    vlm_mode: Optional[str] = None,
    provider: Optional[str] = None,
    scene_audio: Optional[Dict[str, Any]] = None,
    audio_features: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
    frames_manifest: Optional[Dict[str, Any]] = None,
    motion_manifest: Optional[Dict[str, Any]] = None,
    pipeline_status: Optional[Dict[str, Any]] = None,
    frames_manifest_path: Optional[str] = None,
    motion_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Bloc de provenance AGRÉGÉ, rempli depuis les artefacts en cache (déterministe).

    Principe : on trace ce qu'on SAIT (modèles, langues, chemins, nb frames, erreurs
    connues) et on marque EXPLICITEMENT (null + raison) ce qui n'a pas été enregistré.
    Aucune valeur inventée. Tout artefact absent -> champ null.
    """
    sa = scene_audio if isinstance(scene_audio, dict) else {}
    af = audio_features if isinstance(audio_features, dict) else {}
    fm = frames_manifest if isinstance(frames_manifest, dict) else {}
    mm = motion_manifest if isinstance(motion_manifest, dict) else {}
    ps = pipeline_status if isinstance(pipeline_status, dict) else {}
    summ = summary if isinstance(summary, dict) else {}

    # Frames : total (frames_manifest) + nb par scène (motion_manifest.frame_count).
    frames_list = fm.get("frames") if isinstance(fm.get("frames"), list) else None
    mm_segs = mm.get("segments") if isinstance(mm.get("segments"), list) else None
    per_scene = None
    if mm_segs:
        per_scene = {
            str(s.get("segment_id")): _as_int(s.get("frame_count"))
            for s in mm_segs if isinstance(s, dict) and "segment_id" in s
        }

    # Erreurs CONNUES, agrégées (jamais inventées).
    errors: List[Dict[str, Any]] = []
    for st in (ps.get("steps") or []):
        if isinstance(st, dict) and st.get("ok") is False:
            errors.append({"source": "pipeline_step", "step": st.get("step"),
                           "returncode": st.get("returncode"), "detail": st.get("detail")})
    for seg in (summ.get("results") or []):
        if isinstance(seg, dict) and seg.get("valid") is False:
            errors.append({"source": "vlm_validation", "segment_id": seg.get("segment_id"),
                           "errors": seg.get("errors")})
    for asc in (sa.get("scenes") or []):
        if isinstance(asc, dict) and asc.get("audio_status") not in ("ok", None):
            errors.append({"source": "audio_semantic", "scene_index": asc.get("scene_index"),
                           "status": asc.get("audio_status")})

    return {
        "visual": {
            "mode": _as_str_or_none(vlm_mode),
            "provider": _as_str_or_none(provider),
            "model": None,
            "model_note": "nom de modèle non enregistré / non exposé par le SDK",
        },
        "audio_semantic": {
            "engine": "gemini" if sa else None,
            "model": _as_str_or_none(sa.get("model_used")),
            "status": _as_str_or_none(sa.get("status")),
        },
        "audio_measured": {
            "engine": "librosa" if af else None,
            "params": af.get("params") if isinstance(af.get("params"), dict) else None,
        },
        "frames": {
            "total": len(frames_list) if frames_list is not None else None,
            "per_scene": per_scene,
            "manifest": _as_str_or_none(frames_manifest_path),
        },
        "composites": {
            "manifest": _as_str_or_none(motion_manifest_path),
            "available": bool(mm_segs) if mm_segs is not None else None,
        },
        "errors": errors,
        # Champs DÉFINITIVEMENT non enregistrés pour ce run : null + raison (jamais inventés).
        "lost": {
            "model_visuel_exact": {"value": None, "reason": "non enregistré ; SDK default"},
            "prompt_version": {"value": None, "reason": "prompts non versionnés pour ce run"},
            "temps_execution": {"value": None, "reason": "non chronométré pour ce run"},
            "retries": {"value": None, "reason": "non comptés ; status audio global = partiel"},
        },
    }


def _build_metadata(
    inspection: Optional[Dict[str, Any]],
    total_duration_s: Optional[float],
    schema_version: str,
    vlm_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Métadonnées techniques (LECTURE PURE de l'inspection) + provenance minimale.

    Aucune valeur inventée : tout champ absent de l'inspection -> null. Si `inspection`
    est None, les champs techniques sont null (audio_present=False) mais la provenance
    (generation.schema_version / generated_at) reste remplie.
    """
    insp = inspection if isinstance(inspection, dict) else {}
    fblk = insp.get("file") if isinstance(insp.get("file"), dict) else {}
    vblk = insp.get("video") if isinstance(insp.get("video"), dict) else {}
    ablk = insp.get("audio") if isinstance(insp.get("audio"), dict) else {}
    dblk = insp.get("derived") if isinstance(insp.get("derived"), dict) else {}

    width = _as_int(vblk.get("width"))
    height = _as_int(vblk.get("height"))
    fps = _as_num(vblk.get("fps_used"))
    if fps is None:
        fps = _as_num(vblk.get("avg_fps"))
    duration = _as_num(fblk.get("duration_s"))
    if duration is None:
        duration = _as_num(total_duration_s)

    return {
        "resolution": {"width": width, "height": height},
        "orientation": _as_str_or_none(vblk.get("orientation")),
        "duration_s": duration,
        "fps": fps,
        "video_codec": _as_str_or_none(vblk.get("codec")),
        "audio_present": bool(ablk.get("present")),
        "audio_codec": _as_str_or_none(ablk.get("codec")),
        "sample_rate": _as_int(ablk.get("sample_rate")),
        "channels": _as_int(ablk.get("channels")),
        "bitrate_kbps": _as_num(fblk.get("bitrate_kbps")),
        "size_mb": _as_num(fblk.get("size_mb")),
        "platform_hint": _as_str_or_none(dblk.get("platform_hint")),
        "format_hint": _infer_format_hint(width, height, duration),
        "generation": {
            "schema_version": schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "vlm_mode": vlm_mode if isinstance(vlm_mode, str) else None,
        },
    }


def _norm_screen(text: str) -> str:
    return (text or "").strip().casefold()


def _normalize_screen_text(items: Any) -> List[Dict[str, Any]]:
    """Nettoie + déduplique (flou) le `texte_ecran` LU PAR LE VLM pour une scène.

    Chaque item attendu : {text, lang, location}. Jette : texte strippé < 2 caractères
    ou sans aucun caractère alphanumérique. Dédup floue : SequenceMatcher.ratio >= 0.7
    sur texte normalisé (garde la première occurrence). Déterministe, aucune invention.
    """
    cleaned: List[Dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        text = it.get("text")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if len(stripped) < 2 or not any(c.isalnum() for c in stripped):
            continue
        cleaned.append({
            "text": stripped,
            "lang": it.get("lang") if isinstance(it.get("lang"), str) else None,
            "location": it.get("location") if isinstance(it.get("location"), str) else None,
        })

    kept: List[Dict[str, Any]] = []
    for it in cleaned:
        n = _norm_screen(it["text"])
        if not any(SequenceMatcher(None, n, _norm_screen(k["text"])).ratio() >= 0.7 for k in kept):
            kept.append(it)
    return kept


def _aggregate_screen_text(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Union dédupliquée (floue) du texte écran VLM, avec scenes[] des indices.

    items : entrées {text, lang, location, scene_index}. Regroupe par texte normalisé
    (SequenceMatcher.ratio >= 0.7) ; conserve la première graphie/langue rencontrée et
    la liste triée des scènes où le texte apparaît. Ordre = première apparition.
    """
    groups: List[Dict[str, Any]] = []
    for it in items:
        n = _norm_screen(it["text"])
        placed = False
        for g in groups:
            if SequenceMatcher(None, n, _norm_screen(g["text"])).ratio() >= 0.7:
                g["scenes"].add(it["scene_index"])
                placed = True
                break
        if not placed:
            groups.append({"text": it["text"], "lang": it.get("lang"),
                           "scenes": {it["scene_index"]}})
    return [{"text": g["text"], "lang": g["lang"], "scenes": sorted(g["scenes"])}
            for g in groups]


def _compute_montage_rythme(
    scenes: List[Dict[str, Any]], total_duration_s: Optional[float]
) -> Dict[str, Any]:
    """Rythme de montage — 100% déterministe (aucun LLM). Métriques toujours exposées.

    Le label est dérivé de la durée moyenne d'une scène. Si la moyenne n'est pas
    calculable (n>1 mais durées indisponibles), le label est « indetermine » (jamais
    affirmer un rythme sans preuve) et les métriques sont exposées à null.
    """
    n = len(scenes)
    cuts = max(0, n - 1)
    durs = [
        sc.get("duration_s") for sc in scenes
        if isinstance(sc.get("duration_s"), (int, float)) and not isinstance(sc.get("duration_s"), bool)
    ]
    # Durée totale : total_duration_s si exploitable (>0), sinon repli sur la somme des durées.
    total = (
        total_duration_s
        if isinstance(total_duration_s, (int, float)) and not isinstance(total_duration_s, bool)
        and total_duration_s > 0
        else None
    )
    if total is None and durs:
        total = round(sum(durs), 3)
    avg = round(total / n, 4) if (total and total > 0 and n > 0) else None
    cpm = round(cuts * 60.0 / total, 4) if (total and total > 0) else None

    if n <= 1:
        label = "plan_unique"
    elif avg is None:
        label = "indetermine"  # n>1 mais durées indisponibles : pas de rythme prouvable
    elif avg >= 10:
        label = "tres_lent"
    elif avg >= 5:
        label = "lent"
    elif avg >= 2.5:
        label = "modere"
    else:
        label = "rapide"

    return {
        "label": label,
        "scene_count": n,
        "average_scene_duration_s": avg,
        "cuts": cuts,
        "cuts_per_minute": cpm,
    }


# --- Croisement son↔visuel (étape 5b) : FALLBACK par polarité ----------------------
#
# Principe : le mood VISUEL (VLM) est le SOCLE — il est TOUJOURS l'émotion de la
# scène. Le mood AUDIO (Gemini) vient seulement ENRICHIR / vérifier la cohérence quand
# il est exploitable. Le verdict "indéterminé" est SUPPRIMÉ : il mélangeait deux causes
# opposées (pas de musique = légitime ; mot non reconnu = bug). Désormais :
#   - audio exploitable -> on compare les POLARITÉS canoniques -> "cohérent"/"contraste" ;
#   - audio sans musique -> "pas_de_musique" ;
#   - audio absent/échoué/vide/non exploitable -> "audio_non_évalué".
# Dans tous les cas le mood visuel reste l'émotion de la scène (on n'efface rien).

_AV_COHERENT = "cohérent"
_AV_CONTRASTE = "contraste"
_AV_PAS_DE_MUSIQUE = "pas_de_musique"
_AV_NON_EVALUE = "audio_non_évalué"

# Table de POLARITÉ canonique (positif / négatif / tension / neutre). Sous-chaînes
# normalisées sans accent, en RACINES pour capter les formes fléchies (« joyeu » ->
# joyeux/joyeuse). TOLÉRANTE par conception : tout terme inconnu tombe sur « neutre »
# (jamais d'erreur, plus jamais d'« indéterminé »).
#
# ORDRE IMPORTANT (résolution des collisions de sous-chaîne) : négatif et tension AVANT
# positif — ex. « déception légère » contient 'leger' (positif) ET 'decept' (négatif) ;
# en testant négatif d'abord, 'decept' l'emporte correctement.
_POLARITE_TABLE: Dict[str, tuple] = {
    # Tristesse / détresse / doute / inquiétude — états « bas », non agressifs.
    "négatif": (
        "detress", "detresse", "compassion", "triste", "tristesse", "melancol",
        "supplica", "deni", "desarroi", "gene", "decept", "inquiet", "inquietude",
        "doute", "scept", "anxi", "hesit", "morne", "chagrin", "abattement",
        "resignation", "solitude", "culpabilite", "honte", "amertume", "lassitude",
        "peine", "desespoir", "angoiss", "malaise", "sad", "depress",
    ),
    # Conflit / agression / menace / urgence — états « tendus », actifs.
    "tension": (
        "tension", "tendu", "menace", "menacant", "coler", "agac", "exasper",
        "indign", "fureur", "rage", "hostil", "conflit", "stress", "oppress",
        "panique", "urgence", "nervos", "dramatique", "suspense", "danger",
        "impatien", "tense", "dark", "ominous", "menacing",
    ),
    # Émotions « hautes » / agréables (calme et tendresse inclus, par consigne).
    "positif": (
        "joie", "joyeu", "enthousias", "fier", "soulag", "gratitude", "reconnaiss",
        "satisfa", "confian", "espoir", "optimis", "calme", "leger", "tendre",
        "tendresse", "emerveil", "plaisir", "humour", "humor", "ironi", "amus",
        "gai", "heureu", "ludique", "festi", "triomph", "serein", "apais", "enjoue",
        "complicite", "affection", "amour", "chaleur", "epique", "positif", "positive",
        "playful", "happy", "joyful", "cheerful", "upbeat", "fun",
    ),
    # Ni positif ni négatif marqué — observation / attente / surprise.
    "neutre": (
        "neutre", "neutral", "surpris", "surprise", "incredul", "curi", "anticipa",
        "nostalg", "contempl", "observ", "factuel", "descriptif", "indiff",
    ),
}

# Signe de valence par polarité (pour décider cohérent vs contraste). « neutre » = 0 :
# il ne contredit jamais rien -> jamais de contraste avec une polarité neutre.
_POLARITE_SIGNE: Dict[str, int] = {"positif": 1, "neutre": 0, "négatif": -1, "tension": -1}

# Humeurs audio « pas de musique » (normalisées).
_NO_MUSIC_MOODS = {"aucun", "aucune", "pas de musique", "sans musique", "silence",
                   "vide", "absente", "absent", "aucune musique"}
# Humeurs audio non exploitables (musique peut-être présente mais ambiance indécise).
_UNEXPLOITABLE_MOODS = {"incertain", "inconnu", "indetermine", "indetermines", "n/a",
                        "na", "none", "?"}


def _polarite(mood: Any) -> str:
    """Normalise un mood (libre, visuel OU audio) vers une polarité canonique.

    TOLÉRANTE : terme vide/None/inconnu -> « neutre » (jamais d'erreur, jamais
    d'« indéterminé »). C'est un champ de TRAVAIL pour le croisement, pas un
    remplacement de la description visuelle riche (qui reste intacte dans `mood`).
    """
    if not isinstance(mood, str) or not mood.strip():
        return "neutre"
    norm = _normalize_loose(mood)
    for cat, subs in _POLARITE_TABLE.items():
        if any(sub in norm for sub in subs):
            return cat
    return "neutre"  # inconnu -> neutre (tolérance totale)


def _audio_mood_exploitable(a_mood: Any) -> bool:
    """True si le mood audio est une humeur réellement exploitable (≠ vide/incertain)."""
    if not isinstance(a_mood, str) or not a_mood.strip():
        return False
    return _normalize_loose(a_mood) not in _UNEXPLOITABLE_MOODS


def _is_no_music(a_mood: Any, music: Dict[str, Any]) -> bool:
    """True si l'audio Gemini indique explicitement une ABSENCE de musique."""
    if isinstance(music, dict) and music.get("present") is False:
        return True
    if not isinstance(a_mood, str) or not a_mood.strip():
        return False
    n = _normalize_loose(a_mood)
    return (n in _NO_MUSIC_MOODS or n.startswith("aucun")
            or "pas de musique" in n or "sans musique" in n)


def _av_verdict(a_pol: str, v_pol: str) -> str:
    """Verdict par polarité : « cohérent » (même signe / neutre) ou « contraste » (signes opposés).

    « neutre » (signe 0) ne contredit personne -> jamais de contraste avec du neutre ;
    « négatif » et « tension » partagent le signe négatif -> cohérents entre eux.
    """
    if a_pol == v_pol:
        return _AV_COHERENT
    if _POLARITE_SIGNE.get(a_pol, 0) * _POLARITE_SIGNE.get(v_pol, 0) < 0:
        return _AV_CONTRASTE
    return _AV_COHERENT


def _compute_av_coherence(sc: Dict[str, Any]):
    """Croisement musique↔mood visuel pour UNE scène. Déterministe, ne lève jamais.

    Renvoie (block, note) :
      - block : le dict `audio_visual_coherence` (le mood VISUEL est toujours présent
        comme SOCLE ; la polarité canonique est un champ de TRAVAIL) ;
      - note : incertitude à ajouter à la scène — UNIQUEMENT pour « contraste » (pour
        ne pas polluer le rapport quand l'audio n'est simplement pas évalué).

    Plus jamais de « indéterminé » : tout mood inconnu tombe sur la polarité « neutre ».
    """
    av = sc.get("audio_scene") or {}
    audio_ok = av.get("source") == "gemini-audio" and av.get("status") == "ok"
    events = av.get("events") if isinstance(av.get("events"), dict) else {}
    music = events.get("music") if isinstance(events.get("music"), dict) else {}
    a_mood = music.get("mood") if isinstance(music.get("mood"), str) and music.get("mood").strip() else None
    a_conf = _as_num(music.get("mood_confidence"))

    # SOCLE visuel : le mood du VLM EST l'émotion de la scène (toujours présent si
    # le VLM en a fourni). On retient le 1er mood pour l'affichage ; sa polarité sert
    # uniquement au croisement (le champ `mood` riche de la scène reste inchangé).
    visual_moods = [m for m in _as_list(sc.get("mood")) if isinstance(m, str) and m.strip()]
    visual_tone = visual_moods[0] if visual_moods else None
    v_pol = _polarite(visual_tone)
    v_conf = _as_num(sc.get("confidence"))

    block = {
        "visual_tone": visual_tone,                 # SOCLE : émotion de la scène (riche)
        "visual_tone_source": "vlm-mood" if visual_moods else None,
        "visual_polarity": v_pol,                   # champ de TRAVAIL (canonique)
        "audio_mood": a_mood if audio_ok else None,
        "audio_mood_confidence": a_conf if audio_ok else None,
        "audio_polarity": None,
        "verdict": _AV_NON_EVALUE,
        "confidence": 0.0,
        "explication": "",
    }

    # 1) Audio absent / échoué (503 / échec_gracieux / absent) -> audio_non_évalué.
    if not audio_ok:
        block["explication"] = (
            f"Mood visuel « {visual_tone or 'non fourni'} » (socle) ; audio non évalué "
            "(analyse Gemini absente ou en échec)."
        )
        return block, None

    # 2) Gemini a tourné mais SANS musique -> pas_de_musique (le visuel reste le socle).
    if _is_no_music(a_mood, music):
        block["explication"] = (
            f"Mood visuel « {visual_tone or 'non fourni'} » (socle) ; "
            "Gemini n'a pas détecté de musique sur cette scène."
        )
        block["verdict"] = _AV_PAS_DE_MUSIQUE
        return block, None

    # 3) Mood audio vide / non exploitable (« incertain »…) -> audio_non_évalué.
    if not _audio_mood_exploitable(a_mood):
        block["explication"] = (
            f"Mood visuel « {visual_tone or 'non fourni'} » (socle) ; "
            f"ambiance sonore non exploitable (« {a_mood} »)."
        )
        return block, None

    # 4) Audio exploitable -> comparaison de POLARITÉ (mood visuel toujours le socle).
    a_pol = _polarite(a_mood)
    block["audio_polarity"] = a_pol
    verdict = _av_verdict(a_pol, v_pol)
    block["verdict"] = verdict
    if isinstance(a_conf, (int, float)) and not isinstance(a_conf, bool) \
            and isinstance(v_conf, (int, float)) and not isinstance(v_conf, bool):
        block["confidence"] = round(min(a_conf, v_conf), 4)

    if verdict == _AV_COHERENT:
        block["explication"] = (
            f"Mood visuel « {visual_tone} » ({v_pol}) et ambiance sonore « {a_mood} » "
            f"({a_pol}) vont dans le même sens."
        )
        return block, None

    # contraste
    block["explication"] = (
        f"Mood visuel « {visual_tone} » ({v_pol}) en CONTRASTE avec l'ambiance sonore "
        f"« {a_mood} » ({a_pol})."
    )
    note = (f"Contraste audio↔visuel : mood visuel « {visual_tone} » ({v_pol}) vs "
            f"ambiance sonore « {a_mood} » ({a_pol}) — à vérifier.")
    return block, note


def build_final_report(
    vlm_results: Dict[int, Dict[str, Any]],
    fusion: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    sources: Optional[Dict[str, Optional[str]]] = None,
    inspection: Optional[Dict[str, Any]] = None,
    scene_audio: Optional[Dict[str, Any]] = None,
    audio_features: Optional[Dict[str, Any]] = None,
    vlm_mode: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    arc_generator: Optional[
        Callable[[Dict[int, Dict[str, Any]], Dict[str, Any]], Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """Agrège les analyses VLM + la fusion en un rapport vidéo final (dict).

    Args:
        vlm_results: résultats VLM validés, indexés par segment_id.
        fusion: manifest de fusion audio-visuelle (timestamps, combined_text, langue).
        summary: résumé de validation VLM (facultatif ; pour la confiance moyenne).
        sources: chemins sources reportés dans le rapport.
    """
    if not isinstance(vlm_results, dict):
        raise FinalReportError("`vlm_results` doit être un dictionnaire.")
    if not vlm_results and not (
        isinstance(validation, dict) and validation.get("status") == "failed"
    ):
        raise FinalReportError("Aucune analyse VLM exploitable.")
    if not isinstance(fusion, dict):
        raise FinalReportError("`fusion` doit être un dictionnaire.")

    fusion_by_id = {
        int(s.get("segment_id")): s for s in fusion.get("segments", []) or []
    }
    language = fusion.get("language")
    video_path = fusion.get("video_path", "")
    audio_activity_mode = (fusion.get("summary", {}) or {}).get("audio_activity_mode")
    # Donnée AUTORITAIRE pour décider de la présence d'audio (≠ uses_speech_any).
    audio_active = audio_activity_mode == "active_audio"

    scenes: List[Dict[str, Any]] = []
    for seg_id in sorted(vlm_results):
        scenes.append(_build_scene(seg_id, vlm_results[seg_id], fusion_by_id.get(seg_id)))

    uses_speech_any = any(sc["uses_speech"] for sc in scenes)
    segments_with_speech = sum(1 for sc in scenes if sc["uses_speech"])
    non_speech_audio_present = audio_active and any(
        (not sc["uses_speech"])
        and _segment_audio_active(fusion_by_id.get(sc["segment_id"]), audio_active)
        for sc in scenes
    )

    # Correction #2 — Quand l'audio est ACTIF (donnée autoritaire), le rapport ne doit
    # JAMAIS affirmer une absence d'audio. Au niveau SCÈNE : on retire (inconditionnel)
    # les incertitudes « absence d'audio / aucune relation audio-visuelle » des segments
    # à audio actif ; pour un segment SANS parole dont on a justement retiré une telle
    # revendication, on la REMPLACE par un énoncé honnête (audio non verbal actif, nature
    # non identifiée). Aucune invention de musique/effets : on n'affirme que ce que
    # l'analyse audio sait (actif/silencieux, parole oui/non).
    if audio_active:
        for sc in scenes:
            if not _segment_audio_active(fusion_by_id.get(sc["segment_id"]), audio_active):
                continue
            kept = [u for u in sc["uncertainties"] if not _is_no_audio_uncertainty(u)]
            removed_absence = len(kept) != len(sc["uncertainties"])
            if (not sc["uses_speech"]) and removed_absence:
                kept.append(_NON_VERBAL_AUDIO_NOTE)
            sc["uncertainties"] = _dedup_preserve(kept)

    all_people = _dedup_preserve([p for sc in scenes for p in sc["people_objects"]])
    all_actions = _dedup_preserve([a for sc in scenes for a in sc["actions"]])
    # L'agrégat est rebâti à partir des incertitudes de scène DÉJÀ filtrées : aucune
    # « absence d'audio » d'un segment à audio actif ne peut donc y subsister.
    all_uncert = _dedup_preserve([u for sc in scenes for u in sc["uncertainties"]])

    tone_texts = [sc["scene_description"] for sc in scenes] + [
        sc["audio_visual_relation"] for sc in scenes
    ]
    tone = _detect_tone_hints(tone_texts)

    # Émotions dominantes de la vidéo : union dédupliquée des mood des scènes (ordre préservé).
    dominant_moods = _dedup_preserve([m for sc in scenes for m in sc["mood"]])

    confidences = [sc["confidence"] for sc in scenes if isinstance(sc["confidence"], (int, float))]
    if summary and isinstance(summary.get("summary", {}).get("average_confidence"), (int, float)):
        avg_conf = summary["summary"]["average_confidence"]
    else:
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    # Confiance moyenne de transcription (depuis la fusion) — pour qualifier l'audio.
    tconfs = [
        float(t["confidence"])
        for fseg in (fusion.get("segments", []) or [])
        for t in (fseg.get("transcript_segments", []) or [])
        if isinstance(t.get("confidence"), (int, float)) and not isinstance(t.get("confidence"), bool)
    ]
    avg_transcript_conf = round(sum(tconfs) / len(tconfs), 4) if tconfs else None

    end_values = [sc["end_s"] for sc in scenes if isinstance(sc["end_s"], (int, float))]
    total_duration = round(max(end_values), 3) if end_values else None
    montage_rythme = _compute_montage_rythme(scenes, total_duration)
    metadata = _build_metadata(inspection, total_duration, SCHEMA_VERSION, vlm_mode=vlm_mode)
    # Provenance : utilise le bloc fourni (from_files, complet) sinon en construit un
    # minimal depuis ce qu'on a sous la main (jamais inventé ; champs manquants -> null).
    provenance_block = provenance if isinstance(provenance, dict) else _build_provenance(
        vlm_mode=vlm_mode, scene_audio=scene_audio, audio_features=audio_features,
        summary=summary,
    )

    # Texte à l'écran : lu NATIVEMENT par le VLM (champ `texte_ecran` par scène, déjà
    # normalisé dans _build_scene). On agrège juste l'union dédupliquée (déterministe,
    # aucun appel modèle). Plus d'OCR ni de corroboration : la lecture EST l'analyse VLM.
    texte_ecran_detecte = _aggregate_screen_text(
        [{**d, "scene_index": sc["segment_id"]} for sc in scenes for d in sc["texte_ecran"]]
    )

    # Audio par scène (étape 5a) : attache le bloc audio_events de scene_audio à chaque
    # scène, apparié par scene_index == segment_id. Provenance marquée ("gemini-audio" +
    # modèle). PAS de croisement son↔visuel ici (étape 5b). Dégradation gracieuse :
    # artefact absent/invalide -> chaque scène garde son bloc `audio_scene` « absent ».
    audio_ok = isinstance(scene_audio, dict) and isinstance(scene_audio.get("scenes"), list)
    audio_model = scene_audio.get("model_used") if audio_ok else None
    audio_by_scene: Dict[int, Dict[str, Any]] = {}
    if audio_ok:
        for sa in scene_audio.get("scenes", []) or []:
            if not isinstance(sa, dict):
                continue
            try:
                audio_by_scene[int(sa.get("scene_index"))] = sa
            except (TypeError, ValueError):
                continue
    for sc in scenes:
        sa = audio_by_scene.get(sc["segment_id"])
        if isinstance(sa, dict):
            events = sa.get("audio_events") if isinstance(sa.get("audio_events"), dict) else {}
            status = _as_str(sa.get("audio_status")) or "absent"
            sc["audio_scene"] = {
                "source": "gemini-audio",
                "model": _as_str_or_none(audio_model),
                "status": status,
                "events": events,
                "measured_features": _default_measured_features(),
            }

    # Features audio MESURÉES (librosa, local) — sous-objet `audio_scene.measured_features`
    # (étape librosa-2). Apparié par scene_index == segment_id. TOUJOURS écrit (même quand
    # scene_audio a remplacé le bloc audio_scene), défaut « absent », provenance librosa.
    af_ok = isinstance(audio_features, dict) and isinstance(audio_features.get("scenes"), list)
    af_method_notes = audio_features.get("method_notes") if af_ok else None
    af_by_scene: Dict[int, Dict[str, Any]] = {}
    if af_ok:
        for fa in audio_features.get("scenes", []) or []:
            if not isinstance(fa, dict):
                continue
            try:
                af_by_scene[int(fa.get("scene_index"))] = fa
            except (TypeError, ValueError):
                continue
    for sc in scenes:
        mf = _default_measured_features()
        fa = af_by_scene.get(sc["segment_id"])
        if isinstance(fa, dict):
            feats = fa.get("features") if isinstance(fa.get("features"), dict) else {}
            st = _as_str(fa.get("audio_status")) or "absent"
            if feats and st == "ok":
                mf = {
                    "source": "librosa",
                    "status": st,
                    "tempo_bpm": _as_num(feats.get("tempo_bpm")),
                    "tempo_reason": _as_str_or_none(feats.get("tempo_reason")),
                    "tempo_note": _as_str_or_none(feats.get("tempo_note")),
                    "energy_rms": _as_num(feats.get("energy_rms")),
                    "beat_count": _as_int(feats.get("beat_count")),
                    "beat_density_per_s": _as_num(feats.get("beat_density_per_s")),
                    "audio_activity": _as_str_or_none(feats.get("audio_activity")),
                    "method_notes": af_method_notes if isinstance(af_method_notes, dict) else None,
                }
            else:
                mf["status"] = st  # ex. features_échec : reflété, sans invention
        if isinstance(sc.get("audio_scene"), dict):
            sc["audio_scene"]["measured_features"] = mf

    # Croisement son↔visuel (étape 5b) — FALLBACK par polarité, 100% déterministe.
    # Le mood VISUEL est le socle (toujours présent) ; l'audio enrichit quand il est
    # exploitable. Verdicts possibles : cohérent / contraste / pas_de_musique /
    # audio_non_évalué (plus jamais « indéterminé »). Seul « contraste » ajoute une note.
    audit = {
        "total_scenes": len(scenes),
        "cohérent": 0, "contraste": 0, "pas_de_musique": 0, "audio_non_évalué": 0,
        "scenes_croisées": 0, "couverture": 0.0, "summary": "",
    }
    for sc in scenes:
        block, note = _compute_av_coherence(sc)
        sc["audio_visual_coherence"] = block
        audit[block["verdict"]] = audit.get(block["verdict"], 0) + 1
        if note:  # uniquement pour « contraste »
            sc["uncertainties"] = _dedup_preserve(sc["uncertainties"] + [note])
    # Scènes réellement CROISÉES audio↔visuel = cohérent + contraste.
    audit["scenes_croisées"] = audit["cohérent"] + audit["contraste"]
    audit["couverture"] = (
        round(audit["scenes_croisées"] / audit["total_scenes"], 4)
        if audit["total_scenes"] else 0.0
    )
    audit["summary"] = (
        f"{audit['cohérent']} cohérente(s), {audit['contraste']} en contraste, "
        f"{audit['pas_de_musique']} sans musique, {audit['audio_non_évalué']} audio non évalué ; "
        f"couverture {audit['scenes_croisées']}/{audit['total_scenes']} scène(s) croisée(s) "
        f"({round(audit['couverture'] * 100, 1)}%)."
    )

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "final_video_report",
        "video_path": str(video_path),
        "language": language,
        "sources": sources or {},
        "metadata": metadata,
        "provenance": provenance_block,
        "scene_count": len(scenes),
        "total_duration_s": total_duration,
        "scenes": scenes,
        "aggregate": {
            "people_objects": all_people,
            "actions": all_actions,
            "tone": tone,
            "dominant_moods": dominant_moods,
            "languages": _dedup_preserve([language]) if language else [],
            "uses_speech_any": uses_speech_any,
            "segments_with_speech": segments_with_speech,
            "audio_activity_mode": audio_activity_mode,
            "non_speech_audio_present": non_speech_audio_present,
            "average_confidence": avg_conf,
            "average_transcript_confidence": avg_transcript_conf,
            "uncertainties": all_uncert,
            "montage_rythme": montage_rythme,
            "genre": [],
            "texte_ecran_detecte": texte_ecran_detecte,
        },
        "sound_design_audit": audit,
    }

    # Arc narratif (correction #1) : 5 champs TOUJOURS présents (vides par défaut),
    # insérés juste après `aggregate` et avant `next_step`. Généré par UN appel LLM
    # texte optionnel (arc_generator) — jamais d'images, jamais de ré-analyse VLM.
    arc_raw: Optional[Dict[str, Any]] = None
    if arc_generator is not None and vlm_results:
        try:
            arc_raw = arc_generator(vlm_results, fusion)
        except Exception:
            arc_raw = None  # ne JAMAIS planter le rapport à cause de l'arc
    valid_ids = {int(sc["segment_id"]) for sc in scenes}
    arc = _normalize_arc_fields(arc_raw, valid_ids=valid_ids)
    report["arc_narratif"] = arc["arc_narratif"]
    report["arc_emotionnel"] = arc["arc_emotionnel"]
    report["resume_humain"] = arc["resume_humain"]
    report["arc_confidence"] = arc["arc_confidence"]
    report["arc_evidence"] = arc["arc_evidence"]
    validation_block = validation or {
        "status": "complete",
        "next_step": None,
        "expected_segment_count": len(scenes),
        "valid_segment_count": len(scenes),
        "invalid_segment_count": 0,
        "missing_segment_count": 0,
        "expected_segments": sorted(vlm_results),
        "valid_segments": sorted(vlm_results),
        "invalid_segments": [],
        "missing_segments": [],
        "extra_segments": [],
        "duplicate_segments": [],
        "errors_by_segment": {},
        "warnings_by_segment": {},
        "global_errors": [],
        "global_warnings": [],
    }
    report["status"] = validation_block.get("status", "failed")
    report["validation"] = validation_block
    report["next_step"] = validation_block.get("next_step")

    # Genre (correction #3) : fondu dans le MÊME appel arc. On écrase la VALEUR du
    # placeholder aggregate["genre"] (ordre des clés inchangé). Liste de str dédupliquée.
    genre = _dedup_preserve(
        [g for g in _as_list((arc_raw or {}).get("genre")) if isinstance(g, str) and g.strip()]
    )
    report["aggregate"]["genre"] = genre

    # Résumé global lisible (une phrase de synthèse + le bloc détaillé).
    report["global_summary"] = (
        f"Vidéo en {language or 'langue inconnue'} de "
        f"{_fmt_ts(total_duration)} s en {len(scenes)} scène(s) : "
        + " ".join(_truncate(sc["scene_description"], 160) for sc in scenes)
    ).strip()
    report["readable_summary"] = _build_readable_summary(report)
    return report


def build_final_report_from_files(
    results_dir: str,
    fusion_manifest_path: str,
    summary_path: Optional[str] = None,
    include_arc: bool = True,
    model: Optional[str] = None,
    inspection_path: Optional[str] = None,
    scene_audio_path: Optional[str] = None,
    audio_features_path: Optional[str] = None,
    vlm_pack_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Charge les artefacts depuis le disque et construit le rapport vidéo final.

    Si `include_arc` (défaut), câble le vrai générateur d'arc narratif (UN appel LLM
    texte via Gemini, sans image). Sinon, les champs d'arc restent vides
    et aucun appel modèle n'est fait (mode 100% hors-ligne).

    L'inspection technique et l'audio par scène sont chargés depuis `inspection_path` /
    `scene_audio_path` si fournis, sinon best-effort `outputs/inspection_<stem>.json` et
    `outputs/audio_events_<stem>.json` (stem dérivé du video_path de la fusion ou du
    results_dir) ; absents -> champs correspondants null/[]/bloc audio vide (jamais
    d'invention). Le texte à l'écran provient du VLM (pas d'artefact OCR séparé).
    """
    fusion = load_json(fusion_manifest_path, "manifest de fusion")
    expected_manifest = fusion
    if vlm_pack_path is not None:
        expected_manifest = load_json(vlm_pack_path, "pack VLM")

    # Toujours recalculer depuis les fichiers actuels. Le résumé enregistré n'est
    # jamais une source d'autorité et sert uniquement au contrôle d'obsolescence.
    try:
        current_validation, vlm_results = validate_results_dir_with_data(
            results_dir, expected_manifest
        )
    except VlmResultValidationError as exc:
        raise FinalReportError(f"Validation VLM impossible : {exc}") from exc
    provided_summary = None
    if summary_path is not None and Path(summary_path).exists():
        provided_summary = load_json(summary_path, "résumé de validation")
    current_validation["summary_compatibility"] = compare_validation_summary(
        provided_summary, current_validation
    )
    summary = current_validation

    # Stem pour la résolution best-effort des artefacts annexes.
    stem = None
    vp = fusion.get("video_path")
    if isinstance(vp, str) and vp:
        stem = Path(vp).stem
    if not stem:
        stem = Path(results_dir).name
    base_dir = Path(fusion_manifest_path).parent

    insp_path = inspection_path
    if insp_path is None and stem:
        cand = base_dir / f"inspection_{stem}.json"
        if cand.exists():
            insp_path = str(cand)
    inspection = load_json(insp_path, "inspection") if insp_path and Path(insp_path).exists() else None

    sa_path = scene_audio_path
    if sa_path is None and stem:
        cand = base_dir / f"audio_events_{stem}.json"
        if cand.exists():
            sa_path = str(cand)
    # Gracieux : artefact absent/invalide -> scene_audio None (rapport construit sans audio).
    scene_audio = None
    if sa_path and Path(sa_path).exists():
        try:
            scene_audio = load_json(sa_path, "scene_audio")
        except FinalReportError:
            scene_audio = None

    af_path = audio_features_path
    if af_path is None and stem:
        cand = base_dir / f"audio_features_{stem}.json"
        if cand.exists():
            af_path = str(cand)
    # Gracieux : artefact absent/invalide -> audio_features None (rapport sans features).
    audio_features = None
    if af_path and Path(af_path).exists():
        try:
            audio_features = load_json(af_path, "audio_features")
        except FinalReportError:
            audio_features = None

    # Provenance (traçabilité) : backfill best-effort depuis les artefacts EN CACHE.
    def _load_opt(path: Optional[str]) -> Optional[Dict[str, Any]]:
        if path and Path(path).exists():
            try:
                return load_json(path, "artefact")
            except FinalReportError:
                return None
        return None

    fm_path = str(base_dir / f"frames_manifest_{stem}.json") if stem else None
    mm_path = str(base_dir / f"motion_manifest_{stem}.json") if stem else None
    frames_manifest = _load_opt(fm_path)
    motion_manifest = _load_opt(mm_path)
    pipeline_status = _load_opt(str(base_dir / f"pipeline_status_{stem}.json")) if stem else None
    run_plan = _load_opt(str(base_dir / "vlm_runs" / stem / "run_plan.json")) if stem else None
    # Le VLM visuel de CE pipeline est Gemini (étape vlm_analysis via google-genai).
    vlm_mode = "gemini"
    provider = (run_plan or {}).get("provider") or "gemini"
    provenance = _build_provenance(
        vlm_mode=vlm_mode, provider=provider,
        scene_audio=scene_audio, audio_features=audio_features,
        summary=summary, frames_manifest=frames_manifest, motion_manifest=motion_manifest,
        pipeline_status=pipeline_status,
        frames_manifest_path=fm_path if frames_manifest else None,
        motion_manifest_path=mm_path if motion_manifest else None,
    )

    sources = {
        "results_dir": results_dir,
        "fusion_manifest": fusion_manifest_path,
        "vlm_pack": vlm_pack_path,
        "validation_summary": summary_path,
    }
    arc_generator = None
    if include_arc and vlm_results:
        from .narrative_arc import generate_narrative_arc

        def arc_generator(results, fus):  # noqa: E731 - closure lisible
            return generate_narrative_arc(results, fus, model=model)
    return build_final_report(
        vlm_results, fusion, summary, sources,
        inspection=inspection, scene_audio=scene_audio,
        audio_features=audio_features, vlm_mode=vlm_mode, provenance=provenance,
        validation=current_validation,
        arc_generator=arc_generator,
    )
