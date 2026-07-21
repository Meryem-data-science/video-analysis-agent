"""Correction #1 — Arc narratif du rapport vidéo final.

Génère un ARC NARRATIF (situation initiale → climax → dénouement), une trajectoire
émotionnelle, un résumé humain, une confiance et des preuves (segment_id), à partir
des SCÈNES DÉJÀ VALIDÉES (descriptions, mood, paroles, timecodes). UN SEUL appel LLM
texte via **Gemini** (google-genai) — AUCUNE image, AUCUNE ré-analyse VLM, AUCUN Claude.

Robuste : si Gemini est indisponible/non configuré ou la réponse inexploitable, renvoie
un arc vide (champs présents mais vides) — ne fait JAMAIS échouer le rapport.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


def build_arc_schema() -> Dict[str, Any]:
    """json_schema de l'arc — même style que build_output_schema (vlm_schema)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "arc_narratif", "arc_emotionnel", "resume_humain",
            "arc_confidence", "arc_evidence", "genre",
        ],
        "properties": {
            "arc_narratif": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "situation_initiale", "element_declencheur",
                    "developpement", "climax", "denouement",
                ],
                "properties": {
                    "situation_initiale": {"type": "string"},
                    "element_declencheur": {"type": "string"},
                    "developpement": {"type": "string"},
                    "climax": {"type": "string"},
                    "denouement": {"type": "string"},
                },
            },
            "arc_emotionnel": {"type": "string"},
            "resume_humain": {"type": "string"},
            "arc_confidence": {"type": "number"},
            "arc_evidence": {"type": "array", "items": {"type": "integer"}},
            "genre": {"type": "array", "items": {"type": "string"}},
        },
    }


_ARC_SUBKEYS = (
    "situation_initiale", "element_declencheur", "developpement", "climax", "denouement",
)


def _empty_arc() -> Dict[str, Any]:
    """Arc vide mais structurellement complet (tous les champs présents/typés)."""
    return {
        "arc_narratif": {k: "" for k in _ARC_SUBKEYS},
        "arc_emotionnel": "",
        "resume_humain": "",
        "arc_confidence": 0.0,
        "arc_evidence": [],
        "genre": [],
    }


def _as_str(x: Any) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _as_str_list(x: Any) -> List[str]:
    """Liste de chaînes nettoyée et dédupliquée (insensible casse), ordre préservé."""
    if not isinstance(x, list):
        return []
    seen, out = set(), []
    for i in x:
        if isinstance(i, str):
            v = i.strip()
            if v and v.casefold() not in seen:
                seen.add(v.casefold())
                out.append(v)
    return out


def _as_int_list(x: Any) -> List[int]:
    out: List[int] = []
    if isinstance(x, list):
        for i in x:
            if isinstance(i, bool):
                continue
            if isinstance(i, int):
                out.append(i)
            elif isinstance(i, float) and i.is_integer():
                out.append(int(i))
    return out


def _coerce_arc(raw: Any) -> Dict[str, Any]:
    """Normalise une réponse brute en arc structurellement complet."""
    if not isinstance(raw, dict):
        return _empty_arc()
    arc_src = raw.get("arc_narratif") if isinstance(raw.get("arc_narratif"), dict) else {}
    conf = raw.get("arc_confidence")
    conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.0
    return {
        "arc_narratif": {k: _as_str(arc_src.get(k)) for k in _ARC_SUBKEYS},
        "arc_emotionnel": _as_str(raw.get("arc_emotionnel")),
        "resume_humain": _as_str(raw.get("resume_humain")),
        "arc_confidence": conf,
        "arc_evidence": _as_int_list(raw.get("arc_evidence")),
        "genre": _as_str_list(raw.get("genre")),
    }


def _ordered_scene_digest(
    vlm_results: Dict[int, Dict[str, Any]], fusion: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Résumé ordonné (chronologique) des scènes : entrée du prompt de l'arc."""
    if not isinstance(vlm_results, dict) or not vlm_results:
        return []
    segs = fusion.get("segments", []) if isinstance(fusion, dict) else []
    fusion_by_id: Dict[int, Dict[str, Any]] = {}
    for s in segs or []:
        try:
            fusion_by_id[int(s.get("segment_id"))] = s
        except (TypeError, ValueError):
            continue
    digest: List[Dict[str, Any]] = []
    for seg_id in sorted(vlm_results):
        vlm = vlm_results[seg_id] if isinstance(vlm_results[seg_id], dict) else {}
        fseg = fusion_by_id.get(seg_id, {})
        digest.append({
            "segment_id": int(seg_id),
            "start_s": fseg.get("start_s"),
            "end_s": fseg.get("end_s"),
            "scene_description": _as_str(vlm.get("scene_description")),
            "people_objects": [x for x in (vlm.get("people_objects") or []) if isinstance(x, str)],
            "actions": [x for x in (vlm.get("actions") or []) if isinstance(x, str)],
            "mood": [x for x in (vlm.get("mood") or []) if isinstance(x, str)],
            "combined_text": _as_str(fseg.get("combined_text")),
        })
    return digest


_ARC_INSTRUCTIONS = (
    "Tu es un analyste vidéo. À partir UNIQUEMENT du résumé ordonné des scènes ci-dessous "
    "(déjà analysées et validées), reconstitue l'ARC NARRATIF de la vidéo.\n\n"
    "RÈGLES STRICTES (anti-hallucination) :\n"
    "- Fonde-toi EXCLUSIVEMENT sur les informations fournies. N'INVENTE RIEN : aucun "
    "personnage, lieu, action, dialogue ou émotion qui ne figure pas dans les scènes.\n"
    "- Si une information manque ou n'est pas applicable, laisse le champ vide (\"\" pour "
    "les textes, [] pour les listes). Ne comble JAMAIS un vide par une supposition.\n"
    "- Rédige tous les champs descriptifs EN FRANÇAIS. NE TRADUIS PAS les paroles citées "
    "(« Parole »), le texte à l'écran cité verbatim, ni les noms propres (personnes, "
    "marques, clubs, équipes, lieux) : garde-les dans leur langue d'origine.\n"
    "- ATTRIBUTION DE LA PAROLE : il n'existe PAS de diarisation (identification des "
    "locuteurs). N'attribue JAMAIS une parole à un personnage précis (n'écris pas "
    "« X dit … »). Écris plutôt « on entend « … » ». De plus, si une scène signale que la "
    "parole ne correspond pas clairement au visuel ou a une confiance faible, NE la "
    "présente PAS comme un fait établi — reste prudent (« une parole peu claire « … » est "
    "audible »).\n\n"
    "CHAMPS À PRODUIRE :\n"
    "- arc_narratif.situation_initiale : l'état de départ.\n"
    "- arc_narratif.element_declencheur : ce qui lance l'action.\n"
    "- arc_narratif.developpement : la progression.\n"
    "- arc_narratif.climax : le point culminant.\n"
    "- arc_narratif.denouement : la résolution / fin.\n"
    "- arc_emotionnel : UNE phrase décrivant la trajectoire émotionnelle (d'après les mood).\n"
    "- resume_humain : UN seul paragraphe fluide qui raconte la vidéo du début à la fin.\n"
    "- arc_confidence : nombre entre 0 et 1 (ta confiance globale dans cet arc).\n"
    "- arc_evidence : liste des segment_id (entiers) des scènes qui justifient l'arc.\n"
    "- genre : liste courte de genres applicables, étiquettes ATOMIQUES (ex : humour, "
    "sport, famille, documentaire, drame, action, tranche de vie/quotidien, scène de "
    "rue...), fondée UNIQUEMENT sur le contenu des scènes. [] si indéterminable. "
    "N'INVENTE PAS de genre non étayé. N'applique une étiquette QUE si elle décrit "
    "fidèlement le contenu observé. N'utilise PAS une étiquette d'exemple qui ne "
    "correspond pas (ex : ne mets PAS « scène de rue » pour une scène d'intérieur).\n\n"
    "RÉPONDS UNIQUEMENT EN JSON VALIDE conforme au schéma, sans markdown ni texte autour.\n"
)


def _fmt_ts(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def build_arc_prompt(scenes: List[Dict[str, Any]], language: Optional[str]) -> str:
    """Construit le prompt texte (instructions + digest ordonné des scènes)."""
    lines: List[str] = [_ARC_INSTRUCTIONS]
    lines.append(f"Langue de la vidéo : {language or 'inconnue'}")
    lines.append(f"Nombre de scènes : {len(scenes)}\n")
    lines.append("--- SCÈNES (ordre chronologique) ---")
    for sc in scenes:
        lines.append(
            f"\nScène {sc['segment_id']} "
            f"[{_fmt_ts(sc['start_s'])}–{_fmt_ts(sc['end_s'])}s]"
        )
        if sc["scene_description"]:
            lines.append(f"  Description : {sc['scene_description']}")
        if sc["people_objects"]:
            lines.append(f"  Personnages/objets : {', '.join(sc['people_objects'])}")
        if sc["actions"]:
            lines.append(f"  Actions : {', '.join(sc['actions'])}")
        if sc["mood"]:
            lines.append(f"  Émotion (mood) : {', '.join(sc['mood'])}")
        if sc["combined_text"]:
            lines.append(f"  Parole : {sc['combined_text']}")
    return "\n".join(lines)


def generate_narrative_arc(
    vlm_results: Dict[int, Dict[str, Any]],
    fusion: Dict[str, Any],
    model: Optional[str] = None,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    """Génère l'arc narratif via UN appel LLM texte. Renvoie un arc vide si échec."""
    scenes = _ordered_scene_digest(vlm_results, fusion)
    if not scenes:
        return _empty_arc()
    language = fusion.get("language") if isinstance(fusion, dict) else None
    prompt = build_arc_prompt(scenes, language)
    schema = build_arc_schema()
    try:
        from .gemini_text_runner import analyze_text_with_gemini
        raw = analyze_text_with_gemini(prompt, schema, model=model, timeout_s=timeout_s)
    except Exception as exc:  # pragma: no cover - dépend du SDK/réseau
        logging.warning("Arc narratif non généré (%s) : %s", type(exc).__name__, exc)
        return _empty_arc()
    return _coerce_arc(raw)
