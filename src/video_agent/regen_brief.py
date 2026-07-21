"""Regeneration brief — seconde analyse SUR le rapport final (non destructif).

Lit le rapport final (`final_report_<id>.json`, schema 1.10) + le transcript + la
vidéo source, et produit un « regen brief » : redécoupage en clips ≤ 8 s, et pour
chaque clip tout ce qu'il faut pour le régénérer dans Google Flow puis tout remonter.

GARANTIE NON DESTRUCTIVE : lecture seule sur toutes les entrées ; n'écrit QUE
``outputs/regen_brief_<id>.json``, ``outputs/regen_worksheet_<id>.md`` et le dossier
``outputs/regen_frames_<id>/``. Les mtimes des entrées sont vérifiés inchangés en fin
de run. Aucun appel modèle/API : agrégation déterministe (+ ffmpeg pour les frames).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Paramètres (en tête de fichier, comme demandé) --------------------------------
MAX_CLIP_S = 8.0
MIN_CLIP_S = 2.0
SPLIT_MODE = "scene_aware"  # alternative documentée : "grid" (blocs rigides 0-8,8-16…)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"


class RegenBriefError(Exception):
    """Erreur de construction du regen brief."""


# --- Helpers généraux ---------------------------------------------------------------


def _strip_accents_lower(text: str) -> str:
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RegenBriefError(f"Fichier {label} introuvable : {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegenBriefError(f"JSON {label} invalide ({path}) : {exc}") from exc


def _aspect_ratio(w: Optional[int], h: Optional[int]) -> str:
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        return "unknown"
    g = math.gcd(w, h)
    return f"{w // g}:{h // g}"


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    return float(x) if isinstance(x, (int, float)) else None


# --- Registre d'assets (canonicalisation des personnages/objets) -------------------

_FR_COLOR = {
    "rouge": "red", "bleu": "blue", "bleue": "blue", "vert": "green", "verte": "green",
    "noir": "black", "noire": "black", "blanc": "white", "blanche": "white",
    "jaune": "yellow",
}
_COLOR_HEX = {
    "white": "#FFFFFF", "red": "#E53935", "blue": "#1E88E5",
    "green": "#43A047", "black": "#111111", "yellow": "#FDD835",
}
# type interne -> (nom canonique EN, gabarit de descripteur EN)
_TYPE_INFO = {
    "capsule": ("capsule", "glossy {c}medical capsule, hollow when open, floating on white"),
    "laptop": ("laptop", "modern laptop computer shown front-on on white"),
    "usb_device": ("USB device", "small glossy {c}USB-like device"),
    "robotic_arms": ("robotic arms", "minimalist robotic assembly arms entering from the top"),
    "rocket": ("rocket", "stylized {c}rocket assembled on a launch platform"),
    "platform": ("platform", "clean launch/assembly platform"),
    "animated_text": ("animated text", "animated English caption text appearing word by word"),
    "progress_bar": ("progress bar", "{c}vertical fill/progress bar"),
    "logo": ("logo", "small abstract black logo mark"),
}
# rôle sémantique par couleur (color_semantics)
_COLOR_ROLE = {"red": "volume/visibility", "blue": "design/quality", "green": "synthesis"}


def _entity_key(name: str) -> Tuple[str, Optional[str], str]:
    """Retourne (asset_id, color, type) canoniques pour un libellé people_objects."""
    n = _strip_accents_lower(name)
    color = None
    for fr, en in _FR_COLOR.items():
        if re.search(r"\b" + fr, n):
            color = en
            break
    if "pilule" in n or "capsule" in n:
        typ = "capsule"
    elif "ordinateur" in n or "laptop" in n or "portable" in n:
        typ = "laptop"
    elif "usb" in n or "appareil" in n:
        typ = "usb_device"
    elif "bras" in n and "robot" in n:
        typ = "robotic_arms"
    elif "fusee" in n:
        typ = "rocket"
    elif "plateforme" in n:
        typ = "platform"
    elif "texte" in n:
        typ = "animated_text"
    elif "barre" in n:
        typ = "progress_bar"
    elif "logo" in n:
        typ = "logo"
    else:
        typ = (re.sub(r"[^a-z0-9]+", "_", n).strip("_")[:24]) or "object"
    parts = [p for p in (color, typ) if p]
    asset_id = "ASSET_" + "_".join(parts).upper()
    return asset_id, color, typ


def _descriptor(color: Optional[str], typ: str) -> Tuple[str, str, str]:
    """Retourne (canonical_name, descriptor, role) en anglais."""
    base_name, tmpl = _TYPE_INFO.get(typ, (typ.replace("_", " "), "{c}3D object on white"))
    cprefix = f"{color} " if color else ""
    canonical_name = f"{color} {base_name}" if color else base_name
    descriptor = tmpl.format(c=cprefix)
    role = _COLOR_ROLE.get(color or "", "")
    return canonical_name, descriptor, role


def build_asset_registry(
    scenes: List[Dict[str, Any]], clips: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Dédoublonne tous les people_objects en entités canoniques avec appears_in_clips."""
    scene_to_clips: Dict[int, List[str]] = {}
    for c in clips:
        scene_to_clips.setdefault(int(c["source_scene_id"]), []).append(c["clip_id"])

    registry: Dict[str, Dict[str, Any]] = {}
    scene_assets: Dict[int, List[str]] = {}
    for sc in scenes:
        sid = int(sc["segment_id"])
        ids_here: List[str] = []
        for name in sc.get("people_objects", []) or []:
            if not isinstance(name, str) or not name.strip():
                continue
            asset_id, color, typ = _entity_key(name)
            if asset_id not in registry:
                cname, desc, role = _descriptor(color, typ)
                registry[asset_id] = {
                    "asset_id": asset_id, "canonical_name": cname,
                    "descriptor": desc, "role": role,
                    "appears_in_clips": [], "_scenes": set(),
                }
            registry[asset_id]["_scenes"].add(sid)
            if asset_id not in ids_here:
                ids_here.append(asset_id)
        scene_assets[sid] = ids_here

    for asset in registry.values():
        clip_ids: List[str] = []
        for sid in sorted(asset["_scenes"]):
            clip_ids += scene_to_clips.get(sid, [])
        asset["appears_in_clips"] = clip_ids
        del asset["_scenes"]

    # mémorise les assets par scène pour les sujets de clip.
    build_asset_registry._scene_assets = scene_assets  # type: ignore[attr-defined]
    return list(registry.values())


# --- Grille de clips -----------------------------------------------------------------


def build_clip_grid(scenes: List[Dict[str, Any]], total_duration_s: float) -> List[Dict[str, Any]]:
    """Découpe scene-aware en clips ≤ MAX_CLIP_S ; scènes < MIN -> static_card."""
    clips: List[Dict[str, Any]] = []
    idx = 1
    for sc in scenes:
        sid = int(sc["segment_id"])
        start = _num(sc.get("start_s")) or 0.0
        end = _num(sc.get("end_s"))
        if end is None:
            continue
        d = round(end - start, 3)
        if d < MIN_CLIP_S:
            clips.append({
                "clip_id": f"clip_{idx:02d}", "source_scene_id": sid,
                "start_s": round(start, 3), "end_s": round(end, 3),
                "target_duration_s": d, "type": "static_card",
            })
            idx += 1
        elif d <= MAX_CLIP_S:
            clips.append({
                "clip_id": f"clip_{idx:02d}", "source_scene_id": sid,
                "start_s": round(start, 3), "end_s": round(end, 3),
                "target_duration_s": d, "type": "generated",
            })
            idx += 1
        else:
            n = math.ceil(d / MAX_CLIP_S)
            step = d / n
            for i in range(n):
                cs = start + i * step
                ce = end if i == n - 1 else start + (i + 1) * step
                clips.append({
                    "clip_id": f"clip_{idx:02d}", "source_scene_id": sid,
                    "start_s": round(cs, 3), "end_s": round(ce, 3),
                    "target_duration_s": round(ce - cs, 3), "type": "generated",
                })
                idx += 1
    return clips


# --- Transitions / raccords ----------------------------------------------------------


def assign_transitions(clips: List[Dict[str, Any]]) -> None:
    """Renseigne transition_from_previous + start_frame_source (raccord remontage)."""
    for i, c in enumerate(clips):
        if i == 0:
            c["transition_from_previous"] = "none"
            c["start_frame_source"] = "extracted"
            continue
        prev = clips[i - 1]
        if c["source_scene_id"] == prev["source_scene_id"] and c["type"] == "generated":
            c["transition_from_previous"] = "continuation"
            c["start_frame_source"] = f"{prev['clip_id']}.end_frame"
        else:
            c["transition_from_previous"] = "hard_cut"
            c["start_frame_source"] = "extracted"


# --- Dialogue (transcript timestampé) -----------------------------------------------


def assign_dialogue(
    clips: List[Dict[str, Any]], transcript: Optional[Dict[str, Any]],
    scenes_by_id: Dict[int, Dict[str, Any]], voice_profile_id: str,
) -> bool:
    """Assigne le dialogue par clip. Retourne True si timestamps exacts utilisés."""
    segs = (transcript or {}).get("segments") if isinstance(transcript, dict) else None
    exact = isinstance(segs, list) and len(segs) > 0
    if exact:
        for c in clips:
            texts: List[str] = []
            for s in segs:
                a, b = _num(s.get("start_s")), _num(s.get("end_s"))
                if a is None or b is None:
                    continue
                mid = (a + b) / 2.0
                if c["start_s"] <= mid < c["end_s"]:
                    t = s.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
            c["dialogue"] = {
                "text": " ".join(texts), "language": "en",
                "approximate": False, "voice_profile_id": voice_profile_id,
            }
        return True
    # Repli : répartition proportionnelle du combined_text de la scène.
    for c in clips:
        sc = scenes_by_id.get(int(c["source_scene_id"]), {})
        combined = (sc.get("combined_text") or "").strip()
        c["dialogue"] = {
            "text": combined, "language": "en", "approximate": True,
            "voice_profile_id": voice_profile_id,
            "note": "Pas de transcript timestampé : combined_text de la scène (approximatif).",
        }
    return False


# --- Texte écran (sanitisation) -----------------------------------------------------

_DROP_PATTERNS = [
    re.compile(r"^F\d+\b"),                     # ex. "F26 35.325s"
    re.compile(r"^\s*[\d.]+\s*s\s*$"),          # ex. "35.325s"
    re.compile(r"^\s*\d+\s*$"),                 # numéro nu
]


def _is_technical_label(text: str) -> bool:
    t = (text or "").strip()
    return any(p.search(t) for p in _DROP_PATTERNS)


def sanitize_on_screen_text(
    scene: Dict[str, Any], dropped_log: List[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in scene.get("texte_ecran", []) or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if _is_technical_label(text):
            dropped_log.append(text.strip())
            continue
        out.append({"text": text.strip(), "location": item.get("location") or ""})
    return out


# --- Audio : voix & musique ----------------------------------------------------------


def detect_voice_gender(scenes: List[Dict[str, Any]]) -> str:
    blob = " ".join(
        _strip_accents_lower((sc.get("audio_scene") or {}).get("events", {}).get("evidence", ""))
        for sc in scenes
    )
    if "feminin" in blob or "femme" in blob:
        return "female"
    if "masculin" in blob or "homme" in blob:
        return "male"
    return "unknown"


def _music_of_scene(scene: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    av = scene.get("audio_scene") or {}
    events = av.get("events") if isinstance(av.get("events"), dict) else {}
    music = events.get("music") if isinstance(events.get("music"), dict) else {}
    mf = av.get("measured_features") if isinstance(av.get("measured_features"), dict) else {}
    return music, mf


def _music_prompt_en(mood_fr: Optional[str], bpm: Optional[float]) -> str:
    bpm_s = f", ~{int(round(bpm))} BPM" if isinstance(bpm, (int, float)) else ""
    return f"minimal corporate synth, calm professional{bpm_s}"


# --- Frames de référence (ffmpeg) ---------------------------------------------------


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_frame(video: Path, t: float, out: Path, notes: List[str]) -> bool:
    """Extrait 1 frame au timestamp t. Retourne True si le fichier est créé."""
    t = max(0.0, t)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(out)]
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True, errors="replace")
    except FileNotFoundError:
        notes.append("ffmpeg introuvable : frames non extraites.")
        return False
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        notes.append(f"Extraction échouée @ {t:.3f}s ({out.name}).")
        return False
    return True


def _file_sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


# --- Flow prompt (bloc unique copier-coller, EN) ------------------------------------

_ACTION_LEX = [
    ("flott", "floating"), ("ouvert", "opening"), ("ouvr", "opening"),
    ("apparition", "appearing"), ("apparait", "appearing"), ("estomp", "fading out"),
    ("connexion", "connecting devices"), ("connect", "connecting"),
    ("assembl", "assembling"), ("apporte", "bringing parts in"),
    ("remplissage", "filling up"), ("surlign", "highlighting words"),
    ("positionn", "positioning"), ("statique", "static"), ("immobile", "static"),
    ("affich", "being displayed"), ("balayage", "sweeping"),
]


def _scenario_en(scene: Dict[str, Any], subj_names: List[str]) -> str:
    blob = _strip_accents_lower(
        " ".join(scene.get("actions", []) or []) + " " + (scene.get("motion_description") or "")
    )
    verbs: List[str] = []
    for key, en in _ACTION_LEX:
        if key in blob and en not in verbs:
            verbs.append(en)
        if len(verbs) >= 4:
            break
    subj = ", ".join(subj_names[:4]) if subj_names else "the elements"
    if verbs:
        return f"the {subj} {', '.join(verbs)}"
    return f"the {subj} are presented on screen"


def _camera_en(scene: Dict[str, Any]) -> str:
    blob = _strip_accents_lower(scene.get("motion_description") or "")
    if "statique" in blob or "immobile" in blob:
        return "static locked-off framing"
    return "static front framing, slow push-in"


def _background_en(scene: Dict[str, Any]) -> str:
    """Fond vert UNIQUEMENT si la description parle du FOND vert (pas d'un objet vert)."""
    blob = _strip_accents_lower(scene.get("scene_description") or "")
    # « fond/arrière-plan/background … vert/green » à courte distance, ou « vert … fond ».
    green_bg = (
        re.search(r"(fond|arriere[- ]?plan|background)[^.]{0,30}\b(vert|green)", blob)
        or re.search(r"\b(vert|green)[^.]{0,15}(fond|arriere[- ]?plan|background)", blob)
    )
    return "light green background" if green_bg else "pure white background"


# Consigne négative : Flow ne génère QUE l'image muette (le reste est ajouté au montage).
_NEGATIVE_DIRECTIVE = (
    "No on-screen text, no captions, no subtitles, no UI, no logos. "
    "No spoken words, no voice, no music, no sound effects. Silent footage only."
)


def _build_flow_prompt(
    clip: Dict[str, Any], scene: Dict[str, Any], registry_by_id: Dict[str, Dict[str, Any]],
    aspect: str, rendering: str,
) -> str:
    """flow_prompt VISUEL UNIQUEMENT : personnages + décor + mouvement + caméra + style +
    format + durée, suivi d'une consigne négative (pas de texte/voix/musique/SFX).

    Le texte écran, la voix, la musique et les effets sonores sont gérés au montage
    (cf. clip["post_production"] et brief["post_production_global"]).
    """
    subj_desc = "; ".join(
        registry_by_id[a]["descriptor"] for a in clip["subjects"][:4] if a in registry_by_id
    )
    scenario = clip["scenario"]
    camera = clip["camera"]
    bg = clip["style_tokens"]["background"]
    lighting = clip["style_tokens"]["lighting"]
    parts: List[str] = [
        f"Vertical {aspect} video, {rendering}.",
        f"{scenario[:1].upper()}{scenario[1:]}." if scenario else "",
    ]
    if subj_desc:
        parts.append(f"Elements: {subj_desc}.")
    parts.append(f"Camera: {camera}; {bg}, {lighting}.")
    parts.append(f"{aspect} vertical, about {int(round(clip['target_duration_s']))} seconds.")
    parts.append(_NEGATIVE_DIRECTIVE)
    return " ".join(p for p in parts if p)


# --- Construction principale ---------------------------------------------------------


def build_regen_brief(
    video_id: str, outputs_dir: Optional[Path] = None, *, now_iso: Optional[str] = None
) -> Dict[str, Any]:
    """Construit le regen brief complet et écrit les 3 sorties. Retourne le rapport d'auto-test."""
    out_dir = Path(outputs_dir) if outputs_dir else OUTPUTS
    report_path = out_dir / f"final_report_{video_id}.json"
    report = _load_json(report_path, "rapport final")
    if report.get("report_type") != "final_video_report":
        raise RegenBriefError(f"report_type inattendu : {report.get('report_type')!r}")

    # Entrées annexes (lecture seule ; transcript/motion/frames optionnels).
    input_paths: List[Path] = [report_path]
    transcript_path = out_dir / f"transcript_{video_id}.json"
    transcript = _load_json(transcript_path, "transcript") if transcript_path.exists() else None
    if transcript_path.exists():
        input_paths.append(transcript_path)
    for opt in (f"frames_manifest_{video_id}.json", f"motion_manifest_{video_id}.json"):
        p = out_dir / opt
        if p.exists():
            input_paths.append(p)

    # Vidéo source (depuis le rapport, repli vidéos/<id>.mp4).
    video = Path(report.get("video_path", ""))
    if not video.exists():
        for cand in (PROJECT_ROOT / "vidéos" / f"{video_id}.mp4",
                     PROJECT_ROOT / "videos" / f"{video_id}.mp4"):
            if cand.exists():
                video = cand
                break
    video_ok = video.exists()
    if video_ok:
        input_paths.append(video)

    # mtimes AVANT (vérif non-destructif).
    mtimes_before = {str(p): p.stat().st_mtime_ns for p in input_paths}

    meta = report.get("metadata", {}) or {}
    res = meta.get("resolution", {}) or {}
    width, height = res.get("width"), res.get("height")
    duration_s = _num(report.get("total_duration_s")) or 0.0
    fps = _num(meta.get("fps"))
    aspect = _aspect_ratio(width if isinstance(width, int) else None,
                           height if isinstance(height, int) else None)
    language = report.get("language") or "en"

    scenes = report.get("scenes", []) or []
    scenes_by_id = {int(s["segment_id"]): s for s in scenes if "segment_id" in s}

    # 1) Grille de clips + 2) transitions.
    clips = build_clip_grid(scenes, duration_s)
    assign_transitions(clips)

    # 3) Registre d'assets + sujets par clip.
    registry = build_asset_registry(scenes, clips)
    registry_by_id = {a["asset_id"]: a for a in registry}
    scene_assets = getattr(build_asset_registry, "_scene_assets", {})
    for c in clips:
        c["subjects"] = list(scene_assets.get(int(c["source_scene_id"]), []))

    # 4) Voix (mutualisée) + 5) musique globale.
    gender = detect_voice_gender(scenes)
    overall_tone = "informative, professional"
    voice_profiles = [{
        "voice_profile_id": "vp_main", "gender": gender, "language": language,
        "tone": overall_tone, "pace": "moderate",
    }]
    # bpm médian mesuré pour la piste globale.
    bpms = [v for v in (_num(_music_of_scene(s)[1].get("tempo_bpm")) for s in scenes) if v]
    global_bpm = round(sorted(bpms)[len(bpms) // 2]) if bpms else None
    music_present_scenes = sum(1 for s in scenes if _music_of_scene(s)[0].get("present") is True)
    continuous = music_present_scenes >= max(1, len(scenes) - 1)
    global_music_prompt = _music_prompt_en("calm professional", global_bpm)
    music_tracks = [{
        "track_id": "mt_main", "continuous": continuous, "mood": "calm professional",
        "bpm": global_bpm, "energy": "medium", "music_prompt": global_music_prompt,
    }]

    rendering = "clean 3D motion design, glossy objects, generous negative space"
    palette = ["#FFFFFF"]
    seen_colors = {c for a in registry for c in [a["asset_id"]] }  # placeholder
    colors_present = set()
    for a in registry:
        for fr, en in _FR_COLOR.items():
            if en in a["asset_id"].lower():
                colors_present.add(en)
    for col in ("red", "blue", "green", "black", "yellow"):
        if col in colors_present and _COLOR_HEX[col] not in palette:
            palette.append(_COLOR_HEX[col])
    moods_first = [(_first_mood(s)) for s in scenes if _first_mood(s)]
    mood_arc = " → ".join(dict.fromkeys(moods_first)) if moods_first else ""
    global_style = {
        "palette_hint": palette, "rendering": rendering,
        "color_semantics": {"red": "volume/visibility", "blue": "design/quality", "green": "synthesis"},
        "mood_arc": mood_arc,
    }

    # 6) Dialogue.
    dialogue_exact = assign_dialogue(clips, transcript, scenes_by_id, "vp_main")

    # Per-clip : on_screen_text, music, sfx, scenario, camera, style, flow_prompt.
    dropped_log: List[str] = []
    audio_false_neg: List[str] = []
    prev_music: Optional[Dict[str, Any]] = None
    for c in clips:
        sc = scenes_by_id.get(int(c["source_scene_id"]), {})
        # 9) texte écran sanitisé.
        c["on_screen_text"] = sanitize_on_screen_text(sc, dropped_log)
        # 7) musique.
        music, mf = _music_of_scene(sc)
        present = music.get("present")
        bpm = _num(mf.get("tempo_bpm"))
        energy_rms = _num(mf.get("energy_rms"))
        mood_fr = music.get("mood")
        if present is True:
            c_present: Any = True
        elif (present is False and isinstance(energy_rms, (int, float)) and energy_rms > 0.0
              and c["target_duration_s"] < MIN_CLIP_S and prev_music is not None):
            c_present = "inherited"
            bpm = bpm or prev_music.get("bpm")
            mood_fr = mood_fr if mood_fr and mood_fr != "incertain" else prev_music.get("mood")
            audio_false_neg.append(c["clip_id"])
        else:
            c_present = bool(present)
        c["music"] = {
            "track_id": "mt_main", "present": c_present, "mood": mood_fr,
            "bpm": bpm or global_bpm, "energy": music.get("energy"),
            "music_prompt": _music_prompt_en(mood_fr, bpm or global_bpm),
        }
        if c_present in (True, "inherited"):
            prev_music = c["music"]
        # 8) SFX.
        c["sound_effects"] = [
            {"label": s.get("label"), "confidence": s.get("confidence")}
            for s in (music and sc.get("audio_scene", {}).get("events", {}).get("sound_effects", []) or [])
            if isinstance(s, dict)
        ]
        # 5) scénario + caméra + style.
        subj_names = [registry_by_id[a]["canonical_name"] for a in c["subjects"] if a in registry_by_id]
        c["scenario"] = _scenario_en(sc, subj_names)
        c["camera"] = _camera_en(sc)
        c["style_tokens"] = {
            "background": _background_en(sc), "lighting": "soft studio", "rendering": "glossy 3D",
        }
        c["mood"] = list(sc.get("mood", []) or [])
        c["prompt_granularity"] = "scene_exact" if _scene_clip_count(clips, c) == 1 else "scene_inherited"
        # 11) flow_prompt = VISUEL UNIQUEMENT (image muette).
        c["flow_prompt"] = _build_flow_prompt(c, sc, registry_by_id, aspect, rendering)
        # Regroupement « montage » : ce qui a été retiré du flow_prompt (valeurs déjà calculées).
        c["post_production"] = {
            "on_screen_text": c["on_screen_text"],
            "voiceover_text": (c.get("dialogue") or {}).get("text", ""),
            "music": c["music"],
            "sound_effects": c["sound_effects"],
        }

    # 10) Frames de référence (ffmpeg) + raccords physiques.
    frames_dirname = f"regen_frames_{video_id}"
    frames_dir = out_dir / frames_dirname
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_notes: List[str] = []
    ffmpeg_ok = _ffmpeg_available() and video_ok
    if not ffmpeg_ok:
        frame_notes.append("ffmpeg indisponible ou vidéo absente : frames non extraites.")

    prev_end_path: Optional[Path] = None
    for c in clips:
        ref: Dict[str, Any] = {}
        cid = c["clip_id"]
        if c["type"] == "static_card":
            card = frames_dir / f"{cid}_card.jpg"
            if ffmpeg_ok:
                _extract_frame(video, (c["start_s"] + c["end_s"]) / 2.0, card, frame_notes)
            ref["card_frame"] = f"{frames_dirname}/{card.name}"
            c["reference_images"] = ref
            prev_end_path = None
            continue
        # end_frame TOUJOURS extrait (alimente le clip suivant).
        end_path = frames_dir / f"{cid}_end.jpg"
        if ffmpeg_ok:
            _extract_frame(video, max(0.0, c["end_s"] - 0.05), end_path, frame_notes)
        mid_path = frames_dir / f"{cid}_mid.jpg"
        if ffmpeg_ok:
            _extract_frame(video, (c["start_s"] + c["end_s"]) / 2.0, mid_path, frame_notes)
        start_path = frames_dir / f"{cid}_start.jpg"
        if c["transition_from_previous"] == "continuation" and prev_end_path and prev_end_path.exists():
            # raccord fluide : COPIE de la fin du clip précédent.
            shutil.copyfile(prev_end_path, start_path)
        elif ffmpeg_ok:
            _extract_frame(video, c["start_s"], start_path, frame_notes)
        ref = {
            "start_frame": f"{frames_dirname}/{start_path.name}",
            "mid_frame": f"{frames_dirname}/{mid_path.name}",
            "end_frame": f"{frames_dirname}/{end_path.name}",
        }
        c["reference_images"] = ref
        # reassembly_note pour les continuations.
        if c["transition_from_previous"] == "continuation":
            c["reassembly_note"] = (
                f"Dans Flow, utiliser {frames_dirname}/{start_path.name} comme image de "
                f"départ (= copie de la fin du clip précédent) pour un raccord fluide."
            )
        prev_end_path = end_path if (ffmpeg_ok and end_path.exists()) else None

    # Reassembly.
    reassembly = {
        "order": [c["clip_id"] for c in clips],
        "total_duration_s": round(duration_s, 3),
        "instructions": (
            "Concaténer les clips dans l'ordre. Régler la durée Flow de chaque clip sur "
            "target_duration_s. Pour une transition 'continuation', utiliser l'image de "
            "départ fournie (copie de la fin du clip précédent) comme image de départ Flow. "
            "Ajouter ENSUITE au montage : voix off, texte écran, musique et effets sonores "
            "(cf. post_production par clip et post_production_global)."
        ),
    }

    # Post-production GLOBALE : voix off complète + musique + timeline texte écran.
    segs_pg = (transcript or {}).get("segments") if isinstance(transcript, dict) else None
    if isinstance(segs_pg, list) and segs_pg:
        seg_texts = [s.get("text", "").strip() for s in segs_pg
                     if isinstance(s.get("text"), str) and s.get("text").strip()]
    else:
        seg_texts = [(c.get("dialogue") or {}).get("text", "").strip() for c in clips
                     if (c.get("dialogue") or {}).get("text", "").strip()]
    full_voiceover_script = " ".join(seg_texts)
    last_segment_text = seg_texts[-1] if seg_texts else ""
    vp = voice_profiles[0]
    mt = music_tracks[0]
    post_production_global = {
        "full_voiceover_script": full_voiceover_script,
        "voice_profile": {"gender": vp["gender"], "language": vp["language"],
                          "tone": vp["tone"], "pace": vp["pace"]},
        "music_track": {"mood": mt["mood"], "bpm": mt["bpm"],
                        "music_prompt": mt["music_prompt"], "continuous": mt["continuous"]},
        "on_screen_text_timeline": [
            {"clip_id": c["clip_id"], "start_s": c["start_s"], "end_s": c["end_s"],
             "texts": [o["text"] for o in c["on_screen_text"]]}
            for c in clips if c["on_screen_text"]
        ],
    }

    brief = {
        "brief_version": "1.1",
        "video_id": video_id,
        "source_report": report_path.as_posix(),
        "fidelity_target": "semantic_structural",
        "split": {"mode": SPLIT_MODE, "max_clip_s": MAX_CLIP_S, "min_clip_s": MIN_CLIP_S},
        "video": {
            "duration_s": round(duration_s, 3), "fps": int(fps) if fps else None,
            "resolution": {"width": width, "height": height},
            "aspect_ratio": aspect, "language": language,
        },
        "global_style": global_style,
        "voice_profiles": voice_profiles,
        "music_tracks": music_tracks,
        "asset_registry": registry,
        "reassembly": reassembly,
        "post_production_global": post_production_global,
        "clips": clips,
        "sanitization_log": {
            "dropped_on_screen_text": dropped_log,
            "audio_false_negatives_fixed": audio_false_neg,
        },
        "build": {
            "generated_at": now_iso or datetime.now(timezone.utc).isoformat(),
            "inputs_untouched": True,
            "frame_notes": frame_notes,
            "dialogue_exact": dialogue_exact,
        },
    }

    # Écritures (3 sorties).
    brief_path = out_dir / f"regen_brief_{video_id}.json"
    brief_path.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")
    worksheet_path = out_dir / f"regen_worksheet_{video_id}.md"
    worksheet_path.write_text(_render_worksheet(brief, frames_dirname), encoding="utf-8")

    # mtimes APRÈS (vérif non-destructif).
    untouched = all(p.stat().st_mtime_ns == mtimes_before[str(p)] for p in input_paths)
    brief["build"]["inputs_untouched"] = untouched
    if not untouched:
        brief_path.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    return _self_test(brief, brief_path, worksheet_path, frames_dir, mtimes_before,
                      input_paths, last_segment_text)


def _first_mood(scene: Dict[str, Any]) -> str:
    m = scene.get("mood", []) or []
    return m[0] if m and isinstance(m[0], str) else ""


def _scene_clip_count(clips: List[Dict[str, Any]], clip: Dict[str, Any]) -> int:
    return sum(1 for c in clips if c["source_scene_id"] == clip["source_scene_id"])


# --- Worksheet Markdown --------------------------------------------------------------


def _render_worksheet(brief: Dict[str, Any], frames_dirname: str) -> str:
    v = brief["video"]
    clips = brief["clips"]
    L: List[str] = []
    L.append(f"# Regeneration worksheet — {brief['video_id']}")
    L.append("")
    L.append(f"- Durée totale à retrouver : **{v['duration_s']} s** | Clips : **{len(clips)}** "
             f"| Format : **{v['aspect_ratio']}** | Langue : {v['language']}")
    L.append("- ⚠️ Dans Flow, **régler la durée de chaque clip sur la durée cible** indiquée.")
    L.append("- Workflow : aller au clip → copier le bloc → uploader les 3 images → générer "
             "→ comparer à l'original → clip suivant.")
    L.append("")
    for c in clips:
        raccord = {"none": "début", "continuation": "continuation", "hard_cut": "coupe franche"}.get(
            c.get("transition_from_previous", ""), c.get("transition_from_previous", ""))
        L.append(f"## {c['clip_id'].replace('clip_', 'Clip ')} — {c['start_s']:.2f}s → "
                 f"{c['end_s']:.2f}s (durée ~{c['target_duration_s']:.1f}s) — raccord: {raccord} "
                 f"[{c['type']}]")
        L.append("")
        L.append("**À copier-coller dans Flow (image muette) :**")
        L.append("")
        L.append("> " + c["flow_prompt"])
        L.append("")
        ref = c.get("reference_images", {})
        L.append("**Images de référence (uploader dans Flow) :**")
        if c["type"] == "static_card":
            L.append(f"- card : {ref.get('card_frame', '(non extraite)')}  ← image fixe (non générée)")
        else:
            start_note = "← image de départ Flow"
            if c.get("transition_from_previous") == "continuation":
                prev = f"clip_{int(c['clip_id'].split('_')[1]) - 1:02d}"
                start_note = (f"← image de départ = COPIE de la fin du clip précédent "
                              f"({prev}_end.jpg), raccord fluide")
            L.append(f"- start : {ref.get('start_frame', '(non extraite)')}   {start_note}")
            L.append(f"- mid   : {ref.get('mid_frame', '(non extraite)')}")
            L.append(f"- end   : {ref.get('end_frame', '(non extraite)')}")
        L.append("")
        # Section MONTAGE (texte écran / voix / musique / SFX retirés du flow_prompt).
        pp = c.get("post_production", {})
        L.append("**À ajouter au montage (PAS dans Flow) :**")
        osts = pp.get("on_screen_text", [])
        if osts:
            txt = " ; ".join(f'« {o["text"]} » ({o.get("location") or "?"})' for o in osts)
            L.append(f"- Texte écran : {txt}")
        else:
            L.append("- Texte écran : (aucun)")
        L.append(f"- Voix (parlée) : « {pp.get('voiceover_text') or '(aucune)'} »")
        mus = pp.get("music", {})
        if mus.get("present") in (True, "inherited"):
            etat = "héritée" if mus.get("present") == "inherited" else "présente"
            L.append(f"- Musique ({etat}) : {mus.get('mood') or '?'} — {mus.get('music_prompt') or ''}")
        else:
            L.append("- Musique : (absente)")
        sfx = pp.get("sound_effects", [])
        L.append("- Effets sonores : "
                 + (", ".join(s.get("label", "?") for s in sfx) if sfx else "(aucun)"))
        L.append("")
        subj = ", ".join(_subj_names(brief, c)) or "(aucun)"
        L.append(f"Personnages : {subj}")
        L.append("Identique à l'original ? [ ]")
        L.append("")
    L.append("---")
    L.append("")
    L.append("## Ordre de remontage")
    L.append("")
    L.append(" → ".join(brief["reassembly"]["order"]))
    L.append("")
    L.append(f"**Durée totale à retrouver : {brief['reassembly']['total_duration_s']} s.**")
    L.append("")
    # Voix off complète (à générer en une seule fois) + rappel de pose au montage.
    pg = brief.get("post_production_global", {})
    L.append("## Voix off complète (à générer en une seule fois)")
    L.append("")
    L.append("> " + (pg.get("full_voiceover_script") or "(aucune)"))
    L.append("")
    vp = pg.get("voice_profile", {})
    L.append(f"Voix : {vp.get('gender','?')} / {vp.get('language','?')} — "
             f"{vp.get('tone','')} (débit {vp.get('pace','')}).")
    L.append("➡️ Générer cette voix off en UNE fois, puis la poser sur la vidéo remontée, "
             "avec la musique de fond et le texte écran de chaque clip.")
    L.append("")
    return "\n".join(L)


def _subj_names(brief: Dict[str, Any], clip: Dict[str, Any]) -> List[str]:
    reg = {a["asset_id"]: a["canonical_name"] for a in brief["asset_registry"]}
    return [reg.get(a, a) for a in clip.get("subjects", [])]


# --- Auto-test -----------------------------------------------------------------------


def _self_test(
    brief: Dict[str, Any], brief_path: Path, worksheet_path: Path, frames_dir: Path,
    mtimes_before: Dict[str, int], input_paths: List[Path], last_segment_text: str = "",
) -> Dict[str, Any]:
    checks: List[Tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    clips = brief["clips"]
    # 1) 3 sorties + mtimes inchangés.
    chk("JSON écrit", brief_path.exists())
    chk("Worksheet écrite", worksheet_path.exists())
    chk("Dossier frames présent", frames_dir.is_dir())
    untouched = all(p.stat().st_mtime_ns == mtimes_before[str(p)] for p in input_paths)
    chk("Entrées non modifiées (mtimes)", untouched, f"{len(input_paths)} entrées")

    # 2) durées.
    total_target = round(sum(c["target_duration_s"] for c in clips), 3)
    chk("sum(target_duration) ≈ duration_s (±0.1)",
        abs(total_target - brief["video"]["duration_s"]) <= 0.1,
        f"{total_target} vs {brief['video']['duration_s']}")
    chk("aucun generated > 8 s",
        all(c["target_duration_s"] <= MAX_CLIP_S + 1e-6 for c in clips if c["type"] == "generated"))

    # 3) sanitisation texte écran.
    bad = []
    for c in clips:
        for o in c.get("on_screen_text", []):
            if _is_technical_label(o["text"]):
                bad.append(o["text"])
    chk("aucun on_screen_text technique (F\\d+ / timestamp nu)", not bad, str(bad))

    # 4) frames generated + continuation = copie identique.
    frames_ok = True
    cont_ok = True
    cont_count = 0
    prev_end: Optional[Path] = None
    fdir = frames_dir
    for c in clips:
        if c["type"] == "static_card":
            prev_end = None
            continue
        sp = fdir / Path(c["reference_images"]["start_frame"]).name
        ep = fdir / Path(c["reference_images"]["end_frame"]).name
        if not (sp.exists() and ep.exists()):
            frames_ok = False
        if c["transition_from_previous"] == "continuation":
            cont_count += 1
            if prev_end and sp.exists() and prev_end.exists():
                if _file_sha(sp) != _file_sha(prev_end):
                    cont_ok = False
                if c.get("start_frame_source", "").endswith("end_frame") is False:
                    cont_ok = False
            else:
                cont_ok = False
        elif c["transition_from_previous"] == "hard_cut":
            if c.get("start_frame_source") != "extracted":
                cont_ok = False
        prev_end = ep if ep.exists() else None
    chk("generated ont start_frame ET end_frame sur disque", frames_ok)
    chk(f"continuation = copie identique (hash) [{cont_count} cas]", cont_ok)
    # static_card : image finale présente sur disque.
    cards = [c for c in clips if c["type"] == "static_card"]
    card_ok = all((frames_dir / Path(c["reference_images"].get("card_frame", "x")).name).exists()
                  for c in cards)
    chk(f"static_card a son image finale sur disque [{len(cards)}]", card_ok if cards else True)

    # 5) flow_prompt = VISUEL pur (aspect + durée + consigne négative ; SANS voix/texte/musique).
    aspect = brief["video"]["aspect_ratio"]
    fp_visual_ok = True
    fp_clean_ok = True
    # Phrases SPÉCIFIQUES aux lignes retirées (la consigne négative mentionne ces mots
    # de façon négative, donc on cible les formulations d'AJOUT uniquement).
    banned = ("says:", "voiceover says", "on-screen text appears", "background:")
    for c in clips:
        fp = c.get("flow_prompt", "")
        low = fp.lower()
        if not fp or aspect not in fp or "second" not in low or "silent footage only" not in low:
            fp_visual_ok = False
        if any(b in low for b in banned):
            fp_clean_ok = False
    chk("flow_prompt visuel : aspect+durée+consigne négative (Silent footage only)", fp_visual_ok)
    chk("flow_prompt SANS voix/texte/musique (says:/voiceover/On-screen text/Background:)", fp_clean_ok)

    # post_production par clip : 4 clés + voiceover_text == dialogue d'origine.
    pp_ok = True
    for c in clips:
        pp = c.get("post_production")
        if not isinstance(pp, dict) or not all(
            k in pp for k in ("on_screen_text", "voiceover_text", "music", "sound_effects")
        ):
            pp_ok = False
            continue
        if pp["voiceover_text"] != (c.get("dialogue") or {}).get("text", ""):
            pp_ok = False
    chk("chaque clip a post_production (4 clés) + voiceover_text == dialogue", pp_ok)

    # post_production_global : voix off complète et non tronquée.
    pg = brief.get("post_production_global", {})
    fv = pg.get("full_voiceover_script", "")
    vo_ok = bool(fv) and (not last_segment_text or last_segment_text in fv)
    chk("full_voiceover_script non vide + contient le DERNIER segment (non tronqué)", vo_ok,
        f"last='{last_segment_text[:30]}'")

    # 6) reassembly couvre tous les clips ; worksheet a une section par clip.
    chk("reassembly.order couvre tous les clips",
        brief["reassembly"]["order"] == [c["clip_id"] for c in clips])
    ws = worksheet_path.read_text(encoding="utf-8")
    chk("worksheet : une section par clip",
        all(c["clip_id"].replace("clip_", "Clip ") in ws for c in clips))

    # 7) JSON valide + sanitization_log présent.
    try:
        json.loads(brief_path.read_text(encoding="utf-8"))
        json_ok = True
    except json.JSONDecodeError:
        json_ok = False
    chk("JSON relu valide", json_ok)
    chk("sanitization_log présent", "sanitization_log" in brief)

    passed = sum(1 for _, ok, _ in checks if ok)
    return {
        "ok": passed == len(checks),
        "passed": passed, "total": len(checks),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "outputs": {
            "json": brief_path.as_posix(), "worksheet": worksheet_path.as_posix(),
            "frames_dir": frames_dir.as_posix(),
        },
        "clip_count": len(clips),
        "generated_count": sum(1 for c in clips if c["type"] == "generated"),
        "static_cards": sum(1 for c in clips if c["type"] == "static_card"),
    }
