"""Niveau 8 — Construction du pack VLM.

À partir du manifest de fusion audio-visuelle (Niveau 7), prépare un « pack »
par segment visuel destiné à un futur VLM : sous-ensemble de frames sélectionnées
+ images de mouvement + parole + instructions/questions/schéma de sortie attendu.

Ce module **ne lance aucun modèle** : il ne fait que structurer les entrées
(pas de VLM/OCR/YOLO, pas de transcription, pas d'analyse sémantique).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


class VlmPackError(Exception):
    """Erreur générique de la construction du pack VLM."""


_DEFAULT_MAX_FRAMES = 6


def load_json(path: str, label: str) -> Dict[str, Any]:
    """Charge un fichier JSON avec des erreurs claires (introuvable / invalide)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise VlmPackError(f"Fichier {label} introuvable : {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VlmPackError(f"JSON {label} invalide ({p}) : {exc}") from exc


def select_frames(
    frames: List[Dict[str, Any]], max_frames: int = _DEFAULT_MAX_FRAMES
) -> List[Dict[str, Any]]:
    """Sélectionne un sous-ensemble représentatif de frames (logique pure).

    - Si le nombre de frames est <= max_frames, on garde tout.
    - Sinon, échantillonnage uniforme par index incluant toujours la première et la
      dernière (indices round(i*(n-1)/(k-1))), dédupliqués, puis triés par timestamp.
    """
    if max_frames <= 0:
        raise VlmPackError(f"max_frames doit être positif : {max_frames}")

    n = len(frames)
    if n == 0:
        return []
    if n <= max_frames:
        selected = list(frames)
    else:
        indices = sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
        selected = [frames[i] for i in indices]

    selected.sort(key=lambda f: (f.get("timestamp_s") or 0.0))
    return [
        {
            "frame_id": f.get("frame_id"),
            "timestamp_s": f.get("timestamp_s"),
            "path": f.get("path"),
        }
        for f in selected
    ]


def _build_instructions(
    combined_text: str, has_speech: bool, language: Optional[str] = None
) -> List[str]:
    """Instructions destinées au VLM pour un segment."""
    instructions = [
        "Tu reçois, pour ce segment : des frames ordonnées par temps, une grille "
        "temporelle (motion_grid) et un composite de mouvement (motion_composite).",
        "Analyse les actions visibles, les personnes/objets, le mouvement (via le "
        "composite), le contexte de scène, et la relation entre l'audio et le visuel.",
        "Pas de bounding boxes. Lis le texte visible à l'écran (sous-titres, panneaux, "
        "logos, watermarks) mot pour mot, avec sa langue et son emplacement.",
        "Signale explicitement toute incertitude.",
    ]
    if has_speech:
        lang_txt = language or "inconnue"
        instructions.append(
            f"Parole transcrite disponible (langue détectée : {lang_txt}, possiblement "
            "approximative à cause de la musique/du bruit/de plusieurs locuteurs) : "
            f"« {combined_text} ». Vérifie si elle est cohérente avec ce qui est vu."
        )
    else:
        instructions.append(
            "Aucune parole exploitable pour ce segment : base-toi uniquement sur le visuel."
        )
    return instructions


def _build_questions() -> List[str]:
    """Questions posées au VLM (identiques par segment)."""
    return [
        "Que se passe-t-il dans ce segment (description de scène) ?",
        "Quelles personnes/objets sont visibles ?",
        "Quel mouvement se produit (utilise le composite de mouvement) ?",
        "La parole disponible est-elle cohérente avec le visuel ?",
        "Quel est ton niveau de confiance et quelles sont les incertitudes ?",
    ]


def _expected_output_schema() -> Dict[str, Any]:
    """Schéma de sortie attendu du VLM (descriptif, non exécuté ici)."""
    return {
        "segment_id": "integer (copier exactement la valeur fournie)",
        "start_s": "number (copier exactement la valeur fournie)",
        "end_s": "number (copier exactement la valeur fournie)",
        "combined_text": "string (copier exactement, chaîne vide autorisée si fournie vide)",
        "scene_description": "string",
        "actions": ["string"],
        "people_objects": ["string"],
        "motion_description": "string",
        "audio_visual_relation": "string",
        "uses_speech": "bool",
        "mood": ["string"],
        "texte_ecran": [{"text": "string", "lang": "string", "location": "string"}],
        "confidence": "number (0..1)",
        "uncertainties": ["string"],
        "evidence": {"visual": ["string"], "audio": ["string"]},
    }


def _build_segment_pack(
    fusion_segment: Dict[str, Any],
    max_frames: int,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Construit l'entrée pack VLM d'un segment à partir d'un segment de fusion."""
    frames = fusion_segment.get("frames", []) or []
    selected = select_frames(frames, max_frames)
    combined_text = fusion_segment.get("combined_text", "") or ""
    audio_summary = fusion_segment.get("audio_summary", {}) or {}
    has_speech = bool(audio_summary.get("has_speech"))

    return {
        "segment_id": fusion_segment.get("segment_id"),
        "start_s": fusion_segment.get("start_s"),
        "end_s": fusion_segment.get("end_s"),
        "duration_s": fusion_segment.get("duration_s"),
        "motion_grid_path": fusion_segment.get("motion_grid_path"),
        "motion_composite_path": fusion_segment.get("motion_composite_path"),
        "selected_frames": selected,
        "selected_frame_count": len(selected),
        "available_frame_count": len(frames),
        "combined_text": combined_text,
        "transcript_segments": fusion_segment.get("transcript_segments", []) or [],
        "audio_summary": audio_summary,
        "warnings": list(fusion_segment.get("warnings", []) or []),
        "instructions_for_vlm": _build_instructions(combined_text, has_speech, language),
        "questions_for_vlm": _build_questions(),
        "expected_output_schema": _expected_output_schema(),
    }


def build_vlm_pack(
    fusion_manifest: Dict[str, Any],
    max_frames: int = _DEFAULT_MAX_FRAMES,
    skip_not_ready: bool = False,
    source_fusion_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Construit le pack VLM complet à partir d'un manifest de fusion (dict).

    Args:
        fusion_manifest: manifest de fusion (Niveau 7).
        max_frames: nombre max de frames sélectionnées par segment.
        skip_not_ready: exclure les segments dont ready_for_vlm est False.
        source_fusion_manifest_path: chemin source (reporté dans le pack).
    """
    fusion_segments = fusion_manifest.get("segments")
    if not isinstance(fusion_segments, list) or not fusion_segments:
        raise VlmPackError("Aucun segment exploitable dans le manifest de fusion.")

    language = fusion_manifest.get("language")
    segments_out: List[Dict[str, Any]] = []
    for fseg in fusion_segments:
        if skip_not_ready and not fseg.get("ready_for_vlm", False):
            continue
        segments_out.append(_build_segment_pack(fseg, max_frames, language))

    total_selected = sum(s["selected_frame_count"] for s in segments_out)
    ready_count = sum(
        1 for fseg in fusion_segments if fseg.get("ready_for_vlm", False)
    )

    logging.info(
        "build_vlm_pack segments=%d selected_frames=%d max_frames=%d",
        len(segments_out), total_selected, max_frames,
    )

    return {
        "video_path": fusion_manifest.get("video_path", ""),
        "language": fusion_manifest.get("language"),
        "pack_type": "vlm_segment_pack",
        "source_fusion_manifest": source_fusion_manifest_path,
        "global_notes": [
            f"Langue détectée (Whisper) : {language or 'inconnue'} ; transcription possiblement approximative.",
            "Pas de bounding boxes ; le texte à l'écran est lu nativement par le VLM (champ texte_ecran).",
            "Utiliser motion_composite pour le mouvement, motion_grid pour la chronologie.",
        ],
        "segments": segments_out,
        "summary": {
            "segment_count": len(segments_out),
            "total_selected_frames": total_selected,
            "ready_for_vlm_count": ready_count,
            "language": fusion_manifest.get("language"),
            "max_frames_per_segment": max_frames,
        },
        "next_step": "vlm_analysis",
    }


def build_vlm_pack_from_file(
    fusion_manifest_path: str,
    max_frames: int = _DEFAULT_MAX_FRAMES,
    skip_not_ready: bool = False,
) -> Dict[str, Any]:
    """Charge un manifest de fusion depuis un fichier et construit le pack VLM."""
    fusion_manifest = load_json(fusion_manifest_path, "manifest de fusion")
    return build_vlm_pack(
        fusion_manifest,
        max_frames=max_frames,
        skip_not_ready=skip_not_ready,
        source_fusion_manifest_path=fusion_manifest_path,
    )
