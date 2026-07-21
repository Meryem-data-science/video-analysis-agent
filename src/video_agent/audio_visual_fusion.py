"""Niveau 7 — Fusion audio-visuelle.

Aligne, **par timestamps**, les segments de parole (transcript faster-whisper) sur
les segments visuels (segmentation), et regroupe par segment visuel : frames,
images de mouvement (grille + composite), parole chevauchante et résumé audio.

Aucune analyse sémantique ici : pas de VLM/OCR/YOLO, pas de transcription. Le
module se contente de **préparer** un manifest structuré prêt pour le VLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class AudioVisualFusionError(Exception):
    """Erreur générique de la fusion audio-visuelle."""


# En dessous de ce ratio de chevauchement (et seul chevauchement du segment),
# on signale un alignement faible.
_LOW_OVERLAP_RATIO = 0.5


def load_json(path: str, label: str) -> Dict[str, Any]:
    """Charge un fichier JSON avec des erreurs claires (introuvable / invalide)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise AudioVisualFusionError(f"Fichier {label} introuvable : {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AudioVisualFusionError(f"JSON {label} invalide ({p}) : {exc}") from exc


def compute_overlap(
    v_start: float, v_end: float, t_start: float, t_end: float
) -> Tuple[float, float]:
    """Calcule (overlap_s, overlap_ratio) entre un segment visuel et un transcript.

    overlap_s   = durée de chevauchement temporel (0 si bornes seulement jointives).
    overlap_ratio = overlap_s / durée du transcript (part de la parole couverte).
    """
    overlap_s = min(v_end, t_end) - max(v_start, t_start)
    if overlap_s <= 0:
        return 0.0, 0.0
    t_duration = t_end - t_start
    overlap_ratio = overlap_s / t_duration if t_duration > 0 else 0.0
    return round(overlap_s, 3), round(overlap_ratio, 4)


def align_transcript_to_visual_segments(
    visual_segments: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Rattache chaque transcript à tous les segments visuels qu'il chevauche.

    Un transcript peut appartenir à plusieurs segments visuels (chevauchement
    multiple) ; chacun reçoit son propre overlap_s/overlap_ratio. Les listes
    retournées sont triées par start_s.
    """
    aligned: Dict[int, List[Dict[str, Any]]] = {}
    for vseg in visual_segments:
        seg_id = int(vseg["segment_id"])
        v_start = float(vseg["start_s"])
        v_end = float(vseg["end_s"])
        matches: List[Dict[str, Any]] = []
        for tseg in transcript_segments:
            t_start = float(tseg.get("start_s", 0.0))
            t_end = float(tseg.get("end_s", 0.0))
            overlap_s, overlap_ratio = compute_overlap(v_start, v_end, t_start, t_end)
            if overlap_s > 0:
                matches.append(
                    {
                        "start_s": round(t_start, 3),
                        "end_s": round(t_end, 3),
                        "text": (tseg.get("text", "") or "").strip(),
                        "confidence": tseg.get("confidence"),
                        "overlap_s": overlap_s,
                        "overlap_ratio": overlap_ratio,
                    }
                )
        matches.sort(key=lambda m: m["start_s"])
        aligned[seg_id] = matches
    return aligned


def _frames_by_segment(frames_manifest: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Indexe les frames par segment_id (champs réduits : frame_id, timestamp_s, path)."""
    by_seg: Dict[int, List[Dict[str, Any]]] = {}
    for frame in frames_manifest.get("frames", []):
        seg_id = int(frame.get("segment_id"))
        by_seg.setdefault(seg_id, []).append(
            {
                "frame_id": frame.get("frame_id"),
                "timestamp_s": frame.get("timestamp_s"),
                "path": frame.get("path"),
            }
        )
    for seg_id in by_seg:
        by_seg[seg_id].sort(key=lambda f: (f.get("timestamp_s") or 0.0))
    return by_seg


def _motion_by_segment(motion_manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Indexe les chemins motion (grid/composite) par segment_id."""
    by_seg: Dict[int, Dict[str, Any]] = {}
    for seg in motion_manifest.get("segments", []):
        seg_id = int(seg.get("segment_id"))
        by_seg[seg_id] = {
            "motion_grid_path": seg.get("grid_path"),
            "motion_composite_path": seg.get("composite_path"),
        }
    return by_seg


def _build_segment_entry(
    vseg: Dict[str, Any],
    seg_frames: List[Dict[str, Any]],
    seg_motion: Optional[Dict[str, Any]],
    seg_transcripts: List[Dict[str, Any]],
    audio_activity_mode: Optional[str],
) -> Dict[str, Any]:
    """Assemble l'entrée fusionnée d'un segment visuel."""
    seg_id = int(vseg["segment_id"])
    start_s = float(vseg["start_s"])
    end_s = float(vseg["end_s"])

    grid_path = seg_motion.get("motion_grid_path") if seg_motion else None
    composite_path = seg_motion.get("motion_composite_path") if seg_motion else None

    combined_text = " ".join(t["text"] for t in seg_transcripts if t.get("text")).strip()
    has_speech = bool(combined_text)
    speech_duration_s = round(sum(t["overlap_s"] for t in seg_transcripts), 3)

    warnings: List[str] = []
    if not has_speech:
        warnings.append("no_speech_overlap")
    elif len(seg_transcripts) == 1 and seg_transcripts[0]["overlap_ratio"] < _LOW_OVERLAP_RATIO:
        warnings.append("low_overlap")

    # ready_for_vlm : présence des frames + motion_grid + motion_composite.
    # La parole est optionnelle.
    has_visual = bool(seg_frames) and bool(grid_path) and bool(composite_path)
    if not seg_frames:
        warnings.append("no_frames")
    if not grid_path or not composite_path:
        warnings.append("missing_motion_images")

    return {
        "segment_id": seg_id,
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "duration_s": round(end_s - start_s, 3),
        "frames": seg_frames,
        "frame_count": len(seg_frames),
        "motion_grid_path": grid_path,
        "motion_composite_path": composite_path,
        "transcript_segments": seg_transcripts,
        "combined_text": combined_text,
        "audio_summary": {
            "has_speech": has_speech,
            "speech_segment_count": len(seg_transcripts),
            "speech_duration_s": speech_duration_s,
            "audio_activity_mode": audio_activity_mode,
        },
        "warnings": warnings,
        "ready_for_vlm": has_visual,
    }


def build_fusion_manifest(
    video_path: str,
    visual_segments: List[Dict[str, Any]],
    frames_by_seg: Dict[int, List[Dict[str, Any]]],
    motion_by_seg: Dict[int, Dict[str, Any]],
    aligned: Dict[int, List[Dict[str, Any]]],
    language: Optional[str],
    audio_activity_mode: Optional[str],
    sources: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Construit le manifest de fusion complet (segments + summary)."""
    segments_out: List[Dict[str, Any]] = []
    for vseg in sorted(visual_segments, key=lambda s: int(s["segment_id"])):
        seg_id = int(vseg["segment_id"])
        entry = _build_segment_entry(
            vseg,
            frames_by_seg.get(seg_id, []),
            motion_by_seg.get(seg_id),
            aligned.get(seg_id, []),
            audio_activity_mode,
        )
        segments_out.append(entry)

    total_frames = sum(s["frame_count"] for s in segments_out)
    speech_segments = sum(s["audio_summary"]["speech_segment_count"] for s in segments_out)
    segments_without_speech = sum(
        1 for s in segments_out if not s["audio_summary"]["has_speech"]
    )
    ready_count = sum(1 for s in segments_out if s["ready_for_vlm"])

    return {
        "video_path": str(video_path),
        "fusion_type": "timestamp_aligned_audio_visual",
        "language": language,
        "sources": sources,
        "segments": segments_out,
        "summary": {
            "segment_count": len(segments_out),
            "total_frames": total_frames,
            "speech_segment_count": speech_segments,
            "segments_without_speech": segments_without_speech,
            "language": language,
            "ready_for_vlm_count": ready_count,
            "audio_activity_mode": audio_activity_mode,
        },
        "next_step": "vlm_analysis",
    }


def fuse_audio_visual(
    segments_manifest_path: str,
    frames_manifest_path: str,
    motion_manifest_path: str,
    audio_light_analysis_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Fusionne les artefacts visuels et audio en un manifest aligné par timestamps.

    Args:
        segments_manifest_path: segmentation visuelle (Niveau 3).
        frames_manifest_path: manifest de frames (Niveau 4).
        motion_manifest_path: manifest motion (Niveau 5).
        audio_light_analysis_path: analyse audio légère (Niveau 6C), facultatif.
        transcript_path: transcript faster-whisper (Niveau 6D), facultatif.

    Returns:
        Un manifest de fusion dict (voir build_fusion_manifest).
    """
    segments = load_json(segments_manifest_path, "segments")
    frames = load_json(frames_manifest_path, "frames")
    motion = load_json(motion_manifest_path, "motion")

    visual_segments = segments.get("segments")
    if not isinstance(visual_segments, list) or not visual_segments:
        raise AudioVisualFusionError(
            "Aucun segment visuel exploitable dans le manifest de segmentation."
        )

    transcript_segments: List[Dict[str, Any]] = []
    language: Optional[str] = None
    if transcript_path is not None:
        transcript = load_json(transcript_path, "transcript")
        transcript_segments = transcript.get("segments", []) or []
        language = transcript.get("summary", {}).get("detected_language") or transcript.get(
            "language"
        )

    audio_activity_mode: Optional[str] = None
    if audio_light_analysis_path is not None:
        audio_light = load_json(audio_light_analysis_path, "analyse audio légère")
        audio_activity_mode = audio_light.get("classification", {}).get("mode")

    video_path = (
        segments.get("video_path")
        or frames.get("video_path")
        or motion.get("video_path")
        or ""
    )

    frames_by_seg = _frames_by_segment(frames)
    motion_by_seg = _motion_by_segment(motion)
    aligned = align_transcript_to_visual_segments(visual_segments, transcript_segments)

    sources = {
        "segments_manifest": segments_manifest_path,
        "frames_manifest": frames_manifest_path,
        "motion_manifest": motion_manifest_path,
        "audio_light_analysis": audio_light_analysis_path,
        "transcript": transcript_path,
    }

    logging.info(
        "fuse_audio_visual video=%s visual_segments=%d transcript_segments=%d",
        video_path, len(visual_segments), len(transcript_segments),
    )

    return build_fusion_manifest(
        video_path=video_path,
        visual_segments=visual_segments,
        frames_by_seg=frames_by_seg,
        motion_by_seg=motion_by_seg,
        aligned=aligned,
        language=language,
        audio_activity_mode=audio_activity_mode,
        sources=sources,
    )
