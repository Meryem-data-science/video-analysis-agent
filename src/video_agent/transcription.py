"""Niveau 6D — Transcription audio (backend faster-whisper).

Transcrit un WAV (idéalement mono 16 kHz) en segments horodatés via faster-whisper.

IMPORTANT : ``WhisperModel`` n'est PAS importé/instancié au chargement du module.
L'import de faster-whisper et l'instanciation du modèle se font uniquement DANS
``transcribe_audio()`` — un simple ``import`` de ce module ne télécharge donc aucun
modèle et ne charge pas CTranslate2.

Pas de traduction (``task="transcribe"`` uniquement), pas de diarisation, pas de VLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


class TranscriptionError(Exception):
    """Erreur générique de la transcription."""


_VALID_TASKS = {"transcribe"}


def _validate_audio_path(audio_path: str) -> Path:
    """Valide le chemin audio (existant, non vide)."""
    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        raise TranscriptionError(f"Fichier audio introuvable : {path}")
    if path.stat().st_size == 0:
        raise TranscriptionError(f"Fichier audio vide (0 octet) : {path}")
    return path


def _load_optional_json(path: Optional[str], label: str) -> Optional[Dict[str, Any]]:
    """Charge un JSON facultatif ; None si non fourni. Erreur claire si invalide."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise TranscriptionError(f"Fichier {label} introuvable : {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"JSON {label} invalide ({p}) : {exc}") from exc


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convertit en float de façon défensive (default si impossible/None)."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segments_to_json(segments: Any) -> List[Dict[str, Any]]:
    """Convertit les segments faster-whisper en liste JSON sérialisable.

    Chaque segment expose start/end/text et, si disponible, une confiance dérivée
    de ``avg_logprob`` (exp pour repasser en probabilité approximative).
    """
    import math

    out: List[Dict[str, Any]] = []
    for seg in segments:
        avg_logprob = getattr(seg, "avg_logprob", None)
        confidence: Optional[float] = None
        if avg_logprob is not None:
            try:
                confidence = round(math.exp(float(avg_logprob)), 4)
            except (TypeError, ValueError, OverflowError):
                confidence = None
        out.append(
            {
                "start_s": round(_safe_float(getattr(seg, "start", 0.0)), 3),
                "end_s": round(_safe_float(getattr(seg, "end", 0.0)), 3),
                "text": (getattr(seg, "text", "") or "").strip(),
                "confidence": confidence,
            }
        )
    return out


def _build_warnings() -> List[str]:
    """Avertissements prudents sur la qualité de transcription attendue."""
    return [
        "Le Darija (dialecte arabe) peut être transcrit de façon approximative.",
        "La musique ou le bruit de fond peuvent dégrader la transcription.",
        "La présence d'enfants ou de plusieurs locuteurs peut dégrader la qualité.",
        "Pas de diarisation (identification des locuteurs) dans cette version.",
    ]


def _make_summary(
    segments_json: List[Dict[str, Any]],
    detected_language: Optional[str],
    duration_s: float,
    warnings: List[str],
) -> Dict[str, Any]:
    """Construit le bloc summary du manifest de transcription."""
    has_speech = any(s["text"] for s in segments_json)
    return {
        "duration_s": round(_safe_float(duration_s), 3),
        "segment_count": len(segments_json),
        "detected_language": detected_language,
        "has_speech": has_speech,
        "warnings": warnings,
    }


def transcribe_audio(
    audio_path: str,
    audio_manifest_path: Optional[str] = None,
    audio_light_analysis_path: Optional[str] = None,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    task: str = "transcribe",
    beam_size: int = 5,
) -> Dict[str, Any]:
    """Transcrit un audio avec faster-whisper et retourne un manifest.

    Args:
        audio_path: chemin du WAV à transcrire.
        audio_manifest_path / audio_light_analysis_path: chemins sources facultatifs
            (reportés dans le manifest pour la traçabilité).
        model_name: modèle Whisper (défaut "small").
        device: "cpu" par défaut.
        compute_type: "int8" par défaut (quantization CPU).
        language: None = auto-détection ; "ar" pour forcer l'arabe.
        task: doit rester "transcribe" (pas de traduction).
        beam_size: taille du faisceau de décodage.

    Returns:
        Un manifest dict (voir structure en fin de fonction).
    """
    if task not in _VALID_TASKS:
        raise TranscriptionError(
            f"task invalide : {task!r} (seul 'transcribe' est autorisé ; "
            "pas de traduction automatique)."
        )

    audio = _validate_audio_path(audio_path)
    _ = _load_optional_json(audio_manifest_path, "manifest audio")
    _ = _load_optional_json(audio_light_analysis_path, "analyse audio légère")

    # Import et instanciation différés : un import du module ne charge rien.
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper indisponible. Installez les fonctions audio : "
            "`python -m pip install \".[audio]\"`."
        ) from exc

    logging.info(
        "transcribe_audio start audio=%s model=%s device=%s compute=%s lang=%s task=%s",
        audio, model_name, device, compute_type, language, task,
    )

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        raise TranscriptionError(
            f"Échec du chargement du modèle '{model_name}' "
            f"(device={device}, compute_type={compute_type}) : {exc}"
        ) from exc

    try:
        segments_iter, info = model.transcribe(
            str(audio),
            language=language,
            task=task,
            beam_size=beam_size,
        )
        segments_json = _segments_to_json(segments_iter)
    except Exception as exc:
        raise TranscriptionError(f"Erreur pendant la transcription : {exc}") from exc

    detected_language = getattr(info, "language", None)
    duration_s = _safe_float(getattr(info, "duration", 0.0))
    warnings = _build_warnings()

    manifest: Dict[str, Any] = {
        "audio_path": str(audio_path),
        "source_audio_manifest": audio_manifest_path,
        "source_audio_light_analysis": audio_light_analysis_path,
        "transcription_engine": "faster-whisper",
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "task": task,
        "segments": segments_json,
        "summary": _make_summary(segments_json, detected_language, duration_s, warnings),
        "next_step": "audio_visual_fusion",
    }

    logging.info(
        "transcribe_audio done segments=%d detected_language=%s",
        len(segments_json), detected_language,
    )
    return manifest
