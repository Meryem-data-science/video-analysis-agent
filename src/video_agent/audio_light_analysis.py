"""Niveau 6C — Analyse audio légère.

Analyse un WAV PCM (idéalement mono 16 kHz) pour estimer, **prudemment**, s'il
est silencieux ou actif, et recommander ou non le lancement de Whisper ensuite.

Bibliothèques : uniquement la stdlib (`wave`, `struct`, `math`, `statistics`,
`pathlib`). Pas d'`audioop` (retiré en Python 3.13), pas de dépendance externe.

Cette étape ne distingue PAS parole vs musique de façon fiable : elle mesure
seulement l'énergie du signal (RMS / pic / ratio de silence).
"""

from __future__ import annotations

import math
import statistics
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional


class AudioLightAnalysisError(Exception):
    """Erreur générique de l'analyse audio légère."""


# Seuil de silence sur le RMS normalisé [0..1] d'une fenêtre.
_SILENCE_RMS = 0.01
# Bornes de classification sur la proportion de fenêtres actives.
_SILENCE_MAX_ACTIVE = 0.05   # en dessous : quasi tout silencieux -> "silence"
_ACTIVE_MIN_RATIO = 0.30     # au dessus : énergie franche -> "active_audio"

# Largeurs d'échantillon PCM supportées -> (format struct, valeur max absolue).
_PCM_FORMATS = {
    1: ("B", 128.0),   # 8-bit non signé (centré sur 128)
    2: ("h", 32768.0),  # 16-bit signé
    4: ("i", 2147483648.0),  # 32-bit signé
}


def _read_wav(audio_path: str) -> tuple[int, int, int, int, bytes]:
    """Ouvre le WAV et retourne (sample_rate, channels, sample_width, n_frames, raw).

    Lève AudioLightAnalysisError pour fichier absent/vide/non PCM/illisible.
    """
    path = Path(audio_path)
    if not path.exists() or not path.is_file():
        raise AudioLightAnalysisError(f"Fichier audio introuvable : {path}")
    if path.stat().st_size == 0:
        raise AudioLightAnalysisError(f"Fichier audio vide (0 octet) : {path}")

    try:
        with wave.open(str(path), "rb") as wf:
            comptype = wf.getcomptype()
            if comptype != "NONE":
                raise AudioLightAnalysisError(
                    f"Fichier audio non PCM (compression={comptype}) : {path}"
                )
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as exc:
        raise AudioLightAnalysisError(
            f"Format WAV invalide ou illisible ({path}) : {exc}"
        ) from exc
    except AudioLightAnalysisError:
        raise
    except Exception as exc:
        raise AudioLightAnalysisError(
            f"Lecture audio impossible ({path}) : {exc}"
        ) from exc

    return sample_rate, channels, sample_width, n_frames, raw


def _validate_wav_params(sample_rate: int, channels: int, sample_width: int) -> None:
    """Valide les paramètres du WAV (PCM supporté, channels et sample_rate sensés)."""
    if sample_width not in _PCM_FORMATS:
        raise AudioLightAnalysisError(
            f"Largeur d'échantillon non supportée : {sample_width} octet(s) "
            f"(attendu : 1, 2 ou 4)."
        )
    if channels < 1 or channels > 2:
        raise AudioLightAnalysisError(
            f"Nombre de canaux inattendu : {channels} (attendu : 1 mono ou 2 stéréo)."
        )
    if sample_rate <= 0:
        raise AudioLightAnalysisError(f"Sample_rate inattendu : {sample_rate}.")


def _decode_samples(raw: bytes, sample_width: int, channels: int) -> List[float]:
    """Décode les octets PCM en échantillons mono normalisés dans [-1, 1].

    Si stéréo, les canaux sont moyennés en mono.
    """
    fmt_char, max_abs = _PCM_FORMATS[sample_width]
    total = len(raw) // sample_width
    if total == 0:
        return []

    ints = struct.unpack("<" + fmt_char * total, raw[: total * sample_width])

    # 8-bit non signé : recentrer autour de 0.
    if sample_width == 1:
        ints = [v - 128 for v in ints]

    if channels == 1:
        return [v / max_abs for v in ints]

    # Stéréo (ou plus) : moyenne par frame.
    mono: List[float] = []
    for i in range(0, len(ints) - channels + 1, channels):
        frame = ints[i : i + channels]
        mono.append((sum(frame) / channels) / max_abs)
    return mono


def _rms(samples: List[float]) -> float:
    """RMS d'une liste d'échantillons normalisés (0.0 si vide)."""
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _peak(samples: List[float]) -> float:
    """Niveau de pic (|max|) d'une liste d'échantillons normalisés (0.0 si vide)."""
    if not samples:
        return 0.0
    return max(abs(s) for s in samples)


def _compute_energy_profile(
    samples: List[float], sample_rate: int, window_ms: int
) -> List[Dict[str, Any]]:
    """Découpe le signal en fenêtres et calcule rms/pic/silence par fenêtre."""
    if window_ms <= 0:
        raise AudioLightAnalysisError(f"window_ms doit être positif : {window_ms}")

    window_size = max(1, int(sample_rate * window_ms / 1000))
    profile: List[Dict[str, Any]] = []

    for index, start in enumerate(range(0, len(samples), window_size)):
        chunk = samples[start : start + window_size]
        if not chunk:
            continue
        rms = _rms(chunk)
        peak = _peak(chunk)
        profile.append(
            {
                "index": index,
                "start_s": round(start / sample_rate, 3),
                "end_s": round((start + len(chunk)) / sample_rate, 3),
                "rms": round(rms, 6),
                "peak": round(peak, 6),
                "is_silent": rms < _SILENCE_RMS,
            }
        )

    return profile


def _classify_audio(active_ratio: float, rms_mean: float) -> Dict[str, Any]:
    """Classification prudente : silence / active_audio / mixed_or_unknown.

    On ne prétend PAS distinguer parole vs musique sans modèle dédié.
    """
    if active_ratio <= _SILENCE_MAX_ACTIVE:
        return {
            "mode": "silence",
            "confidence": 0.9,
            "reason": (
                f"Quasi aucune fenêtre active (active_ratio={active_ratio:.2f}) : "
                "audio considéré comme silencieux."
            ),
        }
    if active_ratio >= _ACTIVE_MIN_RATIO:
        return {
            "mode": "active_audio",
            "confidence": 0.7,
            "reason": (
                f"Énergie présente sur une part notable du signal "
                f"(active_ratio={active_ratio:.2f}). Parole/musique non distinguées "
                "à ce niveau."
            ),
        }
    return {
        "mode": "mixed_or_unknown",
        "confidence": 0.4,
        "reason": (
            f"Énergie intermédiaire/irrégulière (active_ratio={active_ratio:.2f}) : "
            "impossible de trancher silence vs audio actif de façon fiable."
        ),
    }


def _recommend(classification: Dict[str, Any]) -> Dict[str, Any]:
    """Recommande ou non Whisper selon la classification (prudent)."""
    mode = classification["mode"]
    if mode == "silence":
        return {
            "run_whisper": False,
            "reason": "Audio considéré silencieux : transcription inutile.",
        }
    # active_audio ou mixed_or_unknown : on tente Whisper.
    return {
        "run_whisper": True,
        "reason": (
            "Audio actif détecté. Note : l'audio peut contenir de la parole mêlée "
            "à de la musique/bruit (audio may contain speech mixed with music/noise)."
        ),
    }


def analyze_audio_light(
    audio_path: str,
    audio_manifest: Optional[Dict[str, Any]] = None,
    window_ms: int = 500,
) -> Dict[str, Any]:
    """Analyse légère d'un WAV PCM et retourne un manifest d'analyse.

    Args:
        audio_path: chemin du WAV (idéalement mono 16 kHz PCM 16-bit).
        audio_manifest: manifest audio source facultatif (reporté dans la sortie).
        window_ms: taille des fenêtres d'analyse en millisecondes.

    Returns:
        Un manifest dict (voir structure en fin de fonction).
    """
    sample_rate, channels, sample_width, n_frames, raw = _read_wav(audio_path)
    _validate_wav_params(sample_rate, channels, sample_width)

    samples = _decode_samples(raw, sample_width, channels)
    if not samples:
        raise AudioLightAnalysisError(f"Aucun échantillon décodé : {audio_path}")

    duration_s = n_frames / sample_rate if sample_rate else 0.0
    rms_mean = _rms(samples)
    peak_level = _peak(samples)

    energy_profile = _compute_energy_profile(samples, sample_rate, window_ms)
    window_count = len(energy_profile)
    silent_windows = sum(1 for w in energy_profile if w["is_silent"])
    silence_ratio = silent_windows / window_count if window_count else 1.0
    active_ratio = 1.0 - silence_ratio

    classification = _classify_audio(active_ratio, rms_mean)
    recommendation = _recommend(classification)

    next_step = "whisper_transcription" if recommendation["run_whisper"] else "skip_whisper"

    manifest: Dict[str, Any] = {
        "audio_path": str(audio_path),
        "source_audio_manifest": audio_manifest,
        "analysis_type": "light_audio_energy_analysis",
        "features": {
            "duration_s": round(duration_s, 3),
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "rms_mean": round(rms_mean, 6),
            "peak_level": round(peak_level, 6),
            "silence_ratio": round(silence_ratio, 4),
            "active_ratio": round(active_ratio, 4),
            "window_ms": window_ms,
            "window_count": window_count,
        },
        "classification": classification,
        "recommendation": recommendation,
        "energy_profile": energy_profile,
        "next_step": next_step,
    }

    return manifest
