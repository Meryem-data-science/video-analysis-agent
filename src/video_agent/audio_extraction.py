"""Niveau 6 — Extraction audio (backend FFmpeg).

Extrait la piste audio d'une vidéo et la convertit en WAV mono 16 kHz PCM 16-bit,
format adapté à Whisper / faster-whisper.

Backend : **FFmpeg** (via subprocess). Aucune transcription / Whisper / VLM ici.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Délai max accordé à FFmpeg pour extraire la piste audio.
_FFMPEG_AUDIO_TIMEOUT_S = 300.0


class AudioExtractionError(Exception):
    """Erreur générique de l'extraction audio."""


def _check_ffmpeg_available() -> str:
    """Retourne le chemin de l'exécutable ffmpeg, ou lève une erreur claire."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioExtractionError(
            "FFmpeg est introuvable dans le PATH. Installez FFmpeg et assurez-vous "
            "que la commande `ffmpeg` est accessible."
        )
    return ffmpeg


def _safe_mkdir(path: Path) -> None:
    """Crée un répertoire (et ses parents) sans échouer s'il existe déjà."""
    path.mkdir(parents=True, exist_ok=True)


def _derive_video_stem(video_path: str) -> str:
    """Dérive le `video_stem` depuis le chemin (jamais codé en dur)."""
    stem = Path(video_path).stem
    if not stem:
        raise AudioExtractionError(f"Impossible de dériver le nom depuis : {video_path}")
    return stem


def _build_ffmpeg_audio_command(
    ffmpeg: str,
    video_path: str,
    output_path: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
) -> List[str]:
    """Construit la commande FFmpeg d'extraction audio WAV PCM 16-bit.

    Conceptuellement : ffmpeg -y -i <video> -vn -ac <channels> -ar <sr>
    -c:a pcm_s16le <output.wav>.
    """
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", video_path,
        "-vn",                       # pas de flux vidéo
        "-ac", str(channels),        # nombre de canaux (mono par défaut)
        "-ar", str(sample_rate),     # fréquence d'échantillonnage (16 kHz)
        "-c:a", "pcm_s16le",         # PCM signé 16-bit little-endian
        str(output_path),
    ]


def _validate_output_file(path: Path) -> None:
    """Vérifie que le fichier audio a bien été créé et n'est pas vide."""
    if not path.exists() or not path.is_file():
        raise AudioExtractionError(f"Fichier audio non créé par FFmpeg : {path}")
    if path.stat().st_size == 0:
        raise AudioExtractionError(f"Fichier audio créé mais vide (0 octet) : {path}")


def extract_audio(
    video_path: str,
    output_dir: str = "outputs/audio",
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Extrait la piste audio d'une vidéo en WAV mono 16 kHz PCM 16-bit.

    Args:
        video_path: chemin de la vidéo source.
        output_dir: répertoire racine ; sortie sous ``<output_dir>/<video_stem>/audio.wav``.
        sample_rate: fréquence d'échantillonnage cible (défaut 16000).
        channels: nombre de canaux cible (défaut 1 = mono).
        overwrite: réécrire le WAV existant (sinon réutilise l'existant).

    Returns:
        Un manifest dict (voir structure en fin de fonction).
    """
    video = Path(video_path)
    if not video.exists() or not video.is_file():
        raise AudioExtractionError(f"Vidéo introuvable : {video_path}")

    ffmpeg = _check_ffmpeg_available()
    video_stem = _derive_video_stem(video_path)

    audio_out = Path(output_dir) / video_stem
    _safe_mkdir(audio_out)
    audio_path = audio_out / "audio.wav"

    logging.info(
        "extract_audio start video=%s stem=%s sr=%d ac=%d output=%s",
        video_path, video_stem, sample_rate, channels, audio_path,
    )

    need_extraction = overwrite or not audio_path.exists()
    if need_extraction:
        cmd = _build_ffmpeg_audio_command(
            ffmpeg, str(video), audio_path, sample_rate, channels, overwrite
        )
        logging.info("ffmpeg audio -> %s", audio_path)
        try:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_FFMPEG_AUDIO_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(
                f"FFmpeg a dépassé le délai ({_FFMPEG_AUDIO_TIMEOUT_S}s) pour l'extraction audio."
            ) from exc

        if proc.returncode != 0:
            raise AudioExtractionError(
                f"FFmpeg a échoué (code {proc.returncode}) pour l'extraction audio : "
                f"{proc.stderr.strip()[-500:]}"
            )

    _validate_output_file(audio_path)

    manifest: Dict[str, Any] = {
        "video_path": str(video_path),
        "audio_path": audio_path.as_posix(),
        "audio_extraction_type": "ffmpeg_wav_mono_16khz_pcm_s16le",
        "audio": {
            "present": True,
            "output_format": "wav",
            "sample_rate": sample_rate,
            "channels": channels,
            "codec": "pcm_s16le",
        },
        "summary": {
            "video_stem": video_stem,
            "output_dir": audio_out.as_posix(),
            "backend": "ffmpeg",
        },
        "next_step": "audio_light_analysis",
    }

    logging.info("extract_audio done audio_path=%s", audio_path)
    return manifest
