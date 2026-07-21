"""Niveau 4 — Extraction intelligente des frames (backend FFmpeg).

Ce module extrait des frames représentatives d'une vidéo, segment par segment,
en s'appuyant sur la stratégie (`frame_strategy.mode`) pour décider combien de
frames prélever et où les placer dans chaque segment.

Backend d'extraction : **FFmpeg** (via subprocess). Aucune dépendance OpenCV/cv2
pour l'extraction principale.

Non couvert ici (étapes ultérieures) : OCR, YOLO, VLM, motion composite,
analyse sémantique, tool MCP, script CLI.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Optional


class FrameExtractionError(Exception):
    """Erreur générique de l'extraction de frames."""


# Paramètres de densité par mode de stratégie : (frames/s cible, min, max).
_MODE_DENSITY: Dict[str, tuple[float, int, int]] = {
    "dense_sampling": (0.45, 8, 12),
    "medium_sampling": (0.22, 4, 6),
    "sparse_sampling": (0.10, 2, 3),
}
_DEFAULT_MODE = "medium_sampling"

# Espacement minimum souhaité entre deux frames (garde anti-suréchantillonnage,
# vise surtout les LONGS segments).
_MIN_SPACING_S = 0.5
# Frames minimum par segment : un composite de mouvement à 1 frame n'encode AUCUN
# mouvement (cf. _build_motion_composite). On garantit ≥ ce seuil quand le segment
# peut tenir autant de frames DISTINCTES ; sinon fallback honnête (moins de frames).
_DEFAULT_MIN_FRAMES = 2
# Séparation minimale pour considérer 2 frames comme DISTINCTES (plus fine que la garde
# anti-suréchantillonnage ci-dessus : un écart de ~0.2 s suffit à encoder du mouvement).
_MIN_DISTINCT_SPACING_S = 0.2
# Délai max accordé à FFmpeg pour extraire une seule frame.
_FFMPEG_FRAME_TIMEOUT_S = 60.0


def _get_frame_strategy_mode(strategy: Dict[str, Any]) -> str:
    """Retourne le mode d'échantillonnage de frames depuis la stratégie.

    Tolérant : si la stratégie est absente/malformée ou le mode inconnu, on
    retombe sur un échantillonnage moyen plutôt que d'échouer.
    """
    if not isinstance(strategy, dict):
        return _DEFAULT_MODE
    mode = strategy.get("frame_strategy", {}).get("mode")
    if mode in _MODE_DENSITY:
        return mode
    return _DEFAULT_MODE


def _compute_frame_count(
    duration_s: float,
    mode: str,
    min_frames_per_segment: int = _DEFAULT_MIN_FRAMES,
) -> int:
    """Calcule le nombre de frames à extraire pour un segment.

    Logique générique :
      1. n_brut = round(durée * densité_cible)
      2. n borné par [min, max] du mode
      3. garde anti-suréchantillonnage : au moins ~0.5 s entre deux frames
      4. garantie « mouvement » : au moins ``min_frames_per_segment`` frames SI le
         segment peut tenir autant de frames DISTINCTES (écart ≥ ~0.2 s) ; sinon
         fallback HONNÊTE (moins de frames -> le composite restera 1 frame et la note
         d'incertitude « mouvement non exploité » reste légitime).
    """
    if duration_s <= 0:
        raise FrameExtractionError(
            f"Durée de segment invalide pour le calcul du nombre de frames : {duration_s}"
        )

    density, min_f, max_f = _MODE_DENSITY.get(mode, _MODE_DENSITY[_DEFAULT_MODE])

    n = round(duration_s * density)
    n = max(min_f, min(n, max_f))

    # Garde anti-suréchantillonnage : ne pas dépasser ce que l'espacement min autorise.
    spacing_cap = max(1, floor(duration_s / _MIN_SPACING_S))
    n = min(n, spacing_cap)

    # Garantie « mouvement » : remonter à min_frames_per_segment frames tant que le
    # segment peut les contenir DISTINCTEMENT (séparation ≥ _MIN_DISTINCT_SPACING_S).
    max_distinct = max(1, floor(duration_s / _MIN_DISTINCT_SPACING_S))
    target = min(max(1, int(min_frames_per_segment)), max_distinct)
    n = max(n, target)

    return max(1, int(n))


def _compute_timestamps(start_s: float, end_s: float, frame_count: int) -> List[float]:
    """Répartit `frame_count` timestamps à l'intérieur du segment [start_s, end_s].

    Formule (centres de sous-intervalles) : t_i = start + durée * (i + 0.5) / n.
    Ne touche jamais exactement le début ni la fin du segment. Pour une seule
    frame, prend le centre.
    """
    duration = end_s - start_s
    if duration <= 0:
        raise FrameExtractionError(
            f"Segment invalide pour le calcul des timestamps : start={start_s}, end={end_s}"
        )
    if frame_count <= 1:
        return [round(start_s + duration / 2.0, 6)]

    return [
        round(start_s + duration * (i + 0.5) / frame_count, 6)
        for i in range(frame_count)
    ]


def _check_ffmpeg_available() -> str:
    """Retourne le chemin de l'exécutable ffmpeg, ou lève une erreur claire."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FrameExtractionError(
            "FFmpeg est introuvable dans le PATH. Installez FFmpeg et assurez-vous "
            "que la commande `ffmpeg` est accessible."
        )
    return ffmpeg


def _safe_mkdir(path: Path) -> None:
    """Crée un répertoire (et ses parents) sans échouer s'il existe déjà."""
    path.mkdir(parents=True, exist_ok=True)


def _validate_segments(segments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Valide la structure des segments et retourne la liste exploitable.

    Vérifie : présence d'une liste non vide, et pour chaque segment des champs
    `start_s`/`end_s` numériques avec end_s > start_s.
    """
    if not isinstance(segments, dict):
        raise FrameExtractionError("`segments` doit être un dictionnaire.")

    seg_list = segments.get("segments")
    if not isinstance(seg_list, list) or len(seg_list) == 0:
        raise FrameExtractionError(
            "Aucun segment exploitable : la clé `segments` est vide ou absente."
        )

    for index, seg in enumerate(seg_list, start=1):
        if not isinstance(seg, dict):
            raise FrameExtractionError(f"Segment #{index} invalide (pas un objet).")
        if "start_s" not in seg or "end_s" not in seg:
            raise FrameExtractionError(
                f"Segment #{index} sans `start_s`/`end_s` : {seg!r}"
            )
        try:
            start_s = float(seg["start_s"])
            end_s = float(seg["end_s"])
        except (TypeError, ValueError) as exc:
            raise FrameExtractionError(
                f"Segment #{index} avec start_s/end_s non numériques : {seg!r}"
            ) from exc
        if end_s <= start_s:
            raise FrameExtractionError(
                f"Segment #{index} invalide : end_s ({end_s}) <= start_s ({start_s})."
            )

    return seg_list


def _validate_extracted_file(path: Path) -> None:
    """Vérifie qu'un fichier image a bien été créé et n'est pas vide."""
    if not path.exists() or not path.is_file():
        raise FrameExtractionError(f"Frame non créée par FFmpeg : {path}")
    if path.stat().st_size == 0:
        raise FrameExtractionError(f"Frame créée mais vide (0 octet) : {path}")


def _build_scale_filter(
    max_width: Optional[int], max_height: Optional[int]
) -> Optional[str]:
    """Construit un filtre FFmpeg `scale` borné, en conservant le ratio.

    Retourne None si aucun redimensionnement n'est demandé.
    """
    if not max_width and not max_height:
        return None
    if max_width and max_height:
        # Boîte englobante : réduit pour tenir dans max_width × max_height.
        return (
            f"scale='min({max_width},iw)':'min({max_height},ih)'"
            ":force_original_aspect_ratio=decrease"
        )
    if max_width:
        # -2 : laisse FFmpeg calculer une hauteur paire conservant le ratio.
        return f"scale={max_width}:-2"
    return f"scale=-2:{max_height}"


def _build_ffmpeg_command(
    ffmpeg: str,
    video_path: str,
    timestamp_s: float,
    output_path: Path,
    overwrite: bool,
    max_width: Optional[int],
    max_height: Optional[int],
) -> List[str]:
    """Construit la commande FFmpeg pour extraire une frame unique à un timestamp.

    Conceptuellement : ffmpeg -y -ss <ts> -i <video> -frames:v 1 <output>.
    Le `-ss` placé avant `-i` (seek d'entrée) rend l'extraction rapide.
    """
    cmd: List[str] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-ss", f"{timestamp_s:.6f}",
        "-i", video_path,
        "-frames:v", "1",
    ]

    scale_filter = _build_scale_filter(max_width, max_height)
    if scale_filter:
        cmd += ["-vf", scale_filter]

    # Qualité JPEG (2 = très bonne ; 1 = max). Frame unique.
    cmd += ["-q:v", "2", str(output_path)]
    return cmd


def extract_frames(
    video_path: str,
    segments: Dict[str, Any],
    strategy: Dict[str, Any],
    output_dir: str,
    overwrite: bool = True,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    min_frames_per_segment: int = _DEFAULT_MIN_FRAMES,
) -> Dict[str, Any]:
    """Extrait les frames de chaque segment via FFmpeg et retourne un manifest.

    Args:
        video_path: chemin de la vidéo source.
        segments: dict de segmentation (doit contenir une liste `segments`).
        strategy: dict de stratégie (utilise `frame_strategy.mode`).
        output_dir: répertoire racine de sortie ; les frames sont écrites sous
            ``<output_dir>/<video_stem>/segment_NNN/frame_MMM.jpg``.
        overwrite: réécrire les frames existantes (sinon réutilise l'existant).
        max_width / max_height: redimensionnement optionnel (désactivé par défaut).
        min_frames_per_segment: garantie « mouvement » (défaut 2) — au moins ce nombre
            de frames par segment quand celui-ci peut les contenir distinctement.

    Returns:
        Un manifest dict (voir la structure en fin de fonction).
    """
    video = Path(video_path)
    if not video.exists() or not video.is_file():
        raise FrameExtractionError(f"Vidéo introuvable : {video_path}")

    ffmpeg = _check_ffmpeg_available()
    seg_list = _validate_segments(segments)
    mode = _get_frame_strategy_mode(strategy)

    video_stem = video.stem
    video_out = Path(output_dir) / video_stem
    _safe_mkdir(video_out)

    logging.info(
        "extract_frames start video=%s mode=%s segments=%d output=%s",
        video_path, mode, len(seg_list), video_out,
    )

    frames: List[Dict[str, Any]] = []
    frames_per_segment: Dict[str, int] = {}
    frame_id = 0

    for seg_index, seg in enumerate(seg_list, start=1):
        segment_id = int(seg.get("segment_id", seg_index))
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])
        duration = end_s - start_s

        n = _compute_frame_count(duration, mode, min_frames_per_segment)
        timestamps = _compute_timestamps(start_s, end_s, n)
        frames_per_segment[str(segment_id)] = len(timestamps)

        seg_dir = video_out / f"segment_{segment_id:03d}"
        _safe_mkdir(seg_dir)

        if len(timestamps) == 1:
            # Fallback honnête : segment trop court pour 2 frames distinctes -> le
            # composite restera 1 frame (mouvement non exploitable, à signaler en aval).
            reason = (
                f"{mode} : segment trop court ({duration:.3f}s) pour 2 frames distinctes "
                f"→ 1 frame (mouvement non exploitable)"
            )
        else:
            reason = (
                f"{mode} : {len(timestamps)} frame(s) réparties uniformément "
                f"dans le segment {segment_id} ({duration:.3f}s)"
            )

        for local_index, timestamp in enumerate(timestamps, start=1):
            frame_id += 1
            frame_path = seg_dir / f"frame_{local_index:03d}.jpg"

            need_extraction = overwrite or not frame_path.exists()
            if need_extraction:
                cmd = _build_ffmpeg_command(
                    ffmpeg, str(video), timestamp, frame_path,
                    overwrite, max_width, max_height,
                )
                logging.info("ffmpeg frame ts=%.3f -> %s", timestamp, frame_path)
                try:
                    proc = subprocess.run(
                        cmd,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=_FFMPEG_FRAME_TIMEOUT_S,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise FrameExtractionError(
                        f"FFmpeg a dépassé le délai ({_FFMPEG_FRAME_TIMEOUT_S}s) "
                        f"pour la frame à {timestamp:.3f}s."
                    ) from exc

                if proc.returncode != 0:
                    raise FrameExtractionError(
                        f"FFmpeg a échoué (code {proc.returncode}) pour la frame à "
                        f"{timestamp:.3f}s : {proc.stderr.strip()[-500:]}"
                    )

            _validate_extracted_file(frame_path)

            frames.append(
                {
                    "frame_id": frame_id,
                    "segment_id": segment_id,
                    "timestamp_s": round(timestamp, 3),
                    "path": frame_path.as_posix(),
                    "reason": reason,
                }
            )

    manifest: Dict[str, Any] = {
        "video_path": str(video_path),
        "extraction_type": "strategy_aware_ffmpeg_frame_extraction",
        "frames": frames,
        "summary": {
            "segment_count": len(seg_list),
            "frame_count": len(frames),
            "frames_per_segment": frames_per_segment,
            "output_dir": video_out.as_posix(),
            "extraction_backend": "ffmpeg",
        },
        "next_step": "motion_representation",
    }

    logging.info(
        "extract_frames done frames=%d segments=%d", len(frames), len(seg_list)
    )
    return manifest
