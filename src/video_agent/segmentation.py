import json
import logging
from pathlib import Path
import subprocess
from typing import Tuple
from typing import Any, Dict, List, Optional, Union


def _get_threshold_from_strategy(strategy: Optional[Dict[str, Any]]) -> float:
    if not strategy:
        return 25.0

    video_type = strategy.get("video_type")
    if video_type == "short_vertical_video":
        return 15.0
    if video_type == "long_horizontal_video":
        return 40.0
    return 25.0


def segment_video(video_path: Union[str, Path], strategy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError as exc:
        raise ImportError(
            "PySceneDetect est requis pour la segmentation. Installez-le avec `python -m pip install scenedetect`."
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Fichier vidéo introuvable : {video_path}")

    logging.info("segment_video start video=%s strategy=%s", video_path, strategy)
    threshold = _get_threshold_from_strategy(strategy)
    segmentation_type = "visual_content"
    method = "PySceneDetect-ContentDetector"

    logging.info("opening video")
    video = open_video(str(video_path))
    logging.info("video opened")
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    logging.info("detector added threshold=%s", threshold)

    try:
        logging.info("detecting scenes")
        scene_manager.detect_scenes(video=video, show_progress=False)
        logging.info("scene detection completed")
        scene_list = scene_manager.get_scene_list(start_in_scene=True)
        logging.info("scene list length=%s", len(scene_list))
    finally:
        try:
            video.close()
        except Exception:
            pass

    # If no scene cuts are detected, fall back to a single segment covering entire video.
    if not scene_list:
        duration: Optional[float] = None
        # Ne jamais consulter un ancien artefact stem-based : inspection directe de
        # la vidéo courante uniquement, pour préserver l'isolation des runs.
        try:
            from src.video_agent.inspection import inspect_video as _inspect_video

            insp2 = _inspect_video(video_path)
            duration = insp2.get("file", {}).get("duration_s")
        except Exception:
            duration = None

        # As a final safety, ensure duration is a float (0.0 if unknown).
        try:
            duration = float(duration) if duration is not None else 0.0
        except Exception:
            duration = 0.0

        segments = [
            {
                "segment_id": 1,
                "start_s": 0.0,
                "end_s": round(duration, 3),
                "duration_s": round(duration, 3),
            }
        ]

        return {
            "video_path": str(video_path),
            "segmentation_type": "visual_content",
            "method": "PySceneDetect-ContentDetector",
            "segments": segments,
            "summary": {
                "segment_count": 1,
                "cut_count": 0,
                "single_scene_video": True,
                "fallback_used": True,
                "fallback_reason": "Aucun segment exploitable retourné par PySceneDetect : création d’un segment unique couvrant toute la vidéo.",
                "total_duration_s": round(duration, 3),
                "average_segment_duration_s": round(duration, 3),
            },
            "next_step": "frame_extraction",
        }

    segments: List[Dict[str, Any]] = []
    for segment_id, scene in enumerate(scene_list, start=1):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        duration = end_time - start_time
        segments.append(
            {
                "segment_id": segment_id,
                "start_s": round(start_time, 3),
                "end_s": round(end_time, 3),
                "duration_s": round(duration, 3),
            }
        )

    total_duration = sum(segment["duration_s"] for segment in segments)
    segment_count = len(segments)
    average_duration = round(total_duration / segment_count, 3) if segment_count else 0.0

    # Determine summary flags according to rules
    cut_count = max(0, segment_count - 1)
    single_scene_video = segment_count == 1
    fallback_used = False
    fallback_reason = None

    return {
        "video_path": str(video_path),
        "segmentation_type": segmentation_type,
        "method": method,
        "segments": segments,
        "summary": {
            "segment_count": segment_count,
            "cut_count": cut_count,
            "single_scene_video": single_scene_video,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "total_duration_s": round(total_duration, 3),
            "average_segment_duration_s": average_duration,
        },
        "next_step": "frame_extraction",
    }


def load_strategy(strategy: Any) -> Optional[Dict[str, Any]]:
    if isinstance(strategy, dict):
        return strategy
    if isinstance(strategy, str):
        strategy_path = Path(strategy)
        if strategy_path.exists() and strategy_path.is_file():
            try:
                return json.loads(strategy_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Impossible de lire le fichier de stratégie JSON : {exc}") from exc
        try:
            return json.loads(strategy)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Le paramètre strategy doit être un dictionnaire, un chemin de fichier JSON existant ou un JSON inline valide."
            ) from exc
    return None
