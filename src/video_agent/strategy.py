from typing import Any, Dict


def decide_strategy(inspection: Dict[str, Any]) -> Dict[str, Any]:
    """Décide une stratégie d'exécution à partir du JSON d'inspection vidéo."""
    video = inspection.get("video", {})
    audio = inspection.get("audio", {})
    subtitles = inspection.get("subtitles", {})

    width = video.get("width") or 0
    height = video.get("height") or 0
    duration = inspection.get("file", {}).get("duration_s") or 0
    orientation = video.get("orientation", "unknown")

    # Détermination du type vidéo
    if orientation == "vertical" and duration <= 60:
        video_type = "short_vertical_video"
    elif orientation in ("horizontal", "square") and duration >= 300:
        video_type = "long_horizontal_video"
    else:
        video_type = "generic_video"

    # Audio strategy
    audio_strategy = {
        "mode": "prepare_audio_analysis" if audio.get("present") else "skip_audio_analysis"
    }

    # Subtitle strategy
    subtitle_strategy = {
        "mode": "use_soft_subtitles" if subtitles.get("soft_subtitles_present") else "no_soft_subtitles"
    }

    # Frame strategy based on durée et orientation
    if video_type == "short_vertical_video":
        frame_mode = "dense_sampling"
        frame_reason = "Vidéo courte et verticale : privilégier une couverture dense pour capturer les détails du format portrait."
    elif video_type == "long_horizontal_video":
        frame_mode = "sparse_sampling"
        frame_reason = "Vidéo longue : privilégier un échantillonnage plus clairsemé pour réduire le volume de données tout en conservant la structure temporelle."
    else:
        frame_mode = "medium_sampling"
        frame_reason = "Vidéo standard : équilibre entre couverture temporelle et coût de traitement."

    frame_strategy = {"mode": frame_mode, "reason": frame_reason}

    # Resize strategy
    if width >= 1920 or height >= 1920:
        resize_mode = "resize_before_vlm"
        resize_reason = "Résolution élevée : réduire la taille avant le passage au VLM pour optimiser le traitement tout en conservant l'information visuelle."
    else:
        resize_mode = "keep_original_if_reasonable"
        resize_reason = "Résolution raisonnable : conserver l'aspect d'origine pour le VLM si possible."

    resize_strategy = {"mode": resize_mode, "reason": resize_reason}

    visual_text_strategy = {
        "mode": "vlm_native_only",
        "use_ocr": False,
        "run_ocr_now": False,
        "reason": (
            "La V1 du projet ne prévoit pas d’OCR séparé. Le texte visible dans les frames sera traité plus tard "
            "directement par le VLM, dans le même appel que la compréhension de scène."
        ),
        "failure_recovery": [
            "temporal_resampling",
            "smart_cropping",
            "multi_frame_analysis",
            "image_preprocessing",
            "alternate_vlm",
            "manual_review_if_uncertain",
        ],
    }

    object_detection_strategy = {
        "mode": "vlm_first_detector_only_if_precise_boxes_required",
        "use_yolo_by_default": False,
        "reason": (
            "Le VLM sera utilisé plus tard pour la compréhension générale des objets et de la scène. "
            "Un détecteur spécialisé ne sera envisagé que si le besoin exige des bounding boxes précises."
        ),
    }

    motion_strategy = {
        "mode": "prepare_motion_composite_if_needed",
        "reason": (
            "Les VLM comprennent mieux le mouvement si plusieurs frames sont représentées dans une image composite "
            "ou une grille temporelle."
        ),
    }

    model_strategy = {
        "primary": "frontier_vlm",
        "alternatives": [
            "alternate_frontier_vlm",
            "manual_review_if_uncertain",
        ],
        "specialized_models_by_default": False,
    }

    return {
        "video_type": video_type,
        "audio_strategy": audio_strategy,
        "subtitle_strategy": subtitle_strategy,
        "frame_strategy": frame_strategy,
        "resize_strategy": resize_strategy,
        "visual_text_strategy": visual_text_strategy,
        "object_detection_strategy": object_detection_strategy,
        "motion_strategy": motion_strategy,
        "model_strategy": model_strategy,
        "execution_flags": {
            "use_ocr": False,
            "use_yolo": False,
            "run_vlm": False,
            "run_segmentation": False,
            "extract_frames": False,
        },
        "confidence_policy": {
            "preferred": "model_confidence_threshold",
            "fallback": "manual_review_if_uncertain",
            "threshold": 0.85,
        },
        "next_step": "segmentation",
    }
