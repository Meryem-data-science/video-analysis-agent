"""Niveau 5 — Motion representation (backend Pillow).

À partir d'un manifest de frames (Niveau 4), ce module produit, par segment,
deux représentations visuelles du mouvement :

  1. ``motion_grid.jpg``      — grille temporelle (mosaïque des frames ordonnées,
                                 chaque vignette annotée frame_id + timestamp).
  2. ``motion_composite.jpg`` — composite par superposition transparente (moyenne
                                 des frames) : le décor statique reste net, les
                                 zones en mouvement laissent des traînées.

Backend : **Pillow**. Aucune analyse sémantique / VLM / OCR / YOLO.
"""

from __future__ import annotations

import logging
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


class MotionRepresentationError(Exception):
    """Erreur générique de la construction des représentations de mouvement."""


# Couleur de fond neutre des grilles et marge entre vignettes.
_GRID_BG = (32, 32, 32)
_GRID_PADDING = 6
# Bandeau de label sous chaque vignette.
_LABEL_BG = (0, 0, 0)
_LABEL_FG = (255, 255, 255)
_LABEL_HEIGHT = 22


def _validate_frames_manifest(frames_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Valide le manifest de frames et retourne la liste de frames exploitable.

    Vérifie : structure dict, présence d'une liste `frames` non vide, et pour
    chaque frame les champs `segment_id`, `timestamp_s` et `path`.
    """
    if not isinstance(frames_manifest, dict):
        raise MotionRepresentationError("`frames_manifest` doit être un dictionnaire.")

    frames = frames_manifest.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise MotionRepresentationError(
            "Aucune frame exploitable : la clé `frames` est vide ou absente."
        )

    for index, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict):
            raise MotionRepresentationError(f"Frame #{index} invalide (pas un objet).")
        if "segment_id" not in frame:
            raise MotionRepresentationError(f"Frame #{index} sans `segment_id` : {frame!r}")
        if "timestamp_s" not in frame:
            raise MotionRepresentationError(f"Frame #{index} sans `timestamp_s` : {frame!r}")
        if "path" not in frame:
            raise MotionRepresentationError(f"Frame #{index} sans `path` : {frame!r}")

    return frames


def _group_frames_by_segment(frames: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Groupe les frames par `segment_id` ; chaque groupe est trié par timestamp.

    Retourne un dict ordonné par segment_id croissant.
    """
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for frame in frames:
        try:
            segment_id = int(frame["segment_id"])
        except (TypeError, ValueError) as exc:
            raise MotionRepresentationError(
                f"`segment_id` non entier : {frame!r}"
            ) from exc
        groups.setdefault(segment_id, []).append(frame)

    for segment_id in groups:
        groups[segment_id].sort(key=lambda f: (float(f["timestamp_s"]), int(f.get("frame_id", 0))))

    # Réordonner le dict par segment_id croissant.
    return {sid: groups[sid] for sid in sorted(groups)}


def _compute_grid_layout(frame_count: int) -> tuple[int, int]:
    """Calcule (cols, rows) : cols = ceil(sqrt(N)), rows = ceil(N / cols)."""
    if frame_count <= 0:
        raise MotionRepresentationError(
            f"Nombre de frames invalide pour la grille : {frame_count}"
        )
    cols = ceil(sqrt(frame_count))
    rows = ceil(frame_count / cols)
    return cols, rows


def _load_image(path: str) -> Image.Image:
    """Charge une image en RGB avec des erreurs claires."""
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise MotionRepresentationError(f"Frame introuvable : {image_path}")
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise MotionRepresentationError(
            f"Image illisible ({image_path}) : {exc}"
        ) from exc


def _resize_keep_ratio(
    image: Image.Image,
    target_width: Optional[int] = None,
    max_long_side: Optional[int] = None,
) -> Image.Image:
    """Redimensionne en conservant le ratio.

    - `target_width` : impose la largeur (hauteur calculée au prorata).
    - `max_long_side` : borne le plus grand côté (réduction uniquement).
    """
    w, h = image.size
    if w == 0 or h == 0:
        raise MotionRepresentationError("Image de dimensions nulles.")

    if target_width is not None:
        new_w = max(1, int(target_width))
        new_h = max(1, round(h * new_w / w))
        return image.resize((new_w, new_h), Image.LANCZOS)

    if max_long_side is not None:
        long_side = max(w, h)
        if long_side <= max_long_side:
            return image
        scale = max_long_side / long_side
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        return image.resize((new_w, new_h), Image.LANCZOS)

    return image


def _load_font() -> ImageFont.ImageFont:
    """Charge une police de petite taille (repli sur la police par défaut)."""
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except Exception:
        return ImageFont.load_default()


def _draw_label(image: Image.Image, text: str) -> Image.Image:
    """Ajoute un bandeau de label en bas de la vignette (frame_id + timestamp)."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.rectangle([0, h - _LABEL_HEIGHT, w, h], fill=_LABEL_BG)
    font = _load_font()
    draw.text((4, h - _LABEL_HEIGHT + 4), text, fill=_LABEL_FG, font=font)
    return image


def _frame_label(frame: Dict[str, Any]) -> str:
    """Construit le texte du label : ex. ``F03 · 5.758s``."""
    frame_id = frame.get("frame_id", "?")
    timestamp = frame.get("timestamp_s", "?")
    try:
        fid = f"F{int(frame_id):02d}"
    except (TypeError, ValueError):
        fid = f"F{frame_id}"
    return f"{fid} · {timestamp}s"


def _validate_output_file(path: Path) -> None:
    """Vérifie qu'un fichier de sortie a bien été créé et n'est pas vide."""
    if not path.exists() or not path.is_file():
        raise MotionRepresentationError(f"Fichier de sortie non créé : {path}")
    if path.stat().st_size == 0:
        raise MotionRepresentationError(f"Fichier de sortie vide (0 octet) : {path}")


def _build_motion_grid(
    segment_frames: List[Dict[str, Any]],
    output_path: Path,
    thumbnail_width: int,
) -> tuple[int, int]:
    """Construit la grille temporelle et l'enregistre. Retourne (cols, rows)."""
    n = len(segment_frames)
    cols, rows = _compute_grid_layout(n)

    # Charger + redimensionner les vignettes (largeur cible commune) + labels.
    thumbs: List[Image.Image] = []
    for frame in segment_frames:
        img = _resize_keep_ratio(_load_image(frame["path"]), target_width=thumbnail_width)
        img = _draw_label(img, _frame_label(frame))
        thumbs.append(img)

    # Dimensions de cellule : la plus grande vignette (uniformise la grille).
    cell_w = max(t.size[0] for t in thumbs)
    cell_h = max(t.size[1] for t in thumbs)

    grid_w = cols * cell_w + (cols + 1) * _GRID_PADDING
    grid_h = rows * cell_h + (rows + 1) * _GRID_PADDING
    grid = Image.new("RGB", (grid_w, grid_h), _GRID_BG)

    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = _GRID_PADDING + c * (cell_w + _GRID_PADDING)
        y = _GRID_PADDING + r * (cell_h + _GRID_PADDING)
        # Centrer la vignette dans sa cellule.
        x += (cell_w - thumb.size[0]) // 2
        y += (cell_h - thumb.size[1]) // 2
        grid.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, format="JPEG", quality=90)
    _validate_output_file(output_path)
    return cols, rows


def _build_motion_composite(
    segment_frames: List[Dict[str, Any]],
    output_path: Path,
    composite_long_side: int,
) -> None:
    """Construit le composite transparent (moyenne des frames) et l'enregistre.

    Toutes les frames sont ramenées à une taille commune (celle de la première
    après réduction au côté long max), puis moyennées à poids égal (1/N).
    """
    if not segment_frames:
        raise MotionRepresentationError("Aucune frame pour le composite.")

    # Taille commune : on se cale sur la première frame réduite.
    base = _resize_keep_ratio(
        _load_image(segment_frames[0]["path"]), max_long_side=composite_long_side
    )
    target_size = base.size

    # Cas une seule frame : le composite est simplement cette frame redimensionnée.
    if len(segment_frames) == 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(output_path, format="JPEG", quality=90)
        _validate_output_file(output_path)
        return

    # Accumulation en flottant pour une moyenne propre (sans dépendre de numpy).
    n = len(segment_frames)
    accum: Optional[List[float]] = None
    for frame in segment_frames:
        img = _load_image(frame["path"])
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        pixels = list(img.get_flattened_data())  # liste de tuples (R, G, B)
        if accum is None:
            accum = [0.0] * (len(pixels) * 3)
        for i, (r, g, b) in enumerate(pixels):
            accum[i * 3] += r
            accum[i * 3 + 1] += g
            accum[i * 3 + 2] += b

    assert accum is not None
    averaged = [int(round(v / n)) for v in accum]
    flat = tuple(averaged)
    composite = Image.new("RGB", target_size)
    composite.putdata(list(zip(flat[0::3], flat[1::3], flat[2::3])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path, format="JPEG", quality=90)
    _validate_output_file(output_path)


def build_motion_representations(
    frames_manifest: Dict[str, Any],
    output_dir: str = "outputs/motion",
    thumbnail_width: int = 360,
    composite_long_side: int = 1080,
    overwrite: bool = True,
    source_frames_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Construit grille + composite par segment et retourne un manifest.

    Args:
        frames_manifest: manifest de frames du Niveau 4 (clé `frames`).
        output_dir: répertoire racine ; sortie sous ``<output_dir>/<video_stem>/segment_NNN/``.
        thumbnail_width: largeur cible des vignettes de la grille.
        composite_long_side: côté long max des frames pour le composite.
        overwrite: réécrire les images existantes (sinon réutilise l'existant).
        source_frames_manifest_path: chemin du manifest de frames source ; s'il est
            fourni, il est reporté tel quel dans le manifest retourné (sinon None).

    Returns:
        Un manifest dict (voir structure en fin de fonction).
    """
    frames = _validate_frames_manifest(frames_manifest)

    video_path = frames_manifest.get("video_path", "")
    if not video_path:
        raise MotionRepresentationError("`video_path` absent du manifest de frames.")
    video_stem = Path(video_path).stem

    video_out = Path(output_dir) / video_stem
    video_out.mkdir(parents=True, exist_ok=True)

    grouped = _group_frames_by_segment(frames)
    logging.info(
        "build_motion_representations start video_stem=%s segments=%d output=%s",
        video_stem, len(grouped), video_out,
    )

    segments_out: List[Dict[str, Any]] = []
    image_count = 0

    for segment_id, seg_frames in grouped.items():
        seg_dir = video_out / f"segment_{segment_id:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        grid_path = seg_dir / "motion_grid.jpg"
        composite_path = seg_dir / "motion_composite.jpg"

        if overwrite or not grid_path.exists():
            cols, rows = _build_motion_grid(seg_frames, grid_path, thumbnail_width)
        else:
            cols, rows = _compute_grid_layout(len(seg_frames))
            _validate_output_file(grid_path)
        image_count += 1

        if overwrite or not composite_path.exists():
            _build_motion_composite(seg_frames, composite_path, composite_long_side)
        else:
            _validate_output_file(composite_path)
        image_count += 1

        segments_out.append(
            {
                "segment_id": segment_id,
                "frame_count": len(seg_frames),
                "grid_path": grid_path.as_posix(),
                "composite_path": composite_path.as_posix(),
                "grid_layout": {"cols": cols, "rows": rows},
                "source_frames": [
                    {
                        "frame_id": f.get("frame_id"),
                        "timestamp_s": f.get("timestamp_s"),
                        "path": f.get("path"),
                    }
                    for f in seg_frames
                ],
            }
        )

    manifest: Dict[str, Any] = {
        "video_path": str(video_path),
        "source_frames_manifest": source_frames_manifest_path,
        "motion_type": "grid_and_transparent_composite",
        "segments": segments_out,
        "summary": {
            "segment_count": len(segments_out),
            "image_count": image_count,
            "output_dir": video_out.as_posix(),
            "backend": "pillow",
        },
        "next_step": "vlm_analysis",
    }

    logging.info(
        "build_motion_representations done segments=%d images=%d",
        len(segments_out), image_count,
    )
    return manifest
