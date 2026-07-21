"""Vidéo synthétique et fixtures de démonstration, entièrement locales."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image, ImageDraw, ImageFont

from .run_identity import atomic_write_json, canonical_sha256, sha256_file
from .vlm_result_validation import validate_single_result

DEMO_GENERATOR_VERSION = "1.0"
DEMO_FILENAME = "video_agent_demo.mp4"
DEMO_SPEC: Dict[str, Any] = {
    "generator_version": DEMO_GENERATOR_VERSION,
    "width": 360,
    "height": 640,
    "fps": 6,
    "duration_s": 18.0,
    "audio": False,
    "scenes": [
        {
            "segment_id": 1,
            "start_s": 0.0,
            "end_s": 6.0,
            "label": "CERCLE EN MOUVEMENT",
            "background": "#17324d",
        },
        {
            "segment_id": 2,
            "start_s": 6.0,
            "end_s": 12.0,
            "label": "CARRE EN DESCENTE",
            "background": "#d86f55",
        },
        {
            "segment_id": 3,
            "start_s": 12.0,
            "end_s": 18.0,
            "label": "TRIANGLE FINAL",
            "background": "#12392f",
        },
    ],
}
DEMO_SPEC_SHA256 = canonical_sha256(DEMO_SPEC)


class DemoError(Exception):
    """Erreur contrôlée du mode démonstration."""


def default_demo_path(project_root: Path) -> Path:
    return Path(project_root) / "outputs" / "demo" / DEMO_FILENAME


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - compatibilité Pillow ancienne
        return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    *,
    font: ImageFont.ImageFont,
    fill: str,
    width: int,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)


def render_demo_frame(frame_index: int) -> Image.Image:
    """Produit une frame RGB déterministe à partir du seul index."""
    fps = int(DEMO_SPEC["fps"])
    width, height = int(DEMO_SPEC["width"]), int(DEMO_SPEC["height"])
    total_frames = int(float(DEMO_SPEC["duration_s"]) * fps)
    if frame_index < 0 or frame_index >= total_frames:
        raise DemoError(f"Index de frame hors limites : {frame_index}.")
    second = frame_index / fps
    scene_index = min(2, int(second // 6))
    local = (second - scene_index * 6) / 6
    scene = DEMO_SPEC["scenes"][scene_index]
    image = Image.new("RGB", (width, height), str(scene["background"]))
    draw = ImageDraw.Draw(image)

    for x, y, radius in ((42, 122, 2), (308, 104, 3), (55, 518, 2), (292, 562, 2)):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ffffff55")

    draw.rounded_rectangle((24, 28, width - 24, 81), radius=14, fill="#ffffff18")
    _centered_text(
        draw, f"DEMO LOCALE  /  SCENE {scene_index + 1}", 44,
        font=_font(18), fill="#f6fbf8", width=width,
    )

    if scene_index == 0:
        x = int(58 + local * (width - 116))
        y = int(height * 0.47 + math.sin(local * math.pi * 2) * 48)
        radius = 42
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ffb04a")
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill="#17324d")
    elif scene_index == 1:
        size = 76
        x = int(width * 0.5 - size / 2 + math.sin(local * math.pi * 2) * 70)
        y = int(150 + local * 300)
        draw.rounded_rectangle((x, y, x + size, y + size), radius=13, fill="#17324d")
        draw.line((x + 16, y + 38, x + 60, y + 38), fill="#7ce0c4", width=6)
    else:
        center_x, center_y = width // 2, int(height * 0.45)
        radius = int(58 + 22 * math.sin(local * math.pi))
        points = []
        for angle in (-math.pi / 2, -math.pi / 2 + 2.094, -math.pi / 2 + 4.188):
            points.append((
                center_x + math.cos(angle + local * 0.5) * radius,
                center_y + math.sin(angle + local * 0.5) * radius,
            ))
        draw.polygon(points, fill="#78d9ba")
        draw.ellipse((center_x - 12, center_y - 12, center_x + 12, center_y + 12), fill="#12392f")

    draw.rounded_rectangle((28, height - 142, width - 28, height - 50), radius=18, fill="#081a2799")
    _centered_text(
        draw, str(scene["label"]), height - 117,
        font=_font(23), fill="#ffffff", width=width,
    )
    _centered_text(
        draw, f"{second:04.1f} s  /  18.0 s", height - 82,
        font=_font(15), fill="#a9dccc", width=width,
    )
    return image


def _encode_frames(output_path: Path, ffmpeg_path: str) -> None:
    spec = DEMO_SPEC
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{spec['width']}x{spec['height']}",
        "-framerate", str(spec["fps"]),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-threads", "1",
        "-map_metadata", "-1",
        "-fflags", "+bitexact",
        "-flags:v", "+bitexact",
        "-metadata", "title=Video Analysis Agent synthetic demo",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdin is None:
            raise DemoError("FFmpeg n'a pas ouvert son entrée vidéo.")
        frame_count = int(float(spec["duration_s"]) * int(spec["fps"]))
        for index in range(frame_count):
            process.stdin.write(render_demo_frame(index).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[-600:]
        raise DemoError(f"Échec de génération FFmpeg : {detail or return_code}.")


def generate_demo_video(
    output_path: Path,
    *,
    force: bool = False,
    ffmpeg_path: str | None = None,
) -> Dict[str, Any]:
    """Génère ou réutilise la vidéo synthétique et son manifest vérifiable."""
    output = Path(output_path).resolve()
    manifest_path = output.with_name("demo_manifest.json")
    if output.is_file() and manifest_path.is_file() and not force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("generator_spec_sha256") == DEMO_SPEC_SHA256
                and manifest.get("video_sha256") == sha256_file(output)
            ):
                return manifest
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass

    executable = ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise DemoError("FFmpeg est requis pour générer la vidéo de démonstration.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.generating.mp4")
    temporary.unlink(missing_ok=True)
    try:
        _encode_frames(temporary, executable)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": "1.0",
        "synthetic": True,
        "license": "CC0-1.0",
        "generator": "video_agent.demo",
        "generator_version": DEMO_GENERATOR_VERSION,
        "generator_spec_sha256": DEMO_SPEC_SHA256,
        "video_filename": output.name,
        "video_sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "width": DEMO_SPEC["width"],
        "height": DEMO_SPEC["height"],
        "fps": DEMO_SPEC["fps"],
        "duration_s": DEMO_SPEC["duration_s"],
        "audio": False,
        "scenes": DEMO_SPEC["scenes"],
        "network_used": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _load_fixture_templates(path: Path) -> list[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DemoError(f"Fixtures de démonstration illisibles : {path.name}.") from exc
    if not isinstance(data, dict) or data.get("fixture_type") != "simulated_vlm_demo":
        raise DemoError("Le fichier n'est pas une fixture VLM de démonstration reconnue.")
    templates = data.get("results")
    if not isinstance(templates, list) or not templates:
        raise DemoError("Les fixtures VLM de démonstration sont vides.")
    return [dict(item) for item in templates if isinstance(item, dict)]


def bind_demo_vlm_results(
    *,
    run_manifest: Dict[str, Any],
    vlm_pack: Dict[str, Any],
    demo_manifest: Dict[str, Any],
    fixtures_path: Path,
) -> list[Dict[str, Any]]:
    """Lie les fixtures simulées aux timestamps du run, après vérification du hash."""
    run_hash = run_manifest.get("video_sha256")
    demo_hash = demo_manifest.get("video_sha256")
    if not isinstance(run_hash, str) or run_hash != demo_hash:
        raise DemoError(
            "Fixtures refusées : ce run ne correspond pas à la vidéo synthétique générée."
        )
    if demo_manifest.get("generator_spec_sha256") != DEMO_SPEC_SHA256:
        raise DemoError("Fixtures refusées : version du générateur de démo incompatible.")
    segments = vlm_pack.get("segments")
    if not isinstance(segments, list) or not segments:
        raise DemoError("Le pack VLM du run ne contient aucun segment.")
    templates = _load_fixture_templates(fixtures_path)
    if len(templates) != len(segments):
        raise DemoError(
            "Fixtures refusées : le nombre de scènes détectées diffère du scénario de démo."
        )
    bound: list[Dict[str, Any]] = []
    for template, segment in zip(templates, segments):
        if not isinstance(segment, dict):
            raise DemoError("Segment du pack VLM invalide.")
        result = dict(template)
        result["segment_id"] = segment.get("segment_id")
        result["start_s"] = segment.get("start_s")
        result["end_s"] = segment.get("end_s")
        result["combined_text"] = segment.get("combined_text", "")
        validation = validate_single_result(
            result, int(segment["segment_id"]), segment
        )
        if validation.get("errors"):
            raise DemoError(
                "Fixture interne invalide : " + "; ".join(validation["errors"])
            )
        bound.append(result)
    return bound


def fixture_results_payload(results: Iterable[Dict[str, Any]], video_sha256: str) -> Dict[str, Any]:
    return {
        "fixture_type": "simulated_vlm_demo",
        "simulated": True,
        "model": None,
        "provider": None,
        "video_sha256": video_sha256,
        "notice": "Résultats synthétiques déterministes, non produits par un VLM.",
        "results": list(results),
    }
