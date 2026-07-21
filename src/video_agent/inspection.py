import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


class InspectError(Exception):
    pass


def _parse_frac(frac: str) -> Optional[float]:
    if not frac:
        return None
    if isinstance(frac, (int, float)):
        return float(frac)
    try:
        if "/" in frac:
            num, den = frac.split("/")
            num = float(num)
            den = float(den)
            if den == 0:
                return None
            return num / den
        return float(frac)
    except Exception:
        return None


def _get_rotation(stream: Dict[str, Any]) -> int:
    # rotation may appear in tags or side_data_list
    tags = stream.get("tags", {}) or {}
    rot = tags.get("rotate") or tags.get("rotation")
    if rot is not None:
        try:
            return int(float(rot))
        except Exception:
            pass
    # check side_data_list
    for sd in stream.get("side_data_list", []) or []:
        if sd.get("rotation") is not None:
            try:
                return int(float(sd.get("rotation")))
            except Exception:
                pass
    return 0


def inspect_video(video_path: str | Path, ffprobe_timeout: int = 20) -> Dict[str, Any]:
    """Inspecte techniquement une vidéo via ffprobe.

    Appelle ffprobe avec:
      ffprobe -v error -print_format json -show_format -show_streams -show_chapters <video>

    Retourne un dict normalisé.
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    if not path.is_file():
        raise InspectError(f"Le chemin n'est pas un fichier: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=ffprobe_timeout,
        )
    except FileNotFoundError as e:
        raise InspectError("FFprobe introuvable. Installez FFmpeg/ffprobe et réessayez.") from e
    except subprocess.TimeoutExpired as e:
        raise InspectError("FFprobe a expiré (timeout).") from e
    except subprocess.CalledProcessError as e:
        # ffprobe returned non-zero but may still have stderr
        raise InspectError(f"FFprobe a échoué: {e.stderr or e}") from e

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise InspectError("Sortie FFprobe JSON invalide") from e

    fmt = raw.get("format", {}) or {}
    streams = raw.get("streams", []) or []

    # File section
    file_info: Dict[str, Any] = {}
    file_info["path"] = str(path)
    file_info["name"] = path.name
    file_info["extension"] = path.suffix.lstrip(".").lower()
    file_info["format"] = fmt.get("format_name") or fmt.get("format_long_name")

    # duration
    duration_s = None
    try:
        duration_s = float(fmt.get("duration")) if fmt.get("duration") is not None else None
    except Exception:
        duration_s = None
    file_info["duration_s"] = duration_s

    # size
    try:
        size_bytes = path.stat().st_size
        file_info["size_mb"] = round(size_bytes / (1024 * 1024), 2)
    except Exception:
        file_info["size_mb"] = None

    # bitrate
    try:
        bitrate_kbps = int(fmt.get("bit_rate")) // 1000 if fmt.get("bit_rate") else None
    except Exception:
        bitrate_kbps = None
    file_info["bitrate_kbps"] = bitrate_kbps

    # Streams summary
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

    # Video block
    video_present = len(video_streams) > 0
    video_info: Dict[str, Any] = {
        "present": video_present,
        "stream_count": len(video_streams),
    }

    primary_video = video_streams[0] if video_present else None

    if primary_video:
        v_codec = primary_video.get("codec_name")
        v_profile = primary_video.get("profile")
        width = primary_video.get("width")
        height = primary_video.get("height")
        rotation = _get_rotation(primary_video)

        # Adjust orientation according to rotation
        adj_width, adj_height = width, height
        if rotation in (90, 270) and width and height:
            adj_width, adj_height = height, width

        if adj_width and adj_height:
            if adj_width > adj_height:
                orientation = "horizontal"
            elif adj_width < adj_height:
                orientation = "vertical"
            else:
                orientation = "square"
        else:
            orientation = "unknown"

        pix_fmt = primary_video.get("pix_fmt")

        avg_frame_rate = _parse_frac(primary_video.get("avg_frame_rate"))
        raw_frame_rate = _parse_frac(primary_video.get("r_frame_rate"))

        # fps_used: prefer avg if available, else r_frame_rate
        fps_used = None
        if avg_frame_rate:
            fps_used = avg_frame_rate
        elif raw_frame_rate:
            fps_used = raw_frame_rate

        is_vfr = False
        try:
            # If r_frame_rate and avg_frame_rate differ significantly -> variable
            if avg_frame_rate and raw_frame_rate and abs(avg_frame_rate - raw_frame_rate) > 0.01:
                is_vfr = True
        except Exception:
            is_vfr = False

        estimated_frames = None
        try:
            if primary_video.get("nb_frames") and primary_video.get("nb_frames") != "N/A":
                estimated_frames = int(primary_video.get("nb_frames"))
            elif duration_s and fps_used:
                estimated_frames = int(round(duration_s * fps_used))
        except Exception:
            estimated_frames = None

        video_info.update(
            {
                "codec": v_codec,
                "profile": v_profile,
                "width": width,
                "height": height,
                "rotation": rotation,
                "orientation": orientation,
                "pix_fmt": pix_fmt,
                "avg_fps": avg_frame_rate,
                "raw_fps": raw_frame_rate,
                "fps_used": fps_used,
                "is_variable_fps": is_vfr,
                "estimated_frames": estimated_frames,
            }
        )
    else:
        video_info.update(
            {
                "codec": None,
                "profile": None,
                "width": None,
                "height": None,
                "rotation": None,
                "orientation": None,
                "pix_fmt": None,
                "avg_fps": None,
                "raw_fps": None,
                "fps_used": None,
                "is_variable_fps": None,
                "estimated_frames": None,
            }
        )

    # Audio block
    audio_present = len(audio_streams) > 0
    audio_info: Dict[str, Any] = {
        "present": audio_present,
        "stream_count": len(audio_streams),
    }
    if audio_present:
        primary_audio = audio_streams[0]
        a_codec = primary_audio.get("codec_name")
        a_sample_rate = None
        try:
            a_sample_rate = int(primary_audio.get("sample_rate")) if primary_audio.get("sample_rate") else None
        except Exception:
            a_sample_rate = None
        a_channels = primary_audio.get("channels")
        a_lang = (primary_audio.get("tags") or {}).get("language")
        audio_info.update(
            {
                "codec": a_codec,
                "sample_rate": a_sample_rate,
                "channels": a_channels,
                "language": a_lang,
            }
        )
    else:
        audio_info.update({"codec": None, "sample_rate": None, "channels": None, "language": None})

    # Subtitles
    subs_info: Dict[str, Any] = {
        "soft_subtitles_present": len(subtitle_streams) > 0,
        "stream_count": len(subtitle_streams),
        "streams": [],
    }
    for s in subtitle_streams:
        subs_info["streams"].append(
            {
                "index": s.get("index"),
                "codec": s.get("codec_name"),
                "language": (s.get("tags") or {}).get("language"),
                "title": (s.get("tags") or {}).get("title"),
            }
        )

    # Derived heuristics
    warnings = []
    if not video_present:
        warnings.append("Aucun flux vidéo détecté")
    if not audio_present:
        warnings.append("Aucun flux audio détecté")

    # platform hint
    ext = file_info.get("extension", "")
    if ext in ("mp4", "mov", "m4v"):
        platform_hint = "mp4/mov - compatible web/mobile"
    elif ext in ("mkv",):
        platform_hint = "mkv - conteneur riche (souvent desktop)"
    elif ext in ("avi",):
        platform_hint = "avi - ancien conteneur"
    else:
        platform_hint = "unknown"

    # needs_transcode heuristic: prefer h264/hevc/vp9 as ok
    vcodec = video_info.get("codec")
    needs_transcode = False
    if vcodec and vcodec.lower() not in ("h264", "hevc", "vp9", "vp8", "av1"):
        needs_transcode = True
        warnings.append(f"Codec vidéo non standard détecté: {vcodec}")

    # also recommend transcode if variable fps
    if video_info.get("is_variable_fps"):
        needs_transcode = True
        warnings.append("Variable frame rate détectée")

    recommended_next_step = None
    if needs_transcode:
        recommended_next_step = "Transcodage recommandé vers h264 (si compatibilité requise)"
    else:
        recommended_next_step = "Aucun transcodage requis"

    derived = {
        "platform_hint": platform_hint,
        "needs_transcode": needs_transcode,
        "recommended_next_step": recommended_next_step,
        "warnings": warnings,
    }

    result = {
        "file": file_info,
        "video": video_info,
        "audio": audio_info,
        "subtitles": subs_info,
        "derived": derived,
        "ffprobe_raw": raw,
    }

    return result
