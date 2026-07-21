import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.audio_visual_fusion import (
    AudioVisualFusionError,
    fuse_audio_visual,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fusionner artefacts visuels + audio en un manifest aligné par timestamps."
    )
    parser.add_argument(
        "--segments", required=True, help="Manifest de segmentation visuelle (Niveau 3)"
    )
    parser.add_argument(
        "--frames-manifest", required=True, help="Manifest de frames (Niveau 4)"
    )
    parser.add_argument(
        "--motion-manifest", required=True, help="Manifest motion (Niveau 5)"
    )
    parser.add_argument(
        "--audio-light",
        default=None,
        help="Analyse audio légère (Niveau 6C), facultatif",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="Transcript faster-whisper (Niveau 6D), facultatif",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin de sortie du manifest de fusion JSON",
    )

    args = parser.parse_args()

    try:
        manifest = fuse_audio_visual(
            segments_manifest_path=args.segments,
            frames_manifest_path=args.frames_manifest,
            motion_manifest_path=args.motion_manifest,
            audio_light_analysis_path=args.audio_light,
            transcript_path=args.transcript,
        )
    except AudioVisualFusionError as exc:
        print(f"Erreur de fusion audio-visuelle : {exc}", file=sys.stderr)
        return 1

    output_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(output_text, encoding="utf-8")
    except OSError as exc:
        print(f"Erreur : écriture du manifest impossible ({output_path}) : {exc}", file=sys.stderr)
        return 1

    summary = manifest.get("summary", {})
    print(f"Manifest de fusion écrit : {output_path}", file=sys.stderr)
    print(f"  segment_count      = {summary.get('segment_count')}", file=sys.stderr)
    print(f"  total_frames       = {summary.get('total_frames')}", file=sys.stderr)
    print(f"  speech_segment_count = {summary.get('speech_segment_count')}", file=sys.stderr)
    print(f"  ready_for_vlm_count  = {summary.get('ready_for_vlm_count')}", file=sys.stderr)
    print(f"  next_step          = {manifest.get('next_step')}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
