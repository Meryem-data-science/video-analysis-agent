#!/usr/bin/env python3
"""Script pour inspecter une vidéo et sauvegarder le JSON d'inspection.

Usage:
  python scripts/inspect_video.py vidéos/demo.mp4
  python scripts/inspect_video.py vidéos/demo.mp4 --output outputs/inspection_demo.json
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project src/ is importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video_agent.inspection import inspect_video, InspectError


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspecte une vidéo avec ffprobe")
    parser.add_argument("video", help="Chemin vers la vidéo (ex: vidéos/demo.mp4)")
    parser.add_argument("--output", "-o", help="Fichier de sortie JSON", default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.output) if args.output else None

    try:
        res = inspect_video(video_path)
    except FileNotFoundError as e:
        print(f"Erreur: {e}", file=sys.stderr)
        return 2
    except InspectError as e:
        print(f"Erreur inspection: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Erreur inattendue: {e}", file=sys.stderr)
        return 4

    text = json.dumps(res, indent=2, ensure_ascii=False)
    print(text)

    if out_path:
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"Impossible d'écrire le fichier de sortie: {e}", file=sys.stderr)
            return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
