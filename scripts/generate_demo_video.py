"""CLI historique déléguée par ``video-agent generate-demo``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_agent.demo import DemoError, default_demo_path, generate_demo_video
from video_agent.pipeline import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Générer une vidéo verticale synthétique, déterministe et libre."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_demo_path(PROJECT_ROOT),
        help="Fichier MP4 de sortie (défaut: outputs/demo/video_agent_demo.mp4)",
    )
    parser.add_argument("--force", action="store_true", help="Régénérer même si le cache est valide")
    args = parser.parse_args()
    try:
        manifest = generate_demo_video(args.output, force=args.force)
    except DemoError as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
