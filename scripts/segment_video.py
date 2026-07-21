import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.segmentation import load_strategy, segment_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Segmenter une vidéo avec PySceneDetect.")
    parser.add_argument("video_path", help="Chemin du fichier vidéo à analyser")
    parser.add_argument(
        "--strategy",
        help="Chemin facultatif vers un fichier JSON de stratégie ou JSON inline",
        default=None,
    )
    parser.add_argument(
        "--output",
        help="Chemin de sortie JSON pour les résultats de segmentation",
        default=None,
    )

    args = parser.parse_args()
    strategy = load_strategy(args.strategy) if args.strategy else None
    result = segment_video(args.video_path, strategy=strategy)

    output_text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
