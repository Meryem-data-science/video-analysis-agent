#!/usr/bin/env python3
"""Script pour décider une stratégie à partir d'un JSON d'inspection."""
import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when running this script directly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.video_agent.strategy import decide_strategy


def main() -> int:
    parser = argparse.ArgumentParser(description="Décide une stratégie à partir d'un JSON d'inspection vidéo")
    parser.add_argument("inspection", help="Fichier JSON d'inspection (ex: outputs/inspection_demo.json)")
    parser.add_argument("--output", "-o", help="Fichier de sortie JSON de stratégie", required=True)
    args = parser.parse_args()

    inspection_path = Path(args.inspection)
    output_path = Path(args.output)

    if not inspection_path.exists():
        print(f"Fichier d'inspection introuvable: {inspection_path}", file=sys.stderr)
        return 1

    try:
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Impossible de lire le fichier d'inspection: {e}", file=sys.stderr)
        return 2

    strategy = decide_strategy(inspection)
    text = json.dumps(strategy, indent=2, ensure_ascii=False)
    print(text)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"Impossible d'écrire le fichier de sortie: {e}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
