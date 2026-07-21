"""CLI — construit le regen brief à partir du rapport final d'une vidéo.

Non destructif : lecture seule sur les entrées, n'écrit que regen_brief_<id>.json,
regen_worksheet_<id>.md et regen_frames_<id>/. Affiche le rapport d'auto-test.

Exemple :
    .\\.venv\\Scripts\\python.exe scripts\\build_regen_brief.py --video-id Video6
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from video_agent.regen_brief import RegenBriefError, build_regen_brief


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regeneration brief (clips ≤ 8 s pour Google Flow) depuis le rapport final."
    )
    parser.add_argument("--video-id", default="Video6", help="Identifiant vidéo (ex. Video6)")
    parser.add_argument("--outputs-dir", default=None, help="Dossier outputs (défaut: outputs/)")
    args = parser.parse_args()

    try:
        report = build_regen_brief(
            args.video_id,
            outputs_dir=Path(args.outputs_dir) if args.outputs_dir else None,
        )
    except RegenBriefError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print("\n===== AUTO-TEST regen brief =====", file=sys.stderr)
    print(f"  video_id        : {args.video_id}", file=sys.stderr)
    print(f"  clips           : {report['clip_count']} "
          f"(generated={report['generated_count']}, static_card={report['static_cards']})",
          file=sys.stderr)
    for c in report["checks"]:
        mark = "OK " if c["ok"] else "ÉCHEC"
        detail = f"  [{c['detail']}]" if c["detail"] else ""
        print(f"  [{mark}] {c['name']}{detail}", file=sys.stderr)
    print(f"  ----> {report['passed']}/{report['total']} vérifications passées", file=sys.stderr)
    print(f"  JSON      : {report['outputs']['json']}", file=sys.stderr)
    print(f"  Worksheet : {report['outputs']['worksheet']}", file=sys.stderr)
    print(f"  Frames    : {report['outputs']['frames_dir']}", file=sys.stderr)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
