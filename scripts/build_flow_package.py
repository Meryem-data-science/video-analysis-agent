"""CLI — assemble le dossier autonome Flow_regeneration/<video_id>/ (copie, non destructif).

Exemple :
    .\\.venv\\Scripts\\python.exe scripts\\build_flow_package.py --video-id Video6
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

from video_agent.flow_package import FlowPackageError, build_flow_package


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble un dossier Flow_regeneration/<id>/ self-contained (copie)."
    )
    parser.add_argument("--video-id", default="Video6", help="Identifiant vidéo (défaut Video6)")
    parser.add_argument("--no-frames-source", action="store_true",
                        help="Ne pas archiver regen_frames_<id>/ sous frames_source/")
    args = parser.parse_args()

    try:
        report = build_flow_package(args.video_id, include_frames_source=not args.no_frames_source)
    except FlowPackageError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print("\n===== AUTO-TEST Flow package =====", file=sys.stderr)
    print(f"  video_id : {args.video_id}  | clips : {report['clip_count']}", file=sys.stderr)
    for c in report["checks"]:
        mark = "OK " if c["ok"] else "ÉCHEC"
        detail = f"  [{c['detail']}]" if c["detail"] else ""
        print(f"  [{mark}] {c['name']}{detail}", file=sys.stderr)
    print(f"  ----> {report['passed']}/{report['total']} vérifications passées", file=sys.stderr)
    if report["missing_images"]:
        print(f"  Images manquantes : {report['missing_images']}", file=sys.stderr)
    print(f"  Dossier : {report['dest']}", file=sys.stderr)
    print("\n===== ARBORESCENCE =====", file=sys.stderr)
    print(report["tree"], file=sys.stderr)

    print(json.dumps({k: v for k, v in report.items() if k != "tree"}, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
