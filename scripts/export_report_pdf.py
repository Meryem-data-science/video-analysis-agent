import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.report_pdf import ReportPdfError, build_report_pdf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporter le rapport vidéo final en PDF (arabe rendu correctement)."
    )
    parser.add_argument("--report", default="outputs/final_report_demo_darija.json",
                        help="Rapport final JSON (Niveau 12)")
    parser.add_argument("--output", default=None,
                        help="Chemin PDF de sortie (défaut outputs/final_report_<stem>.pdf)")
    parser.add_argument("--font", default=None,
                        help="Police TTF arabe-capable (sinon auto-détection système)")

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists() or not report_path.is_file():
        print(f"Erreur : rapport introuvable : {report_path}", file=sys.stderr)
        return 1
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Erreur : rapport JSON invalide ({report_path}) : {exc}", file=sys.stderr)
        return 1

    stem = Path(report.get("video_path", "video")).stem or "video"
    output_path = args.output or f"outputs/final_report_{stem}.pdf"

    try:
        out = build_report_pdf(report, output_path, font_path=args.font)
    except ReportPdfError as exc:
        print(f"Erreur d'export PDF : {exc}", file=sys.stderr)
        return 1

    size = Path(out).stat().st_size
    print(f"PDF écrit : {out} ({size} octets, {report.get('scene_count')} scènes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
