"""CLI d'évaluation locale du pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_agent.evaluation import (
    EvaluationError,
    evaluate_dataset,
    evaluate_run,
    write_evaluation_reports,
)
from video_agent.pipeline import OUTPUTS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparer un run à des annotations de référence locales."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Dossier contenant dataset.json")
    source.add_argument("--run-id", help="Identité d'un run sous outputs/runs")
    parser.add_argument("--annotations", type=Path, help="Annotations d'un --run-id")
    parser.add_argument("--output", type=Path, help="Base de sortie sans extension")
    parser.add_argument("--scene-tolerance-ms", type=int, default=500)
    parser.add_argument("--format", choices=("json", "html"), default="json")
    parser.add_argument("--include-performance", action="store_true")
    args = parser.parse_args()
    try:
        if args.dataset:
            report, run_dir, _ = evaluate_dataset(
                args.dataset,
                scene_tolerance_ms=args.scene_tolerance_ms,
                include_performance=args.include_performance,
            )
        else:
            if not args.annotations:
                parser.error("--annotations est requis avec --run-id.")
            run_dir = OUTPUTS / "runs" / args.run_id
            report = evaluate_run(
                run_dir, args.annotations,
                scene_tolerance_ms=args.scene_tolerance_ms,
                include_performance=args.include_performance,
            )
        output = args.output or run_dir / "reports" / "evaluation_report"
        paths = write_evaluation_reports(report, output)
    except EvaluationError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "run_id": report.get("run_id"),
        "status": report.get("summary", {}).get("status"),
        "selected_report": paths[args.format],
        "reports": paths,
        "summary": report.get("summary"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
