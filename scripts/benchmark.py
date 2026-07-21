"""Benchmark reproductible du dataset synthétique."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_agent.evaluation import (
    EvaluationError,
    benchmark_dataset,
    render_benchmark_markdown,
)
from video_agent.run_identity import atomic_write_json, atomic_write_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Répéter le pipeline local et mesurer le temps sur une même machine."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--scene-tolerance-ms", type=int, default=500)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/demo/benchmark_report.json")
    )
    args = parser.parse_args()
    try:
        report = benchmark_dataset(
            args.dataset, repeat=args.repeat, scene_tolerance_ms=args.scene_tolerance_ms
        )
    except EvaluationError as exc:
        parser.error(str(exc))
    markdown_path = args.output.with_suffix(".md")
    report["reports"] = {
        "json": args.output.as_posix(),
        "markdown": markdown_path.as_posix(),
    }
    atomic_write_json(args.output, report)
    atomic_write_text(markdown_path, render_benchmark_markdown(report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
