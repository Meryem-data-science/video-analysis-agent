import json
import sys
from pathlib import Path

import pytest

from scripts import evaluate as evaluate_cli
from src.video_agent import evaluation


def _annotations():
    return {
        "schema_version": "1.0",
        "dataset_name": "test",
        "video_id": "clip",
        "video_sha256": "hash",
        "expected_duration_s": 12.0,
        "synthetic": True,
        "provenance": "test",
        "scenes": [
            {"segment_id": 1, "start_s": 0.0, "end_s": 6.0, "texts": [], "objects": [], "actions": []},
            {"segment_id": 2, "start_s": 6.0, "end_s": 12.0, "texts": [], "objects": [], "actions": []},
        ],
    }


def test_scene_metrics_counts_coverage_gaps_and_overlaps():
    expected = _annotations()["scenes"]
    exact = evaluation.scene_metrics(expected, expected, tolerance_ms=100)
    assert exact["boundary_f1"] == 1.0
    assert exact["temporal_coverage_ratio"] == 1.0
    assert exact["gap_count"] == 0
    assert exact["overlap_count"] == 0

    gapped = [
        {"start_s": 0.0, "end_s": 5.5},
        {"start_s": 6.0, "end_s": 12.0},
    ]
    metrics = evaluation.scene_metrics(expected, gapped, tolerance_ms=100)
    assert metrics["boundary_f1"] == 0.0
    assert metrics["gap_count"] == 1
    assert metrics["gap_duration_s"] == 0.5


def test_scene_boundary_tolerance_is_configurable():
    expected = _annotations()["scenes"]
    detected = [
        {"start_s": 0.0, "end_s": 6.35},
        {"start_s": 6.35, "end_s": 12.0},
    ]
    assert evaluation.scene_metrics(expected, detected, tolerance_ms=400)["boundary_f1"] == 1.0
    assert evaluation.scene_metrics(expected, detected, tolerance_ms=300)["boundary_f1"] == 0.0


def _fake_run(tmp_path):
    run_dir = tmp_path / "run"
    for name in ("scenes", "frames/segment_001", "vlm_results", "reports"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(_annotations()), encoding="utf-8")
    segments_path = run_dir / "scenes/segments.json"
    segments_path.write_text(
        json.dumps({"segments": _annotations()["scenes"]}), encoding="utf-8"
    )
    (run_dir / "frames/segment_001/frame.jpg").write_bytes(b"frame")
    status_path = run_dir / "reports/pipeline_status.json"
    status_path.write_text(
        json.dumps({
            "status": "local_complete",
            "steps": [{"step": "inspection", "ok": True, "elapsed_s": 0.2, "detail": ""}],
            "performance": {
                "step_elapsed_s": {"inspection": 0.2},
                "total_elapsed_s": 0.2,
            },
        }),
        encoding="utf-8",
    )
    manifest = {
        "run_id": "a" * 64,
        "video_sha256": "hash",
        "status": "local_complete",
        "artifacts": {
            "segments": segments_path.as_posix(),
            "frames": (run_dir / "frames").as_posix(),
            "vlm_results": (run_dir / "vlm_results").as_posix(),
            "pipeline_status": status_path.as_posix(),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, annotations_path


def test_evaluation_json_and_html_reports(tmp_path):
    run_dir, annotations = _fake_run(tmp_path)
    report = evaluation.evaluate_run(
        run_dir, annotations, scene_tolerance_ms=200, include_performance=True
    )
    assert report["summary"]["detected_scenes"] == 2
    assert report["summary"]["frame_count"] == 1
    assert report["summary"]["total_elapsed_s"] == 0.2
    assert report["identity"]["exact_match"] is True
    paths = evaluation.write_evaluation_reports(report, tmp_path / "evaluation_report")
    json_report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    html_report = Path(paths["html"]).read_text(encoding="utf-8")
    assert json_report["schema_version"] == "1.0"
    assert "Rapport d'évaluation" in html_report
    assert "Limites méthodologiques" in html_report
    assert "<script" not in html_report


def test_evaluate_cli_writes_both_formats(tmp_path, monkeypatch, capsys):
    run_dir, annotations = _fake_run(tmp_path)
    report = evaluation.evaluate_run(run_dir, annotations)
    monkeypatch.setattr(
        evaluate_cli,
        "evaluate_dataset",
        lambda *_args, **_kwargs: (report, run_dir, annotations),
    )
    output = tmp_path / "custom_report"
    monkeypatch.setattr(sys, "argv", [
        "video-agent evaluate",
        "--dataset", str(tmp_path),
        "--output", str(output),
        "--format", "html",
    ])
    assert evaluate_cli.main() == 0
    body = json.loads(capsys.readouterr().out)
    assert body["selected_report"].endswith(".html")
    assert output.with_suffix(".json").is_file()
    assert output.with_suffix(".html").is_file()


def test_vlm_similarity_is_explicitly_approximate():
    expected = [{
        "segment_id": 1, "start_s": 0.0, "end_s": 1.0,
        "description": "un cercle orange", "objects": ["cercle"],
        "actions": ["bouger"], "texts": ["DEMO"],
    }]
    actual = [{
        "segment_id": 1, "start_s": 0.0, "end_s": 1.0, "combined_text": "",
        "scene_description": "un cercle orange", "people_objects": ["cercle"],
        "actions": ["bouger"], "motion_description": "mouvement",
        "audio_visual_relation": "sans audio", "uses_speech": False,
        "mood": [], "texte_ecran": [{"text": "DEMO", "lang": "fr", "location": "bas"}],
        "confidence": 1.0, "uncertainties": [],
        "evidence": {"visual": [], "audio": []},
    }]
    metrics = evaluation.vlm_metrics(expected, actual, tolerance_ms=10)
    assert metrics["valid_field_ratio"] == 1.0
    assert metrics["object_overlap"]["f1"] == 1.0
    assert "pas une mesure scientifique" in metrics["method_note"]


def test_benchmark_markdown_is_generated_from_report():
    markdown = evaluation.render_benchmark_markdown({
        "repeat": 3,
        "statistics": {
            "median_elapsed_s": 2.5,
            "population_stddev_s": 0.1,
            "min_elapsed_s": 2.4,
            "max_elapsed_s": 2.7,
        },
        "environment": {"python_version": "3.13", "logical_cpu_count": 8},
        "quality_snapshot": {
            "expected_scenes": 3,
            "detected_scenes": 3,
            "boundary_f1": 1.0,
        },
        "comparison_warning": "Même machine uniquement.",
    })
    assert "| Médiane | 2.500 s |" in markdown
    assert "Même machine uniquement." in markdown
