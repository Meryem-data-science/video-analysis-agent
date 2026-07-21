import hashlib
import json
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.video_agent import demo, web_api


def test_rendered_frames_are_deterministic():
    first = hashlib.sha256(demo.render_demo_frame(10).tobytes()).hexdigest()
    second = hashlib.sha256(demo.render_demo_frame(10).tobytes()).hexdigest()
    different = hashlib.sha256(demo.render_demo_frame(11).tobytes()).hexdigest()
    assert first == second
    assert first != different


def test_generated_video_is_deterministic_and_uses_no_network(tmp_path, monkeypatch):
    def refuse_network(*_args, **_kwargs):
        raise AssertionError("Aucun accès réseau n'est autorisé pendant la génération.")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    first = demo.generate_demo_video(tmp_path / "first.mp4", force=True)
    second = demo.generate_demo_video(tmp_path / "second.mp4", force=True)
    assert first["video_sha256"] == second["video_sha256"]
    assert first["network_used"] is False
    assert first["duration_s"] == 18.0
    assert (first["width"], first["height"]) == (360, 640)


def _demo_pack():
    return {
        "segments": [
            {"segment_id": 1, "start_s": 0.0, "end_s": 6.0, "combined_text": ""},
            {"segment_id": 2, "start_s": 6.0, "end_s": 12.0, "combined_text": ""},
            {"segment_id": 3, "start_s": 12.0, "end_s": 18.0, "combined_text": ""},
        ]
    }


def test_demo_fixtures_are_bound_to_exact_video_hash():
    manifest = {"video_sha256": "demo-hash"}
    demo_manifest = {
        "video_sha256": "demo-hash",
        "generator_spec_sha256": demo.DEMO_SPEC_SHA256,
    }
    fixtures = (
        Path(__file__).parents[1]
        / "evaluation/datasets/demo/vlm_fixtures/demo_vlm_results.json"
    )
    results = demo.bind_demo_vlm_results(
        run_manifest=manifest,
        vlm_pack=_demo_pack(),
        demo_manifest=demo_manifest,
        fixtures_path=fixtures,
    )
    assert [item["segment_id"] for item in results] == [1, 2, 3]
    assert results[1]["start_s"] == 6.0

    with pytest.raises(demo.DemoError, match="ne correspond pas"):
        demo.bind_demo_vlm_results(
            run_manifest={"video_sha256": "another-video"},
            vlm_pack=_demo_pack(),
            demo_manifest=demo_manifest,
            fixtures_path=fixtures,
        )


def _make_demo_run(settings, run_id, video_hash="demo-hash"):
    run_dir = settings.runs_root / run_id
    for name in ("scenes", "vlm_pack", "vlm_results", "reports", "frames"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    segments = _demo_pack()["segments"]
    (run_dir / "scenes" / "segments.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )
    (run_dir / "vlm_pack" / "vlm_pack.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )
    (run_dir / "reports" / "pipeline_status.json").write_text(
        json.dumps({"steps": [], "performance": {"total_elapsed_s": 1.5}}),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "video_sha256": video_hash,
        "status": "awaiting_vlm",
        "vlm": {"mode": "manual", "provider": None},
        "normalized_configuration": {"vlm": {"mode": "manual", "provider": None}},
        "artifacts": {
            "segments": (run_dir / "scenes" / "segments.json").as_posix(),
            "vlm_pack": (run_dir / "vlm_pack" / "vlm_pack.json").as_posix(),
            "vlm_results": (run_dir / "vlm_results").as_posix(),
            "frames": (run_dir / "frames").as_posix(),
            "pipeline_status": (run_dir / "reports" / "pipeline_status.json").as_posix(),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_demo_job_creation_uses_manual_pipeline_without_provider(tmp_path, monkeypatch):
    settings = web_api.ApiSettings(
        project_root=tmp_path,
        runs_root=tmp_path / "outputs/runs",
        upload_root=tmp_path / "uploads",
        max_workers=1,
    )
    demo_path = tmp_path / "outputs/demo/video_agent_demo.mp4"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_bytes(b"synthetic-video")
    video_hash = hashlib.sha256(b"synthetic-video").hexdigest()
    monkeypatch.setattr(web_api, "default_demo_path", lambda _root: demo_path)
    monkeypatch.setattr(
        web_api,
        "generate_demo_video",
        lambda _path: {"video_sha256": video_hash, "network_used": False},
    )
    calls = []

    def runner(video_path, **kwargs):
        calls.append({
            "video_path": video_path,
            "config": json.loads(Path(kwargs["config_path"]).read_text(encoding="utf-8")),
        })
        run_id = "d" * 64
        _make_demo_run(settings, run_id, video_hash)
        return {"ok": True, "run_id": run_id, "status": "awaiting_vlm"}

    with TestClient(web_api.create_app(settings, pipeline_runner=runner)) as client:
        response = client.post("/api/demo/jobs", data={"mode": "manual"})
        assert response.status_code == 202
        assert response.json()["demo"] is True
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "completed":
                break
        assert job["status"] == "completed"
    assert calls[0]["config"] == {"mode": "manual", "provider": None, "model": ""}


def test_demo_fixture_endpoint_refuses_a_different_video(tmp_path, monkeypatch):
    settings = web_api.ApiSettings(
        project_root=tmp_path,
        runs_root=tmp_path / "outputs/runs",
        upload_root=tmp_path / "uploads",
    )
    run_id = "e" * 64
    _make_demo_run(settings, run_id, video_hash="user-video")
    demo_path = tmp_path / "outputs/demo/video_agent_demo.mp4"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_bytes(b"demo")
    monkeypatch.setattr(web_api, "default_demo_path", lambda _root: demo_path)
    monkeypatch.setattr(
        web_api,
        "generate_demo_video",
        lambda _path: {
            "video_sha256": "demo-video",
            "generator_spec_sha256": demo.DEMO_SPEC_SHA256,
        },
    )
    with TestClient(web_api.create_app(settings)) as client:
        response = client.post(f"/api/demo/runs/{run_id}/vlm-fixtures")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "demo_fixture_refused"


def test_demo_fixtures_validate_and_evaluation_reports_are_exposed(tmp_path, monkeypatch):
    settings = web_api.ApiSettings(
        project_root=tmp_path,
        runs_root=tmp_path / "outputs/runs",
        upload_root=tmp_path / "uploads",
    )
    run_id = "f" * 64
    run_dir = _make_demo_run(settings, run_id, video_hash="demo-video")
    demo_path = tmp_path / "outputs/demo/video_agent_demo.mp4"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_bytes(b"demo")
    monkeypatch.setattr(web_api, "default_demo_path", lambda _root: demo_path)
    monkeypatch.setattr(
        web_api,
        "generate_demo_video",
        lambda _path: {
            "video_sha256": "demo-video",
            "generator_spec_sha256": demo.DEMO_SPEC_SHA256,
        },
    )
    with TestClient(web_api.create_app(settings)) as client:
        fixtures = client.post(f"/api/demo/runs/{run_id}/vlm-fixtures")
        assert fixtures.status_code == 200
        assert fixtures.json()["status"] == "complete"
        assert fixtures.json()["simulated"] is True
        assert len(list((run_dir / "vlm_results").glob("vlm_result_segment_*.json"))) == 3

        evaluation = client.post(f"/api/demo/runs/{run_id}/evaluate")
        assert evaluation.status_code == 200
        assert evaluation.json()["report"]["summary"]["detected_scenes"] == 3
        assert client.get(f"/api/runs/{run_id}/evaluation").status_code == 200
        html_report = client.get(f"/api/runs/{run_id}/evaluation/html")
        assert html_report.status_code == 200
        assert "Rapport d'évaluation" in html_report.text
