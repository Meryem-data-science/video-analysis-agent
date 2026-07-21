import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.video_agent import web_api

RUN_MANUAL = "a" * 64
RUN_DISABLED = "b" * 64


def _result(segment_id: int) -> dict:
    start = float((segment_id - 1) * 10)
    end = float(segment_id * 10)
    text = f"texte {segment_id}"
    return {
        "segment_id": segment_id,
        "start_s": start,
        "end_s": end,
        "combined_text": text,
        "scene_description": f"Description scène {segment_id}",
        "people_objects": ["personne"],
        "actions": ["marcher"],
        "motion_description": "mouvement léger",
        "audio_visual_relation": "la parole accompagne l'image",
        "uses_speech": True,
        "mood": ["calme"],
        "texte_ecran": [],
        "confidence": 0.7,
        "uncertainties": ["détail incertain"],
        "evidence": {"visual": ["frame_001"], "audio": ["parole"]},
    }


def _make_run(settings, run_id=RUN_MANUAL, mode="manual", count=2, status=None):
    run_dir = settings.runs_root / run_id
    for name in (
        "inspection", "scenes", "frames", "audio", "transcription", "fusion",
        "vlm_pack/prompts", "vlm_results", "reports",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    segments = [
        {
            "segment_id": i, "start_s": float((i - 1) * 10),
            "end_s": float(i * 10), "duration_s": 10.0, "combined_text": f"texte {i}",
        }
        for i in range(1, count + 1)
    ]
    (run_dir / "inspection" / "inspection.json").write_text(
        json.dumps({"audio": {"present": True}, "duration_s": count * 10}), encoding="utf-8"
    )
    (run_dir / "scenes" / "segments.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )
    (run_dir / "fusion" / "fusion.json").write_text(
        json.dumps({
            "video_path": "uploads/source.mp4", "language": "fr",
            "summary": {"audio_activity_mode": "active_audio"}, "segments": segments,
        }), encoding="utf-8"
    )
    (run_dir / "vlm_pack" / "vlm_pack.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )
    (run_dir / "vlm_pack" / "prompts" / "prompt_segment_001.md").write_text(
        "prompt", encoding="utf-8"
    )
    (run_dir / "reports" / "pipeline_status.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    artifacts = {
        "inspection": (run_dir / "inspection" / "inspection.json").as_posix(),
        "segments": (run_dir / "scenes" / "segments.json").as_posix(),
        "fusion": (run_dir / "fusion" / "fusion.json").as_posix(),
        "vlm_pack": (run_dir / "vlm_pack" / "vlm_pack.json").as_posix(),
        "vlm_prompts": (run_dir / "vlm_pack" / "prompts").as_posix(),
        "vlm_results": (run_dir / "vlm_results").as_posix(),
        "pipeline_status": (run_dir / "reports" / "pipeline_status.json").as_posix(),
        "final_report": (run_dir / "reports" / "final_report.json").as_posix(),
    }
    manifest = {
        "manifest_schema_version": "1.0", "run_id": run_id,
        "original_filename": "source.mp4", "video_metadata": {"duration_s": count * 10},
        "normalized_configuration": {"vlm": {"mode": mode, "provider": None}},
        "vlm": {"mode": mode, "provider": None},
        "status": status or ("local_complete" if mode == "disabled" else "awaiting_vlm"),
        "current_step": None, "artifacts": artifacts,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


@pytest.fixture
def api_env(tmp_path):
    settings = web_api.ApiSettings(
        project_root=tmp_path, runs_root=tmp_path / "outputs" / "runs",
        upload_root=tmp_path / "uploads", max_upload_bytes=1024 * 1024,
    )
    calls = []

    def runner(video_path, **kwargs):
        config = json.loads(Path(kwargs["config_path"]).read_text(encoding="utf-8"))
        calls.append({"video_path": video_path, "config": config, **kwargs})
        mode = config["mode"]
        run_id = RUN_DISABLED if mode == "disabled" else RUN_MANUAL
        _make_run(settings, run_id=run_id, mode=mode)
        return {"run_id": run_id, "status": "local_complete" if mode == "disabled" else "awaiting_vlm"}

    app = web_api.create_app(settings, pipeline_runner=runner)
    return TestClient(app), settings, calls


def test_health_check(api_env):
    client, _, _ = api_env
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["default_vlm_mode"] == "manual"
    assert {"audio", "gemini", "pdf", "web"} <= body["extras"].keys()


def test_health_detects_missing_ffmpeg(api_env, monkeypatch):
    client, _, _ = api_env
    monkeypatch.setattr(web_api.shutil, "which", lambda _name: None)
    body = client.get("/api/health").json()
    assert body["ffmpeg"]["available"] is False
    assert body["ffprobe"]["available"] is False


@pytest.mark.parametrize(
    ("mode", "run_id", "status"),
    [("manual", RUN_MANUAL, "awaiting_vlm"), ("disabled", RUN_DISABLED, "local_complete")],
)
def test_create_local_run_has_no_gemini_provider(api_env, mode, run_id, status):
    client, _, calls = api_env
    response = client.post(
        "/api/runs", files={"video": ("clip.mp4", b"video", "video/mp4")},
        data={"mode": mode},
    )
    assert response.status_code == 201
    assert response.json()["run_id"] == run_id
    assert response.json()["status"] == status
    assert calls[-1]["config"] == {"mode": mode, "provider": None, "model": ""}


def test_unknown_provider_is_rejected_before_pipeline(api_env):
    client, _, calls = api_env
    response = client.post(
        "/api/runs", files={"video": ("clip.mp4", b"video", "video/mp4")},
        data={"mode": "automatic", "provider": "claude"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "vlm_configuration_invalid"
    assert calls == []


def test_explicit_gemini_provider_reaches_existing_pipeline_contract(api_env):
    client, _, calls = api_env
    response = client.post(
        "/api/runs", files={"video": ("clip.mp4", b"video", "video/mp4")},
        data={"mode": "automatic", "provider": "gemini"},
    )
    assert response.status_code == 201
    assert calls[-1]["config"]["provider"] == "gemini"


def test_manifest_unknown_run_and_awaiting_report(api_env):
    client, settings, _ = api_env
    _make_run(settings)
    body = client.get(f"/api/runs/{RUN_MANUAL}").json()
    assert body["configuration"]["vlm"]["mode"] == "manual"
    report = client.get(f"/api/runs/{RUN_MANUAL}/report").json()
    assert report["status"] == "awaiting_vlm" and report["report"] is None
    assert report["next_step"]["pack"].endswith("vlm_pack/vlm_pack.json")
    missing = client.get(f"/api/runs/{'c' * 64}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "run_not_found"


def test_scenes_and_secure_declared_artifacts(api_env, tmp_path):
    client, settings, _ = api_env
    run_dir = _make_run(settings)
    secret = tmp_path / ".env"
    secret.write_text("GEMINI_API_KEY=secret", encoding="utf-8")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["secret"] = secret.as_posix()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    scenes = client.get(f"/api/runs/{RUN_MANUAL}/scenes").json()
    assert scenes["source"] == "segmentation" and len(scenes["scenes"]) == 2
    listed = client.get(f"/api/runs/{RUN_MANUAL}/artifacts").json()["artifacts"]
    names = {item["name"] for item in listed}
    assert "inspection" in names and "secret" not in names
    assert client.get(f"/api/runs/{RUN_MANUAL}/artifacts/inspection").status_code == 200
    assert client.get(f"/api/runs/{RUN_MANUAL}/artifacts/not-declared").status_code == 404


def test_artifact_path_traversal_is_refused(api_env):
    client, settings, _ = api_env
    _make_run(settings)
    response = client.get(f"/api/runs/{RUN_MANUAL}/artifacts/..%5C.env")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "artifact_name_invalid"


def test_invalid_upload_extension(api_env):
    client, _, calls = api_env
    response = client.post(
        "/api/runs", files={"video": ("../../.env", b"secret", "text/plain")}
    )
    assert response.status_code == 415 and calls == []


def test_upload_size_limit_is_enforced(api_env):
    client, settings, calls = api_env
    response = client.post(
        "/api/runs",
        files={"video": ("clip.mp4", b"x" * (settings.max_upload_bytes + 1), "video/mp4")},
    )
    assert response.status_code == 413 and calls == []


def test_import_invalid_json_shape_is_rejected(api_env):
    client, settings, _ = api_env
    _make_run(settings)
    bad = _result(1)
    bad["people_objects"] = "personne"
    response = client.post(f"/api/runs/{RUN_MANUAL}/vlm-results", json={"results": [bad]})
    assert response.status_code == 422
    assert not (settings.runs_root / RUN_MANUAL / "vlm_results" / "vlm_result_segment_001.json").exists()


def test_import_unreadable_json_is_structured(api_env):
    client, settings, _ = api_env
    _make_run(settings)
    response = client.post(
        f"/api/runs/{RUN_MANUAL}/vlm-results", content="{",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_invalid"


def test_import_validate_and_build_partial_report(api_env):
    client, settings, _ = api_env
    _make_run(settings, count=2)
    imported = client.post(
        f"/api/runs/{RUN_MANUAL}/vlm-results", json={"run_id": RUN_MANUAL, "results": [_result(1)]}
    )
    assert imported.status_code == 200
    assert imported.json()["validation"]["status"] == "partial"
    validated = client.post(f"/api/runs/{RUN_MANUAL}/validate").json()
    assert validated["missing_segments"] == [2]
    built = client.post(f"/api/runs/{RUN_MANUAL}/build-report")
    assert built.status_code == 200
    assert built.json()["status"] == "partial"
    assert built.json()["report"]["scene_count"] == 1


def test_import_and_build_complete_report(api_env):
    client, settings, _ = api_env
    _make_run(settings, count=2)
    imported = client.post(
        f"/api/runs/{RUN_MANUAL}/vlm-results", json={"results": [_result(1), _result(2)]}
    )
    assert imported.status_code == 200
    assert imported.json()["validation"]["status"] == "complete"
    built = client.post(f"/api/runs/{RUN_MANUAL}/build-report")
    assert built.status_code == 200
    assert built.json()["status"] == "complete"
    report = client.get(f"/api/runs/{RUN_MANUAL}/report").json()
    assert report["status"] == "complete" and report["report"]["scene_count"] == 2
