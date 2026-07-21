import hashlib
import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.video_agent import web_api
from src.video_agent.pipeline import PipelineError


def _settings(tmp_path, max_workers=1):
    return web_api.ApiSettings(
        project_root=tmp_path,
        runs_root=tmp_path / "outputs" / "runs",
        upload_root=tmp_path / "uploads",
        max_upload_bytes=1024 * 1024,
        max_workers=max_workers,
        upload_ttl_seconds=3600,
    )


def _make_run(settings, run_id, mode, pipeline_status):
    run_dir = settings.runs_root / run_id
    (run_dir / "scenes").mkdir(parents=True)
    (run_dir / "reports").mkdir()
    marker = run_dir / "scenes" / "segments.json"
    marker.write_text(json.dumps({"segments": []}), encoding="utf-8")
    status_file = run_dir / "reports" / "pipeline_status.json"
    status_file.write_text(json.dumps({"steps": []}), encoding="utf-8")
    manifest = {
        "manifest_schema_version": "1.0",
        "run_id": run_id,
        "video_metadata": {"duration_s": 1.0},
        "normalized_configuration": {
            "vlm": {"mode": mode, "provider": None}
        },
        "vlm": {"mode": mode, "provider": None},
        "status": pipeline_status,
        "current_step": None,
        "artifacts": {
            "segments": marker.as_posix(),
            "pipeline_status": status_file.as_posix(),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir


def _poll(client, job_id, terminal=True, timeout=3):
    deadline = time.time() + timeout
    history = []
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        history.append(body)
        if terminal and body["status"] in {"completed", "failed", "cancelled"}:
            return body, history
        if not terminal and body["status"] == "running":
            return body, history
        time.sleep(0.01)
    raise AssertionError("Le job n'a pas atteint l'état attendu.")


class FakePipeline:
    def __init__(self, settings, *, gate=None, fail_predicate=None):
        self.settings = settings
        self.gate = gate
        self.fail_predicate = fail_predicate or (lambda content: False)
        self.started = threading.Event()
        self.calls = []

    def __call__(self, video_path, **kwargs):
        path = Path(video_path)
        content = path.read_bytes()
        config = json.loads(Path(kwargs["config_path"]).read_text(encoding="utf-8"))
        callback = kwargs["progress_callback"]
        run_id = hashlib.sha256(content + config["mode"].encode()).hexdigest()
        self.calls.append({
            "video_path": path, "config": config, "run_id": run_id,
            "progress_callback": callback,
        })
        callback({
            "run_id": run_id, "current_step": "inspection",
            "progress_percent": 8, "message": "Inspection terminée.",
        })
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=3)
        if self.fail_predicate(content):
            raise PipelineError("Échec simulé pendant segmentation.")
        callback({
            "run_id": run_id, "current_step": "segmentation",
            "progress_percent": 22, "message": "Segmentation terminée.",
        })
        pipeline_status = (
            "local_complete" if config["mode"] == "disabled" else "awaiting_vlm"
        )
        run_dir = _make_run(
            self.settings, run_id, config["mode"], pipeline_status
        )
        callback({
            "run_id": run_id, "current_step": "completed",
            "progress_percent": 100, "message": "Pipeline terminé.",
        })
        return {
            "ok": True, "run_id": run_id, "run_dir": run_dir.as_posix(),
            "status": pipeline_status,
        }


def test_create_job_returns_202_then_running_then_completed(tmp_path):
    settings = _settings(tmp_path)
    release = threading.Event()
    runner = FakePipeline(settings, gate=release)
    app = web_api.create_app(settings, pipeline_runner=runner)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"video": ("clip.mp4", b"video", "video/mp4")},
            data={"mode": "manual"},
        )
        assert response.status_code == 202
        queued = response.json()
        assert queued["status"] == "queued"
        assert queued["progress_percent"] == 0
        assert queued["status_url"].endswith(queued["job_id"])

        running, before = _poll(client, queued["job_id"], terminal=False)
        assert running["status"] == "running"
        release.set()
        completed, after = _poll(client, queued["job_id"])
        history = before + after
        percentages = [item["progress_percent"] for item in history]
        assert percentages == sorted(percentages)
        assert completed["status"] == "completed"
        assert completed["run_id"] == runner.calls[0]["run_id"]
        assert completed["pipeline_status"] == "awaiting_vlm"
        assert completed["links"]["manifest"].endswith(completed["run_id"])
        assert completed["created_at"]
        assert completed["started_at"]
        assert completed["finished_at"]
        cannot_cancel = client.post(f"/api/jobs/{queued['job_id']}/cancel")
        assert cannot_cancel.status_code == 409
        assert cannot_cancel.json()["error"]["code"] == "job_already_finished"


def test_manual_job_never_configures_gemini_and_cleans_upload(tmp_path):
    settings = _settings(tmp_path)
    runner = FakePipeline(settings)
    app = web_api.create_app(settings, pipeline_runner=runner)
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={"video": ("manual.mp4", b"manual", "video/mp4")},
        ).json()
        completed, _ = _poll(client, created["job_id"])
        assert runner.calls[0]["config"]["provider"] is None
        assert runner.calls[0]["config"]["mode"] == "manual"
        assert not runner.calls[0]["video_path"].exists()
        run_dir = settings.runs_root / completed["run_id"]
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "scenes" / "segments.json").exists()


def test_disabled_job_finishes_local_complete(tmp_path):
    settings = _settings(tmp_path)
    runner = FakePipeline(settings)
    app = web_api.create_app(settings, pipeline_runner=runner)
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={"video": ("disabled.mp4", b"disabled", "video/mp4")},
            data={"mode": "disabled"},
        ).json()
        completed, _ = _poll(client, created["job_id"])
        assert completed["status"] == "completed"
        assert completed["pipeline_status"] == "local_complete"
        assert runner.calls[0]["config"]["provider"] is None


def test_failed_job_is_structured_cleans_upload_and_server_survives(tmp_path):
    settings = _settings(tmp_path)
    runner = FakePipeline(settings, fail_predicate=lambda content: content == b"fail")
    app = web_api.create_app(settings, pipeline_runner=runner)
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={"video": ("fail.mp4", b"fail", "video/mp4")},
        ).json()
        failed, _ = _poll(client, created["job_id"])
        assert failed["status"] == "failed"
        assert failed["errors"][0]["code"] == "pipeline_rejected"
        assert "Traceback" not in json.dumps(failed)
        assert not runner.calls[0]["video_path"].exists()
        assert client.get("/api/health").status_code == 200


def test_unknown_job_and_invalid_provider_are_structured(tmp_path):
    settings = _settings(tmp_path)
    runner = FakePipeline(settings)
    app = web_api.create_app(settings, pipeline_runner=runner)
    with TestClient(app) as client:
        missing = client.get("/api/jobs/inconnu")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "job_not_found"
        invalid = client.post(
            "/api/jobs",
            files={"video": ("clip.mp4", b"x", "video/mp4")},
            data={"mode": "automatic", "provider": "claude"},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "vlm_configuration_invalid"
        assert runner.calls == []


def test_job_list_filters_and_ids_are_distinct(tmp_path):
    settings = _settings(tmp_path)
    runner = FakePipeline(settings, fail_predicate=lambda content: content == b"fail")
    app = web_api.create_app(settings, pipeline_runner=runner)
    with TestClient(app) as client:
        first = client.post(
            "/api/jobs", files={"video": ("ok.mp4", b"ok", "video/mp4")}
        ).json()
        second = client.post(
            "/api/jobs", files={"video": ("fail.mp4", b"fail", "video/mp4")}
        ).json()
        assert first["job_id"] != second["job_id"]
        _poll(client, first["job_id"])
        _poll(client, second["job_id"])
        completed = client.get("/api/jobs", params={"status": "completed"}).json()
        failed = client.get("/api/jobs", params={"status": "failed"}).json()
        assert completed["count"] == 1
        assert failed["count"] == 1
        assert completed["persistent"] is False


def test_queued_job_can_be_cancelled_and_its_upload_removed(tmp_path):
    settings = _settings(tmp_path, max_workers=1)
    release = threading.Event()
    runner = FakePipeline(settings, gate=release)
    app = web_api.create_app(settings, pipeline_runner=runner)
    with TestClient(app) as client:
        first = client.post(
            "/api/jobs", files={"video": ("one.mp4", b"one", "video/mp4")}
        ).json()
        assert runner.started.wait(timeout=1)
        second = client.post(
            "/api/jobs", files={"video": ("two.mp4", b"two", "video/mp4")}
        ).json()
        cancelled = client.post(f"/api/jobs/{second['job_id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        deadline = time.time() + 1
        while time.time() < deadline and len(list(settings.upload_root.iterdir())) != 1:
            time.sleep(0.01)
        assert len(list(settings.upload_root.iterdir())) == 1
        running_cancel = client.post(f"/api/jobs/{first['job_id']}/cancel")
        assert running_cancel.status_code == 409
        assert running_cancel.json()["error"]["code"] == "job_cancellation_unsupported"
        release.set()
        _poll(client, first["job_id"])


def test_lifespan_shuts_down_executor_and_queue_is_not_persistent(tmp_path):
    settings = _settings(tmp_path)
    runner = FakePipeline(settings)
    app = web_api.create_app(settings, pipeline_runner=runner)
    manager = app.state.job_manager
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["jobs"] == {
            "max_workers": 1, "persistent": False, "accepting": True,
        }
    assert manager.is_shutdown is True
    assert manager.accepting is False


def test_readme_documents_restart_and_cancellation_limits():
    readme = Path("README.md").read_text(encoding="utf-8").lower()
    assert "video_agent_max_workers" in readme
    assert "redémarrage" in readme
    assert "annulation" in readme
