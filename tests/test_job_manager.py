import os
import threading
import time

import pytest

from src.video_agent.job_manager import (
    JobExecutionError,
    LocalJobManager,
    cleanup_abandoned_uploads,
    parse_max_workers,
)


def _wait_terminal(manager, job_id, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = manager.get(job_id)
        if snapshot["status"] in {"completed", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("Le job n'a pas atteint un état terminal.")


@pytest.mark.parametrize("value", ["0", "-1", "abc", "9"])
def test_max_workers_is_strictly_bounded(value):
    with pytest.raises(ValueError, match="VIDEO_AGENT_MAX_WORKERS"):
        parse_max_workers(value)


def test_max_workers_reads_environment(monkeypatch):
    monkeypatch.setenv("VIDEO_AGENT_MAX_WORKERS", "2")
    assert parse_max_workers() == 2


def test_job_progress_is_monotone_and_completed():
    manager = LocalJobManager(max_workers=1)

    def task(progress):
        progress({"current_step": "inspection", "progress_percent": 20})
        progress({"current_step": "segmentation", "progress_percent": 10})
        progress({"current_step": "report", "progress_percent": 120, "run_id": "a" * 64})
        return {
            "ok": True, "run_id": "a" * 64, "pipeline_status": "awaiting_vlm",
        }

    queued = manager.submit(task)
    completed = _wait_terminal(manager, queued["job_id"])
    assert completed["status"] == "completed"
    assert completed["progress_percent"] == 100
    assert completed["run_id"] == "a" * 64
    manager.shutdown()


def test_failed_job_does_not_stop_executor():
    manager = LocalJobManager(max_workers=1)

    def fail(_progress):
        raise JobExecutionError(
            "pipeline_rejected", "Échec simulé.", step="segmentation"
        )

    failed = manager.submit(fail)
    failed_result = _wait_terminal(manager, failed["job_id"])
    assert failed_result["status"] == "failed"
    assert failed_result["errors"][0]["code"] == "pipeline_rejected"

    succeeding = manager.submit(
        lambda progress: {"ok": True, "pipeline_status": "local_complete"}
    )
    assert _wait_terminal(manager, succeeding["job_id"])["status"] == "completed"
    manager.shutdown()


def test_two_jobs_have_distinct_secure_ids():
    manager = LocalJobManager(max_workers=1)
    first = manager.submit(lambda progress: {"ok": True})
    second = manager.submit(lambda progress: {"ok": True})
    assert first["job_id"] != second["job_id"]
    assert len(first["job_id"]) >= 24
    _wait_terminal(manager, first["job_id"])
    _wait_terminal(manager, second["job_id"])
    manager.shutdown()


def test_queued_job_can_be_cancelled_and_cleaned():
    manager = LocalJobManager(max_workers=1)
    release = threading.Event()
    cleaned = threading.Event()

    def blocking(_progress):
        release.wait(timeout=2)
        return {"ok": True}

    running = manager.submit(blocking)
    queued = manager.submit(
        lambda progress: {"ok": True}, cleanup=lambda: cleaned.set()
    )
    assert manager.cancel(queued["job_id"]) == "cancelled"
    cancelled = _wait_terminal(manager, queued["job_id"])
    assert cancelled["status"] == "cancelled"
    assert cleaned.wait(timeout=1)
    release.set()
    _wait_terminal(manager, running["job_id"])
    manager.shutdown()


def test_list_filters_jobs_by_status():
    manager = LocalJobManager(max_workers=1)
    failed = manager.submit(
        lambda progress: (_ for _ in ()).throw(
            JobExecutionError("failure", "échec")
        )
    )
    completed = manager.submit(lambda progress: {"ok": True})
    _wait_terminal(manager, failed["job_id"])
    _wait_terminal(manager, completed["job_id"])
    assert [job["status"] for job in manager.list(status="failed")] == ["failed"]
    assert [job["status"] for job in manager.list(status="completed")] == ["completed"]
    manager.shutdown()


def test_shutdown_stops_new_submissions():
    manager = LocalJobManager(max_workers=1)
    manager.shutdown()
    assert manager.is_shutdown is True
    assert manager.accepting is False
    with pytest.raises(RuntimeError, match="n'accepte plus"):
        manager.submit(lambda progress: {"ok": True})


def test_cleanup_abandoned_uploads_only_removes_old_entries(tmp_path):
    old = tmp_path / "old"
    recent = tmp_path / "recent"
    old.mkdir()
    recent.mkdir()
    (old / "source.mp4").write_bytes(b"x")
    (recent / "source.mp4").write_bytes(b"x")
    now = time.time()
    os.utime(old, (now - 1000, now - 1000))
    os.utime(recent, (now, now))
    removed = cleanup_abandoned_uploads(
        tmp_path, older_than_seconds=100, now=now
    )
    assert removed == 1
    assert not old.exists()
    assert recent.exists()
