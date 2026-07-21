import json
import importlib.util
import sys
import types
from pathlib import Path

from src.video_agent import run_identity as ri


def _identity(video, config=None, prompts=None, schemas=None, params=None):
    return ri.build_run_identity(
        video,
        config or {"vlm": {"mode": "manual", "provider": None}},
        prompts=prompts or {"vlm": "prompt-v1"},
        schemas=schemas or {"report": "1.0"},
        pipeline_parameters=params or {"segmentation": {"threshold": 25.0}},
    )


def test_same_video_and_same_configuration_same_run_id(tmp_path):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"same-content")
    assert _identity(video)["run_id"] == _identity(video)["run_id"]


def test_same_name_different_content_different_run_id(tmp_path):
    first = tmp_path / "a" / "clip.mp4"; first.parent.mkdir(); first.write_bytes(b"first")
    second = tmp_path / "b" / "clip.mp4"; second.parent.mkdir(); second.write_bytes(b"second")
    one, two = _identity(first), _identity(second)
    assert one["run_id"] != two["run_id"]
    assert one["video_sha256"] != two["video_sha256"]


def test_configuration_prompt_and_provider_change_identity(tmp_path):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"content")
    base = _identity(video)
    changed_config = _identity(video, config={"vlm": {"mode": "manual", "provider": None}, "x": 2})
    changed_prompt = _identity(video, prompts={"vlm": "prompt-v2"})
    changed_provider = _identity(video, config={"vlm": {"mode": "manual", "provider": "gemini"}})
    changed_model = _identity(video, params={"model": "gemini-2.5-pro"})
    assert len({
        base["run_id"], changed_config["run_id"], changed_prompt["run_id"],
        changed_provider["run_id"], changed_model["run_id"],
    }) == 5


def test_checkpoint_detects_deleted_and_modified_artifact(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    source = tmp_path / "source.json"; source.write_text('{"v": 1}', encoding="utf-8")
    output = tmp_path / "output.json"; output.write_text('{"ok": true}', encoding="utf-8")
    kwargs = {
        "inputs": {"source": source}, "outputs": {"output": output},
        "artifact_schema_version": "x/1", "configuration": {"mode": "manual"},
    }
    ri.write_checkpoint(checkpoints, "step", **kwargs, json_output_requirements={"output": ["ok"]})
    assert ri.validate_checkpoint(checkpoints, "step", **kwargs)[0] is True
    output.write_text('{"ok": false}', encoding="utf-8")
    valid, reason = ri.validate_checkpoint(checkpoints, "step", **kwargs)
    assert valid is False and "sortie" in reason
    output.unlink()
    valid, reason = ri.validate_checkpoint(checkpoints, "step", **kwargs)
    assert valid is False and "manquante" in reason


def test_checkpoint_detects_modified_input(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    source = tmp_path / "source.json"; source.write_text('{"v": 1}', encoding="utf-8")
    output = tmp_path / "output.json"; output.write_text('{"ok": true}', encoding="utf-8")
    kwargs = {
        "inputs": {"source": source}, "outputs": {"output": output},
        "artifact_schema_version": "x/1", "configuration": {"mode": "manual"},
    }
    ri.write_checkpoint(checkpoints, "step", **kwargs)
    source.write_text('{"v": 2}', encoding="utf-8")
    valid, reason = ri.validate_checkpoint(checkpoints, "step", **kwargs)
    assert valid is False and "entrée" in reason


def test_checkpoint_detects_wrong_hash_configuration_and_corrupt_json(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    source = tmp_path / "source"; source.write_text("input", encoding="utf-8")
    output = tmp_path / "output.json"; output.write_text('{"ok": true}', encoding="utf-8")
    kwargs = {
        "inputs": {"source": source}, "outputs": {"output": output},
        "artifact_schema_version": "x/1", "configuration": {"model": "a"},
    }
    path = ri.write_checkpoint(checkpoints, "step", **kwargs, json_output_requirements={"output": ["ok"]})
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["output_fingerprints"]["output"]["sha256"] = "0" * 64
    ri.atomic_write_json(path, checkpoint)
    assert ri.validate_checkpoint(checkpoints, "step", **kwargs)[0] is False
    ri.write_checkpoint(checkpoints, "step", **kwargs, json_output_requirements={"output": ["ok"]})
    changed = {**kwargs, "configuration": {"model": "b"}}
    assert ri.validate_checkpoint(checkpoints, "step", **changed)[0] is False
    output.write_text("{broken", encoding="utf-8")
    assert ri.validate_checkpoint(checkpoints, "step", **kwargs)[0] is False


def test_invalidate_checkpoint_and_dependents(tmp_path):
    checkpoints = tmp_path / "checkpoints"; checkpoints.mkdir()
    for name in ("a", "b", "c"):
        (checkpoints / f"{name}.json").write_text("{}", encoding="utf-8")
    ri.invalidate_checkpoints_from(checkpoints, ["a", "b", "c"], "b")
    assert (checkpoints / "a.json").exists()
    assert not (checkpoints / "b.json").exists()
    assert not (checkpoints / "c.json").exists()


def test_run_manifest_is_atomic_and_preserves_created_at(tmp_path):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"content")
    identity = _identity(video)
    manifest_path = tmp_path / "run" / "run_manifest.json"
    args = dict(
        identity=identity, original_filename="clip.mp4",
        normalized_configuration={"vlm": {"mode": "manual", "provider": None}},
        schemas={"report": "1"}, prompts={"vlm": "p"},
        pipeline_parameters={"segmentation": {}},
    )
    first = ri.create_or_update_run_manifest(manifest_path, status="running", **args)
    second = ri.create_or_update_run_manifest(manifest_path, status="awaiting_vlm", **args)
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == identity["run_id"]
    assert loaded["status"] == "awaiting_vlm"
    assert first["created_at"] == second["created_at"]
    assert loaded["original_filename"] == "clip.mp4"
    assert loaded["source_filenames"] == ["clip.mp4"]
    assert not list(manifest_path.parent.glob("*.tmp"))


def test_full_pipeline_cli_returns_run_identity(monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_full_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_full_pipeline_cli_identity", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = {
        "run_id": "abc123", "run_dir": "outputs/runs/abc123", "run_manifest": "outputs/runs/abc123/run_manifest.json",
        "ok": True, "status": "local_complete", "audio_present": False,
        "video_stem": "clip", "steps": [], "reports": {"json": None, "txt": None},
    }
    monkeypatch.setattr(module, "run_full_pipeline", lambda **kwargs: result)
    monkeypatch.setattr(sys, "argv", [str(script), "clip.mp4"])
    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "abc123"
    assert payload["run_dir"] == "outputs/runs/abc123"


def test_mcp_subprocess_returns_run_identity(monkeypatch):
    from src.video_agent import mcp_server

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write(json.dumps({
            "run_id": "mcp-run", "run_dir": "outputs/runs/mcp-run",
            "status": "awaiting_vlm", "ok": True,
        }))
        kwargs["stdout"].flush()
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(mcp_server.subprocess, "run", fake_run)
    result = mcp_server._run_full_pipeline_subprocess(
        "clip.mp4", "config/vlm_config.example.json", None, False, timeout=5
    )
    assert result["run_id"] == "mcp-run"
    assert result["run_dir"] == "outputs/runs/mcp-run"
