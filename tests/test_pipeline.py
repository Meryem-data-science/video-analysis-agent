import json
import subprocess
import types
from pathlib import Path

import pytest

from src.video_agent import pipeline
from src.video_agent.pipeline import PipelineError


# --- _resolve_video -------------------------------------------------------------
def test_resolve_video_missing():
    with pytest.raises(PipelineError):
        pipeline._resolve_video("__does_not_exist__.mp4")


# --- _run_step (subprocess mocké) ----------------------------------------------
def test_run_step_success(monkeypatch):
    monkeypatch.setattr(
        pipeline.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stderr=""),
    )
    st = pipeline._run_step("x", ["cmd"], timeout=5)
    assert st["ok"] is True and st["returncode"] == 0


def test_run_step_failure(monkeypatch):
    monkeypatch.setattr(
        pipeline.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stderr="boom"),
    )
    st = pipeline._run_step("x", ["cmd"], timeout=5)
    assert st["ok"] is False and "boom" in st["detail"]


def test_run_step_agent_rc2_ok(monkeypatch):
    monkeypatch.setattr(
        pipeline.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=2, stderr=""),
    )
    st = pipeline._run_step("vlm_analysis", ["cmd"], timeout=5)
    assert st["ok"] is True  # rc=2 toléré pour l'analyse VLM


def test_run_step_timeout(monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=5)
    monkeypatch.setattr(pipeline.subprocess, "run", _raise)
    st = pipeline._run_step("x", ["cmd"], timeout=5)
    assert st["ok"] is False and "timeout" in st["detail"]


# --- run_full_pipeline (orchestration, étapes simulées) ------------------------
def _install_fake_pipeline(monkeypatch, tmp_path, video_stem, audio_present):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(pipeline, "OUTPUTS", outputs)
    video = tmp_path / f"{video_stem}.mp4"
    video.write_bytes(b"stub")
    monkeypatch.setattr(pipeline, "_resolve_video", lambda name: video)
    monkeypatch.setattr(
        pipeline,
        "_load_vlm_runtime_config",
        lambda path: {"mode": "automatic", "provider": "gemini"},
    )

    cmds = {}

    def fake_step(name, cmd, timeout):
        cmds[name] = cmd
        cmds[f"_to_{name}"] = timeout  # mémorise le timeout passé à _run_step
        if "--output-dir" in cmd:
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
        if name == "prepare_vlm_run":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            (output_dir / "selected_inputs.json").write_text(
                json.dumps({"segments": [{"segment_id": 1, "prompt_path": (output_dir / "prompt_segment_001.md").as_posix()}]}),
                encoding="utf-8",
            )
            (output_dir / "run_plan.json").write_text("{}", encoding="utf-8")
            (output_dir / "prompt_segment_001.md").write_text("prompt", encoding="utf-8")
        if name == "audio_extraction":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            wav = output_dir / video_stem / "audio.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"wav")
        # Crée les fichiers de sortie référencés (pour satisfaire les existences).
        for flag in ("--output", "--manifest", "--text-output", "--summary-output"):
            if flag in cmd:
                target = Path(cmd[cmd.index(flag) + 1])
                target.parent.mkdir(parents=True, exist_ok=True)
                if name == "inspection":
                    target.write_text(json.dumps({"audio": {"present": audio_present}}),
                                      encoding="utf-8")
                elif name == "strategy":
                    target.write_text(json.dumps({"next_step": "segmentation"}), encoding="utf-8")
                elif name == "segmentation":
                    target.write_text(json.dumps({"segments": [{"segment_id": 1, "start_s": 0.0, "end_s": 1.0}]}), encoding="utf-8")
                elif name == "frames":
                    target.write_text(json.dumps({"frames": []}), encoding="utf-8")
                elif name == "motion":
                    target.write_text(json.dumps({"segments": []}), encoding="utf-8")
                elif name == "audio_extraction":
                    target.write_text(json.dumps({"audio_path": (Path(cmd[cmd.index("--output-dir") + 1]) / video_stem / "audio.wav").as_posix()}), encoding="utf-8")
                elif name == "audio_light_analysis":
                    target.write_text(json.dumps({"recommendation": {"run_whisper": True}}),
                                      encoding="utf-8")
                elif name == "transcription":
                    target.write_text(json.dumps({"segments": []}), encoding="utf-8")
                elif name == "fusion":
                    target.write_text(json.dumps({"segments": []}), encoding="utf-8")
                elif name == "vlm_pack":
                    target.write_text(json.dumps({"segments": []}), encoding="utf-8")
                elif name == "final_report" and flag == "--output":
                    target.write_text(
                        json.dumps({"status": "complete", "next_step": None, "validation": {}}),
                        encoding="utf-8",
                    )
                else:
                    target.write_text("{}", encoding="utf-8")
        return {"step": name, "ok": True, "returncode": 0, "detail": ""}

    cmds["_fake_step"] = fake_step

    monkeypatch.setattr(pipeline, "_run_step", fake_step)

    # scene_audio tourne EN PROCESSUS (pas via _run_step) -> on mocke l'appel direct
    # pour garder les tests hors-ligne (aucun ffmpeg/réseau réel).
    def fake_build_scene_audio(video_path, scenes, output_path=None, model=None, **kwargs):
        cmds["_scene_audio_call"] = {"scenes": scenes, "model": model, "output": output_path}
        if output_path:
            pth = Path(output_path)
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(
                json.dumps({"schema": "scene_audio/1.0", "status": "ok", "scenes": []}),
                encoding="utf-8",
            )
        return {"status": "ok", "artifact_path": output_path,
                "scenes_ok": 0, "scenes_total": 0, "model_used": model or "gemini-x"}

    monkeypatch.setattr(pipeline, "build_scene_audio", fake_build_scene_audio)

    # scene_audio_features (librosa) tourne aussi EN PROCESSUS -> on mocke l'appel direct.
    def fake_build_scene_audio_features(video_path, scenes, output_path=None, **kwargs):
        cmds["_audio_features_call"] = {"scenes": scenes, "output": output_path}
        if output_path:
            pth = Path(output_path)
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(
                json.dumps({"schema": "audio_features/1.0", "status": "ok", "scenes": []}),
                encoding="utf-8",
            )
        return {"status": "ok", "artifact_path": output_path, "scenes_ok": 0, "scenes_total": 0}

    monkeypatch.setattr(pipeline, "build_scene_audio_features", fake_build_scene_audio_features)
    return outputs, cmds


def test_pipeline_with_audio(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipA", audio_present=True)
    result = pipeline.run_full_pipeline("clipA.mp4")
    assert result["ok"] is True
    assert result["audio_present"] is True
    names = [s["step"] for s in result["steps"]]
    # branche audio présente
    for n in ("audio_extraction", "audio_light_analysis", "transcription"):
        assert n in names
    # fusion inclut --audio-light et --transcript
    assert "--audio-light" in cmds["fusion"]
    assert "--transcript" in cmds["fusion"]
    # rapports + statut produits
    assert result["reports"]["json"] is not None
    assert Path(result["run_dir"], "reports", "pipeline_status.json").exists()
    assert result["run_id"]


def test_pipeline_no_audio(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipB", audio_present=False)
    result = pipeline.run_full_pipeline("clipB.mp4")
    assert result["ok"] is True
    assert result["audio_present"] is False
    names = [s["step"] for s in result["steps"]]
    # branche audio ABSENTE
    for n in ("audio_extraction", "audio_light_analysis", "transcription"):
        assert n not in names
    # fusion sans audio
    assert "--audio-light" not in cmds["fusion"]
    assert "--transcript" not in cmds["fusion"]
    assert result["reports"]["json"] is not None


def test_pipeline_scene_audio_between_scenes_and_report(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipA2", audio_present=True)
    result = pipeline.run_full_pipeline("clipA2.mp4", audio_model="gemini-zzz")
    names = [s["step"] for s in result["steps"]]
    # scene_audio s'exécute après que les scènes sont disponibles (après fusion) et AVANT le rapport.
    assert "scene_audio" in names
    assert names.index("fusion") < names.index("scene_audio") < names.index("final_report")
    assert result["audio_status"] == "ok"
    assert result["audio_events"] is not None
    # L'override de modèle est transmis à build_scene_audio.
    assert cmds["_scene_audio_call"]["model"] == "gemini-zzz"
    # Le rapport final reste produit (final_report NON modifié à ce stade).
    assert result["reports"]["json"] is not None


def test_pipeline_scene_audio_failure_non_fatal(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipA3", audio_present=False)

    def boom(*a, **k):  # build_scene_audio lève -> le pipeline doit CONTINUER quand même
        raise RuntimeError("503 boom inattendu")

    monkeypatch.setattr(pipeline, "build_scene_audio", boom)
    result = pipeline.run_full_pipeline("clipA3.mp4")
    assert result["ok"] is True                            # échec audio mais pipeline CONTINUE
    names = [s["step"] for s in result["steps"]]
    assert "scene_audio" in names
    assert names.index("scene_audio") < names.index("final_report")
    assert result["audio_status"] == "échec"
    audio_step = [s for s in result["steps"] if s["step"] == "scene_audio"][0]
    assert audio_step["ok"] is False
    assert result["reports"]["json"] is not None           # rapport produit malgré l'échec audio


def test_pipeline_audio_disabled_skips(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipA4", audio_present=True)
    result = pipeline.run_full_pipeline("clipA4.mp4", enable_audio=False)
    names = [s["step"] for s in result["steps"]]
    assert "scene_audio" not in names                      # étape sautée
    assert "_scene_audio_call" not in cmds                 # build_scene_audio jamais appelé
    assert result["audio_status"] == "désactivé"
    assert result["audio_events"] is None
    assert result["reports"]["json"] is not None


def test_pipeline_audio_features_between_scenes_and_report(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipF1", audio_present=True)
    result = pipeline.run_full_pipeline("clipF1.mp4")
    names = [s["step"] for s in result["steps"]]
    # scene_audio_features s'exécute après les scènes (après fusion) et AVANT le rapport.
    assert "scene_audio_features" in names
    assert names.index("fusion") < names.index("scene_audio_features") < names.index("final_report")
    assert result["audio_features_status"] == "ok"
    assert result["audio_features"] is not None
    assert "_audio_features_call" in cmds                   # build_scene_audio_features appelé
    # le rapport consomme l'artefact features.
    assert "--audio-features" in cmds["final_report"]
    assert result["reports"]["json"] is not None


def test_pipeline_audio_features_failure_non_fatal(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipF2", audio_present=False)

    def boom(*a, **k):  # build_scene_audio_features lève -> le pipeline doit CONTINUER.
        raise RuntimeError("librosa boom inattendu")

    monkeypatch.setattr(pipeline, "build_scene_audio_features", boom)
    result = pipeline.run_full_pipeline("clipF2.mp4")
    assert result["ok"] is True
    names = [s["step"] for s in result["steps"]]
    assert "scene_audio_features" in names
    assert names.index("scene_audio_features") < names.index("final_report")
    assert result["audio_features_status"] == "échec"
    feat_step = [s for s in result["steps"] if s["step"] == "scene_audio_features"][0]
    assert feat_step["ok"] is False
    assert result["reports"]["json"] is not None


def test_pipeline_audio_features_disabled_skips(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipF3", audio_present=True)
    result = pipeline.run_full_pipeline("clipF3.mp4", enable_audio_features=False)
    names = [s["step"] for s in result["steps"]]
    assert "scene_audio_features" not in names
    assert "_audio_features_call" not in cmds
    assert result["audio_features_status"] == "désactivé"
    assert result["audio_features"] is None
    assert "--audio-features" not in cmds["final_report"]
    assert result["reports"]["json"] is not None


def test_pipeline_stops_on_critical_failure(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(pipeline, "OUTPUTS", outputs)
    video = tmp_path / "clipC.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "_resolve_video", lambda name: video)
    monkeypatch.setattr(
        pipeline,
        "_load_vlm_runtime_config",
        lambda path: {"mode": "automatic", "provider": "gemini"},
    )

    def fail_at_segmentation(name, cmd, timeout):
        if name == "inspection":
            Path(cmd[cmd.index("--output") + 1]).write_text(
                json.dumps({"audio": {"present": False}}), encoding="utf-8")
            return {"step": name, "ok": True, "returncode": 0, "detail": ""}
        if name == "strategy":
            Path(cmd[cmd.index("--output") + 1]).write_text(
                json.dumps({"next_step": "segmentation"}), encoding="utf-8")
            return {"step": name, "ok": True, "returncode": 0, "detail": ""}
        if name == "segmentation":
            return {"step": name, "ok": False, "returncode": 1, "detail": "seg fail"}
        return {"step": name, "ok": True, "returncode": 0, "detail": ""}

    monkeypatch.setattr(pipeline, "_run_step", fail_at_segmentation)
    result = pipeline.run_full_pipeline("clipC.mp4")
    assert result["ok"] is False
    names = [s["step"] for s in result["steps"]]
    assert names == ["inspection", "strategy", "segmentation"]  # s'arrête après l'échec
    assert result["reports"]["json"] is None


def test_pipeline_manual_stops_without_any_gemini_call(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipManual", False)
    monkeypatch.setattr(
        pipeline,
        "_load_vlm_runtime_config",
        lambda path: {"mode": "manual", "provider": None},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Gemini ne doit pas être appelé en mode manual")

    monkeypatch.setattr(pipeline, "build_scene_audio", forbidden)
    result = pipeline.run_full_pipeline("clipManual.mp4")
    names = [s["step"] for s in result["steps"]]
    assert result["ok"] is True
    assert result["status"] == "awaiting_vlm"
    assert result["next_step"] == "manual_vlm_analysis"
    assert result["reports"] == {"json": None, "txt": None}
    assert "prepare_vlm_run" in names
    assert "vlm_analysis" not in names
    assert "scene_audio" not in names
    assert "final_report" not in names
    assert result["manual_vlm"]["pack"].endswith("vlm_pack/vlm_pack.json")


def test_pipeline_disabled_stops_without_any_gemini_call(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipDisabled", False)
    monkeypatch.setattr(
        pipeline,
        "_load_vlm_runtime_config",
        lambda path: {"mode": "disabled", "provider": None},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Gemini ne doit pas être appelé en mode disabled")

    monkeypatch.setattr(pipeline, "build_scene_audio", forbidden)
    result = pipeline.run_full_pipeline("clipDisabled.mp4")
    names = [s["step"] for s in result["steps"]]
    assert result["ok"] is True
    assert result["status"] == "local_complete"
    assert result["next_step"] == "vlm_disabled"
    assert result["reports"] == {"json": None, "txt": None}
    assert "vlm_pack" in names
    assert "prepare_vlm_run" not in names
    assert "vlm_analysis" not in names
    assert "scene_audio" not in names


def test_pipeline_automatic_passes_explicit_gemini_provider(monkeypatch, tmp_path):
    outputs, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "clipAuto", False)
    result = pipeline.run_full_pipeline("clipAuto.mp4")
    assert result["status"] == "complete"
    assert "--provider" in cmds["vlm_analysis"]
    assert cmds["vlm_analysis"][cmds["vlm_analysis"].index("--provider") + 1] == "gemini"


def test_pipeline_progress_callback_is_optional_real_and_monotone(monkeypatch, tmp_path):
    _install_fake_pipeline(monkeypatch, tmp_path, "clipProgress", audio_present=True)
    events = []
    result = pipeline.run_full_pipeline(
        "clipProgress.mp4", progress_callback=events.append
    )
    percentages = [event["progress_percent"] for event in events]
    completed_steps = {
        event["current_step"] for event in events
        if "en cours" not in event["message"]
        and event["current_step"] != "run_initialized"
    }
    assert result["ok"] is True
    assert percentages == sorted(percentages)
    assert all(0 <= value <= 100 for value in percentages)
    assert events[0]["current_step"] == "run_initialized"
    assert events[-1]["current_step"] == "completed"
    assert events[-1]["progress_percent"] == 100
    assert {
        "inspection", "segmentation", "frame_extraction", "audio_extraction",
        "transcription", "fusion", "vlm_preparation", "vlm_analysis",
        "validation", "report", "completed",
    } <= completed_steps


def test_pipeline_ignores_progress_callback_failures(monkeypatch, tmp_path):
    _install_fake_pipeline(monkeypatch, tmp_path, "clipProgressError", audio_present=False)

    def broken_callback(_event):
        raise RuntimeError("consumer unavailable")

    result = pipeline.run_full_pipeline(
        "clipProgressError.mp4", progress_callback=broken_callback
    )
    assert result["ok"] is True


def test_two_same_named_videos_with_different_content_never_share_run(monkeypatch, tmp_path):
    outputs, _ = _install_fake_pipeline(monkeypatch, tmp_path, "clip", False)
    first = pipeline.run_full_pipeline("clip.mp4")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"different-content")
    second = pipeline.run_full_pipeline("clip.mp4")
    assert first["run_id"] != second["run_id"]
    assert first["run_dir"] != second["run_dir"]
    first_root, second_root = Path(first["run_dir"]), Path(second["run_dir"])
    assert first_root.exists() and second_root.exists()
    assert first_root.parent == second_root.parent == outputs / "runs"
    for path in first["artifacts"].values():
        assert Path(path).is_relative_to(first_root)
        assert not Path(path).is_relative_to(second_root)
    for path in second["artifacts"].values():
        assert Path(path).is_relative_to(second_root)
        assert not Path(path).is_relative_to(first_root)


def test_same_run_resumes_from_verified_checkpoints(monkeypatch, tmp_path):
    _, _ = _install_fake_pipeline(monkeypatch, tmp_path, "resume", False)
    first = pipeline.run_full_pipeline("resume.mp4")
    second = pipeline.run_full_pipeline("resume.mp4")
    assert first["run_id"] == second["run_id"]
    assert any(step["detail"] == "checkpoint validé" for step in second["steps"])


def test_pipeline_configuration_change_creates_new_run(monkeypatch, tmp_path):
    _, _ = _install_fake_pipeline(monkeypatch, tmp_path, "config-change", False)
    first = pipeline.run_full_pipeline("config-change.mp4", enable_arc=True)
    second = pipeline.run_full_pipeline("config-change.mp4", enable_arc=False)
    assert first["run_id"] != second["run_id"]


def test_deleted_artifact_invalidates_step_and_downstream(monkeypatch, tmp_path):
    _, _ = _install_fake_pipeline(monkeypatch, tmp_path, "deleted", False)
    first = pipeline.run_full_pipeline("deleted.mp4")
    segments = Path(first["run_dir"]) / "scenes" / "segments.json"
    segments.unlink()
    second = pipeline.run_full_pipeline("deleted.mp4")
    details = {step["step"]: step["detail"] for step in second["steps"]}
    assert details["inspection"] == "checkpoint validé"
    assert details["strategy"] == "checkpoint validé"
    assert details["segmentation"] != "checkpoint validé"
    assert details["frames"] != "checkpoint validé"


def test_interrupted_run_resumes_from_last_verified_step(monkeypatch, tmp_path):
    _, cmds = _install_fake_pipeline(monkeypatch, tmp_path, "interrupted", False)
    normal_step = cmds["_fake_step"]
    state = {"failed": False}

    def fail_frames_once(name, cmd, timeout):
        if name == "frames" and not state["failed"]:
            state["failed"] = True
            return {"step": name, "ok": False, "returncode": 1, "detail": "interruption simulée"}
        return normal_step(name, cmd, timeout)

    monkeypatch.setattr(pipeline, "_run_step", fail_frames_once)
    first = pipeline.run_full_pipeline("interrupted.mp4")
    assert first["ok"] is False

    monkeypatch.setattr(pipeline, "_run_step", normal_step)
    resumed = pipeline.run_full_pipeline("interrupted.mp4")
    assert resumed["ok"] is True
    details = {step["step"]: step["detail"] for step in resumed["steps"]}
    assert details["inspection"] == "checkpoint validé"
    assert details["strategy"] == "checkpoint validé"
    assert details["segmentation"] == "checkpoint validé"
    assert details["frames"] != "checkpoint validé"
