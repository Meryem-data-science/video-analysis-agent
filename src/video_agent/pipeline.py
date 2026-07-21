"""Orchestrateur isolé par identité de run déterministe."""

from __future__ import annotations

import json
import importlib
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .run_identity import (
    atomic_write_json,
    build_run_identity,
    create_or_update_run_manifest,
    fingerprint_path,
    invalidate_checkpoints_from,
    prompt_versions,
    schema_versions,
    validate_checkpoint,
    write_checkpoint,
)
from .scene_audio import build_scene_audio
from .scene_audio_features import build_scene_audio_features
from .vlm_strategy import VlmStrategyError, load_vlm_config, validate_config

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (
    _SOURCE_ROOT
    if (_SOURCE_ROOT / "src" / "video_agent").is_dir()
    else Path.cwd().resolve()
)
_source_scripts = PROJECT_ROOT / "scripts"
SCRIPTS = (
    _source_scripts
    if _source_scripts.is_dir()
    else Path(importlib.import_module("scripts").__file__).resolve().parent
)
OUTPUTS = PROJECT_ROOT / "outputs"

STEP_ORDER: List[str] = [
    "inspection", "strategy", "segmentation", "frames", "motion",
    "audio_extraction", "audio_light_analysis", "transcription",
    "fusion", "vlm_pack", "prepare_vlm_run", "vlm_analysis",
    "scene_audio", "scene_audio_features", "final_report",
]

STEP_SCHEMA_VERSIONS = {name: f"{name}/1.0" for name in STEP_ORDER}

PROGRESS_STEPS: Dict[str, tuple[str, int, str]] = {
    "inspection": ("inspection", 8, "Inspection technique terminée."),
    "strategy": ("strategy", 12, "Stratégie locale déterminée."),
    "segmentation": ("segmentation", 22, "Segmentation des scènes terminée."),
    "frames": ("frame_extraction", 34, "Extraction des frames terminée."),
    "motion": ("motion_representation", 42, "Représentation du mouvement terminée."),
    "audio_extraction": ("audio_extraction", 50, "Extraction audio terminée."),
    "audio_light_analysis": ("audio_light_analysis", 56, "Analyse audio légère terminée."),
    "transcription": ("transcription", 64, "Transcription terminée."),
    "fusion": ("fusion", 72, "Fusion audio-visuelle terminée."),
    "vlm_pack": ("vlm_preparation", 80, "Pack VLM préparé."),
    "prepare_vlm_run": ("vlm_preparation", 86, "Prompts VLM préparés."),
    "vlm_analysis": ("vlm_analysis", 92, "Analyse VLM terminée."),
}

ProgressCallback = Callable[[Dict[str, Any]], None]


class PipelineError(Exception):
    """Erreur d'orchestration du pipeline."""


def _resolve_video_root() -> Path:
    env = os.getenv("VIDEO_ROOT")
    if env:
        candidate = Path(env)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate.resolve()
    for name in ("vidéos", "videos"):
        if (PROJECT_ROOT / name).exists():
            return (PROJECT_ROOT / name).resolve()
    return (PROJECT_ROOT / "vidéos").resolve()


VIDEO_ROOT = _resolve_video_root()


def _resolve_video(video_name: str) -> Path:
    candidate = Path(video_name)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] in {"vidéos", "videos"}:
            candidate = PROJECT_ROOT / candidate
        else:
            candidate = VIDEO_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PipelineError(f"Accès refusé : la vidéo doit rester dans le projet : {candidate}") from exc
    if not candidate.exists() or not candidate.is_file():
        raise PipelineError(f"Vidéo introuvable : {candidate}")
    return candidate


def _py() -> str:
    return sys.executable


def _run_step(name: str, cmd: List[str], timeout: float) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[pipeline] ▶ {name} …", file=sys.stderr, flush=True)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=str(PROJECT_ROOT), env=env,
        )
        rc = proc.returncode
        detail = (proc.stderr or "").strip()[-400:]
    except subprocess.TimeoutExpired:
        return {
            "step": name, "ok": False, "returncode": None,
            "detail": f"timeout {timeout}s",
            "elapsed_s": round(time.perf_counter() - started, 6), "cached": False,
        }
    ok = rc == 0 or (name == "vlm_analysis" and rc == 2)
    return {
        "step": name, "ok": ok, "returncode": rc, "detail": detail if not ok else "",
        "elapsed_s": round(time.perf_counter() - started, 6), "cached": False,
    }


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _load_vlm_runtime_config(config_path: str) -> Dict[str, Any]:
    """Valide la configuration et conserve les options modèle qui influencent le run."""
    try:
        raw = load_vlm_config(config_path)
        normalized = validate_config(raw)
    except VlmStrategyError as exc:
        raise PipelineError(f"Configuration VLM invalide : {exc}") from exc
    normalized["provider_options"] = {
        key: raw.get(key) for key in (
            "gemini_vlm_model", "gemini_text_model", "audio_provider"
        ) if key in raw
    }
    return normalized


def _resolve_config_path(config_path: str) -> str:
    """Résout la config source, puis la config manual embarquée dans la roue."""
    path = Path(config_path)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    if config_path.replace("\\", "/") == "config/vlm_config.example.json":
        packaged = Path(__file__).resolve().parent / "vlm_config.example.json"
        if packaged.is_file():
            return str(packaged)
    return config_path


def _pipeline_parameters(
    *, model: Optional[str], audio_model: Optional[str], enable_audio: bool,
    enable_arc: bool, enable_audio_features: bool,
) -> Dict[str, Any]:
    """Paramètres sémantiques effectivement utilisés, inclus dans l'identité."""
    return {
        "segmentation": {
            "backend": "PySceneDetect-ContentDetector",
            "thresholds": {"short_vertical_video": 15.0, "long_horizontal_video": 40.0, "default": 25.0},
        },
        "frames": {"policy": "strategy_driven", "vlm_pack_max_frames": 6},
        "audio_extraction": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"},
        "transcription": {
            "model": "small", "device": "cpu", "compute_type": "int8",
            "language": None, "task": "transcribe", "beam_size": 5,
        },
        "vlm_model_override": model,
        "scene_audio": {
            "enabled": enable_audio,
            "model_override": audio_model or os.getenv("GEMINI_MODEL"),
        },
        "narrative_arc_enabled": enable_arc,
        "audio_features_enabled": enable_audio_features,
        "environment_model_overrides": {
            "GEMINI_VLM_MODEL": os.getenv("GEMINI_VLM_MODEL"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
        },
    }


def _legacy_artifacts(stem: str) -> List[str]:
    candidates = [
        OUTPUTS / f"inspection_{stem}.json", OUTPUTS / f"strategy_{stem}.json",
        OUTPUTS / f"segments_{stem}.json", OUTPUTS / f"frames_manifest_{stem}.json",
        OUTPUTS / f"motion_manifest_{stem}.json", OUTPUTS / f"transcript_{stem}.json",
        OUTPUTS / f"audio_visual_fusion_{stem}.json", OUTPUTS / f"vlm_pack_{stem}.json",
        OUTPUTS / f"final_report_{stem}.json", OUTPUTS / "frames" / stem,
        OUTPUTS / "motion" / stem, OUTPUTS / "audio" / stem,
        OUTPUTS / "vlm_runs" / stem, OUTPUTS / "vlm_results" / stem,
    ]
    legacy = [path.as_posix() for path in candidates if path.exists()]
    old_checkpoints = OUTPUTS / ".checkpoints"
    if old_checkpoints.exists():
        legacy.extend(path.as_posix() for path in old_checkpoints.glob(f"{stem}_*"))
    return sorted(set(legacy))


def _run_paths(run_dir: Path, stem: str) -> Dict[str, Path]:
    paths = {
        "run_root": run_dir,
        "manifest": run_dir / "run_manifest.json",
        "checkpoints": run_dir / "checkpoints",
        "inspection_dir": run_dir / "inspection",
        "scenes_dir": run_dir / "scenes",
        "frames_dir": run_dir / "frames",
        "audio_dir": run_dir / "audio",
        "transcription_dir": run_dir / "transcription",
        "fusion_dir": run_dir / "fusion",
        "vlm_pack_dir": run_dir / "vlm_pack",
        "results_dir": run_dir / "vlm_results",
        "reports_dir": run_dir / "reports",
    }
    paths.update({
        "inspection": paths["inspection_dir"] / "inspection.json",
        "strategy": paths["scenes_dir"] / "strategy.json",
        "segments": paths["scenes_dir"] / "segments.json",
        "frames_manifest": paths["frames_dir"] / "frames_manifest.json",
        "motion_dir": paths["scenes_dir"] / "motion",
        "motion_manifest": paths["scenes_dir"] / "motion_manifest.json",
        "audio_manifest": paths["audio_dir"] / "audio_manifest.json",
        "audio_wav": paths["audio_dir"] / stem / "audio.wav",
        "audio_light": paths["audio_dir"] / "audio_light.json",
        "transcript": paths["transcription_dir"] / "transcript.json",
        "fusion": paths["fusion_dir"] / "fusion.json",
        "pack": paths["vlm_pack_dir"] / "vlm_pack.json",
        "vlm_run_dir": paths["vlm_pack_dir"] / "prompts",
        "summary": paths["results_dir"] / "vlm_results_summary.json",
        "report_json": paths["reports_dir"] / "final_report.json",
        "report_txt": paths["reports_dir"] / "final_report.txt",
        "audio_events": paths["audio_dir"] / "audio_events.json",
        "audio_features": paths["audio_dir"] / "audio_features.json",
        "status": paths["reports_dir"] / "pipeline_status.json",
    })
    for key in (
        "checkpoints", "inspection_dir", "scenes_dir", "frames_dir", "audio_dir",
        "transcription_dir", "fusion_dir", "vlm_pack_dir", "results_dir", "reports_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _manual_vlm_requirements(paths: Mapping[str, Path]) -> Dict[str, Any]:
    selected = _load_json(paths["vlm_run_dir"] / "selected_inputs.json") or {}
    segments = selected.get("segments") if isinstance(selected.get("segments"), list) else []
    expected = [
        (paths["results_dir"] / f"vlm_result_segment_{int(seg['segment_id']):03d}.json").as_posix()
        for seg in segments if isinstance(seg, dict) and seg.get("segment_id") is not None
    ]
    return {
        "pack": paths["pack"].as_posix(),
        "prompts_dir": paths["vlm_run_dir"].as_posix(),
        "selected_inputs": (paths["vlm_run_dir"] / "selected_inputs.json").as_posix(),
        "prompts": [seg.get("prompt_path") for seg in segments if seg.get("prompt_path")],
        "results_dir": paths["results_dir"].as_posix(),
        "expected_result_files": expected,
    }


def run_full_pipeline(
    video_name: str,
    config_path: str = "config/vlm_config.example.json",
    model: Optional[str] = None,
    overwrite_analysis: bool = False,
    step_timeout: float = 900.0,
    enable_audio: bool = True,
    audio_model: Optional[str] = None,
    enable_arc: bool = True,
    enable_audio_features: bool = True,
    force: bool = False,
    from_step: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    video = _resolve_video(video_name)
    config_path = _resolve_config_path(config_path)
    stem = video.stem
    vlm_config = _load_vlm_runtime_config(config_path)
    vlm_mode = str(vlm_config["mode"])
    vlm_provider = vlm_config.get("provider")
    parameters = _pipeline_parameters(
        model=model, audio_model=audio_model, enable_audio=enable_audio,
        enable_arc=enable_arc, enable_audio_features=enable_audio_features,
    )
    normalized_configuration = {"vlm": vlm_config, "pipeline": parameters}
    all_prompts = prompt_versions(PROJECT_ROOT)
    used_prompt_names = {"vlm_directives", "vlm_pack_prompt"}
    if vlm_mode != "disabled":
        used_prompt_names.add("manual_prompt_renderer")
    if vlm_mode == "automatic" and enable_audio:
        used_prompt_names.add("scene_audio_prompt")
    if vlm_mode == "automatic" and enable_arc:
        used_prompt_names.add("narrative_arc_prompt")
    prompts = {
        name: version for name, version in all_prompts.items()
        if name in used_prompt_names
    }
    schemas = schema_versions()
    identity = build_run_identity(
        video, normalized_configuration, prompts=prompts, schemas=schemas,
        pipeline_parameters=parameters,
    )
    run_id = identity["run_id"]
    run_dir = OUTPUTS / "runs" / run_id
    paths = _run_paths(run_dir, stem)
    checkpoints = paths["checkpoints"]
    legacy = _legacy_artifacts(stem)
    last_progress = 0

    def emit_progress(
        current_step: str,
        progress_percent: int,
        message: str,
        *,
        warnings: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Notifie sans jamais faire échouer le pipeline ni diminuer la progression."""
        nonlocal last_progress
        if progress_callback is None:
            return
        bounded = min(100, max(last_progress, max(0, int(progress_percent))))
        last_progress = bounded
        event = {
            "run_id": run_id,
            "current_step": current_step,
            "progress_percent": bounded,
            "message": message,
        }
        if warnings:
            event["warnings"] = warnings
        try:
            progress_callback(event)
        except Exception:  # noqa: BLE001
            logging.warning("Callback de progression ignoré après une erreur", exc_info=True)

    emit_progress("run_initialized", 0, "Run initialisé.")

    if force:
        shutil.rmtree(checkpoints, ignore_errors=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
    if from_step:
        if from_step not in STEP_ORDER:
            raise PipelineError(f"--from-step inconnu : {from_step!r}")
        invalidate_checkpoints_from(checkpoints, STEP_ORDER, from_step)
        if STEP_ORDER.index(from_step) <= STEP_ORDER.index("scene_audio"):
            shutil.rmtree(checkpoints / "cache" / "scene_audio", ignore_errors=True)

    principal_artifacts = {
        "inspection": paths["inspection"].as_posix(),
        "segments": paths["segments"].as_posix(),
        "frames_manifest": paths["frames_manifest"].as_posix(),
        "frames": paths["frames_dir"].as_posix(),
        "motion_manifest": paths["motion_manifest"].as_posix(),
        "motion": paths["motion_dir"].as_posix(),
        "audio_manifest": paths["audio_manifest"].as_posix(),
        "audio": paths["audio_dir"].as_posix(),
        "audio_light": paths["audio_light"].as_posix(),
        "transcript": paths["transcript"].as_posix(),
        "fusion": paths["fusion"].as_posix(),
        "vlm_pack": paths["pack"].as_posix(),
        "vlm_prompts": paths["vlm_run_dir"].as_posix(),
        "vlm_results": paths["results_dir"].as_posix(),
        "validation": paths["summary"].as_posix(),
        "audio_features": paths["audio_features"].as_posix(),
        "pipeline_status": paths["status"].as_posix(),
        "final_report": paths["report_json"].as_posix(),
        "final_report_text": paths["report_txt"].as_posix(),
    }
    create_or_update_run_manifest(
        paths["manifest"], identity=identity, original_filename=video.name,
        normalized_configuration=normalized_configuration, schemas=schemas, prompts=prompts,
        pipeline_parameters=parameters, status="running", artifacts=principal_artifacts,
        legacy_artifacts=legacy,
    )

    statuses: List[Dict[str, Any]] = []
    audio_status = "désactivé" if not enable_audio else "non_exécuté"
    audio_features_status = "désactivé" if not enable_audio_features else "non_exécuté"
    audio_present = False
    video_input = {
        "type": "file", "exists": True, "size_bytes": identity["video_size_bytes"],
        "sha256": identity["video_sha256"],
    }

    def manifest_update(status: str, current_step: Optional[str] = None, metadata=None) -> None:
        create_or_update_run_manifest(
            paths["manifest"], identity=identity, original_filename=video.name,
            normalized_configuration=normalized_configuration, schemas=schemas, prompts=prompts,
            pipeline_parameters=parameters, status=status, current_step=current_step,
            video_metadata=metadata, artifacts=principal_artifacts, legacy_artifacts=legacy,
        )

    def step(
        name: str, cmd: List[str], *, inputs: Mapping[str, Any], outputs: Mapping[str, Any],
        required_json: Optional[Mapping[str, List[str]]] = None,
        step_configuration: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        configuration = {"run_id": run_id, "step": name, **dict(step_configuration or {})}
        valid, reason = validate_checkpoint(
            checkpoints, name, inputs=inputs, outputs=outputs,
            artifact_schema_version=STEP_SCHEMA_VERSIONS[name], configuration=configuration,
        )
        if not force and valid:
            statuses.append({
                "step": name, "ok": True, "returncode": 0,
                "detail": "checkpoint validé", "elapsed_s": 0.0, "cached": True,
            })
            if name == "final_report":
                emit_progress("validation", 97, "Résultats VLM validés.")
                emit_progress("report", 99, "Rapport final construit.")
            elif name in PROGRESS_STEPS:
                public_step, percent, message = PROGRESS_STEPS[name]
                emit_progress(public_step, percent, message)
            return True
        if (checkpoints / f"{name}.json").exists():
            logging.info("checkpoint %s invalidé: %s", name, reason)
        invalidate_checkpoints_from(checkpoints, STEP_ORDER, name)
        manifest_update("running", name)
        if name == "final_report":
            emit_progress("validation", last_progress, "Validation des résultats VLM en cours.")
        elif name in PROGRESS_STEPS:
            public_step, _, _ = PROGRESS_STEPS[name]
            emit_progress(public_step, last_progress, f"Étape {public_step} en cours.")
        result = _run_step(name, cmd, step_timeout)
        if result["ok"] and not (name == "vlm_analysis" and result.get("returncode") == 2):
            write_checkpoint(
                checkpoints, name, inputs=inputs, outputs=outputs,
                artifact_schema_version=STEP_SCHEMA_VERSIONS[name], configuration=configuration,
                json_output_requirements=required_json,
            )
            checkpoint_ok, checkpoint_reason = validate_checkpoint(
                checkpoints, name, inputs=inputs, outputs=outputs,
                artifact_schema_version=STEP_SCHEMA_VERSIONS[name], configuration=configuration,
            )
            if not checkpoint_ok:
                result.update(ok=False, detail=f"sorties non validables: {checkpoint_reason}")
                invalidate_checkpoints_from(checkpoints, STEP_ORDER, name)
        statuses.append(result)
        succeeded = bool(result["ok"])
        if succeeded:
            if name == "final_report":
                emit_progress("validation", 97, "Résultats VLM validés.")
                emit_progress("report", 99, "Rapport final construit.")
            elif name in PROGRESS_STEPS:
                public_step, percent, message = PROGRESS_STEPS[name]
                emit_progress(public_step, percent, message)
        return succeeded

    def in_process_checkpoint_valid(
        name: str, inputs: Mapping[str, Any], outputs: Mapping[str, Any], config: Mapping[str, Any]
    ) -> bool:
        valid, _ = validate_checkpoint(
            checkpoints, name, inputs=inputs, outputs=outputs,
            artifact_schema_version=STEP_SCHEMA_VERSIONS[name], configuration=config,
        )
        return valid and not force

    def finalize(ok: bool, status: Optional[str] = None, next_step: Any = None, manual_vlm=None) -> Dict[str, Any]:
        final_status = status or ("complete" if ok else "failed")
        expose = final_status in ("complete", "partial", "failed")
        result = {
            "pipeline_type": "full_video_pipeline",
            "run_id": run_id,
            "run_dir": run_dir.as_posix(),
            "run_manifest": paths["manifest"].as_posix(),
            "video_path": str(video), "video_stem": stem,
            "video_sha256": identity["video_sha256"],
            "video_fingerprint": identity["video_sha256"],
            "checkpoint_dir": checkpoints.as_posix(),
            "audio_present": audio_present, "status": final_status, "next_step": next_step,
            "vlm": {"mode": vlm_mode, "provider": vlm_provider},
            "manual_vlm": manual_vlm, "steps": statuses, "ok": ok,
            "performance": {
                "step_elapsed_s": {
                    str(item.get("step")): float(item.get("elapsed_s") or 0.0)
                    for item in statuses
                },
                "total_elapsed_s": round(
                    sum(float(item.get("elapsed_s") or 0.0) for item in statuses), 6
                ),
                "cached_steps": [
                    str(item.get("step")) for item in statuses if item.get("cached") is True
                ],
            },
            "artifacts": principal_artifacts,
            "reports": {
                "json": paths["report_json"].as_posix() if expose and paths["report_json"].exists() else None,
                "txt": paths["report_txt"].as_posix() if expose and paths["report_txt"].exists() else None,
            },
            "summary": paths["summary"].as_posix() if paths["summary"].exists() else None,
            "audio_status": audio_status,
            "audio_events": paths["audio_events"].as_posix() if paths["audio_events"].exists() else None,
            "audio_features_status": audio_features_status,
            "audio_features": paths["audio_features"].as_posix() if paths["audio_features"].exists() else None,
            "legacy_artifacts_detected": legacy,
        }
        atomic_write_json(paths["status"], result)
        manifest_update(final_status, None, _load_json(paths["inspection"]))
        if ok:
            emit_progress("completed", 100, "Pipeline terminé.")
        else:
            failed_step = "pipeline"
            if statuses:
                internal_step = str(statuses[-1].get("step") or "pipeline")
                failed_step = PROGRESS_STEPS.get(
                    internal_step, (internal_step, last_progress, "")
                )[0]
            emit_progress(
                failed_step, last_progress,
                f"Pipeline interrompu pendant l'étape {failed_step}.",
            )
        return result

    py = _py()
    if not step(
        "inspection", [py, str(SCRIPTS / "inspect_video.py"), str(video), "--output", str(paths["inspection"])],
        inputs={"video": video_input}, outputs={"inspection": paths["inspection"]},
        required_json={"inspection": ["audio"]},
    ):
        return finalize(False)
    inspection = _load_json(paths["inspection"]) or {}
    audio_present = bool((inspection.get("audio") or {}).get("present"))
    manifest_update("running", "strategy", inspection)

    if not step(
        "strategy", [py, str(SCRIPTS / "decide_strategy.py"), str(paths["inspection"]), "--output", str(paths["strategy"])],
        inputs={"inspection": paths["inspection"]}, outputs={"strategy": paths["strategy"]},
        required_json={"strategy": ["next_step"]},
    ):
        return finalize(False)
    if not step(
        "segmentation", [py, str(SCRIPTS / "segment_video.py"), str(video), "--strategy", str(paths["strategy"]), "--output", str(paths["segments"])],
        inputs={"video": video_input, "strategy": paths["strategy"]}, outputs={"segments": paths["segments"]},
        required_json={"segments": ["segments"]},
    ):
        return finalize(False)
    if not step(
        "frames", [py, str(SCRIPTS / "extract_frames.py"), str(video), "--segments", str(paths["segments"]), "--strategy", str(paths["strategy"]), "--output-dir", str(paths["frames_dir"]), "--manifest", str(paths["frames_manifest"])],
        inputs={"video": video_input, "segments": paths["segments"], "strategy": paths["strategy"]},
        outputs={"frames_manifest": paths["frames_manifest"], "frames": paths["frames_dir"]},
        required_json={"frames_manifest": ["frames"]},
    ):
        return finalize(False)
    if not step(
        "motion", [py, str(SCRIPTS / "build_motion.py"), "--frames-manifest", str(paths["frames_manifest"]), "--output-dir", str(paths["motion_dir"]), "--manifest", str(paths["motion_manifest"])],
        inputs={"frames_manifest": paths["frames_manifest"], "frames": paths["frames_dir"]},
        outputs={"motion_manifest": paths["motion_manifest"], "motion": paths["motion_dir"]},
        required_json={"motion_manifest": ["segments"]},
    ):
        return finalize(False)

    audio_light_ok = False
    transcript_ok = False
    if audio_present:
        if step(
            "audio_extraction", [py, str(SCRIPTS / "extract_audio.py"), str(video), "--output-dir", str(paths["audio_dir"]), "--manifest", str(paths["audio_manifest"])],
            inputs={"video": video_input}, outputs={"audio_manifest": paths["audio_manifest"], "audio_wav": paths["audio_wav"]},
            required_json={"audio_manifest": ["audio_path"]},
        ):
            audio_light_ok = step(
                "audio_light_analysis", [py, str(SCRIPTS / "analyze_audio_light.py"), str(paths["audio_wav"]), "--audio-manifest", str(paths["audio_manifest"]), "--output", str(paths["audio_light"])],
                inputs={"audio_wav": paths["audio_wav"], "audio_manifest": paths["audio_manifest"]},
                outputs={"audio_light": paths["audio_light"]}, required_json={"audio_light": ["recommendation"]},
            )
            should_transcribe = bool(((_load_json(paths["audio_light"]) or {}).get("recommendation") or {}).get("run_whisper")) if audio_light_ok else False
            if should_transcribe:
                transcript_ok = step(
                    "transcription", [py, str(SCRIPTS / "transcribe_audio.py"), str(paths["audio_wav"]), "--audio-manifest", str(paths["audio_manifest"]), "--audio-light-analysis", str(paths["audio_light"]), "--output", str(paths["transcript"]), "--model", "small"],
                    inputs={"audio_wav": paths["audio_wav"], "audio_manifest": paths["audio_manifest"], "audio_light": paths["audio_light"]},
                    outputs={"transcript": paths["transcript"]}, required_json={"transcript": ["segments"]},
                )

    fusion_inputs: Dict[str, Any] = {
        "segments": paths["segments"], "frames_manifest": paths["frames_manifest"],
        "motion_manifest": paths["motion_manifest"],
    }
    fusion_cmd = [py, str(SCRIPTS / "fuse_audio_visual.py"), "--segments", str(paths["segments"]), "--frames-manifest", str(paths["frames_manifest"]), "--motion-manifest", str(paths["motion_manifest"]), "--output", str(paths["fusion"])]
    if audio_light_ok and paths["audio_light"].exists():
        fusion_cmd += ["--audio-light", str(paths["audio_light"])]
        fusion_inputs["audio_light"] = paths["audio_light"]
    if transcript_ok and paths["transcript"].exists():
        fusion_cmd += ["--transcript", str(paths["transcript"])]
        fusion_inputs["transcript"] = paths["transcript"]
    if not step(
        "fusion", fusion_cmd, inputs=fusion_inputs, outputs={"fusion": paths["fusion"]},
        required_json={"fusion": ["segments"]},
    ):
        return finalize(False)
    if not step(
        "vlm_pack", [py, str(SCRIPTS / "build_vlm_pack.py"), "--fusion", str(paths["fusion"]), "--output", str(paths["pack"])],
        inputs={"fusion": paths["fusion"]}, outputs={"vlm_pack": paths["pack"]},
        required_json={"vlm_pack": ["segments"]},
    ):
        return finalize(False)

    if vlm_mode == "disabled":
        return finalize(True, "local_complete", "vlm_disabled", {"pack": paths["pack"].as_posix()})

    if not step(
        "prepare_vlm_run", [py, str(SCRIPTS / "prepare_vlm_run.py"), "--pack", str(paths["pack"]), "--config", config_path, "--output-dir", str(paths["vlm_run_dir"])],
        inputs={"vlm_pack": paths["pack"], "config": Path(config_path)},
        outputs={"prompts": paths["vlm_run_dir"]},
    ):
        return finalize(False)
    if vlm_mode == "manual":
        return finalize(True, "awaiting_vlm", "manual_vlm_analysis", _manual_vlm_requirements(paths))

    agent_cmd = [py, str(SCRIPTS / "run_vlm_agent.py"), "--run-dir", str(paths["vlm_run_dir"]), "--pack", str(paths["pack"]), "--config", config_path, "--results-dir", str(paths["results_dir"]), "--summary-output", str(paths["summary"]), "--provider", str(vlm_provider)]
    if model:
        agent_cmd += ["--model", model]
    if overwrite_analysis:
        agent_cmd.append("--overwrite")
    if not step(
        "vlm_analysis", agent_cmd,
        inputs={"prompts": paths["vlm_run_dir"], "vlm_pack": paths["pack"], "config": Path(config_path)},
        outputs={"vlm_results": paths["results_dir"]},
    ):
        return finalize(False)

    scene_inputs = {"video": video_input, "segments": paths["segments"]}
    scene_config = {"run_id": run_id, "step": "scene_audio", "model": audio_model}
    if enable_audio:
        if in_process_checkpoint_valid("scene_audio", scene_inputs, {"audio_events": paths["audio_events"]}, scene_config):
            audio_status = "ok"
            statuses.append({"step": "scene_audio", "ok": True, "returncode": 0, "detail": "checkpoint validé"})
        else:
            invalidate_checkpoints_from(checkpoints, STEP_ORDER, "scene_audio")
            segment_data = _load_json(paths["segments"]) or {}
            bounds = [{"index": s["segment_id"], "start": s.get("start_s"), "end": s.get("end_s")} for s in segment_data.get("segments", []) if "segment_id" in s]
            try:
                response = build_scene_audio(str(video), bounds, output_path=str(paths["audio_events"]), model=audio_model, cache_dir=str(checkpoints / "cache" / "scene_audio"))
                audio_status = response.get("status", "échec")
            except Exception as exc:  # noqa: BLE001
                logging.exception("scene_audio non fatal: %s", exc)
                audio_status = "échec"
            ok_audio = audio_status in ("ok", "partiel")
            statuses.append({"step": "scene_audio", "ok": ok_audio, "returncode": 0 if ok_audio else None, "detail": "" if ok_audio else audio_status})
            if audio_status == "ok":
                write_checkpoint(checkpoints, "scene_audio", inputs=scene_inputs, outputs={"audio_events": paths["audio_events"]}, artifact_schema_version=STEP_SCHEMA_VERSIONS["scene_audio"], configuration=scene_config, json_output_requirements={"audio_events": ["status"]})

    feature_config = {"run_id": run_id, "step": "scene_audio_features"}
    if enable_audio_features:
        if in_process_checkpoint_valid("scene_audio_features", scene_inputs, {"audio_features": paths["audio_features"]}, feature_config):
            audio_features_status = "ok"
            statuses.append({"step": "scene_audio_features", "ok": True, "returncode": 0, "detail": "checkpoint validé"})
        else:
            invalidate_checkpoints_from(checkpoints, STEP_ORDER, "scene_audio_features")
            segment_data = _load_json(paths["segments"]) or {}
            bounds = [{"index": s["segment_id"], "start": s.get("start_s"), "end": s.get("end_s")} for s in segment_data.get("segments", []) if "segment_id" in s]
            try:
                response = build_scene_audio_features(str(video), bounds, output_path=str(paths["audio_features"]))
                audio_features_status = response.get("status", "échec")
            except Exception as exc:  # noqa: BLE001
                logging.exception("scene_audio_features non fatal: %s", exc)
                audio_features_status = "échec"
            ok_features = audio_features_status in ("ok", "partiel")
            statuses.append({"step": "scene_audio_features", "ok": ok_features, "returncode": 0 if ok_features else None, "detail": "" if ok_features else audio_features_status})
            if audio_features_status == "ok":
                write_checkpoint(checkpoints, "scene_audio_features", inputs=scene_inputs, outputs={"audio_features": paths["audio_features"]}, artifact_schema_version=STEP_SCHEMA_VERSIONS["scene_audio_features"], configuration=feature_config, json_output_requirements={"audio_features": ["status"]})

    report_inputs: Dict[str, Any] = {"vlm_results": paths["results_dir"], "fusion": paths["fusion"], "vlm_pack": paths["pack"]}
    final_cmd = [py, str(SCRIPTS / "build_final_report.py"), "--results-dir", str(paths["results_dir"]), "--fusion", str(paths["fusion"]), "--pack", str(paths["pack"]), "--output", str(paths["report_json"]), "--text-output", str(paths["report_txt"])]
    if enable_audio and paths["audio_events"].exists():
        final_cmd += ["--scene-audio", str(paths["audio_events"])]
        report_inputs["audio_events"] = paths["audio_events"]
    if enable_audio_features and paths["audio_features"].exists():
        final_cmd += ["--audio-features", str(paths["audio_features"])]
        report_inputs["audio_features"] = paths["audio_features"]
    if not enable_arc:
        final_cmd.append("--no-arc")
    if not step(
        "final_report", final_cmd, inputs=report_inputs,
        outputs={"report_json": paths["report_json"], "report_txt": paths["report_txt"]},
        required_json={"report_json": ["status", "validation"]},
    ):
        return finalize(False)
    report = _load_json(paths["report_json"]) or {}
    report_status = report.get("status")
    if report_status not in ("complete", "partial", "failed"):
        return finalize(False)
    return finalize(report_status != "failed", report_status, report.get("next_step"))
