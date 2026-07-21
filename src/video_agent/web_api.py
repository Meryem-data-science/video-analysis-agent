"""API FastAPI locale, adossée au pipeline et aux artefacts de runs existants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .demo import (
    DEMO_FILENAME,
    DemoError,
    bind_demo_vlm_results,
    default_demo_path,
    generate_demo_video,
)
from .evaluation import EvaluationError, evaluate_run, write_evaluation_reports
from .final_report import FinalReportError, build_final_report_from_files
from .job_manager import (
    JOB_STATUSES,
    JobExecutionError,
    LocalJobManager,
    cleanup_abandoned_uploads,
    parse_max_workers,
    parse_upload_ttl,
    remove_upload_bundle,
)
from .pipeline import OUTPUTS, PROJECT_ROOT, PipelineError, run_full_pipeline
from .run_identity import atomic_write_json, atomic_write_text, utc_now
from .vlm_provider import DEFAULT_MODE, SUPPORTED_PROVIDERS, VlmProviderError, resolve_vlm_mode
from .vlm_result_validation import (
    VlmResultValidationError,
    load_json,
    validate_results_dir,
    validate_single_result,
)

LOGGER = logging.getLogger(__name__)
RUN_ID_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_DISPLAY_RE = re.compile(r"[^A-Za-z0-9._-]+")
SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|token|secret|password|credential)", re.I)
SAFE_SUFFIXES = {
    ".json", ".md", ".txt", ".html", ".jpg", ".jpeg", ".png", ".webp", ".wav"
}
FORBIDDEN_SUFFIXES = {".env", ".pem", ".key", ".p12", ".pfx"}
LOCAL_UI_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


@dataclass(frozen=True)
class ApiSettings:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT.resolve())
    runs_root: Path = field(default_factory=lambda: (OUTPUTS / "runs").resolve())
    upload_root: Path = field(default_factory=lambda: (PROJECT_ROOT / "uploads" / "api").resolve())
    frontend_dist: Path = field(
        default_factory=lambda: (PROJECT_ROOT / "frontend" / "dist").resolve()
    )
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("VIDEO_AGENT_MAX_UPLOAD_BYTES", 2 * 1024**3))
    )
    max_workers: int = field(default_factory=parse_max_workers)
    upload_ttl_seconds: int = field(default_factory=parse_upload_ttl)
    allowed_extensions: frozenset[str] = frozenset(
        {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    )
    cors_origins: tuple[str, ...] = LOCAL_UI_ORIGINS


class ApiProblem(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        step: Optional[str] = None,
        details: Any = None,
        action: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.step = step
        self.details = details
        self.action = action


class VlmResultsImport(BaseModel):
    run_id: Optional[str] = None
    results: list[Dict[str, Any]] = Field(min_length=1)


def _resolve_vlm_request(mode: str, provider: Optional[str]) -> Dict[str, Optional[str]]:
    try:
        resolved_mode = resolve_vlm_mode(cli_mode=mode)
        resolved_provider = (provider or "").strip().lower() or None
        if resolved_provider is not None and resolved_provider not in SUPPORTED_PROVIDERS:
            raise VlmProviderError(
                f"Provider VLM inconnu : {resolved_provider!r}. "
                f"Valeurs supportées : {SUPPORTED_PROVIDERS}."
            )
        if resolved_mode == "automatic" and resolved_provider is None:
            raise VlmProviderError(
                "Le mode VLM 'automatic' exige un provider explicite."
            )
        if resolved_mode != "automatic":
            resolved_provider = None
        return {"mode": resolved_mode, "provider": resolved_provider}
    except VlmProviderError as exc:
        raise ApiProblem(
            422, "vlm_configuration_invalid", str(exc), step="configuration",
            action="Utiliser manual, disabled, ou automatic avec provider=gemini.",
        ) from exc


def _redact(value: Any, key: str = "") -> Any:
    if SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _problem_payload(problem: ApiProblem) -> Dict[str, Any]:
    return {"error": {
        "code": problem.code,
        "message": problem.message,
        "step": problem.step,
        "details": _redact(problem.details),
        "recommended_action": problem.action,
    }}


def _module_available(name: str) -> bool:
    try:
        if "." in name and importlib.util.find_spec(name.split(".", 1)[0]) is None:
            return False
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def runtime_capabilities() -> Dict[str, Any]:
    """Inspecte les extras sans les importer ni initialiser leurs SDK."""
    whisper = _module_available("faster_whisper")
    librosa = _module_available("librosa")
    return {
        "ffmpeg": {"available": shutil.which("ffmpeg") is not None},
        "ffprobe": {"available": shutil.which("ffprobe") is not None},
        "extras": {
            "audio": {
                "available": whisper and librosa,
                "faster_whisper": whisper,
                "librosa": librosa,
            },
            "gemini": {"available": _module_available("google.genai")},
            "pdf": {"available": _module_available("reportlab")},
            "web": {"available": True},
        },
    }


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ApiProblem(404, "artifact_not_found", f"{label} introuvable.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApiProblem(
            500, "artifact_unreadable", f"{label} est illisible.",
            details={"filename": path.name},
            action="Relancer l'étape qui produit cet artefact.",
        ) from exc
    if not isinstance(value, dict):
        raise ApiProblem(500, "artifact_invalid", f"{label} n'est pas un objet JSON.")
    return value


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ApiProblem(
            400, "invalid_run_id", "Le run_id doit être une empreinte SHA-256 en minuscules.",
            action="Utiliser le run_id retourné par POST /api/runs.",
        )


def _run_context(settings: ApiSettings, run_id: str) -> tuple[Path, Dict[str, Any]]:
    _validate_run_id(run_id)
    root = settings.runs_root.resolve()
    run_dir = (root / run_id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ApiProblem(403, "run_access_denied", "Accès au run refusé.") from exc
    manifest_path = run_dir / "run_manifest.json"
    if not run_dir.is_dir() or not manifest_path.is_file():
        raise ApiProblem(404, "run_not_found", "Run inconnu.", details={"run_id": run_id})
    manifest = _read_object(manifest_path, "run_manifest.json")
    if manifest.get("run_id") != run_id:
        raise ApiProblem(
            409, "run_identity_mismatch", "Le manifest ne correspond pas au run demandé.",
            step="manifest", action="Ne pas déplacer les artefacts entre runs.",
        )
    return run_dir, manifest


def _within_run(run_dir: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ApiProblem(403, "artifact_access_denied", "L'artefact sort du run concerné.") from exc
    return resolved


def _declared_path(settings: ApiSettings, run_dir: Path, raw: Any) -> Optional[Path]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        project_candidate = settings.project_root / candidate
        run_candidate = run_dir / candidate
        candidate = project_candidate if project_candidate.exists() else run_candidate
    return _within_run(run_dir, candidate)


def _safe_artifact_file(path: Path, run_dir: Path) -> bool:
    name = path.name.lower()
    if name == ".env" or name.startswith(".env.") or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if any(part.startswith(".") for part in path.relative_to(run_dir).parts):
        return False
    return path.suffix.lower() in SAFE_SUFFIXES


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json", ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8", ".html": "text/html; charset=utf-8",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
        ".wav": "audio/wav",
    }.get(path.suffix.lower(), "application/octet-stream")


def _artifact_id(category: str, relative: str = "") -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", category).strip("-") or "artifact"
    if not relative:
        return base
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", relative).strip("-")[:80]
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
    return f"{base}--{slug}--{digest}"


def _artifact_catalog(
    settings: ApiSettings, run_dir: Path, manifest: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        return catalog
    for category, raw in declared.items():
        try:
            path = _declared_path(settings, run_dir, raw)
        except ApiProblem:
            LOGGER.warning("Artefact déclaré hors du run ignoré: %s", category)
            continue
        if path is None or not path.exists():
            continue
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file_path in files:
            if not _safe_artifact_file(file_path, run_dir):
                continue
            relative = "" if path.is_file() else file_path.relative_to(path).as_posix()
            name = _artifact_id(str(category), relative)
            catalog[name] = {
                "name": name, "category": str(category), "filename": file_path.name,
                "media_type": _media_type(file_path), "size_bytes": file_path.stat().st_size,
                "url": f"/api/runs/{manifest['run_id']}/artifacts/{name}", "_path": file_path,
            }
    return catalog


def _public_catalog(catalog: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [{k: v for k, v in item.items() if k != "_path"} for item in catalog.values()]


def _logical_artifacts(
    settings: ApiSettings, run_dir: Path, manifest: Dict[str, Any]
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, raw in (manifest.get("artifacts") or {}).items():
        try:
            path = _declared_path(settings, run_dir, raw)
            if path is not None:
                result[str(key)] = f"runs/{manifest['run_id']}/{path.relative_to(run_dir).as_posix()}"
        except (ApiProblem, ValueError):
            continue
    return result


def _status_diagnostics(run_dir: Path) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    path = run_dir / "reports" / "pipeline_status.json"
    if not path.is_file():
        return [], []
    status = _read_object(path, "pipeline_status.json")
    errors, warnings = [], []
    for step in status.get("steps") or []:
        if not isinstance(step, dict):
            continue
        detail = str(step.get("detail") or "").strip()
        item = {"step": step.get("step"), "message": detail or "Étape en échec."}
        if step.get("ok") is False:
            errors.append(item)
        elif detail:
            warnings.append(item)
    return errors, warnings


def _run_response(settings: ApiSettings, run_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    errors, warnings = _status_diagnostics(run_dir)
    return _redact({
        "run_id": manifest["run_id"], "status": manifest.get("status"),
        "run_dir": f"outputs/runs/{manifest['run_id']}",
        "current_step": manifest.get("current_step"),
        "video_metadata": manifest.get("video_metadata"),
        "configuration": manifest.get("normalized_configuration"),
        "artifacts": _logical_artifacts(settings, run_dir, manifest),
        "errors": errors + list(manifest.get("errors") or []),
        "warnings": warnings + list(manifest.get("warnings") or []),
    })


def _update_manifest(
    run_dir: Path, manifest: Dict[str, Any], *, status: str,
    current_step: Optional[str], artifacts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    updated = dict(manifest)
    updated.update(status=status, current_step=current_step, updated_at=utc_now())
    if artifacts:
        updated_artifacts = dict(updated.get("artifacts") or {})
        updated_artifacts.update(artifacts)
        updated["artifacts"] = updated_artifacts
    atomic_write_json(run_dir / "run_manifest.json", updated)
    return updated


def _expected_segments(pack: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    segments = pack.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ApiProblem(409, "vlm_pack_invalid", "Le pack VLM ne contient aucun segment attendu.")
    expected: Dict[int, Dict[str, Any]] = {}
    for segment in segments:
        segment_id = segment.get("segment_id") if isinstance(segment, dict) else None
        if isinstance(segment_id, bool) or not isinstance(segment_id, int) or segment_id in expected:
            raise ApiProblem(409, "vlm_pack_invalid", "Les segment_id attendus sont invalides.")
        expected[segment_id] = segment
    return expected


def _run_artifact_path(
    settings: ApiSettings, run_dir: Path, manifest: Dict[str, Any], key: str, fallback: Path,
) -> Path:
    raw = (manifest.get("artifacts") or {}).get(key)
    path = _declared_path(settings, run_dir, raw) if raw else _within_run(run_dir, fallback)
    return path if path is not None else _within_run(run_dir, fallback)


def _validation_status(
    summary: Dict[str, Any], manifest: Dict[str, Any], results_dir: Path
) -> str:
    mode = str((manifest.get("vlm") or {}).get("mode") or DEFAULT_MODE).lower()
    if mode == "disabled":
        return "local_complete"
    if mode == "manual" and not list(results_dir.glob("vlm_result_segment_*.json")):
        return "awaiting_vlm"
    return str(summary.get("status") or "failed")


def _validation_response(summary: Dict[str, Any], status: str) -> Dict[str, Any]:
    next_step = summary.get("next_step")
    if status == "awaiting_vlm":
        next_step = {"action": "analyze_vlm_packs_manually"}
    elif status == "local_complete":
        next_step = {"action": "none", "reason": "vlm_disabled"}
    return {
        "status": status, "expected_segments": summary.get("expected_segments", []),
        "valid_segments": summary.get("valid_segments", []),
        "missing_segments": summary.get("missing_segments", []),
        "invalid_segments": summary.get("invalid_segments", []),
        "extra_segments": summary.get("extra_segments", []),
        "errors": {"by_segment": summary.get("errors_by_segment", {}),
                   "global": summary.get("global_errors", [])},
        "warnings": {"by_segment": summary.get("warnings_by_segment", {}),
                     "global": summary.get("global_warnings", [])},
        "next_step": next_step,
    }


async def _store_upload(upload: UploadFile, settings: ApiSettings) -> tuple[Path, str]:
    leaf = (upload.filename or "video").replace("\\", "/").split("/")[-1]
    suffix = Path(leaf).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise ApiProblem(
            415, "upload_extension_refused", f"Extension vidéo refusée : {suffix or '(absente)'}.",
            step="upload", details={"allowed_extensions": sorted(settings.allowed_extensions)},
            action="Envoyer un fichier vidéo dans un format explicitement autorisé.",
        )
    clean = SAFE_DISPLAY_RE.sub("_", Path(leaf).stem).strip("._-")[:80] or "video"
    upload_dir = settings.upload_root / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=False)
    destination, temporary = upload_dir / f"source{suffix}", upload_dir / ".upload.tmp"
    total = 0
    try:
        with temporary.open("xb") as stream:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise ApiProblem(
                        413, "upload_too_large", "La vidéo dépasse la taille maximale autorisée.",
                        step="upload", details={"max_bytes": settings.max_upload_bytes},
                        action="Réduire la vidéo ou configurer VIDEO_AGENT_MAX_UPLOAD_BYTES.",
                    )
                stream.write(chunk)
        if total == 0:
            raise ApiProblem(400, "upload_empty", "Le fichier vidéo est vide.", step="upload")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return destination, f"{clean}{suffix}"


def _safe_exception_message(exc: Exception, settings: ApiSettings) -> str:
    message = str(exc)
    replacements = {
        str(settings.project_root): "<project>",
        settings.project_root.as_posix(): "<project>",
        str(settings.upload_root): "<uploads>",
        settings.upload_root.as_posix(): "<uploads>",
    }
    for source, replacement in replacements.items():
        if source:
            message = message.replace(source, replacement)
    return message


def _public_job(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    public = _redact(snapshot)
    result = public.get("result")
    if isinstance(result, dict):
        for key in ("run_id", "run_dir", "pipeline_status", "artifacts", "links"):
            if key in result:
                public[key] = result[key]
    return public


def create_app(
    settings: Optional[ApiSettings] = None,
    pipeline_runner: Callable[..., Dict[str, Any]] = run_full_pipeline,
    job_manager: Optional[LocalJobManager] = None,
    *,
    serve_ui: Optional[bool] = None,
) -> FastAPI:
    cfg = settings or ApiSettings()
    jobs = job_manager or LocalJobManager(max_workers=cfg.max_workers)

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        try:
            cleanup_abandoned_uploads(
                cfg.upload_root, older_than_seconds=cfg.upload_ttl_seconds
            )
        except OSError:
            LOGGER.warning("Nettoyage initial des uploads abandonnés impossible", exc_info=True)
        try:
            yield
        finally:
            jobs.stop_accepting()
            jobs.shutdown(wait=True)

    application = FastAPI(
        title="Video Analysis Agent API", version=__version__, docs_url="/docs",
        redoc_url="/redoc", lifespan=lifespan,
    )
    application.state.job_manager = jobs
    application.state.jobs_persistent = False
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )

    @application.exception_handler(ApiProblem)
    async def api_problem_handler(_request: Request, exc: ApiProblem) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_problem_payload(exc))

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        problem = ApiProblem(
            422, "request_invalid", "La requête ne respecte pas le schéma attendu.",
            details={"errors": exc.errors()}, action="Corriger les champs signalés puis réessayer.",
        )
        return JSONResponse(status_code=422, content=_problem_payload(problem))

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Erreur API non gérée", exc_info=exc)
        problem = ApiProblem(
            500, "internal_error", "Une erreur interne est survenue.",
            action="Consulter les logs locaux du serveur et réessayer.",
        )
        return JSONResponse(status_code=500, content=_problem_payload(problem))

    @application.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok", "version": __version__, **runtime_capabilities(),
            "default_vlm_mode": DEFAULT_MODE,
            "jobs": {
                "max_workers": jobs.max_workers,
                "persistent": False,
                "accepting": jobs.accepting,
            },
        }

    @application.post("/api/runs", status_code=201)
    async def create_run(
        video: UploadFile = File(...), mode: str = Form(DEFAULT_MODE),
        provider: Optional[str] = Form(None), overwrite_analysis: bool = Form(False),
        step_timeout: float = Form(900.0), enable_audio: bool = Form(True),
        enable_audio_features: bool = Form(True), enable_arc: bool = Form(False),
    ) -> Dict[str, Any]:
        vlm = _resolve_vlm_request(mode, provider)
        uploaded_path, original_name = await _store_upload(video, cfg)
        config_path = uploaded_path.parent / "vlm_config.json"
        atomic_write_json(
            config_path, {"mode": vlm["mode"], "provider": vlm["provider"], "model": ""}
        )
        try:
            result = pipeline_runner(
                str(uploaded_path), config_path=str(config_path),
                overwrite_analysis=overwrite_analysis, step_timeout=step_timeout,
                enable_audio=enable_audio, enable_audio_features=enable_audio_features,
                enable_arc=enable_arc,
            )
        except (PipelineError, VlmProviderError) as exc:
            raise ApiProblem(
                422, "pipeline_rejected", str(exc), step="pipeline",
                action="Vérifier FFmpeg, les extras installés et les paramètres du run.",
            ) from exc
        run_id = result.get("run_id")
        if not isinstance(run_id, str):
            raise ApiProblem(500, "pipeline_contract_error", "Le pipeline n'a pas retourné de run_id.")
        run_dir, manifest = _run_context(cfg, run_id)
        response = _run_response(cfg, run_dir, manifest)
        response["original_filename"] = original_name
        return response

    @application.post("/api/jobs", status_code=202)
    async def create_job(
        video: UploadFile = File(...), mode: str = Form(DEFAULT_MODE),
        provider: Optional[str] = Form(None), overwrite_analysis: bool = Form(False),
        step_timeout: float = Form(900.0), enable_audio: bool = Form(True),
        enable_audio_features: bool = Form(True), enable_arc: bool = Form(False),
    ) -> Dict[str, Any]:
        if not jobs.accepting:
            raise ApiProblem(
                503, "job_queue_stopping", "La file locale est en cours d'arrêt.",
                step="queue", action="Redémarrer le serveur avant de soumettre un nouveau job.",
            )
        vlm = _resolve_vlm_request(mode, provider)
        uploaded_path, original_name = await _store_upload(video, cfg)
        config_path = uploaded_path.parent / "vlm_config.json"
        try:
            atomic_write_json(
                config_path, {
                    "mode": vlm["mode"], "provider": vlm["provider"], "model": "",
                }
            )

            def execute(progress_callback: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
                latest_step = {"value": "pipeline"}

                def track_progress(event: Dict[str, Any]) -> None:
                    step = event.get("current_step") or event.get("step")
                    if isinstance(step, str) and step:
                        latest_step["value"] = step
                    progress_callback(event)

                try:
                    raw_result = pipeline_runner(
                        str(uploaded_path), config_path=str(config_path),
                        overwrite_analysis=overwrite_analysis, step_timeout=step_timeout,
                        enable_audio=enable_audio,
                        enable_audio_features=enable_audio_features,
                        enable_arc=enable_arc,
                        progress_callback=track_progress,
                    )
                except (PipelineError, VlmProviderError) as exc:
                    raise JobExecutionError(
                        "pipeline_rejected", _safe_exception_message(exc, cfg),
                        step=latest_step["value"],
                        action=(
                            "Vérifier FFmpeg, les extras installés et les paramètres du job."
                        ),
                    ) from exc
                run_id = raw_result.get("run_id")
                if not isinstance(run_id, str):
                    raise JobExecutionError(
                        "pipeline_contract_error",
                        "Le pipeline n'a pas retourné de run_id.",
                        step="pipeline",
                        action="Consulter les logs locaux puis relancer l'analyse.",
                    )
                try:
                    run_dir, manifest = _run_context(cfg, run_id)
                except ApiProblem as exc:
                    raise JobExecutionError(
                        exc.code, exc.message, step=exc.step,
                        details=exc.details, action=exc.action,
                    ) from exc
                public_run = _run_response(cfg, run_dir, manifest)
                pipeline_status = str(
                    raw_result.get("status") or public_run.get("status") or "failed"
                )
                return {
                    "ok": raw_result.get("ok", pipeline_status != "failed"),
                    "run_id": run_id,
                    "run_dir": public_run["run_dir"],
                    "pipeline_status": pipeline_status,
                    "artifacts": public_run["artifacts"],
                    "links": {
                        "manifest": f"/api/runs/{run_id}",
                        "scenes": f"/api/runs/{run_id}/scenes",
                        "report": f"/api/runs/{run_id}/report",
                        "artifacts": f"/api/runs/{run_id}/artifacts",
                    },
                    "warnings": public_run.get("warnings", []),
                    "original_filename": original_name,
                }

            snapshot = jobs.submit(
                execute,
                cleanup=lambda: remove_upload_bundle(uploaded_path, cfg.upload_root),
            )
        except ApiProblem:
            remove_upload_bundle(uploaded_path, cfg.upload_root)
            raise
        except RuntimeError as exc:
            remove_upload_bundle(uploaded_path, cfg.upload_root)
            raise ApiProblem(
                503, "job_queue_stopping", str(exc), step="queue",
                action="Redémarrer le serveur avant de soumettre un nouveau job.",
            ) from exc
        except Exception:
            remove_upload_bundle(uploaded_path, cfg.upload_root)
            raise
        return {
            "job_id": snapshot["job_id"],
            "status": snapshot["status"],
            "progress_percent": snapshot["progress_percent"],
            "status_url": f"/api/jobs/{snapshot['job_id']}",
        }

    @application.post("/api/demo/jobs", status_code=202)
    async def create_demo_job(mode: str = Form(DEFAULT_MODE)) -> Dict[str, Any]:
        if mode not in {"manual", "disabled"}:
            raise ApiProblem(
                422,
                "demo_mode_invalid",
                "La démonstration accepte uniquement les modes manual ou disabled.",
                step="configuration",
                action="Choisir manual pour le parcours complet, ou disabled pour le local seul.",
            )
        demo_path = default_demo_path(cfg.project_root)
        try:
            demo_manifest = generate_demo_video(demo_path)
            upload = UploadFile(filename=DEMO_FILENAME, file=demo_path.open("rb"))
            response = await create_job(
                video=upload,
                mode=mode,
                provider=None,
                overwrite_analysis=False,
                step_timeout=900.0,
                enable_audio=False,
                enable_audio_features=False,
                enable_arc=False,
            )
        except DemoError as exc:
            raise ApiProblem(
                503,
                "demo_generation_failed",
                str(exc),
                step="demo_generation",
                action="Vérifier que FFmpeg est installé et accessible dans le PATH.",
            ) from exc
        response.update({
            "demo": True,
            "synthetic": True,
            "network_used": False,
            "video_sha256": demo_manifest["video_sha256"],
            "notice": (
                "Mode démonstration local — aucune donnée envoyée vers un service externe."
            ),
        })
        return response

    @application.get("/api/jobs")
    def list_jobs(status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        if status is not None and status not in JOB_STATUSES:
            raise ApiProblem(
                422, "job_status_invalid", f"Statut de job inconnu : {status!r}.",
                step="queue", details={"allowed_statuses": sorted(JOB_STATUSES)},
            )
        try:
            records = jobs.list(status=status, limit=limit)
        except ValueError as exc:
            raise ApiProblem(
                422, "job_list_invalid", str(exc), step="queue",
                action="Utiliser une limite comprise entre 1 et 200.",
            ) from exc
        return {
            "jobs": [_public_job(record) for record in records],
            "count": len(records),
            "persistent": False,
        }

    @application.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        snapshot = jobs.get(job_id)
        if snapshot is None:
            raise ApiProblem(
                404, "job_not_found", "Job inconnu.", details={"job_id": job_id},
                action="Utiliser le job_id retourné par POST /api/jobs.",
            )
        response = _public_job(snapshot)
        response["status_url"] = f"/api/jobs/{job_id}"
        return response

    @application.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        outcome = jobs.cancel(job_id)
        if outcome == "not_found":
            raise ApiProblem(
                404, "job_not_found", "Job inconnu.", details={"job_id": job_id},
            )
        if outcome == "running":
            raise ApiProblem(
                409, "job_cancellation_unsupported",
                "Le job a déjà démarré et ne peut pas être interrompu de force.",
                step="pipeline",
                action="Attendre sa fin ; seuls les jobs encore en attente sont annulables.",
            )
        if outcome == "terminal":
            raise ApiProblem(
                409, "job_already_finished",
                "Un job terminé, échoué ou déjà annulé ne peut pas être annulé.",
                step="queue",
            )
        snapshot = jobs.get(job_id)
        return _public_job(snapshot or {"job_id": job_id, "status": "cancelled"})

    @application.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        run_dir, manifest = _run_context(cfg, run_id)
        return _run_response(cfg, run_dir, manifest)

    @application.get("/api/runs/{run_id}/scenes")
    def get_scenes(run_id: str) -> Dict[str, Any]:
        run_dir, manifest = _run_context(cfg, run_id)
        report_path = _run_artifact_path(
            cfg, run_dir, manifest, "final_report", run_dir / "reports" / "final_report.json"
        )
        if report_path.is_file():
            report = _read_object(report_path, "rapport final")
            scenes = report.get("scenes") if isinstance(report.get("scenes"), list) else []
            return {"run_id": run_id, "source": "final_report", "scenes": scenes}
        segments_path = _run_artifact_path(
            cfg, run_dir, manifest, "segments", run_dir / "scenes" / "segments.json"
        )
        segments = _read_object(segments_path, "manifest des scènes")
        scenes = segments.get("scenes")
        if not isinstance(scenes, list):
            scenes = segments.get("segments") if isinstance(segments.get("segments"), list) else []
        return {"run_id": run_id, "source": "segmentation", "scenes": scenes}

    @application.get("/api/runs/{run_id}/report")
    def get_report(run_id: str) -> Dict[str, Any]:
        run_dir, manifest = _run_context(cfg, run_id)
        report_path = _run_artifact_path(
            cfg, run_dir, manifest, "final_report", run_dir / "reports" / "final_report.json"
        )
        if report_path.is_file():
            report = _read_object(report_path, "rapport final")
            return {"run_id": run_id, "status": report.get("status", "failed"), "report": report}
        status = str(manifest.get("status") or "failed")
        if status == "complete":
            status = "failed"
        next_step: Any = manifest.get("next_step")
        if status == "awaiting_vlm":
            artifacts = _logical_artifacts(cfg, run_dir, manifest)
            next_step = {
                "action": "analyze_vlm_packs_manually", "pack": artifacts.get("vlm_pack"),
                "prompts": artifacts.get("vlm_prompts"),
            }
        elif status == "local_complete":
            next_step = {"action": "none", "reason": "vlm_disabled"}
        return {"run_id": run_id, "status": status, "report": None, "next_step": next_step}

    @application.get("/api/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str) -> Dict[str, Any]:
        run_dir, manifest = _run_context(cfg, run_id)
        catalog = _artifact_catalog(cfg, run_dir, manifest)
        return {"run_id": run_id, "artifacts": _public_catalog(catalog)}

    @application.get("/api/runs/{run_id}/artifacts/{artifact_name}")
    def get_artifact(run_id: str, artifact_name: str):
        if (
            ".." in artifact_name or "/" in artifact_name or "\\" in artifact_name
            or Path(artifact_name).is_absolute()
        ):
            raise ApiProblem(400, "artifact_name_invalid", "Nom d'artefact invalide.")
        run_dir, manifest = _run_context(cfg, run_id)
        entry = _artifact_catalog(cfg, run_dir, manifest).get(artifact_name)
        if entry is None:
            raise ApiProblem(404, "artifact_not_declared", "Artefact inconnu ou non déclaré.")
        path = _within_run(run_dir, entry["_path"])
        if path.suffix.lower() == ".json":
            return JSONResponse(content=_redact(_read_object(path, path.name)))
        return FileResponse(path, media_type=entry["media_type"], filename=entry["filename"])

    def validate_run_data(run_id: str) -> tuple[Path, Dict[str, Any], Dict[str, Any], str]:
        run_dir, manifest = _run_context(cfg, run_id)
        results_dir = _run_artifact_path(
            cfg, run_dir, manifest, "vlm_results", run_dir / "vlm_results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        pack_path = _run_artifact_path(
            cfg, run_dir, manifest, "vlm_pack", run_dir / "vlm_pack" / "vlm_pack.json"
        )
        try:
            pack = load_json(str(pack_path), "pack VLM")
            summary = validate_results_dir(str(results_dir), pack)
        except VlmResultValidationError as exc:
            raise ApiProblem(
                409, "vlm_validation_impossible", str(exc), step="vlm_validation",
                action="Reconstruire le pack VLM ou corriger les résultats importés.",
            ) from exc
        status = _validation_status(summary, manifest, results_dir)
        summary["status"] = status
        if status == "awaiting_vlm":
            summary["next_step"] = {"action": "analyze_vlm_packs_manually"}
        elif status == "local_complete":
            summary["next_step"] = {"action": "none", "reason": "vlm_disabled"}
        summary_path = results_dir / "vlm_results_summary.json"
        atomic_write_json(summary_path, summary)
        run_status = "partial" if status == "complete" else status
        _update_manifest(
            run_dir, manifest, status=run_status,
            current_step="report_pending" if status == "complete" else "vlm_validation",
            artifacts={"validation": summary_path.as_posix()},
        )
        return run_dir, manifest, summary, status

    def demo_asset(relative: str) -> Path:
        source_candidate = (
            cfg.project_root / "evaluation" / "datasets" / "demo" / relative
        )
        package_candidate = Path(__file__).resolve().parent / "demo_assets" / Path(relative).name
        candidate = source_candidate if source_candidate.is_file() else package_candidate
        if not candidate.is_file():
            raise ApiProblem(
                503,
                "demo_asset_missing",
                "Une fixture interne de démonstration est absente.",
                step="demo_fixture",
                details={"asset": Path(relative).name},
                action="Réinstaller le package ou vérifier evaluation/datasets/demo.",
            )
        return candidate

    @application.post("/api/demo/runs/{run_id}/vlm-fixtures")
    def load_demo_fixtures(run_id: str) -> Dict[str, Any]:
        run_dir, manifest = _run_context(cfg, run_id)
        if str((manifest.get("vlm") or {}).get("mode")) != "manual":
            raise ApiProblem(
                409,
                "demo_fixture_mode_invalid",
                "Les fixtures VLM de démonstration exigent un run en mode manual.",
                step="demo_fixture",
            )
        demo_path = default_demo_path(cfg.project_root)
        try:
            demo_manifest = generate_demo_video(demo_path)
            pack_path = _run_artifact_path(
                cfg, run_dir, manifest, "vlm_pack",
                run_dir / "vlm_pack" / "vlm_pack.json",
            )
            pack = load_json(str(pack_path), "pack VLM")
            results = bind_demo_vlm_results(
                run_manifest=manifest,
                vlm_pack=pack,
                demo_manifest=demo_manifest,
                fixtures_path=demo_asset("vlm_fixtures/demo_vlm_results.json"),
            )
        except (DemoError, VlmResultValidationError) as exc:
            raise ApiProblem(
                409,
                "demo_fixture_refused",
                str(exc),
                step="demo_fixture",
                action="Utiliser uniquement le run créé par POST /api/demo/jobs.",
            ) from exc
        results_dir = _run_artifact_path(
            cfg, run_dir, manifest, "vlm_results", run_dir / "vlm_results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            segment_id = int(result["segment_id"])
            atomic_write_json(
                results_dir / f"vlm_result_segment_{segment_id:03d}.json", result
            )
        marked = dict(manifest)
        marked["demo_vlm_fixture_loaded"] = True
        marked["demo_vlm_fixture"] = {
            "simulated": True,
            "provider": None,
            "model": None,
            "video_sha256": demo_manifest["video_sha256"],
            "notice": "Fixture synthétique, non produite par un VLM.",
        }
        atomic_write_json(run_dir / "run_manifest.json", marked)
        _, _, summary, status = validate_run_data(run_id)
        response = _validation_response(summary, status)
        response.update({
            "simulated": True,
            "source": "demo_fixture",
            "notice": "Résultats simulés — aucun modèle VLM n'a été appelé.",
        })
        return response

    @application.post("/api/demo/runs/{run_id}/evaluate")
    def evaluate_demo_run(run_id: str) -> Dict[str, Any]:
        run_dir, manifest = _run_context(cfg, run_id)
        try:
            demo_manifest = generate_demo_video(default_demo_path(cfg.project_root))
            if manifest.get("video_sha256") != demo_manifest.get("video_sha256"):
                raise ApiProblem(
                    409,
                    "demo_evaluation_refused",
                    "Ce run ne correspond pas à la vidéo synthétique de démonstration.",
                    step="evaluation",
                )
            report = evaluate_run(
                run_dir,
                demo_asset("annotations/demo_annotations.json"),
                scene_tolerance_ms=500,
                include_performance=True,
            )
            paths = write_evaluation_reports(
                report, run_dir / "reports" / "evaluation_report"
            )
        except ApiProblem:
            raise
        except (DemoError, EvaluationError) as exc:
            raise ApiProblem(
                409,
                "evaluation_failed",
                str(exc),
                step="evaluation",
                action="Vérifier les artefacts du run puis relancer l'évaluation.",
            ) from exc
        updated = dict(manifest)
        updated_artifacts = dict(updated.get("artifacts") or {})
        updated_artifacts.update({
            "evaluation_json": paths["json"],
            "evaluation_html": paths["html"],
        })
        updated["artifacts"] = updated_artifacts
        atomic_write_json(run_dir / "run_manifest.json", updated)
        return {
            "run_id": run_id,
            "report": report,
            "links": {
                "json": f"/api/runs/{run_id}/evaluation",
                "html": f"/api/runs/{run_id}/evaluation/html",
            },
        }

    @application.get("/api/runs/{run_id}/evaluation")
    def get_evaluation(run_id: str) -> Dict[str, Any]:
        run_dir, _ = _run_context(cfg, run_id)
        path = run_dir / "reports" / "evaluation_report.json"
        return {
            "run_id": run_id,
            "report": _read_object(path, "rapport d'évaluation"),
            "links": {"html": f"/api/runs/{run_id}/evaluation/html"},
        }

    @application.get("/api/runs/{run_id}/evaluation/html")
    def get_evaluation_html(run_id: str):
        run_dir, _ = _run_context(cfg, run_id)
        path = _within_run(run_dir, run_dir / "reports" / "evaluation_report.html")
        if not path.is_file():
            raise ApiProblem(404, "evaluation_not_found", "Rapport d'évaluation absent.")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @application.post("/api/runs/{run_id}/vlm-results")
    def import_vlm_results(run_id: str, payload: VlmResultsImport) -> Dict[str, Any]:
        run_dir, manifest = _run_context(cfg, run_id)
        if payload.run_id is not None and payload.run_id != run_id:
            raise ApiProblem(
                409, "run_identity_mismatch", "Le run_id du contenu ne correspond pas à l'URL.",
                step="vlm_import",
            )
        pack_path = _run_artifact_path(
            cfg, run_dir, manifest, "vlm_pack", run_dir / "vlm_pack" / "vlm_pack.json"
        )
        expected = _expected_segments(_read_object(pack_path, "pack VLM"))
        validated: Dict[int, Dict[str, Any]] = {}
        rejected: Dict[str, list[str]] = {}
        for index, result in enumerate(payload.results):
            segment_id = result.get("segment_id")
            if isinstance(segment_id, bool) or not isinstance(segment_id, int):
                rejected[str(index)] = ["segment_id doit être un entier."]
            elif segment_id in validated:
                rejected[str(segment_id)] = ["segment_id dupliqué dans l'import."]
            elif segment_id not in expected:
                rejected[str(segment_id)] = ["segment_id supplémentaire non attendu."]
            else:
                check = validate_single_result(result, segment_id, expected[segment_id])
                if check["valid"]:
                    validated[segment_id] = result
                else:
                    rejected[str(segment_id)] = list(check["errors"])
        if rejected:
            raise ApiProblem(
                422, "vlm_result_invalid", "Aucun résultat invalide n'a été enregistré.",
                step="vlm_import", details={"errors": rejected},
                action="Corriger les JSON selon le schéma du pack VLM.",
            )
        results_dir = _run_artifact_path(
            cfg, run_dir, manifest, "vlm_results", run_dir / "vlm_results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for segment_id, result in validated.items():
            destination = results_dir / f"vlm_result_segment_{segment_id:03d}.json"
            atomic_write_json(destination, result)
            written.append(destination.name)
        _, _, summary, status = validate_run_data(run_id)
        return {
            "run_id": run_id, "imported": sorted(written),
            "validation": _validation_response(summary, status),
        }

    @application.post("/api/runs/{run_id}/validate")
    def validate_run(run_id: str) -> Dict[str, Any]:
        _, _, summary, status = validate_run_data(run_id)
        return {"run_id": run_id, **_validation_response(summary, status)}

    @application.post("/api/runs/{run_id}/build-report")
    def build_report(run_id: str) -> Dict[str, Any]:
        run_dir, _manifest, summary, status = validate_run_data(run_id)
        if status in {"awaiting_vlm", "local_complete", "failed"}:
            return {"run_id": run_id, **_validation_response(summary, status), "report": None}
        manifest = _read_object(run_dir / "run_manifest.json", "run_manifest.json")
        results_dir = _run_artifact_path(
            cfg, run_dir, manifest, "vlm_results", run_dir / "vlm_results"
        )
        fusion_path = _run_artifact_path(
            cfg, run_dir, manifest, "fusion", run_dir / "fusion" / "fusion.json"
        )
        pack_path = _run_artifact_path(
            cfg, run_dir, manifest, "vlm_pack", run_dir / "vlm_pack" / "vlm_pack.json"
        )
        summary_path = results_dir / "vlm_results_summary.json"
        report_path = run_dir / "reports" / "final_report.json"
        report_text_path = run_dir / "reports" / "final_report.txt"
        optional_paths = {
            "inspection_path": run_dir / "inspection" / "inspection.json",
            "scene_audio_path": run_dir / "audio" / "audio_events.json",
            "audio_features_path": run_dir / "audio" / "audio_features.json",
        }
        kwargs = {key: str(path) for key, path in optional_paths.items() if path.is_file()}
        try:
            report = build_final_report_from_files(
                results_dir=str(results_dir), fusion_manifest_path=str(fusion_path),
                summary_path=str(summary_path), include_arc=False,
                vlm_pack_path=str(pack_path), **kwargs,
            )
        except FinalReportError as exc:
            raise ApiProblem(
                409, "report_build_failed", str(exc), step="final_report",
                action="Corriger les segments signalés puis relancer la validation.",
            ) from exc
        atomic_write_json(report_path, report)
        atomic_write_text(report_text_path, str(report.get("readable_summary") or ""))
        final_status = str(report.get("status") or status)
        _update_manifest(
            run_dir, manifest, status=final_status, current_step=None,
            artifacts={
                "final_report": report_path.as_posix(),
                "final_report_text": report_text_path.as_posix(),
                "validation": summary_path.as_posix(),
            },
        )
        return {"run_id": run_id, "status": final_status, "report": report}

    frontend_index = cfg.frontend_dist / "index.html"
    ui_available = frontend_index.is_file() and serve_ui is not False
    application.state.ui_available = ui_available
    application.state.frontend_dist = cfg.frontend_dist

    if ui_available:
        frontend_root = cfg.frontend_dist.resolve()

        @application.get("/", include_in_schema=False)
        @application.get("/{frontend_path:path}", include_in_schema=False)
        def serve_frontend(frontend_path: str = ""):
            if frontend_path == "api" or frontend_path.startswith("api/"):
                raise ApiProblem(
                    404,
                    "api_route_not_found",
                    "Route API inconnue.",
                    action="Vérifier l'URL et consulter /docs.",
                )
            requested = (frontend_root / frontend_path).resolve()
            try:
                requested.relative_to(frontend_root)
            except ValueError as exc:
                raise ApiProblem(
                    400, "frontend_path_invalid", "Chemin frontend invalide."
                ) from exc
            if requested.is_file():
                return FileResponse(requested)
            return FileResponse(frontend_index)

    return application


app = create_app()
