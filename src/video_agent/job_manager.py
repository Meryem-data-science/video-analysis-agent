"""File locale en mémoire pour exécuter un nombre borné d'analyses."""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

LOGGER = logging.getLogger(__name__)
JOB_STATUSES = {"queued", "running", "completed", "failed", "cancelled"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}
DEFAULT_MAX_WORKERS = 1
MAX_WORKERS_LIMIT = 8
DEFAULT_UPLOAD_TTL_SECONDS = 24 * 60 * 60

ProgressCallback = Callable[[Dict[str, Any]], None]
JobTask = Callable[[ProgressCallback], Dict[str, Any]]
CleanupCallback = Callable[[], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_max_workers(value: Optional[str] = None) -> int:
    raw = value if value is not None else os.getenv(
        "VIDEO_AGENT_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)
    )
    try:
        workers = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "VIDEO_AGENT_MAX_WORKERS doit être un entier compris entre "
            f"1 et {MAX_WORKERS_LIMIT}."
        ) from exc
    if not 1 <= workers <= MAX_WORKERS_LIMIT:
        raise ValueError(
            "VIDEO_AGENT_MAX_WORKERS doit être un entier compris entre "
            f"1 et {MAX_WORKERS_LIMIT}."
        )
    return workers


def parse_upload_ttl(value: Optional[str] = None) -> int:
    raw = value if value is not None else os.getenv(
        "VIDEO_AGENT_UPLOAD_TTL_SECONDS", str(DEFAULT_UPLOAD_TTL_SECONDS)
    )
    try:
        ttl = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "VIDEO_AGENT_UPLOAD_TTL_SECONDS doit être un entier strictement positif."
        ) from exc
    if ttl <= 0:
        raise ValueError(
            "VIDEO_AGENT_UPLOAD_TTL_SECONDS doit être un entier strictement positif."
        )
    return ttl


class JobExecutionError(Exception):
    """Erreur utilisateur structurée remontée par une tâche."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        step: Optional[str] = None,
        details: Any = None,
        action: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.error = {
            "code": code,
            "message": message,
            "step": step,
            "details": details,
            "recommended_action": action,
        }


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"
    progress_percent: int = 0
    current_step: Optional[str] = None
    message: str = "Analyse placée dans la file locale."
    created_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    run_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    errors: list[Dict[str, Any]] = field(default_factory=list)
    warnings: list[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "run_id": self.run_id,
            "result": self.result,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class LocalJobManager:
    """Registre en mémoire et ThreadPoolExecutor borné."""

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        if not 1 <= max_workers <= MAX_WORKERS_LIMIT:
            raise ValueError(
                f"max_workers doit être compris entre 1 et {MAX_WORKERS_LIMIT}."
            )
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="video-agent-job"
        )
        self._jobs: Dict[str, JobRecord] = {}
        self._futures: Dict[str, Future[None]] = {}
        self._cleanups: Dict[str, CleanupCallback] = {}
        self._cleanup_done: set[str] = set()
        self._lock = threading.RLock()
        self._accepting = True
        self._shutdown = False

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting and not self._shutdown

    @property
    def is_shutdown(self) -> bool:
        with self._lock:
            return self._shutdown

    def submit(
        self, task: JobTask, *, cleanup: Optional[CleanupCallback] = None
    ) -> Dict[str, Any]:
        with self._lock:
            if not self.accepting:
                raise RuntimeError("La file locale n'accepte plus de nouveaux jobs.")
            job_id = secrets.token_urlsafe(24)
            while job_id in self._jobs:
                job_id = secrets.token_urlsafe(24)
            record = JobRecord(job_id=job_id)
            self._jobs[job_id] = record
            if cleanup is not None:
                self._cleanups[job_id] = cleanup
            future = self._executor.submit(self._execute, job_id, task)
            self._futures[job_id] = future
            future.add_done_callback(
                lambda completed, current_id=job_id: self._future_done(
                    current_id, completed
                )
            )
            return record.snapshot()

    def _progress_callback(self, job_id: str) -> ProgressCallback:
        def update(event: Dict[str, Any]) -> None:
            with self._lock:
                record = self._jobs[job_id]
                if record.status in TERMINAL_JOB_STATUSES:
                    return
                raw_percent = event.get("progress_percent", record.progress_percent)
                try:
                    percent = int(raw_percent)
                except (TypeError, ValueError):
                    percent = record.progress_percent
                record.progress_percent = max(
                    record.progress_percent, min(100, max(0, percent))
                )
                step = event.get("current_step") or event.get("step")
                if isinstance(step, str) and step:
                    record.current_step = step
                message = event.get("message")
                if isinstance(message, str) and message:
                    record.message = message
                run_id = event.get("run_id")
                if isinstance(run_id, str) and run_id:
                    record.run_id = run_id
                warnings = event.get("warnings")
                if isinstance(warnings, list):
                    record.warnings.extend(
                        item for item in warnings if isinstance(item, dict)
                    )

        return update

    def _execute(self, job_id: str, task: JobTask) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if record.status == "cancelled":
                return
            record.status = "running"
            record.started_at = _utc_now()
            record.message = "Analyse locale démarrée."
        try:
            result = task(self._progress_callback(job_id))
            self._cleanup(job_id)
            with self._lock:
                record = self._jobs[job_id]
                if record.status == "cancelled":
                    return
                pipeline_failed = result.get("ok") is False or result.get(
                    "pipeline_status"
                ) == "failed"
                if pipeline_failed:
                    record.status = "failed"
                    record.message = "Le pipeline d'analyse a échoué."
                    record.errors.append({
                        "code": "pipeline_failed",
                        "message": "Le pipeline d'analyse n'a pas produit de résultat exploitable.",
                        "step": record.current_step,
                        "details": None,
                        "recommended_action": (
                            "Consulter les diagnostics du run et relancer l'étape en échec."
                        ),
                    })
                else:
                    record.status = "completed"
                    record.progress_percent = 100
                    record.current_step = "completed"
                    record.message = "Analyse locale terminée."
                record.result = result
                run_id = result.get("run_id")
                if isinstance(run_id, str):
                    record.run_id = run_id
                warnings = result.get("warnings")
                if isinstance(warnings, list):
                    record.warnings.extend(
                        item for item in warnings if isinstance(item, dict)
                    )
                record.finished_at = _utc_now()
        except JobExecutionError as exc:
            self._cleanup(job_id)
            with self._lock:
                record = self._jobs[job_id]
                record.status = "failed"
                record.message = "L'analyse locale a échoué."
                record.errors.append(exc.error)
                record.finished_at = _utc_now()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Job %s en échec", job_id)
            self._cleanup(job_id)
            with self._lock:
                record = self._jobs[job_id]
                record.status = "failed"
                record.message = "L'analyse locale a échoué."
                record.errors.append({
                    "code": "job_execution_failed",
                    "message": "Une erreur interne a interrompu l'analyse.",
                    "step": record.current_step,
                    "details": {"exception_type": type(exc).__name__},
                    "recommended_action": (
                        "Consulter les logs locaux puis relancer l'analyse."
                    ),
                })
                record.finished_at = _utc_now()
        finally:
            self._cleanup(job_id)

    def _future_done(self, job_id: str, future: Future[None]) -> None:
        if future.cancelled():
            with self._lock:
                record = self._jobs[job_id]
                record.status = "cancelled"
                record.current_step = None
                record.message = "Job annulé avant son démarrage."
                record.finished_at = record.finished_at or _utc_now()
            self._cleanup(job_id)

    def _cleanup(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._cleanup_done:
                return
            cleanup = self._cleanups.get(job_id)
            self._cleanup_done.add(job_id)
        if cleanup is None:
            return
        try:
            cleanup()
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Nettoyage de l'upload du job %s impossible", job_id, exc_info=True
            )
            with self._lock:
                self._jobs[job_id].warnings.append({
                    "code": "upload_cleanup_failed",
                    "message": "Le nettoyage de l'upload temporaire a échoué.",
                })

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._jobs.get(job_id)
            return record.snapshot() if record is not None else None

    def list(self, *, status: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
        if status is not None and status not in JOB_STATUSES:
            raise ValueError(f"Statut de job inconnu : {status!r}.")
        if not 1 <= limit <= 200:
            raise ValueError("La limite doit être comprise entre 1 et 200.")
        with self._lock:
            records = [
                record for record in self._jobs.values()
                if status is None or record.status == status
            ]
            records.sort(key=lambda item: item.created_at, reverse=True)
            return [record.snapshot() for record in records[:limit]]

    def cancel(self, job_id: str) -> str:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return "not_found"
            if record.status == "queued":
                future = self._futures[job_id]
                if future.cancel():
                    record.status = "cancelled"
                    record.message = "Job annulé avant son démarrage."
                    record.finished_at = _utc_now()
                    return "cancelled"
                return "running"
            if record.status == "running":
                return "running"
            return "terminal"

    def stop_accepting(self) -> None:
        with self._lock:
            self._accepting = False

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._accepting = False
            self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=True)


def remove_upload_bundle(path: Path, upload_root: Path) -> None:
    """Supprime uniquement le dossier contrôlé contenant un upload de job."""
    path = Path(path)
    root = Path(upload_root).resolve()
    bundle = (path if path.is_dir() else path.parent).resolve()
    try:
        bundle.relative_to(root)
    except ValueError as exc:
        raise ValueError("Le bundle d'upload sort de la racine contrôlée.") from exc
    if bundle == root:
        raise ValueError("La racine globale des uploads ne peut pas être supprimée.")
    if bundle.exists():
        shutil.rmtree(bundle)


def cleanup_abandoned_uploads(
    upload_root: Path,
    *,
    older_than_seconds: int = DEFAULT_UPLOAD_TTL_SECONDS,
    now: Optional[float] = None,
) -> int:
    """Supprime les bundles d'upload anciens sous la racine contrôlée."""
    if older_than_seconds <= 0:
        raise ValueError("older_than_seconds doit être strictement positif.")
    root = Path(upload_root).resolve()
    if not root.exists():
        return 0
    current_time = time.time() if now is None else now
    removed = 0
    for child in root.iterdir():
        try:
            resolved = child.resolve()
            resolved.relative_to(root)
            age = current_time - child.stat().st_mtime
            if age < older_than_seconds:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        except OSError:
            LOGGER.warning("Upload temporaire abandonné non supprimé : %s", child)
    return removed
