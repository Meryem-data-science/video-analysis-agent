"""Identité déterministe, manifest et checkpoints vérifiables d'un run vidéo."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


RUN_MANIFEST_SCHEMA_VERSION = "1.0"
CHECKPOINT_SCHEMA_VERSION = "1.0"
PIPELINE_IDENTITY_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path) -> Dict[str, Any]:
    """Empreinte un fichier ou un arbre de fichiers, sans dépendre de son chemin absolu."""
    path = Path(path)
    if not path.exists():
        return {"type": "missing", "exists": False}
    if path.is_file():
        stat = path.stat()
        return {
            "type": "file", "exists": True, "size_bytes": stat.st_size,
            "sha256": sha256_file(path),
        }
    if not path.is_dir():
        return {"type": "unsupported", "exists": True}
    entries = []
    for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        rel = child.relative_to(path).as_posix()
        entries.append({"path": rel, "size_bytes": child.stat().st_size, "sha256": sha256_file(child)})
    return {
        "type": "directory", "exists": True, "entry_count": len(entries),
        "sha256": canonical_sha256(entries), "entries": entries,
    }


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


def prompt_versions(project_root: Path) -> Dict[str, str]:
    """Versionne le contenu des sources qui définissent réellement les prompts."""
    rel_paths = {
        "vlm_directives": "src/video_agent/vlm_schema.py",
        "vlm_pack_prompt": "src/video_agent/vlm_pack_builder.py",
        "manual_prompt_renderer": "scripts/prepare_vlm_run.py",
        "scene_audio_prompt": "src/video_agent/scene_audio.py",
        "narrative_arc_prompt": "src/video_agent/narrative_arc.py",
    }
    package_root = Path(__file__).resolve().parent
    fallbacks = {
        "vlm_directives": package_root / "vlm_schema.py",
        "vlm_pack_prompt": package_root / "vlm_pack_builder.py",
        "scene_audio_prompt": package_root / "scene_audio.py",
        "narrative_arc_prompt": package_root / "narrative_arc.py",
    }
    manual_spec = importlib.util.find_spec("scripts.prepare_vlm_run")
    if manual_spec and manual_spec.origin:
        fallbacks["manual_prompt_renderer"] = Path(manual_spec.origin)

    versions: Dict[str, str] = {}
    for name, rel in rel_paths.items():
        path = Path(project_root) / rel
        if not path.exists():
            path = fallbacks.get(name, path)
        versions[name] = sha256_file(path) if path.exists() else "missing"
    return versions


def schema_versions() -> Dict[str, str]:
    from .final_report import SCHEMA_VERSION
    from .vlm_result_validation import VALIDATION_SCHEMA_VERSION
    from .vlm_schema import build_output_schema

    return {
        "run_manifest": RUN_MANIFEST_SCHEMA_VERSION,
        "checkpoint": CHECKPOINT_SCHEMA_VERSION,
        "pipeline_identity": PIPELINE_IDENTITY_VERSION,
        "final_report": SCHEMA_VERSION,
        "vlm_validation": VALIDATION_SCHEMA_VERSION,
        "vlm_output_schema_sha256": canonical_sha256(
            build_output_schema(0, 0.0, 1.0, "")
        ),
    }


def build_run_identity(
    video_path: Path,
    normalized_configuration: Mapping[str, Any],
    *,
    prompts: Mapping[str, str],
    schemas: Mapping[str, str],
    pipeline_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Calcule l'identité uniquement depuis le contenu et la configuration sémantique."""
    video_path = Path(video_path)
    video_hash = sha256_file(video_path)
    payload = {
        "identity_version": PIPELINE_IDENTITY_VERSION,
        "video_sha256": video_hash,
        "configuration": _jsonable(dict(normalized_configuration)),
        "schema_versions": _jsonable(dict(schemas)),
        "prompt_versions": _jsonable(dict(prompts)),
        "pipeline_parameters": _jsonable(dict(pipeline_parameters)),
    }
    return {
        "run_id": canonical_sha256(payload),
        "video_sha256": video_hash,
        "video_size_bytes": video_path.stat().st_size,
        "identity_payload": payload,
    }


def create_or_update_run_manifest(
    manifest_path: Path,
    *,
    identity: Mapping[str, Any],
    original_filename: str,
    normalized_configuration: Mapping[str, Any],
    schemas: Mapping[str, str],
    prompts: Mapping[str, str],
    pipeline_parameters: Mapping[str, Any],
    status: str,
    video_metadata: Optional[Mapping[str, Any]] = None,
    current_step: Optional[str] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    legacy_artifacts: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    existing: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = {}
    now = utc_now()
    source_filenames = set(existing.get("source_filenames") or [])
    source_filenames.add(original_filename)
    manifest = {
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": identity["run_id"],
        "original_filename": existing.get("original_filename") or original_filename,
        "source_filenames": sorted(source_filenames),
        "video_sha256": identity["video_sha256"],
        "video_size_bytes": identity["video_size_bytes"],
        "video_metadata": _jsonable(video_metadata) if video_metadata is not None else existing.get("video_metadata"),
        "normalized_configuration": _jsonable(dict(normalized_configuration)),
        "schema_versions": _jsonable(dict(schemas)),
        "prompt_versions": _jsonable(dict(prompts)),
        "pipeline_parameters": _jsonable(dict(pipeline_parameters)),
        "vlm": {
            "mode": normalized_configuration.get("vlm", {}).get("mode"),
            "provider": normalized_configuration.get("vlm", {}).get("provider"),
        },
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "status": status,
        "current_step": current_step,
        "artifacts": _jsonable(dict(artifacts or existing.get("artifacts") or {})),
        "legacy_artifacts_detected": sorted(set(legacy_artifacts or existing.get("legacy_artifacts_detected") or [])),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _fingerprint_mapping(values: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for label, value in sorted(values.items()):
        if isinstance(value, (str, Path)):
            result[label] = fingerprint_path(Path(value))
        elif isinstance(value, dict) and "sha256" in value:
            result[label] = _jsonable(value)
        else:
            result[label] = {"type": "value", "sha256": canonical_sha256(value), "value": _jsonable(value)}
    return result


def _json_schema_ok(path: Path, required_keys: Iterable[str]) -> bool:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    return all(key in data for key in required_keys)


def write_checkpoint(
    checkpoint_dir: Path,
    step: str,
    *,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    artifact_schema_version: str,
    configuration: Mapping[str, Any],
    json_output_requirements: Optional[Mapping[str, Iterable[str]]] = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "input_fingerprints": _fingerprint_mapping(inputs),
        "output_fingerprints": _fingerprint_mapping(outputs),
        "artifact_schema_version": artifact_schema_version,
        "configuration": _jsonable(dict(configuration)),
        "json_output_requirements": {
            key: list(value) for key, value in (json_output_requirements or {}).items()
        },
        "executed_at": utc_now(),
    }
    path = checkpoint_dir / f"{step}.json"
    atomic_write_json(path, checkpoint)
    return path


def validate_checkpoint(
    checkpoint_dir: Path,
    step: str,
    *,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    artifact_schema_version: str,
    configuration: Mapping[str, Any],
) -> Tuple[bool, str]:
    path = Path(checkpoint_dir) / f"{step}.json"
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"checkpoint absent ou illisible: {exc}"
    if not isinstance(checkpoint, dict):
        return False, "checkpoint non objet"
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False, "version de checkpoint incompatible"
    if checkpoint.get("step") != step:
        return False, "étape de checkpoint incompatible"
    if checkpoint.get("artifact_schema_version") != artifact_schema_version:
        return False, "version de schéma modifiée"
    if checkpoint.get("configuration") != _jsonable(dict(configuration)):
        return False, "configuration modifiée"
    current_inputs = _fingerprint_mapping(inputs)
    if checkpoint.get("input_fingerprints") != current_inputs:
        return False, "empreintes d'entrée modifiées"
    current_outputs = _fingerprint_mapping(outputs)
    if any(value.get("exists") is False for value in current_outputs.values() if isinstance(value, dict)):
        return False, "sortie manquante"
    if checkpoint.get("output_fingerprints") != current_outputs:
        return False, "empreintes de sortie modifiées"
    requirements = checkpoint.get("json_output_requirements") or {}
    for label, keys in requirements.items():
        output = outputs.get(label)
        if output is None or not _json_schema_ok(Path(output), keys):
            return False, f"schéma JSON invalide pour {label}"
    return True, "checkpoint valide"


def invalidate_checkpoints_from(checkpoint_dir: Path, step_order: Iterable[str], step: str) -> None:
    order = list(step_order)
    if step not in order:
        raise ValueError(f"Étape inconnue : {step}")
    checkpoint_dir = Path(checkpoint_dir)
    for name in order[order.index(step):]:
        (checkpoint_dir / f"{name}.json").unlink(missing_ok=True)
