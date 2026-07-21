"""Évaluation locale, reproductible et séparée du pipeline métier."""

from __future__ import annotations

import html
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .demo import generate_demo_video
from .pipeline import OUTPUTS, PROJECT_ROOT, run_full_pipeline
from .run_identity import atomic_write_json, atomic_write_text, utc_now
from .vlm_schema import REQUIRED_FIELDS

EVALUATION_SCHEMA_VERSION = "1.0"
DATASET_SCHEMA_VERSION = "1.0"


class EvaluationError(Exception):
    """Erreur de configuration ou de lecture d'une évaluation."""


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"{label} introuvable : {path}.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{label} illisible : {path.name}.") from exc
    if not isinstance(data, dict):
        raise EvaluationError(f"{label} doit être un objet JSON.")
    return data


def load_dataset(dataset_dir: Path) -> Dict[str, Any]:
    root = Path(dataset_dir).resolve()
    dataset = _read_object(root / "dataset.json", "dataset")
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise EvaluationError(
            f"Version de dataset incompatible : {dataset.get('schema_version')!r}."
        )
    videos = dataset.get("videos")
    if not isinstance(videos, list) or not videos:
        raise EvaluationError("Le dataset ne déclare aucune vidéo.")
    dataset["_root"] = root
    return dataset


def prepare_dataset_video(dataset: Dict[str, Any], video_entry: Dict[str, Any]) -> Path:
    root = Path(dataset["_root"])
    relative = video_entry.get("path")
    if not isinstance(relative, str):
        raise EvaluationError("Chemin de vidéo absent du dataset.")
    video_path = (root / relative).resolve()
    try:
        video_path.relative_to(root)
    except ValueError as exc:
        raise EvaluationError("La vidéo du dataset sort de son dossier.") from exc
    if video_entry.get("generated_by") == "video_agent.demo":
        generate_demo_video(video_path)
    if not video_path.is_file():
        raise EvaluationError(f"Vidéo du dataset introuvable : {video_path}.")
    return video_path


def _artifact_path(run_dir: Path, manifest: Dict[str, Any], key: str, fallback: str) -> Path:
    raw = (manifest.get("artifacts") or {}).get(key)
    candidate = Path(raw) if isinstance(raw, str) else run_dir / fallback
    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        candidate = project_candidate if project_candidate.exists() else run_dir / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise EvaluationError(f"Artefact {key!r} déclaré hors du run.") from exc
    return resolved


def _segments(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    values = data.get("segments")
    if not isinstance(values, list):
        values = data.get("scenes")
    return [dict(item) for item in values or [] if isinstance(item, dict)]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _intervals(scenes: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    result = []
    for scene in scenes:
        start, end = _number(scene.get("start_s")), _number(scene.get("end_s"))
        if start is not None and end is not None and end >= start:
            result.append((start, end))
    return sorted(result)


def scene_metrics(
    expected: Sequence[Mapping[str, Any]],
    detected: Sequence[Mapping[str, Any]],
    *,
    tolerance_ms: int = 500,
    expected_duration_s: float | None = None,
) -> Dict[str, Any]:
    """Compare les limites avec appariement glouton dans une tolérance documentée."""
    if tolerance_ms < 0:
        raise EvaluationError("La tolérance temporelle doit être positive.")
    tolerance_s = tolerance_ms / 1000
    expected_intervals, detected_intervals = _intervals(expected), _intervals(detected)
    expected_boundaries = [end for _, end in expected_intervals[:-1]]
    detected_boundaries = [end for _, end in detected_intervals[:-1]]
    remaining = set(range(len(detected_boundaries)))
    matches: list[Dict[str, Any]] = []
    for expected_boundary in expected_boundaries:
        candidates = sorted(
            (
                (abs(detected_boundaries[index] - expected_boundary), index)
                for index in remaining
                if abs(detected_boundaries[index] - expected_boundary) <= tolerance_s
            ),
            key=lambda item: (item[0], item[1]),
        )
        if candidates:
            error, index = candidates[0]
            remaining.remove(index)
            matches.append({
                "expected_s": expected_boundary,
                "detected_s": detected_boundaries[index],
                "absolute_error_ms": round(error * 1000, 3),
            })
    precision = len(matches) / len(detected_boundaries) if detected_boundaries else (
        1.0 if not expected_boundaries else 0.0
    )
    recall = len(matches) / len(expected_boundaries) if expected_boundaries else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gaps, overlaps = [], []
    for previous, current in zip(detected_intervals, detected_intervals[1:]):
        delta = current[0] - previous[1]
        if delta > 1e-6:
            gaps.append({"start_s": previous[1], "end_s": current[0], "duration_s": delta})
        elif delta < -1e-6:
            overlaps.append({
                "start_s": current[0], "end_s": previous[1], "duration_s": -delta
            })
    union = 0.0
    if detected_intervals:
        current_start, current_end = detected_intervals[0]
        for start, end in detected_intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                union += current_end - current_start
                current_start, current_end = start, end
        union += current_end - current_start
    duration = expected_duration_s
    if duration is None and expected_intervals:
        duration = max(end for _, end in expected_intervals)
    coverage = min(1.0, union / duration) if duration and duration > 0 else None
    detected_duration = max((end for _, end in detected_intervals), default=0.0)
    return {
        "expected_scene_count": len(expected),
        "detected_scene_count": len(detected),
        "scene_count_delta": len(detected) - len(expected),
        "scene_tolerance_ms": tolerance_ms,
        "expected_boundary_count": len(expected_boundaries),
        "detected_boundary_count": len(detected_boundaries),
        "matched_boundary_count": len(matches),
        "boundary_precision": round(precision, 6),
        "boundary_recall": round(recall, 6),
        "boundary_f1": round(f1, 6),
        "boundary_matches": matches,
        "unmatched_expected_boundaries": len(expected_boundaries) - len(matches),
        "unmatched_detected_boundaries": len(detected_boundaries) - len(matches),
        "detected_duration_s": round(detected_duration, 6),
        "temporal_coverage_ratio": round(coverage, 6) if coverage is not None else None,
        "gap_count": len(gaps),
        "overlap_count": len(overlaps),
        "gap_duration_s": round(sum(item["duration_s"] for item in gaps), 6),
        "overlap_duration_s": round(sum(item["duration_s"] for item in overlaps), 6),
        "gaps": gaps,
        "overlaps": overlaps,
    }


def _normalized_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    output = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("text")
        if isinstance(value, str) and value.strip():
            output.add(" ".join(value.casefold().split()))
    return output


def _set_scores(expected: set[str], actual: set[str]) -> Dict[str, float | int]:
    matched = len(expected & actual)
    precision = matched / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = matched / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected": len(expected), "detected": len(actual), "matched": matched,
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _valid_field(value: Any, field: str) -> bool:
    if field == "segment_id":
        return isinstance(value, int) and not isinstance(value, bool)
    if field in {"start_s", "end_s", "confidence"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field == "uses_speech":
        return isinstance(value, bool)
    if field in {"people_objects", "actions", "mood", "uncertainties"}:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if field == "texte_ecran":
        return isinstance(value, list) and all(
            isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in ("text", "lang", "location"))
            for item in value
        )
    if field == "evidence":
        return isinstance(value, dict) and all(
            isinstance(value.get(key), list)
            and all(isinstance(item, str) for item in value[key])
            for key in ("visual", "audio")
        )
    if field == "combined_text":
        return isinstance(value, str)
    return isinstance(value, str) and bool(value.strip())


def vlm_metrics(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    *,
    tolerance_ms: int,
) -> Dict[str, Any]:
    by_id = {
        item.get("segment_id"): item
        for item in actual
        if isinstance(item.get("segment_id"), int)
    }
    expected_sets = {"objects": set(), "actions": set(), "texts": set()}
    actual_sets = {"objects": set(), "actions": set(), "texts": set()}
    valid_fields = field_total = timestamp_matches = compared = uncertainty_count = 0
    similarities, per_scene = [], []
    for expected_scene in expected:
        segment_id = expected_scene.get("segment_id")
        result = by_id.get(segment_id)
        if result is None:
            per_scene.append({"segment_id": segment_id, "result_present": False})
            continue
        expected_sets["objects"] |= _normalized_set(expected_scene.get("objects"))
        actual_sets["objects"] |= _normalized_set(result.get("people_objects"))
        expected_sets["actions"] |= _normalized_set(expected_scene.get("actions"))
        actual_sets["actions"] |= _normalized_set(result.get("actions"))
        expected_sets["texts"] |= _normalized_set(expected_scene.get("texts"))
        actual_sets["texts"] |= _normalized_set(result.get("texte_ecran"))
        for field in REQUIRED_FIELDS:
            field_total += 1
            valid_fields += int(_valid_field(result.get(field), field))
        expected_start, expected_end = (
            _number(expected_scene.get("start_s")), _number(expected_scene.get("end_s"))
        )
        actual_start, actual_end = _number(result.get("start_s")), _number(result.get("end_s"))
        timestamp_ok = False
        if None not in (expected_start, expected_end, actual_start, actual_end):
            compared += 1
            timestamp_ok = (
                abs(actual_start - expected_start) * 1000 <= tolerance_ms
                and abs(actual_end - expected_end) * 1000 <= tolerance_ms
            )
            timestamp_matches += int(timestamp_ok)
        reference = str(expected_scene.get("description") or "").strip().casefold()
        description = str(result.get("scene_description") or "").strip().casefold()
        similarity = SequenceMatcher(None, reference, description).ratio() if reference else None
        if similarity is not None:
            similarities.append(similarity)
        uncertainties = result.get("uncertainties")
        uncertainty_count += len(uncertainties) if isinstance(uncertainties, list) else 0
        per_scene.append({
            "segment_id": segment_id, "result_present": True,
            "segment_id_match": result.get("segment_id") == segment_id,
            "timestamps_within_tolerance": timestamp_ok,
            "text_similarity": round(similarity, 6) if similarity is not None else None,
        })
    return {
        "available": bool(actual),
        "result_count": len(actual),
        "object_overlap": _set_scores(expected_sets["objects"], actual_sets["objects"]),
        "action_overlap": _set_scores(expected_sets["actions"], actual_sets["actions"]),
        "screen_text_overlap": _set_scores(expected_sets["texts"], actual_sets["texts"]),
        "timestamp_consistency_ratio": round(timestamp_matches / compared, 6) if compared else None,
        "valid_field_ratio": round(valid_fields / field_total, 6) if field_total else None,
        "uncertainty_count": uncertainty_count,
        "mean_text_similarity": round(statistics.mean(similarities), 6) if similarities else None,
        "per_scene": per_scene,
        "method_note": (
            "Les chevauchements sont lexicaux après normalisation. SequenceMatcher "
            "est une similarité de caractères approximative, pas une mesure scientifique "
            "de compréhension sémantique."
        ),
    }


def _frame_count(frames_root: Path) -> int:
    if not frames_root.exists():
        return 0
    return sum(
        1 for path in frames_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _artifact_size(run_dir: Path) -> int:
    return sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())


def _load_vlm_results(results_dir: Path) -> list[Dict[str, Any]]:
    output = []
    for path in sorted(results_dir.glob("vlm_result_segment_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            output.append(value)
    return output


def evaluate_run(
    run_dir: Path,
    annotations_path: Path,
    *,
    scene_tolerance_ms: int = 500,
    include_performance: bool = False,
) -> Dict[str, Any]:
    run_root = Path(run_dir).resolve()
    manifest = _read_object(run_root / "run_manifest.json", "run_manifest")
    annotations = _read_object(Path(annotations_path), "annotations")
    if annotations.get("schema_version") != "1.0":
        raise EvaluationError("Version d'annotations incompatible.")
    detected = _segments(_read_object(
        _artifact_path(run_root, manifest, "segments", "scenes/segments.json"),
        "segments détectés",
    ))
    expected = annotations.get("scenes")
    if not isinstance(expected, list):
        raise EvaluationError("Les annotations ne contiennent pas de scènes.")
    expected_scenes = [item for item in expected if isinstance(item, dict)]
    duration_expected = _number(annotations.get("expected_duration_s"))
    status_path = _artifact_path(
        run_root, manifest, "pipeline_status", "reports/pipeline_status.json"
    )
    pipeline_status = _read_object(status_path, "pipeline_status") if status_path.exists() else {}
    steps = [item for item in pipeline_status.get("steps") or [] if isinstance(item, dict)]
    errors = [
        {"step": item.get("step"), "message": item.get("detail") or "Étape en échec."}
        for item in steps if item.get("ok") is False
    ]
    warnings = [
        {"step": item.get("step"), "message": item.get("detail")}
        for item in steps if item.get("ok") is not False and item.get("detail")
        and item.get("detail") != "checkpoint validé"
    ]
    annotation_hash, run_hash = annotations.get("video_sha256"), manifest.get("video_sha256")
    identity_match = isinstance(annotation_hash, str) and annotation_hash == run_hash
    if not identity_match:
        warnings.append({
            "step": "identity",
            "message": (
                "Le hash MP4 diffère de l'annotation. Pour une démo régénérée, "
                "cela peut refléter une version FFmpeg différente."
            ),
        })
    scene_block = scene_metrics(
        expected_scenes, detected, tolerance_ms=scene_tolerance_ms,
        expected_duration_s=duration_expected,
    )
    frames_root = _artifact_path(run_root, manifest, "frames", "frames")
    results_root = _artifact_path(run_root, manifest, "vlm_results", "vlm_results")
    actual_vlm = _load_vlm_results(results_root)
    performance = pipeline_status.get("performance") if include_performance else None
    if include_performance and not isinstance(performance, dict):
        performance = {
            "step_elapsed_s": {
                str(item.get("step")): float(item.get("elapsed_s") or 0.0)
                for item in steps
            },
            "total_elapsed_s": round(
                sum(float(item.get("elapsed_s") or 0.0) for item in steps), 6
            ),
        }
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "run_id": manifest.get("run_id"),
        "dataset": annotations.get("dataset_name"),
        "video_id": annotations.get("video_id"),
        "sources": {
            "pipeline_outputs": "calculated",
            "annotations": annotations.get("provenance") or "reference",
            "vlm_results": (
                "simulated_fixture" if manifest.get("demo_vlm_fixture_loaded") else "run_results"
            ) if actual_vlm else "absent",
        },
        "identity": {
            "run_video_sha256": run_hash,
            "annotation_video_sha256": annotation_hash,
            "exact_match": identity_match,
            "synthetic": annotations.get("synthetic") is True,
        },
        "summary": {
            "status": manifest.get("status") or pipeline_status.get("status"),
            "expected_scenes": len(expected_scenes),
            "detected_scenes": len(detected),
            "boundary_f1": scene_block["boundary_f1"],
            "temporal_coverage_ratio": scene_block["temporal_coverage_ratio"],
            "frame_count": _frame_count(frames_root),
            "artifact_size_bytes": _artifact_size(run_root),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "total_elapsed_s": (
                performance.get("total_elapsed_s") if isinstance(performance, dict) else None
            ),
        },
        "scene_metrics": scene_block,
        "vlm_metrics": vlm_metrics(
            expected_scenes, actual_vlm, tolerance_ms=scene_tolerance_ms
        ),
        "performance": performance,
        "scene_comparison": [
            {
                "segment_id": expected_scene.get("segment_id"),
                "expected_start_s": expected_scene.get("start_s"),
                "expected_end_s": expected_scene.get("end_s"),
                "detected_start_s": detected[index].get("start_s") if index < len(detected) else None,
                "detected_end_s": detected[index].get("end_s") if index < len(detected) else None,
            }
            for index, expected_scene in enumerate(expected_scenes)
        ],
        "errors": errors,
        "warnings": warnings,
        "methodology": {
            "reliable": [
                "durées et timestamps extraits",
                "comptages de scènes et frames",
                "trous, chevauchements et tailles de fichiers",
                "temps muraux enregistrés sur ce run",
                "validation structurelle et cohérence des identifiants",
            ],
            "approximate": [
                "appariement des limites dans une tolérance choisie",
                "correspondance lexicale exacte des objets, actions et textes",
                "similarité SequenceMatcher, non sémantique",
            ],
            "machine_comparison_warning": (
                "Les temps ne sont comparables qu'à configuration et machine équivalentes."
            ),
        },
    }
    return report


def render_evaluation_html(report: Dict[str, Any]) -> str:
    summary, scene = report.get("summary") or {}, report.get("scene_metrics") or {}
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('segment_id')))}</td>"
        f"<td>{html.escape(str(item.get('expected_start_s')))} → "
        f"{html.escape(str(item.get('expected_end_s')))}</td>"
        f"<td>{html.escape(str(item.get('detected_start_s')))} → "
        f"{html.escape(str(item.get('detected_end_s')))}</td></tr>"
        for item in report.get("scene_comparison") or []
    )
    warnings = "".join(
        f"<li>{html.escape(str(item.get('message') or item))}</li>"
        for item in report.get("warnings") or []
    ) or "<li>Aucun warning.</li>"
    errors = "".join(
        f"<li>{html.escape(str(item.get('message') or item))}</li>"
        for item in report.get("errors") or []
    ) or "<li>Aucune erreur.</li>"
    performance = report.get("performance") or {}
    timings = "".join(
        f"<tr><td>{html.escape(str(step))}</td><td>{float(value):.3f} s</td></tr>"
        for step, value in (performance.get("step_elapsed_s") or {}).items()
    ) or "<tr><td colspan='2'>Performance non incluse.</td></tr>"
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Évaluation · {html.escape(str(report.get("video_id") or "run"))}</title>
<style>
body{{margin:0;background:#f2f5f2;color:#17303a;font:15px system-ui,sans-serif}}
main{{max-width:1040px;margin:auto;padding:48px 24px}}h1{{font:42px Georgia,serif;margin:.2em 0}}
.tag{{color:#08745d;text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:800}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}
.card,section{{background:white;border:1px solid #dbe4df;border-radius:14px;padding:22px}}
.card strong{{display:block;font:28px Georgia,serif;margin-top:8px}}
section{{margin:16px 0}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid #e5ebe8}}
.note{{border-left:4px solid #d79a37;background:#fff8e9;padding:14px}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<span class="tag">Mesures calculées localement</span><h1>Rapport d'évaluation</h1>
<p>Run <code>{html.escape(str(report.get("run_id")))}</code></p>
<div class="grid">
<div class="card">Scènes attendues<strong>{summary.get("expected_scenes")}</strong></div>
<div class="card">Scènes détectées<strong>{summary.get("detected_scenes")}</strong></div>
<div class="card">F1 limites<strong>{float(summary.get("boundary_f1") or 0):.2f}</strong></div>
<div class="card">Couverture<strong>{float(summary.get("temporal_coverage_ratio") or 0)*100:.1f}%</strong></div>
</div>
<section><h2>Scènes attendues et détectées</h2><table><thead><tr><th>ID</th>
<th>Référence</th><th>Pipeline</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Temps par étape</h2><table>{timings}</table></section>
<section><h2>Erreurs</h2><ul>{errors}</ul><h2>Warnings</h2><ul>{warnings}</ul></section>
<section class="note"><strong>Limites méthodologiques</strong><p>
La tolérance de {scene.get("scene_tolerance_ms")} ms influence les scores de limites.
Les correspondances d'objets et actions sont lexicales. SequenceMatcher ne mesure pas
la compréhension sémantique. Les benchmarks ne sont comparables qu'à machine et
configuration équivalentes.</p></section>
</main></body></html>"""


def write_evaluation_reports(report: Dict[str, Any], output_base: Path) -> Dict[str, str]:
    base = Path(output_base)
    if base.suffix.lower() in {".json", ".html"}:
        base = base.with_suffix("")
    json_path, html_path = base.with_suffix(".json"), base.with_suffix(".html")
    atomic_write_json(json_path, report)
    atomic_write_text(html_path, render_evaluation_html(report))
    return {"json": json_path.as_posix(), "html": html_path.as_posix()}


def evaluate_dataset(
    dataset_dir: Path,
    *,
    scene_tolerance_ms: int = 500,
    include_performance: bool = False,
    force_pipeline: bool = False,
) -> tuple[Dict[str, Any], Path, Path]:
    dataset = load_dataset(dataset_dir)
    entry = dataset["videos"][0]
    if not isinstance(entry, dict):
        raise EvaluationError("Entrée vidéo invalide.")
    video_path = prepare_dataset_video(dataset, entry)
    config_path = OUTPUTS / "demo" / "evaluation_disabled.json"
    atomic_write_json(config_path, {"mode": "disabled", "provider": None, "model": ""})
    result = run_full_pipeline(
        str(video_path), config_path=str(config_path), enable_audio=False,
        enable_audio_features=False, enable_arc=False, force=force_pipeline,
    )
    run_dir = Path(result["run_dir"])
    annotations_path = (Path(dataset["_root"]) / str(entry["annotations"])).resolve()
    report = evaluate_run(
        run_dir, annotations_path, scene_tolerance_ms=scene_tolerance_ms,
        include_performance=include_performance,
    )
    return report, run_dir, annotations_path


def _available_memory_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.avail_phys)
        except (AttributeError, OSError):
            return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def _ffmpeg_version() -> str | None:
    executable = shutil.which("ffmpeg")
    if not executable:
        return None
    try:
        output = subprocess.run(
            [executable, "-version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10, check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return output[0] if output else None


def benchmark_dataset(
    dataset_dir: Path,
    *,
    repeat: int = 3,
    scene_tolerance_ms: int = 500,
) -> Dict[str, Any]:
    if repeat < 2 or repeat > 20:
        raise EvaluationError("Le benchmark exige entre 2 et 20 répétitions.")
    dataset = load_dataset(dataset_dir)
    entry = dataset["videos"][0]
    video_path = prepare_dataset_video(dataset, entry)
    config_path = OUTPUTS / "demo" / "benchmark_disabled.json"
    atomic_write_json(config_path, {"mode": "disabled", "provider": None, "model": ""})
    available_memory_at_start = _available_memory_bytes()
    durations, runs = [], []
    for index in range(repeat):
        started = time.perf_counter()
        result = run_full_pipeline(
            str(video_path), config_path=str(config_path), enable_audio=False,
            enable_audio_features=False, enable_arc=False, force=True,
        )
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        runs.append({
            "iteration": index + 1, "elapsed_s": round(elapsed, 6),
            "run_id": result.get("run_id"), "status": result.get("status"),
        })
    report, _, _ = evaluate_dataset(
        dataset_dir, scene_tolerance_ms=scene_tolerance_ms,
        include_performance=True, force_pipeline=False,
    )
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "dataset": dataset.get("name"),
        "repeat": repeat,
        "results": runs,
        "statistics": {
            "median_elapsed_s": round(statistics.median(durations), 6),
            "mean_elapsed_s": round(statistics.mean(durations), 6),
            "population_stddev_s": round(statistics.pstdev(durations), 6),
            "min_elapsed_s": round(min(durations), 6),
            "max_elapsed_s": round(max(durations), 6),
        },
        "environment": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "available_memory_bytes_at_start": available_memory_at_start,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "ffmpeg_version": _ffmpeg_version(),
            "audio_enabled": False,
            "vlm_mode": "disabled",
        },
        "quality_snapshot": report["summary"],
        "comparison_warning": (
            "Ne pas comparer directement ces temps à ceux d'une autre machine ou "
            "d'une autre configuration."
        ),
    }


def render_benchmark_markdown(report: Dict[str, Any]) -> str:
    stats, environment = report.get("statistics") or {}, report.get("environment") or {}
    quality = report.get("quality_snapshot") or {}
    return f"""<!-- Généré par video-agent benchmark ; ne pas éditer les valeurs à la main. -->
# Benchmark local

| Mesure | Valeur |
|---|---:|
| Répétitions | {report.get("repeat")} |
| Médiane | {float(stats.get("median_elapsed_s") or 0):.3f} s |
| Écart-type population | {float(stats.get("population_stddev_s") or 0):.3f} s |
| Minimum | {float(stats.get("min_elapsed_s") or 0):.3f} s |
| Maximum | {float(stats.get("max_elapsed_s") or 0):.3f} s |
| Scènes attendues / détectées | {quality.get("expected_scenes")} / {quality.get("detected_scenes")} |
| F1 des limites | {float(quality.get("boundary_f1") or 0):.3f} |

Environnement : Python {environment.get("python_version")}, {environment.get("logical_cpu_count")}
CPU logiques, audio désactivé, mode VLM `disabled`.

> {report.get("comparison_warning")}
"""
