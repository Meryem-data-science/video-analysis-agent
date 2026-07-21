"""Validation stricte et déterministe des résultats VLM par segment.

Le validateur recalcule toujours son état depuis le manifest attendu et les fichiers
actuellement présents. Il ne normalise pas les réponses et n'appelle aucun modèle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALIDATION_SCHEMA_VERSION = "2.0"
TIMESTAMP_TOLERANCE_S = 0.001
RESULT_FILE_RE = re.compile(r"^vlm_result_segment_(\d+)\.json$")

REQUIRED_FIELDS = {
    "segment_id", "start_s", "end_s", "scene_description", "people_objects",
    "actions", "motion_description", "audio_visual_relation", "uses_speech",
    "mood", "combined_text", "confidence", "uncertainties", "texte_ecran",
    "evidence",
}


class VlmResultValidationError(Exception):
    """Erreur bloquante empêchant d'établir le contrat de validation."""


_ASSERTIVE_PATTERNS = re.compile(
    r"\b(clairement|sans aucun doute|certainement|évidemment|forcément|"
    r"il est certain|c'est sûr)\b", re.IGNORECASE,
)
_STRONG_RELATION_PATTERNS = re.compile(
    r"\b(correspond exactement|prouve|démontre|confirme clairement|"
    r"est directement lié)\b", re.IGNORECASE,
)


def load_json(path: str, label: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise VlmResultValidationError(f"Fichier {label} introuvable : {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VlmResultValidationError(f"JSON {label} illisible ({p}) : {exc}") from exc
    if not isinstance(data, dict):
        raise VlmResultValidationError(f"JSON {label} doit être un objet ({p}).")
    return data


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_string_list(result: Dict[str, Any], key: str, errors: List[str]) -> None:
    value = result.get(key)
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{key}[{index}] doit être une chaîne.")
        elif not item.strip():
            errors.append(f"{key}[{index}] ne doit pas être vide.")


def validate_result_schema(result: Dict[str, Any]) -> List[str]:
    """Valide le contrat complet sans modifier la réponse."""
    if not isinstance(result, dict):
        return ["Le résultat VLM doit être un objet JSON."]
    errors: List[str] = []

    for key in sorted(REQUIRED_FIELDS - set(result)):
        errors.append(f"Champ requis absent : {key}")
    for key in sorted(set(result) - REQUIRED_FIELDS):
        errors.append(f"Propriété non autorisée : {key}")

    if "segment_id" in result and (
        isinstance(result["segment_id"], bool) or not isinstance(result["segment_id"], int)
    ):
        errors.append("Champ segment_id doit être un entier.")
    for key in ("start_s", "end_s"):
        if key in result and not _is_number(result[key]):
            errors.append(f"Champ {key} doit être un nombre.")

    for key in ("scene_description", "motion_description", "audio_visual_relation"):
        if key in result:
            if not isinstance(result[key], str):
                errors.append(f"Champ {key} doit être une chaîne.")
            elif not result[key].strip():
                errors.append(f"Champ texte obligatoire vide : {key}")

    if "combined_text" in result and not isinstance(result["combined_text"], str):
        errors.append("Champ combined_text doit être une chaîne.")

    for key in ("people_objects", "actions", "mood", "uncertainties"):
        if key in result and not isinstance(result[key], list):
            errors.append(f"Champ {key} doit être une liste.")
        _validate_string_list(result, key, errors)

    if "uses_speech" in result and not isinstance(result["uses_speech"], bool):
        errors.append("Champ uses_speech doit être un booléen.")
    if "confidence" in result and not _is_number(result["confidence"]):
        errors.append("Champ confidence doit être un nombre.")

    screen = result.get("texte_ecran")
    if "texte_ecran" in result and not isinstance(screen, list):
        errors.append("Champ texte_ecran doit être une liste.")
    elif isinstance(screen, list):
        for index, item in enumerate(screen):
            if not isinstance(item, dict):
                errors.append(f"texte_ecran[{index}] doit être un objet.")
                continue
            required = {"text", "lang", "location"}
            for key in sorted(required - set(item)):
                errors.append(f"texte_ecran[{index}].{key} est requis.")
            for key in sorted(set(item) - required):
                errors.append(f"texte_ecran[{index}].{key} n'est pas autorisé.")
            for key in required & set(item):
                if not isinstance(item[key], str):
                    errors.append(f"texte_ecran[{index}].{key} doit être une chaîne.")
                elif not item[key].strip():
                    errors.append(f"texte_ecran[{index}].{key} ne doit pas être vide.")

    evidence = result.get("evidence")
    if "evidence" in result and not isinstance(evidence, dict):
        errors.append("Champ evidence doit être un objet.")
    elif isinstance(evidence, dict):
        required = {"visual", "audio"}
        for key in sorted(required - set(evidence)):
            errors.append(f"evidence.{key} est requis.")
        for key in sorted(set(evidence) - required):
            errors.append(f"evidence.{key} n'est pas autorisé.")
        for key in required & set(evidence):
            if not isinstance(evidence[key], list):
                errors.append(f"evidence.{key} doit être une liste.")
            else:
                for index, item in enumerate(evidence[key]):
                    if not isinstance(item, str):
                        errors.append(f"evidence.{key}[{index}] doit être une chaîne.")
                    elif not item.strip():
                        errors.append(f"evidence.{key}[{index}] ne doit pas être vide.")
    return errors


def check_segment_id(result: Dict[str, Any], expected_segment_id: int) -> List[str]:
    if "segment_id" not in result:
        return ["Champ requis absent : segment_id"]
    actual = result["segment_id"]
    if isinstance(actual, bool) or not isinstance(actual, int):
        return [f"segment_id non entier : {actual!r}"]
    if actual != expected_segment_id:
        return [f"segment_id non concordant : attendu {expected_segment_id}, obtenu {actual}."]
    return []


def check_timestamps(result: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in ("start_s", "end_s"):
        actual = result.get(key)
        wanted = expected.get(key)
        if not _is_number(actual) or not _is_number(wanted):
            continue
        if abs(float(actual) - float(wanted)) > TIMESTAMP_TOLERANCE_S:
            errors.append(
                f"{key} non concordant : attendu {wanted}, obtenu {actual} "
                f"(tolérance {TIMESTAMP_TOLERANCE_S}s)."
            )
    return errors


def check_combined_text(result: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    actual = result.get("combined_text")
    wanted = expected.get("combined_text", "")
    if not isinstance(actual, str) or not isinstance(wanted, str):
        return []
    if wanted.strip() and not actual.strip():
        return ["Champ texte obligatoire vide : combined_text"]
    if actual != wanted:
        return ["combined_text ne correspond pas exactement au texte du pack attendu."]
    return []


def check_confidence_range(result: Dict[str, Any]) -> List[str]:
    conf = result.get("confidence")
    if not _is_number(conf):
        return []
    return [] if 0.0 <= float(conf) <= 1.0 else [f"confidence hors plage [0,1] : {conf}"]


def check_evidence(result: Dict[str, Any], segment_pack: Optional[Dict[str, Any]] = None) -> List[str]:
    warnings: List[str] = []
    evidence = result.get("evidence") or {}
    visual = evidence.get("visual") if isinstance(evidence, dict) else None
    audio = evidence.get("audio") if isinstance(evidence, dict) else None
    if isinstance(visual, list) and not visual:
        warnings.append("evidence.visual est vide alors que le segment fournit des images.")
    if result.get("uses_speech") is True and isinstance(audio, list) and not audio:
        warnings.append("uses_speech=true mais evidence.audio est vide.")
    return warnings


def check_anti_hallucination_flags(
    result: Dict[str, Any], segment_pack: Optional[Dict[str, Any]] = None
) -> List[str]:
    warnings: List[str] = []
    confidence = result.get("confidence")
    uncertainties = result.get("uncertainties")
    evidence = result.get("evidence") or {}
    audio = evidence.get("audio") if isinstance(evidence, dict) else []
    scene = result.get("scene_description") or ""
    relation = result.get("audio_visual_relation") or ""
    if _is_number(confidence) and confidence > 0.85 and isinstance(uncertainties, list) and not uncertainties:
        warnings.append("confidence > 0.85 mais uncertainties est vide.")
    if isinstance(scene, str) and _ASSERTIVE_PATTERNS.search(scene) and not (
        isinstance(evidence, dict) and (evidence.get("visual") or evidence.get("audio"))
    ):
        warnings.append("scene_description très affirmative sans evidence.")
    if isinstance(relation, str) and _STRONG_RELATION_PATTERNS.search(relation) and not audio:
        warnings.append("audio_visual_relation affirme une relation forte mais evidence.audio est vide.")
    if segment_pack is not None:
        combined = segment_pack.get("combined_text")
        combined = combined.strip() if isinstance(combined, str) else ""
        if not combined and result.get("uses_speech") is True:
            warnings.append("Le pack indique combined_text vide mais uses_speech=true.")
        if combined and result.get("uses_speech") is False:
            warnings.append("Le pack indique combined_text non vide mais uses_speech=false.")
    return warnings


def validate_single_result(
    result: Dict[str, Any],
    expected_segment_id: int,
    segment_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expected = segment_pack or {"segment_id": expected_segment_id}
    errors = validate_result_schema(result)
    errors += [e for e in check_segment_id(result, expected_segment_id) if e not in errors]
    if segment_pack is not None:
        errors += check_timestamps(result, expected)
        errors += check_combined_text(result, expected)
    errors += check_confidence_range(result)
    warnings: List[str] = []
    if not errors:
        warnings += check_evidence(result, segment_pack)
        warnings += check_anti_hallucination_flags(result, segment_pack)
    conf = result.get("confidence")
    return {
        "valid": not errors,
        "segment_id": result.get("segment_id", expected_segment_id),
        "errors": errors,
        "warnings": warnings,
        "confidence": float(conf) if _is_number(conf) else 0.0,
    }


def _expected_segments(manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    segments = manifest.get("segments") if isinstance(manifest, dict) else None
    if not isinstance(segments, list) or not segments:
        raise VlmResultValidationError("Manifest attendu sans segments exploitables.")
    expected: Dict[int, Dict[str, Any]] = {}
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise VlmResultValidationError(f"Segment attendu #{index} non objet.")
        seg_id = segment.get("segment_id")
        if isinstance(seg_id, bool) or not isinstance(seg_id, int):
            raise VlmResultValidationError(f"segment_id attendu invalide à l'index {index}: {seg_id!r}")
        if seg_id in expected:
            raise VlmResultValidationError(f"segment_id attendu dupliqué dans le manifest : {seg_id}")
        if not _is_number(segment.get("start_s")) or not _is_number(segment.get("end_s")):
            raise VlmResultValidationError(f"Timestamps attendus absents/invalides pour le segment {seg_id}.")
        if float(segment["end_s"]) < float(segment["start_s"]):
            raise VlmResultValidationError(f"Timestamps attendus inversés pour le segment {seg_id}.")
        combined = segment.get("combined_text", "")
        if not isinstance(combined, str):
            raise VlmResultValidationError(f"combined_text attendu invalide pour le segment {seg_id}.")
        expected[seg_id] = segment
    return expected


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validation_fingerprint(expected: Dict[int, Dict[str, Any]], files: List[Path]) -> Tuple[str, Dict[str, str]]:
    expected_signature = [
        {"segment_id": sid, "start_s": seg["start_s"], "end_s": seg["end_s"],
         "combined_text": seg.get("combined_text", "")}
        for sid, seg in sorted(expected.items())
    ]
    hashes: Dict[str, str] = {}
    for path in files:
        try:
            hashes[path.name] = _sha256(path)
        except OSError:
            hashes[path.name] = "<illisible>"
    payload = {"schema": VALIDATION_SCHEMA_VERSION, "expected": expected_signature, "files": hashes}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, hashes


def validate_results_dir_with_data(
    results_dir: str, expected_manifest: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Recalcule la validation et retourne (résumé, résultats strictement valides)."""
    expected = _expected_segments(expected_manifest)
    expected_ids = set(expected)
    base = Path(results_dir)
    if not base.exists() or not base.is_dir():
        raise VlmResultValidationError(f"Dossier de résultats introuvable : {base}")
    files = sorted(base.glob("vlm_result_segment_*.json"))
    fingerprint, file_hashes = _validation_fingerprint(expected, files)

    by_filename_id: Dict[int, List[Path]] = {}
    global_errors: List[str] = []
    global_warnings: List[str] = []
    for path in files:
        match = RESULT_FILE_RE.match(path.name)
        if not match:
            global_errors.append(f"Nom de résultat incompatible : {path.name}")
            continue
        by_filename_id.setdefault(int(match.group(1)), []).append(path)

    extra_ids = sorted(set(by_filename_id) - expected_ids)
    if extra_ids:
        global_warnings.append(f"Résultats de segments supplémentaires : {extra_ids}")

    duplicate_ids = {sid for sid, paths in by_filename_id.items() if len(paths) > 1}
    parsed: Dict[int, Tuple[Path, Dict[str, Any]]] = {}
    unreadable: Dict[int, str] = {}
    for sid, paths in by_filename_id.items():
        if len(paths) != 1:
            continue
        path = paths[0]
        try:
            parsed[sid] = (path, load_json(str(path), f"résultat {path.name}"))
        except VlmResultValidationError as exc:
            unreadable[sid] = str(exc)
            if sid not in expected_ids:
                global_errors.append(str(exc))

    payload_ids: Dict[int, List[int]] = {}
    for filename_id, (_, data) in parsed.items():
        payload_id = data.get("segment_id")
        if isinstance(payload_id, int) and not isinstance(payload_id, bool):
            payload_ids.setdefault(payload_id, []).append(filename_id)
    for payload_id, filename_ids in payload_ids.items():
        if len(filename_ids) > 1:
            duplicate_ids.update(filename_ids)
            global_errors.append(
                f"segment_id dupliqué dans les contenus : {payload_id} "
                f"(fichiers pour segments {sorted(filename_ids)})."
            )

    valid_results: Dict[int, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    missing: List[int] = []
    invalid: List[int] = []
    errors_by_segment: Dict[str, List[str]] = {}
    warnings_by_segment: Dict[str, List[str]] = {}
    confidences: List[float] = []

    for sid in sorted(expected):
        errors: List[str] = []
        warnings: List[str] = []
        if sid not in by_filename_id:
            missing.append(sid)
            errors.append(f"Résultat manquant : vlm_result_segment_{sid:03d}.json")
        elif sid in duplicate_ids or len(by_filename_id[sid]) > 1:
            invalid.append(sid)
            names = [p.name for p in by_filename_id[sid]]
            errors.append(f"Résultats dupliqués pour le segment {sid} : {names}")
        elif sid in unreadable:
            invalid.append(sid)
            errors.append(unreadable[sid])
        else:
            path, data = parsed[sid]
            report = validate_single_result(data, sid, expected[sid])
            errors.extend(report["errors"])
            warnings.extend(report["warnings"])
            if errors:
                invalid.append(sid)
            else:
                valid_results[sid] = data
                confidences.append(report["confidence"])
        if errors:
            errors_by_segment[str(sid)] = errors
        if warnings:
            warnings_by_segment[str(sid)] = warnings
        results.append({
            "segment_id": sid, "valid": not errors, "missing": sid in missing,
            "errors": errors, "warnings": warnings,
            "result_path": str(by_filename_id[sid][0]) if sid in by_filename_id and len(by_filename_id[sid]) == 1 else None,
        })

    valid_ids = sorted(valid_results)
    correction_ids = sorted(set(missing) | set(invalid) | (duplicate_ids & expected_ids))
    if len(valid_ids) == len(expected) and not correction_ids:
        status = "complete"
        next_step: Any = None
    elif valid_ids:
        status = "partial"
        next_step = {"action": "correct_vlm_results", "segment_ids": correction_ids}
    else:
        status = "failed"
        next_step = {"action": "rerun_vlm_analysis", "segment_ids": sorted(expected)}

    average = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    summary = {
        "validation_type": "vlm_results_validation",
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validation_fingerprint": fingerprint,
        "results_dir": str(results_dir),
        "status": status,
        "next_step": next_step,
        "expected_segment_count": len(expected),
        "valid_segment_count": len(valid_ids),
        "invalid_segment_count": len(set(invalid)),
        "missing_segment_count": len(missing),
        "expected_segments": sorted(expected),
        "valid_segments": valid_ids,
        "invalid_segments": sorted(set(invalid)),
        "missing_segments": missing,
        "extra_segments": extra_ids,
        "duplicate_segments": sorted(duplicate_ids),
        "errors_by_segment": errors_by_segment,
        "warnings_by_segment": warnings_by_segment,
        "global_errors": global_errors,
        "global_warnings": global_warnings,
        "result_file_hashes": file_hashes,
        "results": results,
        # Alias de transition pour les consommateurs existants.
        "segment_count": len(expected),
        "valid_count": len(valid_ids),
        "invalid_count": len(set(invalid)) + len(missing),
        "summary": {
            "average_confidence": average,
            "segments_missing_results": missing,
            "segments_with_errors": correction_ids,
            "segments_with_warnings": sorted(int(k) for k in warnings_by_segment),
        },
    }
    return summary, valid_results


def validate_results_dir(results_dir: str, vlm_pack: Dict[str, Any]) -> Dict[str, Any]:
    summary, _ = validate_results_dir_with_data(results_dir, vlm_pack)
    return summary


def compare_validation_summary(provided: Any, current: Dict[str, Any]) -> Dict[str, Any]:
    """Indique si un résumé enregistré décrit exactement les fichiers actuels."""
    warnings: List[str] = []
    if provided is None:
        return {"provided": False, "compatible": True, "warnings": []}
    if not isinstance(provided, dict):
        warnings.append("Résumé de validation fourni absent ou non objet.")
    elif provided.get("schema_version") != VALIDATION_SCHEMA_VERSION:
        warnings.append("Résumé de validation ancien ou version incompatible.")
    elif provided.get("validation_fingerprint") != current.get("validation_fingerprint"):
        warnings.append("Résumé de validation obsolète : manifest ou fichiers de résultats modifiés.")
    return {"provided": True, "compatible": not warnings, "warnings": warnings}
