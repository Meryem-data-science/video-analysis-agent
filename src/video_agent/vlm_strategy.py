"""Niveau 9 — Stratégie d'utilisation du VLM.

Lit une configuration VLM, la valide, vérifie que le pack VLM existe et prépare
un **plan d'exécution** — SANS jamais appeler de modèle ni d'API.

Les modes sont ``disabled``, ``manual`` (défaut, sans réseau) et ``automatic``
(provider obligatoire). Ce module ne lance jamais de modèle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vlm_provider import VlmProviderError, resolve_vlm_settings


class VlmStrategyError(Exception):
    """Erreur générique de la stratégie VLM."""


def load_vlm_config(config_path: str) -> Dict[str, Any]:
    """Charge un fichier de config VLM (JSON) avec des erreurs claires."""
    p = Path(config_path)
    if not p.exists() or not p.is_file():
        raise VlmStrategyError(f"Fichier de config VLM introuvable : {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VlmStrategyError(f"JSON de config VLM invalide ({p}) : {exc}") from exc


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Valide la config VLM et retourne une version normalisée.

    ``automatic`` exige un provider explicite. Les valeurs inconnues échouent
    clairement et ne sont jamais converties en Gemini.
    """
    if not isinstance(config, dict):
        raise VlmStrategyError("La config VLM doit être un objet JSON.")

    try:
        settings = resolve_vlm_settings(config)
    except VlmProviderError as exc:
        raise VlmStrategyError(str(exc)) from exc
    mode = settings["mode"]
    provider = settings["provider"]

    # model : chaîne (éventuellement vide) ; jamais imposée par le code.
    model = config.get("model", "")
    if model is not None and not isinstance(model, str):
        raise VlmStrategyError("Le champ 'model' doit être une chaîne (éventuellement vide).")

    input_policy = config.get("input_policy", {}) or {}
    output_policy = config.get("output_policy", {}) or {}
    if not isinstance(input_policy, dict):
        raise VlmStrategyError("'input_policy' doit être un objet.")
    if not isinstance(output_policy, dict):
        raise VlmStrategyError("'output_policy' doit être un objet.")

    max_frames = input_policy.get("max_frames_per_segment", 6)
    try:
        max_frames = int(max_frames)
    except (TypeError, ValueError) as exc:
        raise VlmStrategyError("'max_frames_per_segment' doit être un entier.") from exc
    if max_frames <= 0:
        raise VlmStrategyError("'max_frames_per_segment' doit être positif.")

    return {
        "mode": mode,
        "provider": provider,
        "model": model or "",
        "input_policy": {
            "use_motion_grid": bool(input_policy.get("use_motion_grid", True)),
            "use_motion_composite": bool(input_policy.get("use_motion_composite", True)),
            "use_selected_frames": bool(input_policy.get("use_selected_frames", True)),
            "use_transcript": bool(input_policy.get("use_transcript", True)),
            "max_frames_per_segment": max_frames,
        },
        "output_policy": {
            "require_json": bool(output_policy.get("require_json", True)),
            "include_confidence": bool(output_policy.get("include_confidence", True)),
            "include_uncertainties": bool(output_policy.get("include_uncertainties", True)),
            "no_hallucination": bool(output_policy.get("no_hallucination", True)),
        },
    }


def check_pack_exists(pack_path: str) -> Dict[str, Any]:
    """Vérifie que le pack VLM existe, est un JSON valide et contient des segments."""
    p = Path(pack_path)
    if not p.exists() or not p.is_file():
        raise VlmStrategyError(f"Pack VLM introuvable : {p}")
    try:
        pack = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VlmStrategyError(f"Pack VLM JSON invalide ({p}) : {exc}") from exc
    if not isinstance(pack.get("segments"), list) or not pack["segments"]:
        raise VlmStrategyError("Pack VLM sans segments exploitables.")
    return pack


def _segment_inputs(segment: Dict[str, Any], input_policy: Dict[str, Any]) -> Dict[str, Any]:
    """Sélectionne les entrées d'un segment selon input_policy (chemins/texte seulement)."""
    inputs: Dict[str, Any] = {
        "segment_id": segment.get("segment_id"),
        "start_s": segment.get("start_s"),
        "end_s": segment.get("end_s"),
    }
    if input_policy["use_motion_grid"]:
        inputs["motion_grid_path"] = segment.get("motion_grid_path")
    if input_policy["use_motion_composite"]:
        inputs["motion_composite_path"] = segment.get("motion_composite_path")
    if input_policy["use_selected_frames"]:
        frames = segment.get("selected_frames", []) or []
        inputs["selected_frame_paths"] = [f.get("path") for f in frames]
    if input_policy["use_transcript"]:
        inputs["combined_text"] = segment.get("combined_text", "")
        inputs["transcript_segments"] = segment.get("transcript_segments", [])
    return inputs


def build_vlm_run_plan(pack: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Construit un plan d'exécution VLM (descriptif) — n'appelle AUCUN modèle.

    Le plan liste, par segment, les entrées retenues selon input_policy. L'exécution
    réelle (manuelle ou via un futur adapter) est hors périmètre de ce module.
    """
    normalized = validate_config(config)
    segments = pack.get("segments")
    if not isinstance(segments, list) or not segments:
        raise VlmStrategyError("Pack VLM sans segments exploitables.")

    plan_segments = [
        _segment_inputs(seg, normalized["input_policy"]) for seg in segments
    ]

    logging.info(
        "build_vlm_run_plan mode=%s provider=%s segments=%d (aucun appel modèle)",
        normalized["mode"], normalized["provider"], len(plan_segments),
    )

    return {
        "plan_type": "vlm_run_plan",
        "mode": normalized["mode"],
        "provider": normalized["provider"],
        "model": normalized["model"],
        "video_path": pack.get("video_path", ""),
        "language": pack.get("language"),
        "input_policy": normalized["input_policy"],
        "output_policy": normalized["output_policy"],
        "segments": plan_segments,
        "segment_count": len(plan_segments),
        "executed": False,
        "next_step": "prepare_vlm_run",
    }


def prepare_strategy(
    pack_path: str, config_path: str
) -> Dict[str, Any]:
    """Charge config + pack, valide, et retourne le plan d'exécution (sans appel modèle)."""
    config = load_vlm_config(config_path)
    pack = check_pack_exists(pack_path)
    return build_vlm_run_plan(pack, config)
