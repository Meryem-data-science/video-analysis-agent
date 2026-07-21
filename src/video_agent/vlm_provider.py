"""Configuration et sélection explicites du fournisseur VLM.

Modes : ``disabled``, ``manual`` (défaut, sans réseau) et ``automatic``.
Gemini est le seul fournisseur automatique actuel et reste importé paresseusement.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Protocol, runtime_checkable

DEFAULT_MODE = "manual"
SUPPORTED_MODES = ("disabled", "manual", "automatic")
SUPPORTED_PROVIDERS = ("gemini",)


class VlmProviderError(Exception):
    """Mode ou fournisseur VLM absent, inconnu ou non applicable."""


@runtime_checkable
class VlmProvider(Protocol):
    """Interface minimale d'un provider automatique extensible."""

    name: str

    def analyze_segment(
        self,
        prompt_text: str,
        image_paths: list[str],
        segment_id: int,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        combined_text: str = "",
        model: Optional[str] = None,
        timeout_s: float = 300.0,
    ) -> Dict[str, Any]: ...


def _config_value(config: Optional[Dict[str, Any]], *keys: str) -> str:
    if not isinstance(config, dict):
        return ""
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_vlm_mode(
    cli_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None
) -> str:
    """Résout le mode : CLI > ``VLM_MODE`` > config > ``manual``."""
    candidate = (
        (cli_mode or "").strip()
        or (os.environ.get("VLM_MODE") or "").strip()
        or _config_value(config, "mode")
        or DEFAULT_MODE
    ).lower()
    if candidate not in SUPPORTED_MODES:
        raise VlmProviderError(
            f"Mode VLM inconnu : {candidate!r}. Valeurs supportées : {SUPPORTED_MODES}."
        )
    return candidate


def resolve_provider_name(
    cli_provider: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    required: bool = False,
) -> Optional[str]:
    """Résout strictement le provider, sans fallback silencieux."""
    candidate = (
        (cli_provider or "").strip()
        or (os.environ.get("VLM_PROVIDER") or "").strip()
        or _config_value(config, "vlm_provider", "provider")
    )
    if not candidate:
        if required:
            raise VlmProviderError(
                "Le mode VLM 'automatic' exige un provider explicite "
                f"parmi : {SUPPORTED_PROVIDERS}."
            )
        return None
    candidate = candidate.lower()
    if candidate not in SUPPORTED_PROVIDERS:
        raise VlmProviderError(
            f"Provider VLM inconnu : {candidate!r}. "
            f"Valeurs supportées : {SUPPORTED_PROVIDERS}."
        )
    return candidate


def resolve_vlm_settings(
    config: Optional[Dict[str, Any]] = None,
    *,
    cli_mode: Optional[str] = None,
    cli_provider: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Valide ensemble le mode et le provider éventuel."""
    mode = resolve_vlm_mode(cli_mode, config)
    provider = resolve_provider_name(
        cli_provider, config, required=(mode == "automatic")
    )
    return {"mode": mode, "provider": provider}


def get_vlm_provider(
    name: Optional[str], *, model: Optional[str] = None
) -> VlmProvider:
    """Instancie un provider automatique explicitement nommé."""
    resolved = resolve_provider_name(name, required=True)
    if resolved == "gemini":
        from .vlm_gemini_runner import GeminiVLMProvider

        return GeminiVLMProvider(model=model)
    raise VlmProviderError(
        f"Provider VLM non implémenté : {resolved!r}. Valeurs : {SUPPORTED_PROVIDERS}."
    )
