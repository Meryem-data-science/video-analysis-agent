"""Tests des modes VLM explicites et de la sélection stricte du provider."""

import sys

import pytest

from src.video_agent import vlm_provider as vp
from src.video_agent.vlm_provider import VlmProviderError


def test_default_mode_is_manual(monkeypatch):
    monkeypatch.delenv("VLM_MODE", raising=False)
    assert vp.resolve_vlm_mode(None, None) == "manual"
    assert vp.resolve_vlm_settings({}) == {"mode": "manual", "provider": None}


def test_mode_cli_over_env_over_config(monkeypatch):
    monkeypatch.setenv("VLM_MODE", "disabled")
    assert vp.resolve_vlm_mode("automatic", {"mode": "manual"}) == "automatic"
    assert vp.resolve_vlm_mode(None, {"mode": "manual"}) == "disabled"


def test_automatic_requires_explicit_provider(monkeypatch):
    monkeypatch.delenv("VLM_PROVIDER", raising=False)
    with pytest.raises(VlmProviderError, match="exige un provider explicite"):
        vp.resolve_vlm_settings({"mode": "automatic"})


def test_provider_unknown_never_falls_back_to_gemini(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "claude_agent")
    with pytest.raises(VlmProviderError, match="Provider VLM inconnu"):
        vp.resolve_vlm_settings({"mode": "automatic", "provider": "gemini"})


def test_manual_and_disabled_do_not_import_gemini(monkeypatch):
    monkeypatch.delenv("VLM_MODE", raising=False)
    monkeypatch.delenv("VLM_PROVIDER", raising=False)
    sys.modules.pop("src.video_agent.vlm_gemini_runner", None)
    assert vp.resolve_vlm_settings({"mode": "manual"})["provider"] is None
    assert vp.resolve_vlm_settings({"mode": "disabled"})["provider"] is None
    assert "src.video_agent.vlm_gemini_runner" not in sys.modules


def test_get_provider_gemini_explicit():
    prov = vp.get_vlm_provider("gemini")
    assert prov.name == "gemini"
    assert isinstance(prov, vp.VlmProvider)


@pytest.mark.parametrize("name", [None, "manual", "openai", "claude"])
def test_get_provider_missing_or_unknown_raises(name):
    with pytest.raises(VlmProviderError):
        vp.get_vlm_provider(name)
