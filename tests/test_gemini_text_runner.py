"""Tests du runner texte Gemini — google.genai MOCKÉ (aucun appel API)."""
import sys
import types as pytypes

import pytest

from src.video_agent import gemini_text_runner as gt
from src.video_agent.gemini_text_runner import GeminiTextError


def _install_fake_genai(monkeypatch, *, response_text, record=None):
    google_mod = pytypes.ModuleType("google")
    genai_mod = pytypes.ModuleType("google.genai")
    types_mod = pytypes.ModuleType("google.genai.types")

    class HttpOptions:
        def __init__(self, **kw):
            pass

    class GenerateContentConfig:
        def __init__(self, **kw):
            if record is not None:
                record["config"] = kw

    types_mod.HttpOptions = HttpOptions
    types_mod.GenerateContentConfig = GenerateContentConfig

    class _Models:
        def generate_content(self, model, contents, config):
            if record is not None:
                record["model"] = model
                record["contents"] = contents
            return pytypes.SimpleNamespace(text=response_text)

    class Client:
        def __init__(self, api_key=None, http_options=None):
            self.models = _Models()

    genai_mod.Client = Client
    genai_mod.types = types_mod
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


_SCHEMA = {
    "type": "object",
    "required": ["resume"],
    "properties": {"resume": {"type": "string"}},
}


def test_returns_parsed_dict(monkeypatch):
    rec = {}
    _install_fake_genai(monkeypatch, response_text='{"resume": "ok"}', record=rec)
    out = gt.analyze_text_with_gemini("prompt", _SCHEMA, model="gemini-2.5-flash")
    assert out == {"resume": "ok"}
    assert rec["model"] == "gemini-2.5-flash"
    assert rec["config"]["response_mime_type"] == "application/json"


def test_strips_fences(monkeypatch):
    _install_fake_genai(monkeypatch, response_text='```json\n{"resume": "ok"}\n```')
    assert gt.analyze_text_with_gemini("p", _SCHEMA)["resume"] == "ok"


def test_empty_response_raises(monkeypatch):
    _install_fake_genai(monkeypatch, response_text="")
    with pytest.raises(GeminiTextError):
        gt.analyze_text_with_gemini("p", _SCHEMA)


def test_bad_json_raises(monkeypatch):
    _install_fake_genai(monkeypatch, response_text="pas du json")
    with pytest.raises(GeminiTextError):
        gt.analyze_text_with_gemini("p", _SCHEMA)


def test_missing_api_key_raises(monkeypatch):
    _install_fake_genai(monkeypatch, response_text='{"resume": "ok"}')
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiTextError):
        gt.analyze_text_with_gemini("p", _SCHEMA)


def test_resolve_text_model_priority(monkeypatch):
    monkeypatch.delenv("GEMINI_TEXT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_VLM_MODEL", raising=False)
    assert gt.resolve_text_model() == gt.DEFAULT_TEXT_MODEL
    monkeypatch.setenv("GEMINI_VLM_MODEL", "vlm")       # fallback
    assert gt.resolve_text_model() == "vlm"
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "text")     # prioritaire
    assert gt.resolve_text_model() == "text"
    assert gt.resolve_text_model("explicit") == "explicit"
