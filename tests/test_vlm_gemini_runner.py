"""Tests du VLM Gemini — google.genai MOCKÉ (aucun appel API, hors-ligne).

Vérifie : JSON valide conservé tel quel, réponse invalide non réparée,
nettoyage des fences ```json et validation stricte hors-ligne.
"""
import sys
import types as pytypes
import json

import pytest

from src.video_agent import vlm_gemini_runner as gr
from src.video_agent.vlm_gemini_runner import VlmGeminiError
from src.video_agent.vlm_result_validation import validate_single_result


def _install_fake_genai(monkeypatch, *, response_text, record=None):
    """Installe un faux `google.genai` (+ types) et une clé GEMINI_API_KEY de test."""
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

    class Part:
        def __init__(self, data, mime_type):
            self.data, self.mime_type = data, mime_type

        @classmethod
        def from_bytes(cls, data, mime_type):
            return cls(data, mime_type)

    types_mod.HttpOptions = HttpOptions
    types_mod.GenerateContentConfig = GenerateContentConfig
    types_mod.Part = Part

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


def _img(tmp_path):
    p = tmp_path / "f.jpg"
    p.write_bytes(b"\xff\xd8stub")
    return str(p)


def _valid_json(segment_id=1, start_s=0.0, end_s=1.0, combined_text="hello"):
    return json.dumps({
        "segment_id": segment_id, "start_s": start_s, "end_s": end_s,
        "combined_text": combined_text,
        "scene_description": "une scène de rue",
        "people_objects": ["enfant", "voiture"], "actions": ["marcher"],
        "motion_description": "léger panoramique",
        "audio_visual_relation": "musique douce", "uses_speech": True,
        "mood": ["calme"],
        "texte_ecran": [{"text": "STOP", "lang": "en", "location": "haut"}],
        "confidence": 0.8, "uncertainties": ["incertain sur la marque"],
        "evidence": {"visual": ["enfant visible"], "audio": ["voix"]},
    }, ensure_ascii=False)


def test_analyze_segment_valid(tmp_path, monkeypatch):
    rec = {}
    _install_fake_genai(monkeypatch, response_text=_valid_json(3), record=rec)
    out = gr.analyze_segment(
        "analyse", [_img(tmp_path)], segment_id=3,
        start_s=0.0, end_s=1.0, combined_text="hello",
    )
    assert out["segment_id"] == 3
    assert out["confidence"] == 0.8
    report = validate_single_result(
        out, 3, {"segment_id": 3, "start_s": 0.0, "end_s": 1.0, "combined_text": "hello"}
    )
    assert report["valid"] is True, report["errors"]
    assert out["texte_ecran"][0]["text"] == "STOP"
    # config JSON demandée + image transmise.
    assert rec["config"]["response_mime_type"] == "application/json"


def test_analyze_segment_strips_markdown_fences(tmp_path, monkeypatch):
    fenced = "```json\n" + _valid_json(1) + "\n```"
    _install_fake_genai(monkeypatch, response_text=fenced)
    out = gr.analyze_segment("analyse", [_img(tmp_path)], segment_id=1)
    assert out["segment_id"] == 1
    assert validate_single_result(out, 1)["valid"] is True


def test_analyze_segment_partial_json_is_not_normalized(tmp_path, monkeypatch):
    _install_fake_genai(monkeypatch, response_text='{"scene_description": "x"}')
    out = gr.analyze_segment("p", [_img(tmp_path)], segment_id=7)
    assert out == {"scene_description": "x"}
    assert validate_single_result(out, 7)["valid"] is False


def test_analyze_segment_no_images_raises():
    with pytest.raises(VlmGeminiError):
        gr.analyze_segment("p", [], segment_id=1)


def test_analyze_segment_empty_response_raises(tmp_path, monkeypatch):
    _install_fake_genai(monkeypatch, response_text="")
    with pytest.raises(VlmGeminiError):
        gr.analyze_segment("p", [_img(tmp_path)], segment_id=1)


def test_missing_api_key_raises(tmp_path, monkeypatch):
    _install_fake_genai(monkeypatch, response_text=_valid_json())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(VlmGeminiError):
        gr.analyze_segment("p", [_img(tmp_path)], segment_id=1)


def test_provider_wrapper(tmp_path, monkeypatch):
    _install_fake_genai(monkeypatch, response_text=_valid_json(2))
    prov = gr.GeminiVLMProvider(model="gemini-2.5-flash")
    assert prov.name == "gemini"
    out = prov.analyze_segment("p", [_img(tmp_path)], segment_id=2)
    assert out["segment_id"] == 2


def test_resolve_vlm_model_priority(monkeypatch):
    monkeypatch.delenv("GEMINI_VLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert gr.resolve_vlm_model() == gr.DEFAULT_VLM_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "audio-model")      # fallback documenté
    assert gr.resolve_vlm_model() == "audio-model"
    monkeypatch.setenv("GEMINI_VLM_MODEL", "vlm-model")    # prioritaire sur GEMINI_MODEL
    assert gr.resolve_vlm_model() == "vlm-model"
    assert gr.resolve_vlm_model("explicit") == "explicit"  # argument prioritaire
