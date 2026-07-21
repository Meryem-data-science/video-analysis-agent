"""Tests de scripts/run_final_report.py — pipeline mocké, contrôle de complétude pur.

Vérifie : (1) tous les composants sont FORCÉS à True à l'appel du pipeline ;
(2) le contrôle de complétude détecte un arc vide, un OCR absent et un audio partiel.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_final_report as rfr  # noqa: E402


# --- Helpers : artefacts synthétiques -----------------------------------------------
def _scene(seg, ocr=True, gem="ok", lib="ok"):
    return {
        "segment_id": seg,
        "texte_ecran": [{"text": "x", "confidence": 0.9}] if ocr else [],
        "audio_scene": {"status": gem, "measured_features": {"status": lib}},
    }


def _report(n=16, arc=True, ocr=True, gem_ok=None, lib_ok=None):
    gem_ok = n if gem_ok is None else gem_ok
    lib_ok = n if lib_ok is None else lib_ok
    scenes = [
        _scene(i, ocr=ocr,
               gem="ok" if i <= gem_ok else "échec_gracieux",
               lib="ok" if i <= lib_ok else "features_échec")
        for i in range(1, n + 1)
    ]
    rep = {"scene_count": n, "scenes": scenes}
    if arc:
        rep["arc_narratif"] = {"situation_initiale": "il était une fois", "climax": "le pic"}
        rep["resume_humain"] = "Un résumé humain non vide."
    else:
        rep["arc_narratif"] = {"situation_initiale": "", "climax": ""}
        rep["resume_humain"] = ""
    return rep


def _result(ok=True, steps=15, audio=True):
    return {"ok": ok, "audio_present": audio,
            "steps": [{"step": f"s{i}", "ok": True} for i in range(steps)],
            "reports": {"json": None, "txt": None}, "summary": None}


_SUMMARY = {"valid_count": 16, "segment_count": 16}


# --- (1) tous les composants forcés à ON --------------------------------------------
def test_forced_flags_constant_all_true():
    assert rfr.FORCED_FLAGS == {
        "enable_arc": True, "enable_audio": True,
        "enable_audio_features": True,
    }


def test_pipeline_called_with_all_components_on():
    captured = {}

    def fake_runner(video, **kw):
        captured.update(kw)
        captured["_video"] = video
        return _result()

    rc = rfr.run_demo("clip.mp4", runner=fake_runner, load_json=lambda p: {})
    assert captured["_video"] == "clip.mp4"
    assert captured["enable_arc"] is True
    assert captured["enable_audio"] is True
    assert captured["enable_audio_features"] is True
    # aucun flag de désactivation ne doit être passé.
    assert not any(k.startswith("no_") or k.startswith("disable") for k in captured)
    assert rc == 0


def test_run_demo_returns_1_when_not_ok():
    rc = rfr.run_demo("clip.mp4",
                      runner=lambda v, **k: _result(ok=False),
                      load_json=lambda p: {})
    assert rc == 1


def test_run_demo_returns_1_when_awaiting_manual_vlm():
    result = _result(ok=True)
    result["status"] = "awaiting_vlm"
    rc = rfr.run_demo(
        "clip.mp4", runner=lambda v, **k: result, load_json=lambda p: {}
    )
    assert rc == 1


# --- (2) contrôle de complétude ------------------------------------------------------
def test_completeness_full_no_warnings():
    check, warnings = rfr.build_completeness(_result(), _report(), _SUMMARY)
    assert warnings == []
    assert check["ok"] is True
    assert check["arc_filled"] is True
    assert check["valid_count"] == 16 and check["segment_count"] == 16
    assert check["gemini_audio_ok"] == 16 and check["librosa_features_ok"] == 16
    assert check["screen_text_scenes"] == 16


def test_completeness_detects_empty_arc():
    check, warnings = rfr.build_completeness(_result(), _report(arc=False), _SUMMARY)
    assert check["arc_filled"] is False
    assert any("arc vide" in w for w in warnings)


def test_completeness_detects_missing_screen_text():
    check, warnings = rfr.build_completeness(_result(), _report(ocr=False), _SUMMARY)
    assert check["screen_text_scenes"] == 0
    assert any("texte écran (VLM) sur 0/16" in w for w in warnings)


def test_completeness_detects_partial_gemini_audio():
    check, warnings = rfr.build_completeness(_result(), _report(gem_ok=5), _SUMMARY)
    assert check["gemini_audio_ok"] == 5
    assert any("audio Gemini partiel 5/16" in w for w in warnings)


def test_completeness_detects_partial_librosa():
    check, warnings = rfr.build_completeness(_result(), _report(lib_ok=10), _SUMMARY)
    assert check["librosa_features_ok"] == 10
    assert any("librosa measured_features partiel 10/16" in w for w in warnings)


def test_audio_warnings_suppressed_when_no_audio():
    # vidéo sans audio : pas d'avertissement Gemini/librosa (audio non attendu).
    check, warnings = rfr.build_completeness(
        _result(audio=False), _report(gem_ok=0, lib_ok=0), _SUMMARY)
    assert check["audio_present"] is False
    assert not any("Gemini" in w or "librosa" in w for w in warnings)


def test_partial_steps_warned():
    result = _result()
    result["steps"][0]["ok"] = False  # une étape échouée
    check, warnings = rfr.build_completeness(result, _report(), _SUMMARY)
    assert check["steps_ok"] == 14 and check["steps_total"] == 15
    assert any("étapes partielles 14/15" in w for w in warnings)


def test_format_completeness_includes_warnings_and_counts():
    check, warnings = rfr.build_completeness(_result(), _report(ocr=False), _SUMMARY)
    text = rfr.format_completeness(check, warnings)
    assert "CONTRÔLE DE COMPLÉTUDE" in text
    assert "valid_count" in text
    assert "⚠️" in text and "texte écran (VLM) sur 0/16" in text
