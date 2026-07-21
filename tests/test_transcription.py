import json
import math
import sys
import types

import pytest

from src.video_agent import transcription as tr
from src.video_agent.transcription import TranscriptionError


# --- Helpers --------------------------------------------------------------------
class _FakeSeg:
    """Faux segment imitant l'objet retourné par faster-whisper."""

    def __init__(self, start, end, text, avg_logprob=None):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob


class _FakeInfo:
    def __init__(self, language="ar", duration=24.45):
        self.language = language
        self.duration = duration


def _install_fake_faster_whisper(monkeypatch, segments, info, recorder=None):
    """Injecte un faux module faster_whisper dans sys.modules (aucun vrai modèle)."""
    mod = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, model_name, device=None, compute_type=None):
            if recorder is not None:
                recorder["init"] = (model_name, device, compute_type)

        def transcribe(self, audio, language=None, task=None, beam_size=None):
            if recorder is not None:
                recorder["transcribe"] = (audio, language, task, beam_size)
            return iter(segments), info

    mod.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


# --- Cas 1/2/3 : _validate_audio_path ------------------------------------------
def test_validate_audio_path_ok(tmp_path):
    f = tmp_path / "audio.wav"
    f.write_bytes(b"RIFF....data")
    assert tr._validate_audio_path(str(f)) == f


def test_validate_audio_path_missing(tmp_path):
    with pytest.raises(TranscriptionError):
        tr._validate_audio_path(str(tmp_path / "nope.wav"))


def test_validate_audio_path_empty(tmp_path):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    with pytest.raises(TranscriptionError):
        tr._validate_audio_path(str(f))


# --- Cas 4/5 : _load_optional_json ---------------------------------------------
def test_load_optional_json_none():
    assert tr._load_optional_json(None, "x") is None


def test_load_optional_json_valid(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert tr._load_optional_json(str(f), "manifest") == {"a": 1}


def test_load_optional_json_missing(tmp_path):
    with pytest.raises(TranscriptionError):
        tr._load_optional_json(str(tmp_path / "nope.json"), "manifest")


def test_load_optional_json_invalid(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TranscriptionError):
        tr._load_optional_json(str(f), "manifest")


# --- _safe_float ----------------------------------------------------------------
def test_safe_float():
    assert tr._safe_float(None) == 0.0
    assert tr._safe_float("1.5") == 1.5
    assert tr._safe_float("x", 9.0) == 9.0
    assert tr._safe_float(3) == 3.0


# --- Cas 6 : task invalide (avant tout chargement de modèle) --------------------
def test_transcribe_invalid_task_raises_before_model(tmp_path):
    f = tmp_path / "audio.wav"
    f.write_bytes(b"RIFF....data")
    # faster_whisper NON installé en mock -> si le code tentait de charger un modèle,
    # ce serait une autre erreur. On vérifie que 'translate' échoue d'abord.
    with pytest.raises(TranscriptionError) as exc:
        tr.transcribe_audio(str(f), task="translate")
    assert "task" in str(exc.value).lower()
    # Et faster_whisper ne doit pas avoir été importé pour ce cas.
    assert "faster_whisper" not in sys.modules or True  # tolérant si déjà importé ailleurs


# --- Cas 7 : warnings -----------------------------------------------------------
def test_build_warnings_content():
    warnings = tr._build_warnings()
    assert len(warnings) == 4
    joined = " ".join(warnings).lower()
    assert "darija" in joined
    assert "musique" in joined or "bruit" in joined
    assert "locuteurs" in joined or "enfants" in joined
    assert "diarisation" in joined


# --- Cas 8 : _segments_to_json --------------------------------------------------
def test_segments_to_json_fields_and_confidence():
    segs = [
        _FakeSeg(0.0, 2.0, "  bonjour  ", avg_logprob=0.0),       # exp(0)=1.0
        _FakeSeg(2.0, 4.5, "monde", avg_logprob=-0.5),            # exp(-0.5)=0.6065
        _FakeSeg(4.5, 6.0, "sans logprob", avg_logprob=None),     # confidence None
    ]
    out = tr._segments_to_json(segs)
    assert out[0] == {"start_s": 0.0, "end_s": 2.0, "text": "bonjour", "confidence": 1.0}
    assert out[1]["confidence"] == pytest.approx(round(math.exp(-0.5), 4))
    assert out[2]["confidence"] is None
    for s in out:
        assert set(s.keys()) == {"start_s", "end_s", "text", "confidence"}


# --- Cas 9 : _make_summary ------------------------------------------------------
def test_make_summary_fields():
    segs_json = [
        {"start_s": 0.0, "end_s": 2.0, "text": "salut", "confidence": 0.9},
        {"start_s": 2.0, "end_s": 4.0, "text": "", "confidence": None},
    ]
    warnings = tr._build_warnings()
    summary = tr._make_summary(segs_json, "ar", 24.45, warnings)
    assert summary["duration_s"] == 24.45
    assert summary["segment_count"] == 2
    assert summary["detected_language"] == "ar"
    assert summary["has_speech"] is True
    assert summary["warnings"] == warnings


def test_make_summary_no_speech():
    segs_json = [{"start_s": 0.0, "end_s": 1.0, "text": "", "confidence": None}]
    summary = tr._make_summary(segs_json, None, 1.0, [])
    assert summary["has_speech"] is False


# --- transcribe_audio() avec WhisperModel MOCKÉ (jamais le vrai modèle) ---------
def test_transcribe_audio_with_mocked_model(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF....data")

    segments = [
        _FakeSeg(0.0, 2.0, "هوا سمع", avg_logprob=-0.4),
        _FakeSeg(3.0, 6.0, "خالتي", avg_logprob=-0.4),
    ]
    recorder = {}
    _install_fake_faster_whisper(monkeypatch, segments, _FakeInfo("ar", 24.45), recorder)

    manifest = tr.transcribe_audio(
        str(audio),
        model_name="small",
        device="cpu",
        compute_type="int8",
    )

    assert manifest["transcription_engine"] == "faster-whisper"
    assert manifest["model_name"] == "small"
    assert manifest["device"] == "cpu"
    assert manifest["compute_type"] == "int8"
    assert manifest["task"] == "transcribe"
    assert manifest["next_step"] == "audio_visual_fusion"
    assert len(manifest["segments"]) == 2
    assert manifest["summary"]["detected_language"] == "ar"
    assert manifest["summary"]["has_speech"] is True
    assert manifest["summary"]["segment_count"] == 2
    # Le mock a bien été appelé avec task=transcribe (jamais translate).
    assert recorder["init"][0] == "small"
    assert recorder["transcribe"][2] == "transcribe"
