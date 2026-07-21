"""Contrat observé entre l'installation minimale et l'extra audio."""

import builtins
import subprocess
import sys
import tomllib
import wave
from pathlib import Path

import pytest

from src.video_agent.transcription import TranscriptionError, transcribe_audio


def test_minimal_import_does_not_load_optional_model_sdks():
    code = (
        "import sys, src.video_agent as video_agent; "
        "assert video_agent.__version__; "
        "assert 'faster_whisper' not in sys.modules; "
        "assert 'google.genai' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_audio_dependencies_are_optional_but_grouped_in_audio_extra():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    core = " ".join(data["project"]["dependencies"]).lower()
    audio = " ".join(data["project"]["optional-dependencies"]["audio"]).lower()
    assert "faster-whisper" not in core and "librosa" not in core
    assert "faster-whisper" in audio and "librosa" in audio


def test_transcription_without_audio_extra_has_actionable_error(tmp_path, monkeypatch):
    wav_path = tmp_path / "silence.wav"
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160)

    original_import = builtins.__import__

    def without_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("simulated minimal installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_whisper)
    with pytest.raises(TranscriptionError, match=r"\.\[audio\]"):
        transcribe_audio(str(wav_path))
