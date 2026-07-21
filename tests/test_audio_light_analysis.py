import math
import struct
import wave

import pytest

from src.video_agent import audio_light_analysis as ala
from src.video_agent.audio_light_analysis import AudioLightAnalysisError


# --- Helpers : génération de petits WAV mono 16 kHz factices ---------------------
def _write_wav(path, samples, sample_rate=16000, channels=1, sample_width=2):
    """Écrit un petit WAV PCM 16-bit à partir d'échantillons entiers."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return str(path)


def _silence(n):
    return [0] * n


def _tone(n, amplitude=12000, sample_rate=16000, freq=220):
    return [
        int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n)
    ]


# --- _rms / _peak ---------------------------------------------------------------
def test_rms_empty_and_zero():
    assert ala._rms([]) == 0.0
    assert ala._rms([0.0, 0.0, 0.0]) == 0.0


def test_rms_known_value():
    # RMS de [0.5, -0.5] = 0.5
    assert ala._rms([0.5, -0.5]) == pytest.approx(0.5)


def test_peak():
    assert ala._peak([]) == 0.0
    assert ala._peak([0.1, -0.8, 0.3]) == pytest.approx(0.8)


# --- _compute_energy_profile ----------------------------------------------------
def test_compute_energy_profile_structure():
    samples = [0.0] * 16000  # 1 s à 16 kHz
    profile = ala._compute_energy_profile(samples, 16000, 500)
    assert len(profile) == 2  # deux fenêtres de 500 ms
    for w in profile:
        assert set(w.keys()) == {"index", "start_s", "end_s", "rms", "peak", "is_silent"}
    assert profile[0]["start_s"] == 0.0
    assert profile[0]["end_s"] == pytest.approx(0.5)


def test_compute_energy_profile_invalid_window():
    with pytest.raises(AudioLightAnalysisError):
        ala._compute_energy_profile([0.0] * 100, 16000, 0)


# --- _classify_audio / _recommend ----------------------------------------------
def test_classify_silence():
    c = ala._classify_audio(0.0, 0.0)
    assert c["mode"] == "silence"
    assert ala._recommend(c)["run_whisper"] is False


def test_classify_active():
    c = ala._classify_audio(0.9, 0.2)
    assert c["mode"] == "active_audio"
    assert ala._recommend(c)["run_whisper"] is True


def test_classify_mixed():
    c = ala._classify_audio(0.15, 0.05)
    assert c["mode"] == "mixed_or_unknown"
    assert ala._recommend(c)["run_whisper"] is True


# --- Cas 1 : audio silencieux ---------------------------------------------------
def test_analyze_silence(tmp_path):
    p = _write_wav(tmp_path / "silence.wav", _silence(16000 * 2))  # 2 s de zéros
    m = ala.analyze_audio_light(p)
    assert m["classification"]["mode"] == "silence"
    assert m["recommendation"]["run_whisper"] is False
    assert m["next_step"] == "skip_whisper"
    assert m["features"]["silence_ratio"] == pytest.approx(1.0)
    assert m["features"]["active_ratio"] == pytest.approx(0.0)


# --- Cas 2 : audio actif --------------------------------------------------------
def test_analyze_active(tmp_path):
    p = _write_wav(tmp_path / "tone.wav", _tone(16000 * 2))  # 2 s de tonalité
    m = ala.analyze_audio_light(p)
    assert m["classification"]["mode"] in ("active_audio", "mixed_or_unknown")
    assert m["recommendation"]["run_whisper"] is True
    assert m["next_step"] == "whisper_transcription"
    assert m["features"]["active_ratio"] > 0.0


# --- Cas 3 : energy_profile dans le manifest ------------------------------------
def test_analyze_energy_profile_fields(tmp_path):
    p = _write_wav(tmp_path / "tone.wav", _tone(16000))
    m = ala.analyze_audio_light(p, window_ms=500)
    assert m["analysis_type"] == "light_audio_energy_analysis"
    assert len(m["energy_profile"]) == m["features"]["window_count"]
    for w in m["energy_profile"]:
        assert set(w.keys()) == {"index", "start_s", "end_s", "rms", "peak", "is_silent"}


# --- Cas 4 : fichier absent -----------------------------------------------------
def test_analyze_missing_file(tmp_path):
    with pytest.raises(AudioLightAnalysisError):
        ala.analyze_audio_light(str(tmp_path / "nope.wav"))


# --- Cas 5 : WAV vide / invalide ------------------------------------------------
def test_analyze_empty_file(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")  # 0 octet
    with pytest.raises(AudioLightAnalysisError):
        ala.analyze_audio_light(str(p))


def test_analyze_not_a_wav(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"this is not a wav file at all")  # contenu non WAV
    with pytest.raises(AudioLightAnalysisError):
        ala.analyze_audio_light(str(p))


# --- Cas 6 : paramètres rapportés -----------------------------------------------
def test_analyze_reported_params(tmp_path):
    p = _write_wav(tmp_path / "tone.wav", _tone(16000 * 2), sample_rate=16000, channels=1)
    m = ala.analyze_audio_light(p)
    f = m["features"]
    assert f["sample_rate"] == 16000
    assert f["channels"] == 1
    assert f["sample_width_bytes"] == 2
    assert f["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert f["silence_ratio"] + f["active_ratio"] == pytest.approx(1.0)


def test_analyze_source_manifest_reported(tmp_path):
    p = _write_wav(tmp_path / "tone.wav", _tone(16000))
    src = {"audio_path": "x", "summary": {"video_stem": "x"}}
    m = ala.analyze_audio_light(p, audio_manifest=src)
    assert m["source_audio_manifest"] == src
