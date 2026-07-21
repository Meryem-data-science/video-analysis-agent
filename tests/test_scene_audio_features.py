"""Tests de scene_audio_features — local, déterministe, hors-ligne.

Pour la structure et la dégradation gracieuse : on monkeypatch les seams IO/calcul
(_extract_scene_wav, _compute_features). Pour la logique réelle (scène trop courte,
énergie), on appelle _features_from_signal sur de PETITS signaux numpy synthétiques
(librosa réel, mais aucun fichier / réseau).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.video_agent import scene_audio_features as saf


SCENES = [
    {"index": 1, "start": 0.0, "end": 6.0},
    {"index": 2, "start": 6.0, "end": 8.5},
    {"index": 3, "start": 8.5, "end": 14.0},
]


def _fake_features(activity="actif", tempo=120.0):
    return {
        "source": "librosa",
        "audio_activity": activity,
        "energy_rms": 0.12,
        "tempo_bpm": tempo,
        "tempo_reason": None,
        "tempo_note": saf._TEMPO_NOTE,
        "beat_count": 12,
        "beat_density_per_s": 2.0,
    }


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Structure de l'artefact + provenance -------------------------------------------

def test_artifact_structure_and_provenance(tmp_path, monkeypatch):
    def fake_extract(video_path, start, end, tmp_dir, sr=saf.DEFAULT_SR):
        p = Path(tmp_dir) / f"fake_{start:.3f}.wav"
        p.write_bytes(b"\x00\x01")
        return p

    monkeypatch.setattr(saf, "_extract_scene_wav", fake_extract)
    monkeypatch.setattr(saf, "_compute_features",
                        lambda wav, dur, **k: _fake_features())

    out = tmp_path / "audio_features_demo.json"
    res = saf.build_scene_audio_features("videos/demo.mp4", SCENES, str(out))

    assert res["status"] == "ok"
    assert res["scenes_ok"] == 3 and res["scenes_total"] == 3
    assert res["artifact_path"] == str(out)

    art = _read(out)
    assert art["schema"] == "audio_features/1.0"
    assert art["video_path"] == "videos/demo.mp4"
    assert art["status"] == "ok"
    assert set(art["params"]) == {"sr", "tempo_min_duration_s", "energy_activity_threshold"}
    assert "tempo" in art["method_notes"] and "audio_activity" in art["method_notes"]
    assert len(art["scenes"]) == 3
    sc = art["scenes"][0]
    assert set(sc) == {"scene_index", "start_s", "end_s", "duration_s", "features", "audio_status"}
    assert sc["audio_status"] == "ok"
    assert sc["duration_s"] == 6.0
    assert sc["features"]["source"] == "librosa"           # provenance
    assert sc["features"]["tempo_bpm"] == 120.0


# --- Scène trop courte -> tempo null + raison (logique réelle, signal synthétique) ---

def test_short_scene_tempo_null_with_reason():
    sr = saf.DEFAULT_SR
    dur = 1.0  # < 4 s
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = (0.3 * np.sin(2 * math.pi * 220 * t)).astype(np.float32)
    feat = saf._features_from_signal(y, sr, dur)
    assert feat["tempo_bpm"] is None
    assert feat["beat_count"] is None and feat["beat_density_per_s"] is None
    assert "trop courte" in feat["tempo_reason"]
    assert feat["source"] == "librosa"
    # énergie quand même mesurée (ne dépend pas du tempo)
    assert isinstance(feat["energy_rms"], float) and feat["energy_rms"] > 0


# --- Dégradation gracieuse ----------------------------------------------------------

def test_graceful_degradation_unreadable_audio(tmp_path, monkeypatch):
    # ffmpeg échoue pour toutes les scènes -> features_échec partout, statut échec.
    monkeypatch.setattr(saf, "_extract_scene_wav",
                        lambda *a, **k: None)
    out = tmp_path / "audio_features_demo.json"
    res = saf.build_scene_audio_features("videos/demo.mp4", SCENES, str(out))
    assert res["status"] == "échec"
    assert res["scenes_ok"] == 0
    art = _read(out)
    assert all(s["audio_status"] == "features_échec" for s in art["scenes"])
    assert all(s["features"] == {} for s in art["scenes"])


def test_graceful_degradation_compute_raises(tmp_path, monkeypatch):
    # extraction OK mais librosa lève sur 1 scène -> cette scène échoue, les autres ok.
    def fake_extract(video_path, start, end, tmp_dir, sr=saf.DEFAULT_SR):
        p = Path(tmp_dir) / f"fake_{start:.3f}.wav"
        p.write_bytes(b"\x00\x01")
        return p

    def fake_compute(wav, dur, **k):
        if abs(dur - 2.5) < 1e-6:   # la scène 2 dure 2.5 s -> on simule un échec
            raise RuntimeError("signal corrompu")
        return _fake_features()

    monkeypatch.setattr(saf, "_extract_scene_wav", fake_extract)
    monkeypatch.setattr(saf, "_compute_features", fake_compute)

    out = tmp_path / "audio_features_demo.json"
    res = saf.build_scene_audio_features("videos/demo.mp4", SCENES, str(out))
    assert res["status"] == "partiel"
    assert res["scenes_ok"] == 2
    art = _read(out)
    by_idx = {s["scene_index"]: s for s in art["scenes"]}
    assert by_idx[2]["audio_status"] == "features_échec" and by_idx[2]["features"] == {}
    assert by_idx[1]["audio_status"] == "ok" and by_idx[3]["audio_status"] == "ok"


# --- Signal synthétique : assertions LÂCHES -----------------------------------------

def test_energy_strong_signal_greater_than_silence():
    sr = saf.DEFAULT_SR
    dur = 5.0  # >= 4 s : tempo calculé
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    strong = (0.5 * np.sin(2 * math.pi * 110 * t)).astype(np.float32)
    silence = np.zeros(int(sr * dur), dtype=np.float32)

    f_strong = saf._features_from_signal(strong, sr, dur)
    f_silence = saf._features_from_signal(silence, sr, dur)

    assert f_strong["energy_rms"] > f_silence["energy_rms"]
    assert f_strong["audio_activity"] == "actif"
    assert f_silence["audio_activity"] == "quasi_silence"
    # tempo : un nombre ou null, jamais autre chose.
    assert f_strong["tempo_bpm"] is None or isinstance(f_strong["tempo_bpm"], float)


def test_empty_signal_raises():
    with pytest.raises(ValueError):
        saf._features_from_signal(np.array([], dtype=np.float32), saf.DEFAULT_SR, 5.0)
