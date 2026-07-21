"""Tests de scene_audio — appels Gemini MOCKÉS (hors-ligne, gratuits, rapides).

Aucun vrai appel réseau, aucun ffmpeg réel : on monkeypatch les trois seams
(_init_gemini, _extract_scene_audio, _analyze_scene_call). On vérifie :
- la structure de l'artefact écrit ;
- la dégradation gracieuse par scène (503 simulé sur UNE scène) ;
- le respect de l'override de modèle (argument + $GEMINI_MODEL).
"""
import json
from pathlib import Path

import pytest

from src.video_agent import scene_audio


SCENES = [
    {"index": 1, "start": 0.0, "end": 1.5},
    {"index": 2, "start": 1.5, "end": 4.0},
    {"index": 3, "start": 4.0, "end": 6.0},
]


def _valid_events(tag="ok"):
    """Un dict d'événements audio valide (forme du RESPONSE_SCHEMA)."""
    return {
        "music": {
            "present": True,
            "confidence": 0.8,
            "mood": tag,
            "mood_confidence": 0.7,
            "energy": "moyenne",
            "tempo_feel": "modéré",
        },
        "sound_effects": [{"label": "whoosh", "confidence": 0.6}],
        "speech_overlap": False,
        "evidence": "musique de fond continue",
        "uncertainties": ["incertain sur le genre"],
    }


@pytest.fixture
def patched(monkeypatch):
    """Patche les seams réseau/ffmpeg ; renvoie un dict pour configurer le comportement."""
    state = {"raise_on_start": None, "models_seen": [], "call_count": 0}

    # Pas de vrai client : un sentinel suffit (les appels sont interceptés).
    monkeypatch.setattr(scene_audio, "_init_gemini", lambda: ("CLIENT", "TYPES"))

    # Pas de ffmpeg : on fabrique un petit fichier réel (dont le nom encode le start)
    # pour que unlink fonctionne ET que la scène soit identifiable côté appel.
    def fake_extract(video_path, start, end, tmp_dir):
        p = Path(tmp_dir) / f"fake_{start:.3f}_{end:.3f}.mp3"
        p.write_bytes(b"\x00\x01")
        return p

    monkeypatch.setattr(scene_audio, "_extract_scene_audio", fake_extract)

    # Appel Gemini factice : enregistre le modèle, lève un 503 pour la scène ciblée
    # par son START (identité stable -> tous les retries de cette scène échouent).
    def fake_call(client, types, audio_path, model):
        state["call_count"] += 1
        state["models_seen"].append(model)
        start = float(audio_path.stem.split("_")[1])
        if state["raise_on_start"] is not None and abs(start - state["raise_on_start"]) < 1e-6:
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return _valid_events(tag=f"start{start:.3f}")

    monkeypatch.setattr(scene_audio, "_analyze_scene_call", fake_call)
    return state


def _read_artifact(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Structure de l'artefact --------------------------------------------------------

def test_artifact_structure_all_ok(tmp_path, patched):
    out = tmp_path / "audio_events_demo.json"
    res = scene_audio.build_scene_audio(
        "videos/demo.mp4", SCENES, str(out),
        model="gemini-x", pause_s=0, retry_pause_s=0,
    )

    assert res["status"] == "ok"
    assert res["scenes_ok"] == 3 and res["scenes_total"] == 3
    assert res["artifact_path"] == str(out)
    assert out.is_file()

    art = _read_artifact(out)
    assert art["schema"] == "scene_audio/1.0"
    assert art["video_path"] == "videos/demo.mp4"
    assert art["status"] == "ok"
    assert len(art["scenes"]) == 3
    first = art["scenes"][0]
    assert set(first) == {"scene_index", "start_s", "end_s", "audio_events", "audio_status"}
    assert first["scene_index"] == 1
    assert first["start_s"] == 0.0 and first["end_s"] == 1.5
    assert first["audio_status"] == "ok"
    # Le contenu d'événements respecte la forme attendue.
    assert first["audio_events"]["music"]["present"] is True
    assert "uncertainties" in first["audio_events"]


# --- Dégradation gracieuse par scène ------------------------------------------------

def test_graceful_degradation_one_scene_503(tmp_path, patched):
    patched["raise_on_start"] = 1.5  # 503 (épuise les retries) sur la scène 2 (start=1.5)
    out = tmp_path / "audio_events_demo.json"
    res = scene_audio.build_scene_audio(
        "videos/demo.mp4", SCENES, str(out),
        model="gemini-x", max_retries=2, pause_s=0, retry_pause_s=0,
    )

    assert res["status"] == "partiel"
    assert res["scenes_ok"] == 2 and res["scenes_total"] == 3

    art = _read_artifact(out)
    by_idx = {s["scene_index"]: s for s in art["scenes"]}
    # Seule la scène 2 échoue, malgré les retries ; les autres réussissent.
    assert by_idx[2]["audio_status"] == "échec_gracieux"
    assert by_idx[1]["audio_status"] == "ok"
    assert by_idx[3]["audio_status"] == "ok"
    # La scène en échec a un dict d'événements vide, pas inventé.
    assert by_idx[2]["audio_events"] == {}


def test_retry_then_give_up_makes_scene_fail(tmp_path, patched, monkeypatch):
    """Une scène qui lève 503 à CHAQUE essai finit en échec_gracieux (pas de crash)."""
    # Force toutes les tentatives à lever : la 1re scène échoue après ses retries.
    def always_503(client, types, audio_path, model):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(scene_audio, "_analyze_scene_call", always_503)
    out = tmp_path / "audio_events_demo.json"
    res = scene_audio.build_scene_audio(
        "videos/demo.mp4", SCENES[:1], str(out),
        max_retries=2, pause_s=0, retry_pause_s=0,
    )
    assert res["status"] == "échec"
    assert res["scenes_ok"] == 0
    art = _read_artifact(out)
    assert art["scenes"][0]["audio_status"] == "échec_gracieux"


def test_no_client_all_graceful(tmp_path, monkeypatch):
    """SDK/clé absents -> toutes les scènes en échec_gracieux, statut global échec."""
    monkeypatch.setattr(scene_audio, "_init_gemini", lambda: (None, None))
    out = tmp_path / "audio_events_demo.json"
    res = scene_audio.build_scene_audio(
        "videos/demo.mp4", SCENES, str(out), pause_s=0, retry_pause_s=0,
    )
    assert res["status"] == "échec"
    assert res["scenes_ok"] == 0
    art = _read_artifact(out)
    assert all(s["audio_status"] == "échec_gracieux" for s in art["scenes"])


# --- Override de modèle -------------------------------------------------------------

def test_model_override_explicit_arg(tmp_path, patched, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
    out = tmp_path / "audio_events_demo.json"
    res = scene_audio.build_scene_audio(
        "videos/demo.mp4", SCENES, str(out),
        model="gemini-explicit", pause_s=0, retry_pause_s=0,
    )
    # L'argument explicite l'emporte sur l'env.
    assert res["model_used"] == "gemini-explicit"
    assert _read_artifact(out)["model_used"] == "gemini-explicit"
    assert set(patched["models_seen"]) == {"gemini-explicit"}


def test_model_override_from_env(tmp_path, patched, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
    out = tmp_path / "audio_events_demo.json"
    res = scene_audio.build_scene_audio(
        "videos/demo.mp4", SCENES, str(out), pause_s=0, retry_pause_s=0,
    )
    # Pas d'argument -> $GEMINI_MODEL utilisé.
    assert res["model_used"] == "gemini-from-env"
    assert set(patched["models_seen"]) == {"gemini-from-env"}


def test_default_model_is_25_flash(tmp_path, patched, monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    out = tmp_path / "audio_events_demo.json"
    res = scene_audio.build_scene_audio(
        "videos/demo.mp4", SCENES, str(out), pause_s=0, retry_pause_s=0,
    )
    assert res["model_used"] == scene_audio.DEFAULT_MODEL
    assert scene_audio.DEFAULT_MODEL == "gemini-2.5-flash"
