"""Tests narrative_arc — génération de l'arc via Gemini texte MOCKÉ (aucun appel API).

On remplace `gemini_text_runner.analyze_text_with_gemini` par un faux : aucune
dépendance Claude, aucun réseau. Vérifie aussi le fallback (arc vide) si Gemini échoue.
"""
import src.video_agent.narrative_arc as na


def _patch_gemini(monkeypatch, *, returns=None, raises=None, recorder=None):
    """Patche analyze_text_with_gemini importé dans narrative_arc (import différé)."""
    def fake(prompt_text, schema, model=None, timeout_s=180.0):
        if recorder is not None:
            recorder["prompt"] = prompt_text
            recorder["schema"] = schema
            recorder["model"] = model
        if raises is not None:
            raise raises
        return returns
    # narrative_arc fait `from .gemini_text_runner import analyze_text_with_gemini`
    monkeypatch.setattr(
        "src.video_agent.gemini_text_runner.analyze_text_with_gemini", fake
    )


def _vlm(seg_id):
    return {
        "segment_id": seg_id,
        "scene_description": f"scène {seg_id}",
        "people_objects": ["enfant"],
        "actions": ["jouer"],
        "mood": ["joie"],
    }


def _fusion():
    return {
        "video_path": "v.mp4",
        "language": "ar",
        "segments": [
            {"segment_id": 1, "start_s": 0.0, "end_s": 5.0, "combined_text": "مرحبا"},
            {"segment_id": 2, "start_s": 5.0, "end_s": 9.0, "combined_text": ""},
        ],
    }


def _valid_arc():
    return {
        "arc_narratif": {
            "situation_initiale": "s", "element_declencheur": "e",
            "developpement": "d", "climax": "c", "denouement": "f",
        },
        "arc_emotionnel": "trajectoire",
        "resume_humain": "résumé",
        "arc_confidence": 0.8,
        "arc_evidence": [1, 2],
    }


# --- build_arc_schema ----------------------------------------------------------
def test_build_arc_schema_required():
    s = na.build_arc_schema()
    for k in ["arc_narratif", "arc_emotionnel", "resume_humain", "arc_confidence", "arc_evidence"]:
        assert k in s["required"]
    arc = s["properties"]["arc_narratif"]
    for k in ["situation_initiale", "element_declencheur", "developpement", "climax", "denouement"]:
        assert k in arc["required"]
        assert arc["properties"][k]["type"] == "string"
    assert s["properties"]["arc_evidence"]["items"]["type"] == "integer"
    assert s["properties"]["arc_confidence"]["type"] == "number"


def test_arc_schema_includes_genre():
    s = na.build_arc_schema()
    assert "genre" in s["required"]
    assert s["properties"]["genre"]["type"] == "array"
    assert s["properties"]["genre"]["items"]["type"] == "string"


def test_empty_arc_has_genre():
    assert na._empty_arc()["genre"] == []


def test_coerce_arc_genre_normalized():
    assert na._coerce_arc({"genre": ["Sport", "sport", "famille", 3, ""]})["genre"] == ["Sport", "famille"]


def test_arc_prompt_mentions_genre():
    digest = na._ordered_scene_digest({1: _vlm(1)}, _fusion())
    assert "genre" in na.build_arc_prompt(digest, "ar").lower()


def test_generate_arc_includes_genre(monkeypatch):
    arc = _valid_arc(); arc["genre"] = ["humour", "famille"]
    _patch_gemini(monkeypatch, returns=arc)
    assert na.generate_narrative_arc({1: _vlm(1)}, _fusion())["genre"] == ["humour", "famille"]


# --- build_arc_prompt ----------------------------------------------------------
def test_build_arc_prompt_anti_hallucination_and_json():
    digest = na._ordered_scene_digest({1: _vlm(1), 2: _vlm(2)}, _fusion())
    p = na.build_arc_prompt(digest, "ar")
    low = p.lower()
    assert "invente rien" in low or "n'invente" in low
    assert "json" in low
    assert "ne traduis pas" in low
    assert "scène 1" in low
    assert "scène 2" in low
    assert "مرحبا" in p  # parole conservée telle quelle


def test_build_arc_prompt_speech_attribution_rule():
    digest = na._ordered_scene_digest({1: _vlm(1)}, _fusion())
    p = na.build_arc_prompt(digest, "ar")
    low = p.lower()
    assert "diarisation" in low
    assert "on entend" in low


def test_ordered_scene_digest_empty():
    assert na._ordered_scene_digest({}, _fusion()) == []


# --- generate_narrative_arc (Gemini texte mocké) -------------------------------
def test_generate_arc_success(monkeypatch):
    rec = {}
    _patch_gemini(monkeypatch, returns=_valid_arc(), recorder=rec)
    out = na.generate_narrative_arc({1: _vlm(1), 2: _vlm(2)}, _fusion(), model="gemini-2.5-flash")
    assert out["arc_confidence"] == 0.8
    assert out["arc_evidence"] == [1, 2]
    assert out["arc_narratif"]["climax"] == "c"
    # Le modèle est transmis et le prompt (texte seul) contient le digest des scènes.
    assert rec["model"] == "gemini-2.5-flash"
    assert "arc_narratif" in rec["schema"]["required"]
    assert "scène 1" in rec["prompt"].lower()


def test_generate_arc_malformed_returns_empty(monkeypatch):
    # Gemini lève (réponse non exploitable) -> arc vide, jamais de plantage.
    from src.video_agent.gemini_text_runner import GeminiTextError
    _patch_gemini(monkeypatch, raises=GeminiTextError("boom"))
    out = na.generate_narrative_arc({1: _vlm(1)}, _fusion())
    assert out == na._empty_arc()


def test_generate_arc_unconfigured_returns_empty(monkeypatch):
    # Gemini non configuré (clé absente) -> exception attrapée -> arc vide.
    from src.video_agent.gemini_text_runner import GeminiTextError
    _patch_gemini(monkeypatch, raises=GeminiTextError("GEMINI_API_KEY absente"))
    out = na.generate_narrative_arc({1: _vlm(1)}, _fusion())
    assert out == na._empty_arc()


def test_generate_arc_empty_scenes_no_call(monkeypatch):
    # Court-circuit : scènes vides -> aucun appel Gemini.
    def boom(*a, **k):
        raise AssertionError("Gemini ne doit pas être appelé sans scènes")
    monkeypatch.setattr("src.video_agent.gemini_text_runner.analyze_text_with_gemini", boom)
    assert na.generate_narrative_arc({}, _fusion()) == na._empty_arc()


def test_generate_arc_partial_coerced(monkeypatch):
    partial = {"resume_humain": "ok", "arc_evidence": [1, "x", 2.0]}
    _patch_gemini(monkeypatch, returns=partial)
    out = na.generate_narrative_arc({1: _vlm(1), 2: _vlm(2)}, _fusion())
    assert out["resume_humain"] == "ok"
    assert out["arc_evidence"] == [1, 2]
    assert out["arc_narratif"] == {k: "" for k in na._ARC_SUBKEYS}
    assert out["arc_confidence"] == 0.0
