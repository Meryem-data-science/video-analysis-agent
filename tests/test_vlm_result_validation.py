import json

import pytest

from src.video_agent import vlm_result_validation as vrv
from src.video_agent.vlm_result_validation import VlmResultValidationError


# --- Helpers --------------------------------------------------------------------
def _valid_result(segment_id=1):
    return {
        "segment_id": segment_id,
        "start_s": float((segment_id - 1) * 10),
        "end_s": float(segment_id * 10),
        "combined_text": "مرحبا",
        "scene_description": "deux enfants jouent dans une pièce",
        "people_objects": ["enfants", "canapé"],
        "actions": ["jouer"],
        "motion_description": "mouvement vers la gauche",
        "audio_visual_relation": "la voix accompagne la scène",
        "uses_speech": True,
        "mood": ["joie", "tendresse"],
        "texte_ecran": [{"text": "BONJOUR", "lang": "fr", "location": "centre"}],
        "confidence": 0.6,
        "uncertainties": ["identité des personnes incertaine"],
        "evidence": {"visual": ["frame_001"], "audio": ["parole 0-2s"]},
    }


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _segment_pack(combined_text="مرحبا"):
    return {"segment_id": 1, "combined_text": combined_text}


# --- load_json ------------------------------------------------------------------
def test_load_json_valid(tmp_path):
    p = _write_json(tmp_path / "r.json", {"a": 1})
    assert vrv.load_json(p, "résultat") == {"a": 1}


def test_load_json_missing(tmp_path):
    with pytest.raises(VlmResultValidationError):
        vrv.load_json(str(tmp_path / "nope.json"), "résultat")


def test_load_json_invalid(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(VlmResultValidationError):
        vrv.load_json(str(bad), "résultat")


# --- validate_result_schema : chaque règle bloquante ---------------------------
def test_schema_valid_no_errors():
    assert vrv.validate_result_schema(_valid_result()) == []


def test_schema_not_a_dict():
    assert vrv.validate_result_schema(["x"]) != []


@pytest.mark.parametrize(
    "key",
    [
        "scene_description", "people_objects", "actions", "motion_description",
        "audio_visual_relation", "uses_speech", "mood", "confidence", "uncertainties",
        "evidence", "segment_id", "start_s", "end_s", "combined_text", "texte_ecran",
    ],
)
def test_schema_missing_required_field(key):
    r = _valid_result()
    del r[key]
    errors = vrv.validate_result_schema(r)
    assert any(key in e for e in errors)


def test_schema_mood_wrong_type():
    r = _valid_result(); r["mood"] = "joie"  # chaîne au lieu d'une liste
    assert any("mood" in e for e in vrv.validate_result_schema(r))


def test_schema_mood_empty_ok():
    r = _valid_result(); r["mood"] = []  # liste vide -> valide (toujours présent)
    assert vrv.validate_result_schema(r) == []


def test_schema_scene_description_wrong_type():
    r = _valid_result(); r["scene_description"] = 123
    assert any("scene_description" in e for e in vrv.validate_result_schema(r))


def test_schema_people_objects_wrong_type():
    r = _valid_result(); r["people_objects"] = "pas une liste"
    assert any("people_objects" in e for e in vrv.validate_result_schema(r))


def test_schema_actions_wrong_type():
    r = _valid_result(); r["actions"] = {}
    assert any("actions" in e for e in vrv.validate_result_schema(r))


def test_schema_motion_description_wrong_type():
    r = _valid_result(); r["motion_description"] = 5
    assert any("motion_description" in e for e in vrv.validate_result_schema(r))


def test_schema_audio_visual_relation_wrong_type():
    r = _valid_result(); r["audio_visual_relation"] = 0
    assert any("audio_visual_relation" in e for e in vrv.validate_result_schema(r))


def test_schema_uses_speech_not_bool():
    r = _valid_result(); r["uses_speech"] = "true"
    assert any("uses_speech" in e for e in vrv.validate_result_schema(r))


def test_schema_confidence_not_number():
    r = _valid_result(); r["confidence"] = "0.6"
    assert any("confidence" in e for e in vrv.validate_result_schema(r))


def test_schema_confidence_bool_rejected():
    # bool est sous-classe de int : doit être refusé comme nombre.
    r = _valid_result(); r["confidence"] = True
    assert any("confidence" in e for e in vrv.validate_result_schema(r))


def test_schema_uncertainties_wrong_type():
    r = _valid_result(); r["uncertainties"] = "aucune"
    assert any("uncertainties" in e for e in vrv.validate_result_schema(r))


def test_schema_evidence_not_dict():
    r = _valid_result(); r["evidence"] = ["x"]
    assert any("evidence" in e for e in vrv.validate_result_schema(r))


def test_schema_evidence_visual_missing_or_wrong():
    r = _valid_result(); r["evidence"] = {"audio": []}
    assert any("evidence.visual" in e for e in vrv.validate_result_schema(r))
    r2 = _valid_result(); r2["evidence"] = {"visual": "x", "audio": []}
    assert any("evidence.visual" in e for e in vrv.validate_result_schema(r2))


def test_schema_evidence_audio_missing_or_wrong():
    r = _valid_result(); r["evidence"] = {"visual": []}
    assert any("evidence.audio" in e for e in vrv.validate_result_schema(r))
    r2 = _valid_result(); r2["evidence"] = {"visual": [], "audio": {}}
    assert any("evidence.audio" in e for e in vrv.validate_result_schema(r2))


# --- check_segment_id -----------------------------------------------------------
def test_check_segment_id_match():
    assert vrv.check_segment_id(_valid_result(1), 1) == []


def test_check_segment_id_mismatch():
    assert vrv.check_segment_id(_valid_result(2), 1) != []


def test_check_segment_id_missing():
    r = _valid_result(); del r["segment_id"]
    assert vrv.check_segment_id(r, 1) != []


def test_check_segment_id_not_int():
    r = _valid_result(); r["segment_id"] = "abc"
    assert vrv.check_segment_id(r, 1) != []


# --- check_confidence_range -----------------------------------------------------
def test_confidence_in_range():
    assert vrv.check_confidence_range(_valid_result()) == []


@pytest.mark.parametrize("conf", [-0.1, 1.1, 2.0])
def test_confidence_out_of_range(conf):
    r = _valid_result(); r["confidence"] = conf
    assert vrv.check_confidence_range(r) != []


# --- check_evidence (warnings) --------------------------------------------------
def test_evidence_visual_empty_warns():
    r = _valid_result(); r["evidence"] = {"visual": [], "audio": ["x"]}
    warns = vrv.check_evidence(r)
    assert any("visual" in w for w in warns)


def test_evidence_audio_empty_with_speech_warns():
    r = _valid_result(); r["uses_speech"] = True
    r["evidence"] = {"visual": ["x"], "audio": []}
    warns = vrv.check_evidence(r)
    assert any("audio" in w for w in warns)


def test_evidence_audio_empty_without_speech_ok():
    r = _valid_result(); r["uses_speech"] = False
    r["evidence"] = {"visual": ["x"], "audio": []}
    warns = vrv.check_evidence(r)
    assert not any("audio" in w for w in warns)


# --- check_anti_hallucination_flags (warnings) ---------------------------------
def test_high_confidence_no_uncertainties_warns():
    r = _valid_result(); r["confidence"] = 0.95; r["uncertainties"] = []
    warns = vrv.check_anti_hallucination_flags(r)
    assert any("uncertainties" in w for w in warns)


def test_assertive_scene_without_evidence_warns():
    r = _valid_result()
    r["scene_description"] = "il y a clairement un chien"
    r["evidence"] = {"visual": [], "audio": []}
    warns = vrv.check_anti_hallucination_flags(r)
    assert any("affirmative" in w.lower() for w in warns)


def test_strong_relation_without_audio_evidence_warns():
    r = _valid_result()
    r["audio_visual_relation"] = "la parole prouve la scène"
    r["evidence"] = {"visual": ["x"], "audio": []}
    warns = vrv.check_anti_hallucination_flags(r)
    assert any("relation forte" in w for w in warns)


def test_pack_combined_text_empty_but_uses_speech_warns():
    r = _valid_result(); r["uses_speech"] = True
    warns = vrv.check_anti_hallucination_flags(r, segment_pack=_segment_pack(combined_text=""))
    assert any("uses_speech=true" in w for w in warns)


def test_pack_combined_text_present_but_not_uses_speech_warns():
    r = _valid_result(); r["uses_speech"] = False
    warns = vrv.check_anti_hallucination_flags(r, segment_pack=_segment_pack(combined_text="مرحبا"))
    assert any("uses_speech=false" in w for w in warns)


# --- validate_single_result -----------------------------------------------------
def test_validate_single_valid():
    rep = vrv.validate_single_result(_valid_result(1), 1)
    assert rep["valid"] is True
    assert rep["errors"] == []
    assert rep["confidence"] == 0.6


def test_validate_single_invalid():
    bad = {"segment_id": 2, "confidence": 1.5}  # mauvais id, champs manquants, conf KO
    rep = vrv.validate_single_result(bad, 1)
    assert rep["valid"] is False
    assert len(rep["errors"]) > 0


def test_validate_single_warning_but_valid():
    r = _valid_result(1)
    r["confidence"] = 0.95
    r["uncertainties"] = []
    r["evidence"] = {"visual": [], "audio": []}
    rep = vrv.validate_single_result(r, 1)
    assert rep["valid"] is True
    assert len(rep["warnings"]) > 0


# --- validate_results_dir -------------------------------------------------------
def _pack(n=2):
    return {"segments": [
        {"segment_id": i, "start_s": float((i - 1) * 10), "end_s": float(i * 10),
         "combined_text": "مرحبا"}
        for i in range(1, n + 1)
    ]}


def test_validate_results_dir_all_present(tmp_path):
    _write_json(tmp_path / "vlm_result_segment_001.json", _valid_result(1))
    _write_json(tmp_path / "vlm_result_segment_002.json", _valid_result(2))
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["validation_type"] == "vlm_results_validation"
    assert out["segment_count"] == 2
    assert out["valid_count"] == 2
    assert out["invalid_count"] == 0
    assert out["status"] == "complete"
    assert out["next_step"] is None
    assert out["summary"]["segments_missing_results"] == []


def test_validate_results_dir_missing_segment(tmp_path):
    _write_json(tmp_path / "vlm_result_segment_001.json", _valid_result(1))
    # segment 2 absent
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["valid_count"] == 1
    assert out["invalid_count"] == 1
    assert out["status"] == "partial"
    assert out["summary"]["segments_missing_results"] == [2]
    assert 2 in out["summary"]["segments_with_errors"]


def test_validate_results_dir_invalid_json(tmp_path):
    _write_json(tmp_path / "vlm_result_segment_001.json", _valid_result(1))
    (tmp_path / "vlm_result_segment_002.json").write_text("{bad", encoding="utf-8")
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["valid_count"] == 1
    assert out["invalid_count"] == 1
    assert out["status"] == "partial"
    assert 2 in out["summary"]["segments_with_errors"]


def test_validate_results_dir_empty_pack(tmp_path):
    with pytest.raises(VlmResultValidationError):
        vrv.validate_results_dir(str(tmp_path), {"segments": []})


def test_validate_results_dir_average_confidence(tmp_path):
    r1 = _valid_result(1); r1["confidence"] = 0.4
    r2 = _valid_result(2); r2["confidence"] = 0.8
    _write_json(tmp_path / "vlm_result_segment_001.json", r1)
    _write_json(tmp_path / "vlm_result_segment_002.json", r2)
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["summary"]["average_confidence"] == pytest.approx(0.6)


def test_all_segments_invalid_is_failed(tmp_path):
    r1 = _valid_result(1); r1["scene_description"] = ""
    r2 = _valid_result(2); r2["actions"] = [42]
    _write_json(tmp_path / "vlm_result_segment_001.json", r1)
    _write_json(tmp_path / "vlm_result_segment_002.json", r2)
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["status"] == "failed"
    assert out["valid_segment_count"] == 0
    assert out["invalid_segment_count"] == 2
    assert out["next_step"]["action"] == "rerun_vlm_analysis"


def test_extra_result_is_explicit_warning(tmp_path):
    _write_json(tmp_path / "vlm_result_segment_001.json", _valid_result(1))
    _write_json(tmp_path / "vlm_result_segment_002.json", _valid_result(2))
    _write_json(tmp_path / "vlm_result_segment_099.json", _valid_result(99))
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["status"] == "complete"
    assert out["extra_segments"] == [99]
    assert any("supplémentaires" in warning for warning in out["global_warnings"])


def test_duplicate_segment_files_are_invalid(tmp_path):
    _write_json(tmp_path / "vlm_result_segment_001.json", _valid_result(1))
    _write_json(tmp_path / "vlm_result_segment_0001.json", _valid_result(1))
    _write_json(tmp_path / "vlm_result_segment_002.json", _valid_result(2))
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["status"] == "partial"
    assert 1 in out["duplicate_segments"]
    assert 1 in out["invalid_segments"]


def test_required_text_empty_is_invalid(tmp_path):
    bad = _valid_result(1); bad["motion_description"] = "   "
    _write_json(tmp_path / "vlm_result_segment_001.json", bad)
    out = vrv.validate_results_dir(str(tmp_path), _pack(1))
    assert out["status"] == "failed"
    assert any("texte obligatoire vide" in e for e in out["errors_by_segment"]["1"])


def test_wrong_item_type_in_list_is_invalid(tmp_path):
    bad = _valid_result(1); bad["people_objects"] = ["enfant", {"objet": "ballon"}]
    _write_json(tmp_path / "vlm_result_segment_001.json", bad)
    out = vrv.validate_results_dir(str(tmp_path), _pack(1))
    assert out["status"] == "failed"
    assert any("people_objects[1]" in e for e in out["errors_by_segment"]["1"])


def test_timestamp_mismatch_is_invalid(tmp_path):
    bad = _valid_result(1); bad["end_s"] = 9.5
    _write_json(tmp_path / "vlm_result_segment_001.json", bad)
    out = vrv.validate_results_dir(str(tmp_path), _pack(1))
    assert out["status"] == "failed"
    assert any("end_s non concordant" in e for e in out["errors_by_segment"]["1"])


def test_combined_text_mismatch_is_invalid(tmp_path):
    bad = _valid_result(1); bad["combined_text"] = "texte réécrit"
    _write_json(tmp_path / "vlm_result_segment_001.json", bad)
    out = vrv.validate_results_dir(str(tmp_path), _pack(1))
    assert out["status"] == "failed"
    assert any("combined_text" in e for e in out["errors_by_segment"]["1"])


def test_duplicate_payload_segment_id_invalidates_both_files(tmp_path):
    _write_json(tmp_path / "vlm_result_segment_001.json", _valid_result(1))
    duplicate = _valid_result(2); duplicate["segment_id"] = 1
    _write_json(tmp_path / "vlm_result_segment_002.json", duplicate)
    out = vrv.validate_results_dir(str(tmp_path), _pack(2))
    assert out["status"] == "failed"
    assert out["duplicate_segments"] == [1, 2]
    assert any("segment_id dupliqué" in e for e in out["global_errors"])
