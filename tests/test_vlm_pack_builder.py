import json

import pytest

from src.video_agent import vlm_pack_builder as vpb
from src.video_agent.vlm_pack_builder import VlmPackError


# --- Helpers --------------------------------------------------------------------
def _frames(n):
    return [
        {"frame_id": i, "segment_id": 1, "timestamp_s": float(i), "path": f"f/{i}.jpg"}
        for i in range(1, n + 1)
    ]


def _fusion_segment(seg_id=1, frames_n=8, has_speech=True, ready=True, text="alpha beta"):
    return {
        "segment_id": seg_id,
        "start_s": 0.0,
        "end_s": 10.0,
        "duration_s": 10.0,
        "frames": _frames(frames_n),
        "frame_count": frames_n,
        "motion_grid_path": f"m/s{seg_id}/grid.jpg",
        "motion_composite_path": f"m/s{seg_id}/comp.jpg",
        "transcript_segments": (
            [{"start_s": 0.0, "end_s": 2.0, "text": text, "confidence": 0.6,
              "overlap_s": 2.0, "overlap_ratio": 1.0}]
            if has_speech else []
        ),
        "combined_text": text if has_speech else "",
        "audio_summary": {
            "has_speech": has_speech,
            "speech_segment_count": 1 if has_speech else 0,
            "speech_duration_s": 2.0 if has_speech else 0.0,
            "audio_activity_mode": "active_audio",
        },
        "warnings": [] if has_speech else ["no_speech_overlap"],
        "ready_for_vlm": ready,
    }


def _fusion_manifest(segments):
    return {"video_path": "vidéos/clipX.mp4", "language": "ar", "segments": segments}


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


# --- A. select_frames -----------------------------------------------------------
def test_select_frames_keep_all_when_under_max():
    frames = _frames(3)
    out = vpb.select_frames(frames, 6)
    assert [f["frame_id"] for f in out] == [1, 2, 3]


def test_select_frames_cap_8_to_6_keep_first_last():
    frames = _frames(8)
    out = vpb.select_frames(frames, 6)
    ids = [f["frame_id"] for f in out]
    assert len(out) == 6
    assert ids[0] == 1 and ids[-1] == 8  # première + dernière conservées
    assert ids == sorted(ids)            # ordre croissant
    assert len(ids) == len(set(ids))     # pas de doublons


def test_select_frames_sorted_by_timestamp():
    frames = [
        {"frame_id": 3, "timestamp_s": 9.0, "path": "c"},
        {"frame_id": 1, "timestamp_s": 1.0, "path": "a"},
        {"frame_id": 2, "timestamp_s": 5.0, "path": "b"},
    ]
    out = vpb.select_frames(frames, 6)
    assert [f["timestamp_s"] for f in out] == [1.0, 5.0, 9.0]


def test_select_frames_empty_and_invalid_max():
    assert vpb.select_frames([], 6) == []
    with pytest.raises(VlmPackError):
        vpb.select_frames(_frames(3), 0)


# --- B. _build_instructions -----------------------------------------------------
def test_build_instructions_with_speech():
    instr = vpb._build_instructions("salam alikoum", has_speech=True, language="ar")
    joined = " ".join(instr).lower()
    assert "motion_grid" in joined
    assert "motion_composite" in joined
    assert "actions" in joined
    assert "mouvement" in joined
    assert "audio" in joined and "visuel" in joined
    assert "incertitude" in joined
    assert "bounding box" in joined          # toujours pas de bounding boxes
    assert "texte" in joined and "écran" in joined  # lecture native du texte à l'écran demandée
    # langue DÉTECTÉE injectée, plus de « arabe/Darija » figé
    assert "langue détectée : ar" in " ".join(instr)
    assert "darija" not in joined
    # combined_text injecté
    assert "salam alikoum" in " ".join(instr)


def test_build_instructions_without_speech():
    instr = vpb._build_instructions("", has_speech=False)
    joined = " ".join(instr).lower()
    assert "aucune parole" in joined


# --- C. _build_questions --------------------------------------------------------
def test_build_questions():
    q = " ".join(vpb._build_questions()).lower()
    assert "scène" in q or "scene" in q
    assert "personnes" in q or "objets" in q
    assert "mouvement" in q
    assert "parole" in q or "audio" in q
    assert "confiance" in q or "incertitude" in q


# --- D. _expected_output_schema -------------------------------------------------
def test_expected_output_schema_keys():
    schema = vpb._expected_output_schema()
    for key in [
        "scene_description", "actions", "people_objects", "motion_description",
        "audio_visual_relation", "uses_speech", "confidence", "uncertainties",
    ]:
        assert key in schema


# --- E. _build_segment_pack -----------------------------------------------------
def test_build_segment_pack_fields():
    entry = vpb._build_segment_pack(_fusion_segment(frames_n=8), max_frames=6)
    for key in [
        "segment_id", "start_s", "end_s", "motion_grid_path", "motion_composite_path",
        "selected_frames", "selected_frame_count", "available_frame_count",
        "combined_text", "transcript_segments", "audio_summary", "warnings",
        "instructions_for_vlm", "questions_for_vlm", "expected_output_schema",
    ]:
        assert key in entry
    assert entry["selected_frame_count"] == 6
    assert entry["available_frame_count"] == 8


# --- F. build_vlm_pack ----------------------------------------------------------
def test_build_vlm_pack_summary():
    manifest = _fusion_manifest([
        _fusion_segment(seg_id=1, frames_n=8),
        _fusion_segment(seg_id=2, frames_n=4),
    ])
    pack = vpb.build_vlm_pack(manifest, max_frames=6)
    assert pack["pack_type"] == "vlm_segment_pack"
    assert pack["next_step"] == "vlm_analysis"
    s = pack["summary"]
    assert s["segment_count"] == 2
    assert s["total_selected_frames"] == 6 + 4  # seg1 capé à 6, seg2 gardé 4
    assert s["ready_for_vlm_count"] == 2
    assert s["max_frames_per_segment"] == 6


def test_build_vlm_pack_injects_detected_language():
    # La langue DÉTECTÉE (Whisper, via le manifest) est injectée dans les notes globales
    # et les instructions de segment — plus aucune hypothèse « arabe/Darija » figée.
    manifest = _fusion_manifest([_fusion_segment(seg_id=1, has_speech=True)])
    manifest["language"] = "fr"
    pack = vpb.build_vlm_pack(manifest, max_frames=6)
    notes = " ".join(pack["global_notes"]).lower()
    assert "fr" in notes
    assert "arabe/darija" not in notes and "darija" not in notes
    seg_instr = " ".join(pack["segments"][0]["instructions_for_vlm"])
    assert "langue détectée : fr" in seg_instr
    assert "darija" not in seg_instr.lower()


# --- G. skip_not_ready ----------------------------------------------------------
def test_skip_not_ready_false_keeps_all():
    manifest = _fusion_manifest([
        _fusion_segment(seg_id=1, ready=True),
        _fusion_segment(seg_id=2, ready=False),
    ])
    pack = vpb.build_vlm_pack(manifest, skip_not_ready=False)
    assert pack["summary"]["segment_count"] == 2
    # ready_for_vlm_count compte les segments réellement prêts (1)
    assert pack["summary"]["ready_for_vlm_count"] == 1


def test_skip_not_ready_true_excludes():
    manifest = _fusion_manifest([
        _fusion_segment(seg_id=1, ready=True),
        _fusion_segment(seg_id=2, ready=False),
    ])
    pack = vpb.build_vlm_pack(manifest, skip_not_ready=True)
    assert pack["summary"]["segment_count"] == 1
    assert pack["segments"][0]["segment_id"] == 1


# --- H. build_vlm_pack_from_file ------------------------------------------------
def test_build_vlm_pack_from_file(tmp_path):
    manifest = _fusion_manifest([_fusion_segment(seg_id=1, frames_n=8)])
    path = _write_json(tmp_path / "fusion.json", manifest)
    pack = vpb.build_vlm_pack_from_file(path, max_frames=6)
    assert pack["pack_type"] == "vlm_segment_pack"
    assert pack["source_fusion_manifest"] == path
    assert pack["segments"][0]["selected_frame_count"] == 6
    # arabe préservé via combined_text
    pack2 = vpb.build_vlm_pack_from_file(
        _write_json(tmp_path / "f2.json",
                    _fusion_manifest([_fusion_segment(text="مرحبا")])),
    )
    assert pack2["segments"][0]["combined_text"] == "مرحبا"


# --- I. erreurs -----------------------------------------------------------------
def test_missing_file_raises(tmp_path):
    with pytest.raises(VlmPackError):
        vpb.build_vlm_pack_from_file(str(tmp_path / "nope.json"))


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(VlmPackError):
        vpb.build_vlm_pack_from_file(str(bad))


def test_no_segments_raises(tmp_path):
    path = _write_json(tmp_path / "empty.json", {"video_path": "x", "segments": []})
    with pytest.raises(VlmPackError):
        vpb.build_vlm_pack_from_file(path)
