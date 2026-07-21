import json

import pytest

from src.video_agent import audio_visual_fusion as avf
from src.video_agent.audio_visual_fusion import AudioVisualFusionError


# --- Helpers --------------------------------------------------------------------
def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _segments_manifest():
    return {
        "video_path": "vidéos/clipX.mp4",
        "segments": [
            {"segment_id": 1, "start_s": 0.0, "end_s": 10.0, "duration_s": 10.0},
            {"segment_id": 2, "start_s": 10.0, "end_s": 16.0, "duration_s": 6.0},
        ],
    }


def _frames_manifest():
    return {
        "video_path": "vidéos/clipX.mp4",
        "frames": [
            {"frame_id": 2, "segment_id": 1, "timestamp_s": 5.0, "path": "f/s1/2.jpg"},
            {"frame_id": 1, "segment_id": 1, "timestamp_s": 1.0, "path": "f/s1/1.jpg"},
            {"frame_id": 3, "segment_id": 2, "timestamp_s": 12.0, "path": "f/s2/3.jpg"},
        ],
    }


def _motion_manifest():
    return {
        "video_path": "vidéos/clipX.mp4",
        "segments": [
            {"segment_id": 1, "grid_path": "m/s1/grid.jpg", "composite_path": "m/s1/comp.jpg"},
            {"segment_id": 2, "grid_path": "m/s2/grid.jpg", "composite_path": "m/s2/comp.jpg"},
        ],
    }


def _transcript_manifest():
    return {
        "summary": {"detected_language": "ar"},
        "segments": [
            {"start_s": 0.0, "end_s": 2.0, "text": "alpha", "confidence": 0.6},
            {"start_s": 3.0, "end_s": 6.0, "text": "beta", "confidence": 0.6},
            {"start_s": 10.0, "end_s": 13.0, "text": "gamma", "confidence": 0.6},
        ],
    }


def _audio_light_manifest():
    return {"classification": {"mode": "active_audio"}}


# --- A. compute_overlap ---------------------------------------------------------
def test_overlap_total():
    assert avf.compute_overlap(0, 10, 2, 6) == (4.0, 1.0)


def test_overlap_partial():
    overlap_s, ratio = avf.compute_overlap(0, 10, 8, 12)  # chevauche 8..10
    assert overlap_s == 2.0
    assert ratio == pytest.approx(0.5)


def test_overlap_none():
    assert avf.compute_overlap(0, 10, 11, 13) == (0.0, 0.0)


def test_overlap_boundary_touching():
    # 0-10 et 10-13 : bornes jointives -> overlap nul
    assert avf.compute_overlap(0, 10, 10, 13) == (0.0, 0.0)


def test_overlap_zero_duration_transcript():
    overlap_s, ratio = avf.compute_overlap(0, 10, 5, 5)
    assert ratio == 0.0


# --- B. align_transcript_to_visual_segments ------------------------------------
def test_align_single_segment():
    vsegs = _segments_manifest()["segments"]
    tsegs = [{"start_s": 1.0, "end_s": 2.0, "text": "x"}]
    aligned = avf.align_transcript_to_visual_segments(vsegs, tsegs)
    assert [t["text"] for t in aligned[1]] == ["x"]
    assert aligned[2] == []


def test_align_spanning_two_segments():
    vsegs = _segments_manifest()["segments"]
    tsegs = [{"start_s": 8.0, "end_s": 12.0, "text": "span"}]  # chevauche V1 et V2
    aligned = avf.align_transcript_to_visual_segments(vsegs, tsegs)
    assert [t["text"] for t in aligned[1]] == ["span"]
    assert [t["text"] for t in aligned[2]] == ["span"]
    assert aligned[1][0]["overlap_s"] == 2.0  # 8..10
    assert aligned[2][0]["overlap_s"] == 2.0  # 10..12


def test_align_no_overlap():
    vsegs = _segments_manifest()["segments"]
    tsegs = [{"start_s": 16.0, "end_s": 18.0, "text": "out"}]  # 16.0 = borne fin V2
    aligned = avf.align_transcript_to_visual_segments(vsegs, tsegs)
    assert aligned[1] == []
    assert aligned[2] == []


def test_align_sorted_by_start():
    vsegs = _segments_manifest()["segments"]
    tsegs = [
        {"start_s": 6.0, "end_s": 7.0, "text": "b"},
        {"start_s": 1.0, "end_s": 2.0, "text": "a"},
    ]
    aligned = avf.align_transcript_to_visual_segments(vsegs, tsegs)
    assert [t["text"] for t in aligned[1]] == ["a", "b"]


# --- C. _frames_by_segment ------------------------------------------------------
def test_frames_by_segment_indexed_and_sorted():
    by_seg = avf._frames_by_segment(_frames_manifest())
    assert set(by_seg.keys()) == {1, 2}
    # triées par timestamp
    assert [f["frame_id"] for f in by_seg[1]] == [1, 2]
    for f in by_seg[1]:
        assert set(f.keys()) == {"frame_id", "timestamp_s", "path"}


# --- D. _motion_by_segment ------------------------------------------------------
def test_motion_by_segment():
    by_seg = avf._motion_by_segment(_motion_manifest())
    assert by_seg[1]["motion_grid_path"] == "m/s1/grid.jpg"
    assert by_seg[1]["motion_composite_path"] == "m/s1/comp.jpg"
    assert set(by_seg.keys()) == {1, 2}


# --- E. build_fusion_manifest ---------------------------------------------------
def test_build_fusion_manifest():
    vsegs = _segments_manifest()["segments"]
    frames_by = avf._frames_by_segment(_frames_manifest())
    motion_by = avf._motion_by_segment(_motion_manifest())
    aligned = avf.align_transcript_to_visual_segments(
        vsegs, _transcript_manifest()["segments"]
    )
    manifest = avf.build_fusion_manifest(
        video_path="vidéos/clipX.mp4",
        visual_segments=vsegs,
        frames_by_seg=frames_by,
        motion_by_seg=motion_by,
        aligned=aligned,
        language="ar",
        audio_activity_mode="active_audio",
        sources={"segments_manifest": "x"},
    )
    assert manifest["fusion_type"] == "timestamp_aligned_audio_visual"
    assert manifest["next_step"] == "vlm_analysis"
    s = manifest["summary"]
    assert s["segment_count"] == 2
    assert s["total_frames"] == 3
    # V1 a alpha+beta (2), V2 a gamma (1) -> 3
    assert s["speech_segment_count"] == 3
    assert s["ready_for_vlm_count"] == 2
    # combined_text V1
    seg1 = manifest["segments"][0]
    assert seg1["combined_text"] == "alpha beta"


# --- F & G. warnings + ready_for_vlm -------------------------------------------
def test_no_speech_overlap_warning_but_ready():
    vseg = {"segment_id": 1, "start_s": 0.0, "end_s": 10.0}
    entry = avf._build_segment_entry(
        vseg,
        [{"frame_id": 1, "timestamp_s": 1.0, "path": "a.jpg"}],
        {"motion_grid_path": "g.jpg", "motion_composite_path": "c.jpg"},
        [],  # aucun transcript
        "silence",
    )
    assert "no_speech_overlap" in entry["warnings"]
    assert entry["ready_for_vlm"] is True  # parole optionnelle
    assert entry["audio_summary"]["has_speech"] is False


def test_low_overlap_warning():
    vseg = {"segment_id": 1, "start_s": 0.0, "end_s": 10.0}
    entry = avf._build_segment_entry(
        vseg,
        [{"frame_id": 1, "timestamp_s": 1.0, "path": "a.jpg"}],
        {"motion_grid_path": "g.jpg", "motion_composite_path": "c.jpg"},
        [{"start_s": 0.0, "end_s": 8.0, "text": "x", "confidence": 0.5,
          "overlap_s": 1.0, "overlap_ratio": 0.125}],
        "active_audio",
    )
    assert "low_overlap" in entry["warnings"]


def test_no_frames_makes_not_ready():
    vseg = {"segment_id": 1, "start_s": 0.0, "end_s": 10.0}
    entry = avf._build_segment_entry(
        vseg, [], {"motion_grid_path": "g.jpg", "motion_composite_path": "c.jpg"}, [], None
    )
    assert "no_frames" in entry["warnings"]
    assert entry["ready_for_vlm"] is False


def test_missing_motion_makes_not_ready():
    vseg = {"segment_id": 1, "start_s": 0.0, "end_s": 10.0}
    entry = avf._build_segment_entry(
        vseg,
        [{"frame_id": 1, "timestamp_s": 1.0, "path": "a.jpg"}],
        {"motion_grid_path": None, "motion_composite_path": None},
        [],
        None,
    )
    assert "missing_motion_images" in entry["warnings"]
    assert entry["ready_for_vlm"] is False


# --- H. fuse_audio_visual (manifests factices dans tmp_path) --------------------
def test_fuse_audio_visual_full(tmp_path):
    seg = _write_json(tmp_path / "seg.json", _segments_manifest())
    frm = _write_json(tmp_path / "frm.json", _frames_manifest())
    mot = _write_json(tmp_path / "mot.json", _motion_manifest())
    al = _write_json(tmp_path / "al.json", _audio_light_manifest())
    tr = _write_json(tmp_path / "tr.json", _transcript_manifest())

    manifest = avf.fuse_audio_visual(seg, frm, mot, al, tr)
    assert manifest["language"] == "ar"
    assert manifest["summary"]["audio_activity_mode"] == "active_audio"
    assert manifest["summary"]["segment_count"] == 2
    assert manifest["summary"]["total_frames"] == 3
    assert manifest["next_step"] == "vlm_analysis"


def test_fuse_audio_visual_without_audio(tmp_path):
    seg = _write_json(tmp_path / "seg.json", _segments_manifest())
    frm = _write_json(tmp_path / "frm.json", _frames_manifest())
    mot = _write_json(tmp_path / "mot.json", _motion_manifest())

    manifest = avf.fuse_audio_visual(seg, frm, mot)  # sans audio_light ni transcript
    assert manifest["language"] is None
    assert manifest["summary"]["speech_segment_count"] == 0
    assert manifest["summary"]["segments_without_speech"] == 2
    # visuels présents -> toujours prêts pour le VLM
    assert manifest["summary"]["ready_for_vlm_count"] == 2
    for s in manifest["segments"]:
        assert "no_speech_overlap" in s["warnings"]


# --- I. erreurs -----------------------------------------------------------------
def test_missing_file_raises(tmp_path):
    with pytest.raises(AudioVisualFusionError):
        avf.fuse_audio_visual(
            str(tmp_path / "nope.json"),
            str(tmp_path / "nope2.json"),
            str(tmp_path / "nope3.json"),
        )


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    frm = _write_json(tmp_path / "frm.json", _frames_manifest())
    mot = _write_json(tmp_path / "mot.json", _motion_manifest())
    with pytest.raises(AudioVisualFusionError):
        avf.fuse_audio_visual(str(bad), frm, mot)


def test_empty_segments_raises(tmp_path):
    seg = _write_json(tmp_path / "seg.json", {"video_path": "x", "segments": []})
    frm = _write_json(tmp_path / "frm.json", _frames_manifest())
    mot = _write_json(tmp_path / "mot.json", _motion_manifest())
    with pytest.raises(AudioVisualFusionError):
        avf.fuse_audio_visual(seg, frm, mot)
