import importlib.util
import json
from pathlib import Path

import pytest

from src.video_agent import vlm_strategy as vs
from src.video_agent.vlm_strategy import VlmStrategyError


# --- Helpers --------------------------------------------------------------------
def _valid_config(mode="manual"):
    return {
        "mode": mode,
        "provider": "gemini" if mode == "automatic" else None,
        "model": "",
        "input_policy": {
            "use_motion_grid": True,
            "use_motion_composite": True,
            "use_selected_frames": True,
            "use_transcript": True,
            "max_frames_per_segment": 6,
        },
        "output_policy": {
            "require_json": True,
            "include_confidence": True,
            "include_uncertainties": True,
            "no_hallucination": True,
        },
    }


def _valid_pack(n_segments=2):
    segs = []
    for i in range(1, n_segments + 1):
        segs.append(
            {
                "segment_id": i,
                "start_s": float(i),
                "end_s": float(i) + 1.0,
                "motion_grid_path": f"m/s{i}/grid.jpg",
                "motion_composite_path": f"m/s{i}/comp.jpg",
                "selected_frames": [{"frame_id": 1, "timestamp_s": float(i), "path": f"f/{i}.jpg"}],
                "combined_text": "مرحبا",
                "transcript_segments": [{"start_s": float(i), "end_s": float(i) + 0.5, "text": "مرحبا", "confidence": 0.6}],
                "audio_summary": {"has_speech": True},
                "warnings": [],
                "instructions_for_vlm": ["instr motion_grid motion_composite"],
                "questions_for_vlm": ["q?"],
                "expected_output_schema": {"scene_description": "string"},
            }
        )
    return {"video_path": "vidéos/clipX.mp4", "language": "ar", "segments": segs}


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _load_prepare_script():
    """Charge scripts/prepare_vlm_run.py comme module (pour ses helpers purs)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_vlm_run.py"
    spec = importlib.util.spec_from_file_location("prepare_vlm_run_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- A. load_vlm_config ---------------------------------------------------------
def test_load_vlm_config_valid(tmp_path):
    p = _write_json(tmp_path / "cfg.json", _valid_config())
    cfg = vs.load_vlm_config(p)
    assert cfg["mode"] == "manual"


def test_load_vlm_config_missing(tmp_path):
    with pytest.raises(VlmStrategyError):
        vs.load_vlm_config(str(tmp_path / "nope.json"))


def test_load_vlm_config_invalid(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(VlmStrategyError):
        vs.load_vlm_config(str(bad))


# --- B. validate_config ---------------------------------------------------------
def test_validate_config_manual():
    cfg = vs.validate_config(_valid_config("manual"))
    assert cfg["mode"] == "manual" and cfg["provider"] is None


def test_validate_config_disabled():
    assert vs.validate_config(_valid_config("disabled"))["mode"] == "disabled"


def test_validate_config_automatic_gemini():
    cfg = vs.validate_config(_valid_config("automatic"))
    assert cfg["mode"] == "automatic" and cfg["provider"] == "gemini"


def test_validate_config_automatic_requires_provider():
    cfg = _valid_config("automatic")
    cfg["provider"] = None
    with pytest.raises(VlmStrategyError, match="exige un provider explicite"):
        vs.validate_config(cfg)


def test_validate_config_unknown_mode():
    with pytest.raises(VlmStrategyError):
        vs.validate_config(_valid_config("gpt5"))


def test_validate_config_unknown_provider():
    cfg = _valid_config()
    cfg["provider"] = "some_unknown_provider"
    with pytest.raises(VlmStrategyError):
        vs.validate_config(cfg)


def test_validate_config_model_must_be_string():
    cfg = _valid_config()
    cfg["model"] = 123
    with pytest.raises(VlmStrategyError):
        vs.validate_config(cfg)


def test_validate_config_max_frames_positive_int():
    cfg = _valid_config()
    cfg["input_policy"]["max_frames_per_segment"] = 0
    with pytest.raises(VlmStrategyError):
        vs.validate_config(cfg)
    cfg["input_policy"]["max_frames_per_segment"] = "abc"
    with pytest.raises(VlmStrategyError):
        vs.validate_config(cfg)


# --- C. check_pack_exists -------------------------------------------------------
def test_check_pack_exists_valid(tmp_path):
    p = _write_json(tmp_path / "pack.json", _valid_pack())
    pack = vs.check_pack_exists(p)
    assert len(pack["segments"]) == 2


def test_check_pack_exists_missing(tmp_path):
    with pytest.raises(VlmStrategyError):
        vs.check_pack_exists(str(tmp_path / "nope.json"))


def test_check_pack_exists_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(VlmStrategyError):
        vs.check_pack_exists(str(bad))


def test_check_pack_exists_no_segments(tmp_path):
    p = _write_json(tmp_path / "empty.json", {"video_path": "x", "segments": []})
    with pytest.raises(VlmStrategyError):
        vs.check_pack_exists(p)


# --- D. build_vlm_run_plan ------------------------------------------------------
def test_build_vlm_run_plan_fields():
    plan = vs.build_vlm_run_plan(_valid_pack(3), _valid_config())
    assert plan["plan_type"] == "vlm_run_plan"
    assert plan["executed"] is False
    assert plan["mode"] == "manual"
    assert plan["provider"] is None
    assert plan["model"] == ""
    assert plan["segment_count"] == 3
    assert plan["next_step"] == "prepare_vlm_run"
    seg0 = plan["segments"][0]
    assert "motion_grid_path" in seg0
    assert "motion_composite_path" in seg0
    assert "selected_frame_paths" in seg0
    assert "combined_text" in seg0


def test_build_vlm_run_plan_respects_input_policy():
    cfg = _valid_config()
    cfg["input_policy"]["use_transcript"] = False
    cfg["input_policy"]["use_motion_grid"] = False
    plan = vs.build_vlm_run_plan(_valid_pack(1), cfg)
    seg0 = plan["segments"][0]
    assert "motion_grid_path" not in seg0
    assert "combined_text" not in seg0
    assert "motion_composite_path" in seg0  # toujours activé ici


# --- E. prepare_strategy --------------------------------------------------------
def test_prepare_strategy(tmp_path):
    pack = _write_json(tmp_path / "pack.json", _valid_pack(2))
    cfg = _write_json(tmp_path / "cfg.json", _valid_config())
    plan = vs.prepare_strategy(pack, cfg)
    assert plan["plan_type"] == "vlm_run_plan"
    assert plan["segment_count"] == 2
    assert plan["executed"] is False


def test_prepare_strategy_rejects_unknown_mode(tmp_path):
    pack = _write_json(tmp_path / "pack.json", _valid_pack(1))
    cfg = _write_json(tmp_path / "cfg.json", _valid_config("api"))
    with pytest.raises(VlmStrategyError):
        vs.prepare_strategy(pack, cfg)


# --- F. helpers du script prepare_vlm_run.py -----------------------------------
def test_prepare_script_segment_ready():
    mod = _load_prepare_script()
    ready = {"motion_grid_path": "g", "motion_composite_path": "c", "selected_frames": [{"path": "f"}]}
    not_ready = {"motion_grid_path": None, "motion_composite_path": "c", "selected_frames": []}
    assert mod._segment_ready(ready) is True
    assert mod._segment_ready(not_ready) is False


def test_prepare_script_prompt_content():
    mod = _load_prepare_script()
    seg = _valid_pack(1)["segments"][0]
    md = mod._build_prompt_md(seg, "ar")
    assert "Réponds uniquement en JSON valide" in md
    assert "N'invente rien" in md
    assert "expected_output_schema" not in md  # c'est un titre humain, le schéma est inclus en JSON
    assert "scene_description" in md            # schéma JSON présent
    assert "motion_grid" in md
    assert "motion_composite" in md
    assert "مرحبا" in md                        # combined_text arabe préservé
    assert "segment_id : 1" in md
