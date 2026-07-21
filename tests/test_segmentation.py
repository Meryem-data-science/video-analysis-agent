import sys
import types
from pathlib import Path

import pytest

from src.video_agent import segmentation


class _T:
    def __init__(self, s: float):
        self._s = s

    def get_seconds(self):
        return self._s


def _make_fake_scenedetect(scene_list):
    mod = types.ModuleType("scenedetect")

    # detectors submodule with a dummy ContentDetector
    det = types.ModuleType("scenedetect.detectors")

    class ContentDetector:
        def __init__(self, *args, **kwargs):
            pass

    det.ContentDetector = ContentDetector
    mod.detectors = det

    def open_video(path):
        class V:
            def close(self):
                pass

        return V()

    mod.open_video = open_video

    class SceneManager:
        def __init__(self):
            pass

        def add_detector(self, d):
            pass

        def detect_scenes(self, video=None, show_progress=False):
            pass

        def get_scene_list(self, start_in_scene=True):
            return scene_list

    mod.SceneManager = SceneManager
    return mod


def _patch_scenedetect(monkeypatch, scene_list):
    fake = _make_fake_scenedetect(scene_list)
    monkeypatch.setitem(sys.modules, "scenedetect", fake)
    monkeypatch.setitem(sys.modules, "scenedetect.detectors", fake.detectors)


def test_no_scenes_uses_fallback(monkeypatch):
    # scenedetect returns empty list -> fallback segment created
    _patch_scenedetect(monkeypatch, [])

    res = segmentation.segment_video(Path("vidéos/demo.mp4"))
    s = res["summary"]
    assert s["segment_count"] == 1
    assert s["cut_count"] == 0
    assert s["single_scene_video"] is True
    assert s["fallback_used"] is True
    assert isinstance(s["fallback_reason"], str) and "Aucun segment exploitable" in s["fallback_reason"]


def test_single_scene_no_fallback(monkeypatch):
    # scenedetect returns a single scene -> no fallback
    scene = (_T(0.0), _T(23.033))
    _patch_scenedetect(monkeypatch, [scene])

    res = segmentation.segment_video(Path("vidéos/demo.mp4"))
    s = res["summary"]
    assert s["segment_count"] == 1
    assert s["cut_count"] == 0
    assert s["single_scene_video"] is True
    assert s["fallback_used"] is False
    assert s["fallback_reason"] is None


def test_multiple_scenes(monkeypatch):
    # scenedetect returns multiple scenes -> no fallback
    scene1 = (_T(0.0), _T(10.0))
    scene2 = (_T(10.0), _T(20.0))
    _patch_scenedetect(monkeypatch, [scene1, scene2])

    res = segmentation.segment_video(Path("vidéos/demo.mp4"))
    s = res["summary"]
    assert s["segment_count"] == 2
    assert s["cut_count"] == 1
    assert s["single_scene_video"] is False
    assert s["fallback_used"] is False
    assert s["fallback_reason"] is None
