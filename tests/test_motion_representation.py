import pytest
from PIL import Image

from src.video_agent import motion_representation as mr
from src.video_agent.motion_representation import MotionRepresentationError


# --- Helpers --------------------------------------------------------------------
def _make_image(path, size=(40, 60), color=(120, 80, 200)):
    """Crée une petite image JPG factice (jamais les vraies frames lourdes)."""
    img = Image.new("RGB", size, color)
    img.save(path, format="JPEG", quality=90)
    return str(path)


# --- Cas 1 : layout de grille ---------------------------------------------------
@pytest.mark.parametrize(
    "n, expected",
    [
        (1, (1, 1)),
        (9, (3, 3)),
        (10, (4, 3)),
    ],
)
def test_compute_grid_layout(n, expected):
    assert mr._compute_grid_layout(n) == expected


def test_compute_grid_layout_invalid():
    with pytest.raises(MotionRepresentationError):
        mr._compute_grid_layout(0)


# --- Cas 2 : groupement par segment + tri par timestamp -------------------------
def test_group_frames_by_segment():
    frames = [
        {"frame_id": 5, "segment_id": 2, "timestamp_s": 9.0, "path": "e"},
        {"frame_id": 2, "segment_id": 1, "timestamp_s": 3.4, "path": "b"},
        {"frame_id": 1, "segment_id": 1, "timestamp_s": 1.1, "path": "a"},
        {"frame_id": 4, "segment_id": 2, "timestamp_s": 2.0, "path": "d"},
        {"frame_id": 3, "segment_id": 1, "timestamp_s": 2.2, "path": "c"},
    ]
    groups = mr._group_frames_by_segment(frames)

    # Regroupées par segment_id, dans l'ordre croissant.
    assert list(groups.keys()) == [1, 2]
    # Triées par timestamp_s dans chaque segment.
    assert [f["timestamp_s"] for f in groups[1]] == [1.1, 2.2, 3.4]
    assert [f["timestamp_s"] for f in groups[2]] == [2.0, 9.0]
    # Comptes par segment corrects.
    assert len(groups[1]) == 3
    assert len(groups[2]) == 2


# --- Cas 3 : manifest invalide -> MotionRepresentationError ---------------------
def test_validate_manifest_not_dict():
    with pytest.raises(MotionRepresentationError):
        mr._validate_frames_manifest(["pas un dict"])


def test_validate_manifest_no_frames_key():
    with pytest.raises(MotionRepresentationError):
        mr._validate_frames_manifest({})


def test_validate_manifest_empty_frames():
    with pytest.raises(MotionRepresentationError):
        mr._validate_frames_manifest({"frames": []})


def test_validate_manifest_frame_missing_segment_id():
    with pytest.raises(MotionRepresentationError):
        mr._validate_frames_manifest({"frames": [{"timestamp_s": 1.0, "path": "a"}]})


def test_validate_manifest_frame_missing_timestamp():
    with pytest.raises(MotionRepresentationError):
        mr._validate_frames_manifest({"frames": [{"segment_id": 1, "path": "a"}]})


def test_validate_manifest_frame_missing_path():
    with pytest.raises(MotionRepresentationError):
        mr._validate_frames_manifest({"frames": [{"segment_id": 1, "timestamp_s": 1.0}]})


def test_validate_manifest_valid_returns_list():
    frames = [{"segment_id": 1, "timestamp_s": 1.0, "path": "a"}]
    assert mr._validate_frames_manifest({"frames": frames}) == frames


# --- Cas 4 : redimensionnement conserve le ratio --------------------------------
def test_resize_keep_ratio_target_width():
    img = Image.new("RGB", (200, 100))  # ratio 2:1
    out = mr._resize_keep_ratio(img, target_width=50)
    assert out.size == (50, 25)


def test_resize_keep_ratio_max_long_side_reduces():
    img = Image.new("RGB", (2000, 1000))  # côté long 2000
    out = mr._resize_keep_ratio(img, max_long_side=1080)
    assert max(out.size) == 1080
    # Ratio conservé (2:1).
    assert out.size == (1080, 540)


def test_resize_keep_ratio_max_long_side_no_upscale():
    img = Image.new("RGB", (300, 200))  # déjà sous la limite
    out = mr._resize_keep_ratio(img, max_long_side=1080)
    assert out.size == (300, 200)


# --- Cas 5 : génération complète avec petites images factices -------------------
def test_build_motion_representations_with_fake_images(tmp_path):
    # 3 petites frames factices dans tmp_path (segment unique).
    frames = []
    for i in range(1, 4):
        p = _make_image(tmp_path / f"frame_{i:03d}.jpg", color=(40 * i, 80, 200))
        frames.append(
            {"frame_id": i, "segment_id": 1, "timestamp_s": float(i), "path": p}
        )

    frames_manifest = {"video_path": "videos/clipX.mp4", "frames": frames}
    out_dir = tmp_path / "motion_out"

    manifest = mr.build_motion_representations(
        frames_manifest=frames_manifest,
        output_dir=str(out_dir),
        thumbnail_width=60,
        composite_long_side=120,
        source_frames_manifest_path="some/frames_manifest.json",
    )

    # video_stem dérivé de video_path (générique, pas de 'demo').
    seg_dir = out_dir / "clipX" / "segment_001"
    grid_path = seg_dir / "motion_grid.jpg"
    composite_path = seg_dir / "motion_composite.jpg"

    assert grid_path.exists() and grid_path.stat().st_size > 0
    assert composite_path.exists() and composite_path.stat().st_size > 0

    assert manifest["motion_type"] == "grid_and_transparent_composite"
    assert manifest["summary"]["backend"] == "pillow"
    assert manifest["summary"]["segment_count"] == 1
    assert manifest["summary"]["image_count"] == 2
    assert manifest["next_step"] == "vlm_analysis"
    assert manifest["source_frames_manifest"] == "some/frames_manifest.json"
    assert manifest["segments"][0]["frame_count"] == 3
    assert manifest["segments"][0]["grid_layout"] == {"cols": 2, "rows": 2}


def test_build_single_frame_composite_is_resized_frame(tmp_path):
    # Segment à une seule frame : composite = la frame redimensionnée.
    p = _make_image(tmp_path / "frame_001.jpg", size=(100, 200))
    frames_manifest = {
        "video_path": "videos/solo.mp4",
        "frames": [{"frame_id": 1, "segment_id": 1, "timestamp_s": 0.5, "path": p}],
    }
    out_dir = tmp_path / "motion_out"

    manifest = mr.build_motion_representations(
        frames_manifest=frames_manifest,
        output_dir=str(out_dir),
        composite_long_side=100,
    )

    seg = manifest["segments"][0]
    assert seg["frame_count"] == 1
    assert seg["grid_layout"] == {"cols": 1, "rows": 1}
    assert (out_dir / "solo" / "segment_001" / "motion_composite.jpg").exists()


def test_build_missing_frame_path_raises(tmp_path):
    frames_manifest = {
        "video_path": "videos/x.mp4",
        "frames": [{"frame_id": 1, "segment_id": 1, "timestamp_s": 0.5, "path": str(tmp_path / "nope.jpg")}],
    }
    with pytest.raises(MotionRepresentationError):
        mr.build_motion_representations(
            frames_manifest=frames_manifest, output_dir=str(tmp_path / "out")
        )


def test_build_missing_video_path_raises(tmp_path):
    p = _make_image(tmp_path / "f.jpg")
    frames_manifest = {
        "frames": [{"frame_id": 1, "segment_id": 1, "timestamp_s": 0.5, "path": p}]
    }
    with pytest.raises(MotionRepresentationError):
        mr.build_motion_representations(
            frames_manifest=frames_manifest, output_dir=str(tmp_path / "out")
        )
