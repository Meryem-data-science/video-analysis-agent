import pytest

from src.video_agent import frame_extraction as fe
from src.video_agent.frame_extraction import FrameExtractionError


# --- Cas 1 : dense_sampling sur le segment de la démo (23.033s) ------------------
def test_compute_frame_count_dense_23s():
    assert fe._compute_frame_count(23.033, "dense_sampling") == 10


# --- Cas 2 : garantie « mouvement » -> >= 2 frames si le segment peut les contenir --
@pytest.mark.parametrize("duration", [0.4, 0.5, 0.9, 1.0, 1.415, 1.49])
def test_compute_frame_count_short_now_two_frames(duration):
    # Avant : ces segments courts retournaient 1 frame (composite sans mouvement).
    # Désormais : >= 2 frames distinctes -> le composite encode du mouvement.
    assert fe._compute_frame_count(duration, "dense_sampling") == 2


# --- Fallback HONNÊTE : segment vraiment trop court pour 2 frames distinctes -> 1 ---
@pytest.mark.parametrize("duration", [0.05, 0.2, 0.39])
def test_compute_frame_count_too_short_falls_back_to_one(duration):
    assert fe._compute_frame_count(duration, "dense_sampling") == 1


# --- Seuil configurable (min_frames_per_segment) --------------------------------
def test_compute_frame_count_min_frames_configurable():
    # min=1 : pas de garantie -> on retombe sur le calcul densité/espacement (0.9s -> 1).
    assert fe._compute_frame_count(0.9, "dense_sampling", min_frames_per_segment=1) == 1
    # min=3 : 3 frames si le segment peut les contenir distinctement (0.9s/0.2 = 4 >= 3).
    assert fe._compute_frame_count(0.9, "dense_sampling", min_frames_per_segment=3) == 3
    # min=3 mais segment trop court pour 3 distinctes (0.5s/0.2 = 2) -> plafonné à 2.
    assert fe._compute_frame_count(0.5, "dense_sampling", min_frames_per_segment=3) == 2


# --- Cas 3 : medium_sampling -> entre 4 et 6 ------------------------------------
@pytest.mark.parametrize("duration", [10.0, 23.033, 40.0, 120.0])
def test_compute_frame_count_medium_range(duration):
    n = fe._compute_frame_count(duration, "medium_sampling")
    assert 4 <= n <= 6


# --- Cas 4 : sparse_sampling -> entre 2 et 3 ------------------------------------
@pytest.mark.parametrize("duration", [10.0, 23.033, 40.0, 120.0])
def test_compute_frame_count_sparse_range(duration):
    n = fe._compute_frame_count(duration, "sparse_sampling")
    assert 2 <= n <= 3


# --- Garde anti-suréchantillonnage : segment court ne sur-échantillonne pas -----
def test_compute_frame_count_spacing_guard_short_segment():
    # 2.0s avec dense (min=8) : la garde ~0.5s plafonne à floor(2.0/0.5)=4.
    n = fe._compute_frame_count(2.0, "dense_sampling")
    assert n <= 4
    assert n >= 1


# --- Mode inconnu / stratégie manquante : repli sur medium_sampling -------------
def test_get_frame_strategy_mode_known():
    strategy = {"frame_strategy": {"mode": "dense_sampling"}}
    assert fe._get_frame_strategy_mode(strategy) == "dense_sampling"


@pytest.mark.parametrize(
    "strategy",
    [
        {},
        {"frame_strategy": {}},
        {"frame_strategy": {"mode": "unknown_mode"}},
        None,
        "pas un dict",
    ],
)
def test_get_frame_strategy_mode_fallback(strategy):
    assert fe._get_frame_strategy_mode(strategy) == "medium_sampling"


# --- Cas 5 : timestamps répartis strictement à l'intérieur du segment -----------
def test_compute_timestamps_inside_and_ordered():
    start, end, n = 0.0, 23.033, 10
    ts = fe._compute_timestamps(start, end, n)

    assert len(ts) == n
    # Strictement à l'intérieur : jamais le début ni la fin exacts.
    assert all(start < t < end for t in ts)
    # Ordonnés et strictement croissants.
    assert ts == sorted(ts)
    assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))


def test_compute_timestamps_single_is_center():
    start, end = 10.0, 20.0
    ts = fe._compute_timestamps(start, end, 1)
    assert ts == [15.0]
    assert start < ts[0] < end


def test_compute_timestamps_offset_segment_inside():
    # Segment ne commençant pas à 0 : les timestamps restent dans [start, end].
    start, end, n = 100.0, 130.0, 6
    ts = fe._compute_timestamps(start, end, n)
    assert all(start < t < end for t in ts)
    assert ts == sorted(ts)


# --- Cas 6 : segments invalides -> FrameExtractionError -------------------------
def test_validate_segments_empty_list():
    with pytest.raises(FrameExtractionError):
        fe._validate_segments({"segments": []})


def test_validate_segments_missing_key():
    with pytest.raises(FrameExtractionError):
        fe._validate_segments({})


def test_validate_segments_not_a_dict():
    with pytest.raises(FrameExtractionError):
        fe._validate_segments([{"start_s": 0.0, "end_s": 1.0}])


def test_validate_segments_missing_start():
    with pytest.raises(FrameExtractionError):
        fe._validate_segments({"segments": [{"end_s": 5.0}]})


def test_validate_segments_missing_end():
    with pytest.raises(FrameExtractionError):
        fe._validate_segments({"segments": [{"start_s": 0.0}]})


def test_validate_segments_end_le_start():
    with pytest.raises(FrameExtractionError):
        fe._validate_segments({"segments": [{"start_s": 5.0, "end_s": 5.0}]})
    with pytest.raises(FrameExtractionError):
        fe._validate_segments({"segments": [{"start_s": 5.0, "end_s": 3.0}]})


def test_validate_segments_non_numeric():
    with pytest.raises(FrameExtractionError):
        fe._validate_segments({"segments": [{"start_s": "a", "end_s": 5.0}]})


def test_validate_segments_valid_returns_list():
    seg_list = [{"start_s": 0.0, "end_s": 10.0}, {"start_s": 10.0, "end_s": 20.0}]
    result = fe._validate_segments({"segments": seg_list})
    assert result == seg_list
