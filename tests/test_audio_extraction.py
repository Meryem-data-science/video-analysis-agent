from pathlib import Path

import pytest

from src.video_agent import audio_extraction as ae
from src.video_agent.audio_extraction import AudioExtractionError


# --- Cas 1 : derive video_stem --------------------------------------------------
@pytest.mark.parametrize(
    "video_path, expected",
    [
        ("vidéos/demo_darija.mp4", "demo_darija"),
        ("vidéos/test_audio.mp4", "test_audio"),
        ("C:/x/y/clip.final.mov", "clip.final"),
    ],
)
def test_derive_video_stem(video_path, expected):
    assert ae._derive_video_stem(video_path) == expected


# --- Cas 2 : construction de la commande FFmpeg ---------------------------------
def test_build_ffmpeg_audio_command_contains_expected():
    out = Path("outputs/audio/demo_darija/audio.wav")
    cmd = ae._build_ffmpeg_audio_command(
        "ffmpeg", "vidéos/demo_darija.mp4", out, 16000, 1, True
    )

    assert cmd[0] == "ffmpeg"
    assert "-vn" in cmd
    # -ac 1 (mono) : paire ordonnée
    assert cmd[cmd.index("-ac") + 1] == "1"
    # -ar 16000 (16 kHz)
    assert cmd[cmd.index("-ar") + 1] == "16000"
    # -c:a pcm_s16le
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    # le chemin de sortie .wav est le dernier argument
    assert cmd[-1].endswith("audio.wav")
    # l'entrée -i pointe vers la vidéo
    assert cmd[cmd.index("-i") + 1] == "vidéos/demo_darija.mp4"


def test_build_ffmpeg_audio_command_respects_params():
    out = Path("outputs/audio/x/audio.wav")
    cmd = ae._build_ffmpeg_audio_command("ffmpeg", "x.mp4", out, 44100, 2, False)
    assert cmd[cmd.index("-ar") + 1] == "44100"
    assert cmd[cmd.index("-ac") + 1] == "2"
    # overwrite=False -> -n (pas -y)
    assert "-n" in cmd
    assert "-y" not in cmd


def test_build_ffmpeg_audio_command_overwrite_flag():
    out = Path("outputs/audio/x/audio.wav")
    cmd = ae._build_ffmpeg_audio_command("ffmpeg", "x.mp4", out, 16000, 1, True)
    assert "-y" in cmd
    assert "-n" not in cmd


# --- Cas 3 : fichier audio valide (non vide) ------------------------------------
def test_validate_output_file_accepts_non_empty(tmp_path):
    f = tmp_path / "audio.wav"
    f.write_bytes(b"RIFF....data")  # contenu factice non vide
    # Ne doit pas lever.
    ae._validate_output_file(f)


# --- Cas 4 : fichier audio absent -----------------------------------------------
def test_validate_output_file_missing_raises(tmp_path):
    with pytest.raises(AudioExtractionError):
        ae._validate_output_file(tmp_path / "nope.wav")


# --- Cas 5 : fichier audio vide -------------------------------------------------
def test_validate_output_file_empty_raises(tmp_path):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")  # 0 octet
    with pytest.raises(AudioExtractionError):
        ae._validate_output_file(f)


# --- _safe_mkdir ----------------------------------------------------------------
def test_safe_mkdir_creates_and_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    ae._safe_mkdir(target)
    assert target.exists() and target.is_dir()
    # Idempotent : un second appel ne lève pas.
    ae._safe_mkdir(target)
    assert target.exists()
