"""Serveur MCP « plomberie de données » — expose les artefacts du pipeline.

Serveur AUTONOME, distinct de ``mcp_server.py`` (lequel relance le pipeline via des
sous-processus). Celui-ci ne fait que **LIRE** des artefacts déjà produits et
**SAUVEGARDER** de nouveaux résultats fournis par un client MCP (Gemini pour l'audio
ET le visuel).

Garanties strictes :
- N'appelle AUCUN modèle IA et ne relance AUCUNE étape du pipeline.
- Lecture seule sur les artefacts existants (segments, frames, motion, vlm_results).
- Écritures UNIQUEMENT sous outputs/audio/<stem>/segments/ (clips audio) et
  outputs/audio_gemini/<stem>/ (résultats d'analyse Gemini).

Même framework et même style que ``mcp_server.py`` : paquet ``fastmcp``, décorateur
``@app.tool``, transport stdio via ``app.run_stdio_async()``.
"""

import json
import logging
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

# Racine du projet sur sys.path quand on lance ce module directement.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROJECT_ROOT = ROOT_DIR
OUTPUTS = PROJECT_ROOT / "outputs"

from fastmcp.server import FastMCP

app = FastMCP(
    name="Agent Analyse Vidéo — Artefacts MCP",
    instructions=(
        "Plomberie de données en lecture seule sur les artefacts du pipeline. "
        "list_analysis_units liste les scènes (segment_id + timestamps) ; "
        "get_analysis_unit retourne une scène ; "
        "get_audio_clip_for_unit découpe et sauvegarde le clip audio WAV d'une scène "
        "depuis le WAV global (idempotent) ; "
        "get_visual_context_for_unit retourne le contexte visuel compact d'une scène "
        "(résultat VLM) pour aider l'alignement audio ; "
        "save_gemini_audio_analysis sauvegarde l'analyse audio Gemini d'une scène. "
        "Aucun modèle n'est appelé, aucune étape du pipeline n'est relancée."
    ),
    version="0.1",
)


# --- Helpers ---------------------------------------------------------------

def _safe_stem(stem: str) -> str:
    """Valide ``stem`` : nom simple, sans séparateur ni remontée de chemin."""
    s = str(stem).strip()
    if not s or s != Path(s).name or s in {".", ".."}:
        raise ValueError(f"stem invalide : {stem!r}")
    return s


def _segment_name(segment_id: int) -> str:
    """Convention d'identifiant : segment_NNN (3 chiffres) depuis segment_id (int)."""
    sid = int(segment_id)
    if sid < 0:
        raise ValueError(f"segment_id invalide : {segment_id!r}")
    return f"segment_{sid:03d}"


def _read_json(path: Path, label: str) -> Any:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} introuvable : {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_segments(stem: str) -> list:
    path = OUTPUTS / f"segments_{stem}.json"
    data = _read_json(path, "Fichier de segments")
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Champ 'segments' absent ou invalide dans {path}")
    return segments


def _find_segment(segments: list, segment_id: int) -> dict:
    for seg in segments:
        if int(seg.get("segment_id", -1)) == int(segment_id):
            return seg
    raise ValueError(f"segment_id {segment_id} introuvable")


def _compact_unit(seg: dict) -> dict:
    return {
        "segment_id": int(seg["segment_id"]),
        "start_s": seg.get("start_s"),
        "end_s": seg.get("end_s"),
        "duration_s": seg.get("duration_s"),
    }


def _wav_duration_s(path: Path) -> float:
    """Durée réelle d'un WAV PCM via stdlib ``wave`` : nframes / framerate."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return frames / float(rate) if rate else 0.0


def _ffmpeg_cut_wav(src: Path, start_s: float, end_s: float, out: Path) -> None:
    """Découpe [start_s, end_s] du WAV ``src`` vers ``out`` (mono 16k pcm_s16le).

    Même profil audio que le WAV global du projet. Seek APRÈS ``-i`` pour une
    découpe précise sur du PCM. N'appelle aucun modèle.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg introuvable dans le PATH.")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ss", f"{float(start_s):.3f}",
        "-to", f"{float(end_s):.3f}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out),
    ]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"Découpe ffmpeg échouée [{start_s:.3f}, {end_s:.3f}] : "
            f"{(proc.stderr or '').strip()}"
        )


# --- Tools -----------------------------------------------------------------

@app.tool(
    name="list_analysis_units",
    description="Liste les scènes d'une vidéo : segment_id + timestamps (lecture seule).",
)
def list_analysis_units(stem: str) -> list:
    """Lit outputs/segments_<stem>.json -> [{segment_id, start_s, end_s, duration_s}]."""
    stem = _safe_stem(stem)
    return [_compact_unit(seg) for seg in _load_segments(stem)]


@app.tool(
    name="get_analysis_unit",
    description="Retourne une scène {segment_id, start_s, end_s, duration_s} (lecture seule).",
)
def get_analysis_unit(stem: str, segment_id: int) -> dict:
    """Retourne les bornes temporelles d'une scène depuis segments_<stem>.json."""
    stem = _safe_stem(stem)
    seg = _find_segment(_load_segments(stem), segment_id)
    return _compact_unit(seg)


@app.tool(
    name="get_audio_clip_for_unit",
    description=(
        "Découpe l'audio d'une scène depuis le WAV global et le sauvegarde de façon "
        "persistante (idempotent). Écrit uniquement sous outputs/audio/<stem>/segments/."
    ),
)
def get_audio_clip_for_unit(stem: str, segment_id: int) -> dict:
    """WAV global -> clip [start_s, end_s] persistant ; ne recoupe pas si déjà présent."""
    stem = _safe_stem(stem)
    seg = _find_segment(_load_segments(stem), segment_id)
    start_s = float(seg["start_s"])
    end_s = float(seg["end_s"])

    global_wav = OUTPUTS / "audio" / stem / "audio.wav"
    if not global_wav.is_file():
        raise FileNotFoundError(f"WAV global introuvable : {global_wav}")

    segments_dir = OUTPUTS / "audio" / stem / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    out = segments_dir / f"{_segment_name(segment_id)}.wav"

    # Idempotent : clip déjà présent (non vide) -> on ne recoupe pas.
    if not (out.is_file() and out.stat().st_size > 0):
        _ffmpeg_cut_wav(global_wav, start_s, end_s, out)

    return {
        "segment_id": int(segment_id),
        "start_s": start_s,
        "end_s": end_s,
        "audio_clip_path": out.relative_to(PROJECT_ROOT).as_posix(),
        "exists": True,
    }


@app.tool(
    name="get_visual_context_for_unit",
    description=(
        "Contexte visuel compact d'une scène (résultat VLM) pour aider l'alignement "
        "audio. Lecture seule ; renvoie {error, segment_id} si le contexte manque."
    ),
)
def get_visual_context_for_unit(stem: str, segment_id: int) -> dict:
    """Lit vlm_results/<stem>/vlm_result_segment_NNN.json -> contexte compact."""
    stem = _safe_stem(stem)
    vlm_path = (
        OUTPUTS / "vlm_results" / stem / f"vlm_result_{_segment_name(segment_id)}.json"
    )
    if not vlm_path.is_file():
        return {"error": "visual context not found", "segment_id": int(segment_id)}
    try:
        vlm = _read_json(vlm_path, "Résultat VLM")
    except (json.JSONDecodeError, OSError):
        return {"error": "visual context not found", "segment_id": int(segment_id)}

    # Timestamps depuis les segments (best-effort ; absents -> None).
    start_s = end_s = None
    try:
        seg = _find_segment(_load_segments(stem), segment_id)
        start_s, end_s = seg.get("start_s"), seg.get("end_s")
    except (FileNotFoundError, ValueError):
        pass

    return {
        "segment_id": int(segment_id),
        "start_s": start_s,
        "end_s": end_s,
        "scene_description": vlm.get("scene_description"),
        "actions": vlm.get("actions", []),
    }


@app.tool(
    name="save_gemini_audio_analysis",
    description=(
        "Sauvegarde l'analyse audio Gemini d'une scène sous "
        "outputs/audio_gemini/<stem>/audio_segment_NNN.json (UTF-8, ensure_ascii=False)."
    ),
)
def save_gemini_audio_analysis(stem: str, segment_id: int, analysis: Any) -> dict:
    """Écrit l'analyse fournie par le client (aucun modèle appelé ici)."""
    stem = _safe_stem(stem)
    out_dir = OUTPUTS / "audio_gemini" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"audio_{_segment_name(segment_id)}.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(analysis, fh, ensure_ascii=False, indent=2)
    return {"saved": True, "path": out.relative_to(PROJECT_ROOT).as_posix()}


# --- Entrée ----------------------------------------------------------------

def _smoke_clip(stem: str, segment_id: int) -> None:
    """Découpe une scène et confirme durée_clip ≈ end_s − start_s (via wave)."""
    clip = get_audio_clip_for_unit(stem, segment_id)
    print(f"-- get_audio_clip_for_unit({stem!r}, {segment_id}) --")
    print(json.dumps(clip, ensure_ascii=False, indent=2))
    expected = clip["end_s"] - clip["start_s"]
    measured = _wav_duration_s(PROJECT_ROOT / clip["audio_clip_path"])
    print(
        f"   durée attendue (end-start) = {expected:.3f}s | "
        f"durée réelle clip = {measured:.3f}s | écart = {abs(measured - expected):.3f}s"
    )


def _run_smoke_test() -> None:
    """Smoke test hors MCP : list + découpe + mesure de durée sur demo_scene1."""
    stem = "demo_scene1"
    print(f"== list_analysis_units({stem!r}) ==")
    units = list_analysis_units(stem)
    print(f"{len(units)} scène(s) ; 1ère = {units[0] if units else None}")

    print()
    # Scène à début nul (segment 1) ET scène à début non nul (segment 2).
    _smoke_clip(stem, 1)
    print()
    _smoke_clip(stem, 2)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    if "--smoke" in sys.argv:
        _run_smoke_test()
    else:
        import asyncio

        logging.info(f"Démarrage MCP stdio (artefacts). PROJECT_ROOT={PROJECT_ROOT}")
        asyncio.run(app.run_stdio_async())
