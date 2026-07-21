r"""ÉTAPE 3 — analyse audio SCÈNE PAR SCÈNE avec Gemini (isolé, hors pipeline).

Réutilise PROMPT + RESPONSE_SCHEMA du prototype (gemini_audio_prototype.py).
N'appelle ni Claude ni le VLM. Ne modifie aucun fichier du pipeline. Écrit un
seul fichier scratch : scripts/_audio_events_demo_scene1.json.

Bornes de scènes lues depuis l'artefact EXISTANT outputs/segments_demo_scene1.json
(aucune ré-exécution de détection de scènes).

Usage :
    $env:PYTHONUTF8=1; $env:GEMINI_MODEL="gemini-2.5-flash"; \
        .\.venv\Scripts\python.exe scripts\gemini_audio_per_scene.py; \
        Remove-Item Env:\GEMINI_MODEL
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Réutilise la logique du prototype (même dossier scripts/ → import direct).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_audio_prototype import (  # noqa: E402
    MODEL,
    PROMPT,
    RESPONSE_SCHEMA,
    _eprint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO = PROJECT_ROOT / "vidéos" / "demo_scene1.mp4"
SEGMENTS_JSON = PROJECT_ROOT / "outputs" / "segments_demo_scene1.json"
OUTPUT_JSON = Path(__file__).resolve().parent / "_audio_events_demo_scene1.json"

PAUSE_BETWEEN_SCENES_S = 2.5


def load_scene_bounds() -> list[tuple[float, float]] | None:
    """Lit (start, end) par scène depuis l'artefact de segmentation existant."""
    candidates = [
        SEGMENTS_JSON,
        PROJECT_ROOT / "outputs" / "audio_visual_fusion_demo_scene1.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _eprint(f"Avertissement : lecture impossible de {path} : {exc}")
            continue
        segs = data.get("segments")
        if isinstance(segs, list) and segs and "start_s" in segs[0]:
            bounds = [(float(s["start_s"]), float(s["end_s"])) for s in segs]
            print(f"Bornes de scènes lues depuis : {path.name} ({len(bounds)} scènes)")
            return bounds

    _eprint(
        "ERREUR : bornes de scènes introuvables. J'ai cherché dans :\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nJe NE relance PAS la détection de scènes sans ton accord."
    )
    return None


def extract_scene_audio(video: Path, start: float, end: float) -> Path | None:
    """Extrait [start, end] en .mp3 mono temporaire via ffmpeg. None si échec."""
    fd, tmp_name = tempfile.mkstemp(suffix=".mp3", prefix="gemini_scene_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
    except FileNotFoundError:
        _eprint("ERREUR : ffmpeg introuvable sur le PATH.")
        tmp_path.unlink(missing_ok=True)
        return None
    if proc.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        _eprint(f"ERREUR : extraction ffmpeg échouée pour [{start:.3f}, {end:.3f}].")
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        if tail:
            _eprint(tail)
        tmp_path.unlink(missing_ok=True)
        return None
    return tmp_path


def analyze_scene(client, types, audio_path: Path) -> dict | None:
    """Uploade + appelle Gemini sur un .mp3. Renvoie le dict parsé ou None."""
    uploaded = None
    try:
        try:
            uploaded = client.files.upload(file=str(audio_path))
        except Exception as exc:  # noqa: BLE001
            _eprint(f"  upload échoué : {exc}")
            return None
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[PROMPT, uploaded],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _eprint(f"  generate_content échoué (503 ?) : {exc}")
            return None
        raw = getattr(response, "text", None)
        if not raw:
            _eprint("  réponse vide de Gemini.")
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            _eprint(f"  JSON invalide : {exc}")
            return None
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception as exc:  # noqa: BLE001
                _eprint(f"  avertissement : suppression distante échouée : {exc}")


def summarize(events: dict) -> str:
    """Résumé court : musique oui/non + 1-2 SFX principaux."""
    music = events.get("music", {}) if isinstance(events, dict) else {}
    has_music = "oui" if music.get("present") else "non"
    sfx = events.get("sound_effects", []) if isinstance(events, dict) else []
    sfx_labels = [s.get("label", "?") for s in sfx[:2] if isinstance(s, dict)]
    sfx_str = ", ".join(sfx_labels) if sfx_labels else "—"
    return f"musique={has_music} | SFX: {sfx_str}"


def main() -> int:
    if not VIDEO.is_file():
        _eprint(f"ERREUR : vidéo introuvable : {VIDEO}")
        return 1

    bounds = load_scene_bounds()
    if bounds is None:
        return 1

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _eprint(
            "ERREUR : GEMINI_API_KEY absente.\n"
            'Définissez-la : $env:GEMINI_API_KEY = "votre_cle" puis relancez.'
        )
        return 1

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _eprint("ERREUR : 'google-genai' non installé dans ce venv.")
        return 1

    client = genai.Client(api_key=api_key)
    print(f"Modèle utilisé : {MODEL}\nVidéo : {VIDEO}\n")

    scenes_out: list[dict] = []
    for idx, (start, end) in enumerate(bounds, start=1):
        audio_path = extract_scene_audio(VIDEO, start, end)
        if audio_path is None:
            events, status = {}, "échec_gracieux"
        else:
            parsed = analyze_scene(client, types, audio_path)
            audio_path.unlink(missing_ok=True)
            if parsed is None:
                events, status = {}, "échec_gracieux"
            else:
                events, status = parsed, "ok"

        scenes_out.append(
            {
                "scene_index": idx,
                "start": round(start, 3),
                "end": round(end, 3),
                "audio_events": events,
                "audio_status": status,
            }
        )
        tag = summarize(events) if status == "ok" else "(échec gracieux)"
        print(f"  scène {idx:>2} [{start:6.3f}–{end:6.3f}s]  {status:14s}  {tag}")

        if idx < len(bounds):
            time.sleep(PAUSE_BETWEEN_SCENES_S)

    result = {
        "video": str(VIDEO),
        "model_used": MODEL,
        "scenes": scenes_out,
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ok = sum(1 for s in scenes_out if s["audio_status"] == "ok")
    print(
        f"\n{ok}/{len(scenes_out)} scènes analysées avec succès.\n"
        f"Fichier écrit : {OUTPUT_JSON}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
