r"""PROTOTYPE isolé — analyse audio (musique + effets sonores) avec Gemini.

Hors pipeline : ne touche à aucun autre module/output, n'appelle ni Claude ni le
VLM. Un seul appel Gemini (gemini-3-flash-preview), sortie JSON structurée.

Étapes :
  1. résout la vidéo (par défaut vidéos/demo_scene1.mp4 ; surchargeable en arg) ;
  2. extrait la piste audio en .mp3 mono temporaire via ffmpeg ;
  3. lit GEMINI_API_KEY ; crée le client google-genai ;
  4. uploade l'audio, appelle generate_content (response_schema JSON) ;
  5. parse, ajoute audio_status (ok / échec_gracieux) ;
  6. nettoie le fichier uploadé + le .mp3 local ;
  7. affiche le JSON final indenté + le modèle utilisé.

Usage :
    $env:PYTHONUTF8=1; .\.venv\Scripts\python.exe scripts\gemini_audio_prototype.py
    (optionnel) ... scripts\gemini_audio_prototype.py "chemin\vers\autre.mp4"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Modèle par défaut (confirmé). Surchargeable ponctuellement via $env:GEMINI_MODEL
# sans toucher à ce défaut — utile en cas de 503 transitoire sur le preview.
MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3-flash-preview"

# Racine projet = parent du dossier scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "vidéos"
DEFAULT_VIDEO = VIDEO_DIR / "demo_scene1.mp4"

PROMPT = """Tu es un analyste audio rigoureux. On te fournit UNIQUEMENT la piste
audio d'une courte vidéo. Décris la musique et les effets sonores présents.

Règles STRICTES :
- Donne une confiance (0.0 à 1.0) pour CHAQUE affirmation.
- Liste explicitement tes doutes dans le champ "uncertainties".
- En cas de doute, dis "incertain" plutôt que de deviner.
- N'INVENTE JAMAIS le titre d'un morceau ni le nom d'un artiste/chanteur.
- Reste factuel et qualitatif : pas de BPM précis inventé (utilise tempo_feel).
- Réponds UNIQUEMENT par un JSON valide conforme au schéma imposé.

Champs attendus :
- music.present : y a-t-il de la musique ?
- music.mood : ambiance/humeur générale de la musique (qualitatif).
- music.energy : "faible", "moyenne" ou "élevée".
- music.tempo_feel : "lent", "modéré" ou "rapide".
- sound_effects : liste d'effets sonores identifiés, chacun avec un label et une confiance.
- speech_overlap : y a-t-il de la parole qui se superpose au reste ?
- evidence : ce que tu entends concrètement et qui justifie tes réponses.
- uncertainties : la liste de tes doutes.
"""

# Schéma de sortie structuré (google-genai accepte un dict JSON-Schema-like).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "music": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "confidence": {"type": "number"},
                "mood": {"type": "string"},
                "mood_confidence": {"type": "number"},
                "energy": {"type": "string", "enum": ["faible", "moyenne", "élevée"]},
                "tempo_feel": {"type": "string", "enum": ["lent", "modéré", "rapide"]},
            },
            "required": [
                "present",
                "confidence",
                "mood",
                "mood_confidence",
                "energy",
                "tempo_feel",
            ],
        },
        "sound_effects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "confidence"],
            },
        },
        "speech_overlap": {"type": "boolean"},
        "evidence": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "music",
        "sound_effects",
        "speech_overlap",
        "evidence",
        "uncertainties",
    ],
}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def resolve_video() -> Path | None:
    """Résout le chemin vidéo. Renvoie None si introuvable (après diagnostic)."""
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.is_file():
            return candidate
        _eprint(f"ERREUR : chemin passé en argument introuvable : {candidate}")
        return None

    if DEFAULT_VIDEO.is_file():
        return DEFAULT_VIDEO

    _eprint(
        f"ERREUR : fichier attendu introuvable : {DEFAULT_VIDEO}\n"
        "Je NE choisis PAS d'autre vidéo. Contenu du dossier vidéos/ :"
    )
    if VIDEO_DIR.is_dir():
        for p in sorted(VIDEO_DIR.iterdir()):
            _eprint(f"  - {p.name}")
    else:
        _eprint(f"  (dossier {VIDEO_DIR} introuvable)")
    _eprint(
        "\nMerci de me donner le bon chemin (ou passe-le en argument du script)."
    )
    return None


def extract_audio_mp3(video: Path) -> Path | None:
    """Extrait l'audio mono en .mp3 temporaire via ffmpeg. None si échec."""
    fd, tmp_name = tempfile.mkstemp(suffix=".mp3", prefix="gemini_audio_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",            # pas de vidéo
        "-ac",
        "1",              # mono
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
        _eprint("ERREUR : l'extraction audio ffmpeg a échoué.")
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        if tail:
            _eprint(tail)
        tmp_path.unlink(missing_ok=True)
        return None
    return tmp_path


def main() -> int:
    video = resolve_video()
    if video is None:
        return 1
    print(f"Vidéo utilisée : {video}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _eprint(
            "ERREUR : variable d'environnement GEMINI_API_KEY absente.\n"
            'Définissez-la : $env:GEMINI_API_KEY = "votre_cle" puis relancez.'
        )
        return 1

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        _eprint(
            "ERREUR : 'google-genai' non installé dans ce venv.\n"
            "Installez-le : .\\.venv\\Scripts\\python.exe -m pip install google-genai"
        )
        return 1

    audio_path = extract_audio_mp3(video)
    if audio_path is None:
        return 1
    print(f"Audio extrait : {audio_path} ({audio_path.stat().st_size} octets)")

    client = genai.Client(api_key=api_key)

    uploaded = None
    result_obj: dict | None = None
    audio_status = "échec_gracieux"
    try:
        try:
            uploaded = client.files.upload(file=str(audio_path))
        except Exception as exc:  # noqa: BLE001
            _eprint(f"ERREUR lors de l'upload du fichier audio : {exc}")
            return 1

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
            _eprint(f"ERREUR lors de l'appel generate_content : {exc}")
            response = None

        if response is not None:
            raw = getattr(response, "text", None)
            if not raw:
                _eprint("ERREUR : réponse vide de Gemini.")
            else:
                try:
                    result_obj = json.loads(raw)
                    audio_status = "ok"
                except json.JSONDecodeError as exc:
                    _eprint(f"ERREUR : JSON invalide renvoyé par Gemini : {exc}")
                    _eprint("Réponse brute :")
                    _eprint(raw)
    finally:
        # Nettoyage : fichier distant puis fichier local.
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception as exc:  # noqa: BLE001
                _eprint(f"Avertissement : suppression du fichier distant échouée : {exc}")
        audio_path.unlink(missing_ok=True)

    # NOTRE code ajoute audio_status (non demandé à Gemini).
    if result_obj is None:
        result_obj = {}
    final = {**result_obj, "audio_status": audio_status}

    print("\n=== JSON final ===")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    print(f"\nModèle utilisé : {MODEL}")
    return 0 if audio_status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
