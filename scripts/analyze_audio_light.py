import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.audio_light_analysis import (
    AudioLightAnalysisError,
    analyze_audio_light,
)


def _load_json(path_str: str, label: str) -> dict:
    """Charge un fichier JSON avec des erreurs claires (introuvable / invalide)."""
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Fichier {label} introuvable : {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON {label} invalide ({path}) : {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse audio légère (énergie RMS / silence) d'un WAV PCM."
    )
    parser.add_argument("audio_path", help="Chemin du fichier WAV à analyser")
    parser.add_argument(
        "--audio-manifest",
        default=None,
        help="Manifest audio source facultatif (reporté dans la sortie)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sortie JSON de l'analyse (sinon affichage console uniquement)",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=500,
        help="Taille des fenêtres d'analyse en millisecondes (défaut 500)",
    )

    args = parser.parse_args()

    audio = Path(args.audio_path)
    if not audio.exists() or not audio.is_file():
        print(f"Erreur : fichier audio introuvable : {audio}", file=sys.stderr)
        return 1

    audio_manifest = None
    if args.audio_manifest:
        try:
            audio_manifest = _load_json(args.audio_manifest, "audio-manifest")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            return 1

    try:
        result = analyze_audio_light(
            audio_path=str(audio),
            audio_manifest=audio_manifest,
            window_ms=args.window_ms,
        )
    except AudioLightAnalysisError as exc:
        print(f"Erreur d'analyse audio : {exc}", file=sys.stderr)
        return 1

    output_text = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.write_text(output_text, encoding="utf-8")
        except OSError as exc:
            print(f"Erreur : écriture du fichier de sortie impossible ({output_path}) : {exc}", file=sys.stderr)
            return 1
        print(f"Analyse écrite : {output_path}", file=sys.stderr)

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
