import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.transcription import TranscriptionError, transcribe_audio


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcrire un audio WAV avec faster-whisper (timestamps par segment)."
    )
    parser.add_argument("audio_path", help="Chemin du fichier WAV à transcrire")
    parser.add_argument(
        "--audio-manifest",
        default=None,
        help="Manifest audio source facultatif (reporté dans la sortie)",
    )
    parser.add_argument(
        "--audio-light-analysis",
        default=None,
        help="Analyse audio légère source facultative (reportée dans la sortie)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sortie du transcript JSON (sinon affichage console uniquement)",
    )
    parser.add_argument("--model", default="small", help="Modèle Whisper (défaut small)")
    parser.add_argument("--device", default="cpu", help="Device (défaut cpu)")
    parser.add_argument(
        "--compute-type", default="int8", help="Type de calcul (défaut int8)"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Langue forcée (ex: ar) ; défaut None = auto-détection",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        help="Tâche (défaut transcribe ; pas de traduction)",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Taille du faisceau (défaut 5)"
    )

    args = parser.parse_args()

    audio = Path(args.audio_path)
    if not audio.exists() or not audio.is_file():
        print(f"Erreur : fichier audio introuvable : {audio}", file=sys.stderr)
        return 1

    try:
        result = transcribe_audio(
            audio_path=str(audio),
            audio_manifest_path=args.audio_manifest,
            audio_light_analysis_path=args.audio_light_analysis,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
        )
    except TranscriptionError as exc:
        print(f"Erreur de transcription : {exc}", file=sys.stderr)
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
        print(f"Transcript écrit : {output_path}", file=sys.stderr)

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
