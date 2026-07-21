import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.audio_extraction import AudioExtractionError, extract_audio


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extraire la piste audio d'une vidéo en WAV mono 16 kHz PCM 16-bit (FFmpeg)."
    )
    parser.add_argument("video_path", help="Chemin du fichier vidéo à traiter")
    parser.add_argument(
        "--output-dir",
        default="outputs/audio",
        help="Répertoire racine de sortie (audio sous <output-dir>/<video_stem>/audio.wav)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Chemin de sortie du manifest audio JSON (sinon affichage console uniquement)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Fréquence d'échantillonnage cible (défaut 16000)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Nombre de canaux cible (défaut 1 = mono)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Réutiliser le WAV déjà présent au lieu de le réécrire",
    )

    args = parser.parse_args()

    video = Path(args.video_path)
    if not video.exists() or not video.is_file():
        print(f"Erreur : vidéo introuvable : {video}", file=sys.stderr)
        return 1

    try:
        manifest = extract_audio(
            video_path=str(video),
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            channels=args.channels,
            overwrite=not args.no_overwrite,
        )
    except AudioExtractionError as exc:
        print(f"Erreur d'extraction audio : {exc}", file=sys.stderr)
        return 1

    output_text = json.dumps(manifest, indent=2, ensure_ascii=False)

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            manifest_path.write_text(output_text, encoding="utf-8")
        except OSError as exc:
            print(f"Erreur : écriture du manifest impossible ({manifest_path}) : {exc}", file=sys.stderr)
            return 1
        print(f"Manifest écrit : {manifest_path}", file=sys.stderr)

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
