import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.frame_extraction import FrameExtractionError, extract_frames


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
        description="Extraire des frames d'une vidéo avec FFmpeg, segment par segment."
    )
    parser.add_argument("video_path", help="Chemin du fichier vidéo à traiter")
    parser.add_argument(
        "--segments",
        required=True,
        help="Chemin du fichier JSON de segmentation (clé `segments`)",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="Chemin du fichier JSON de stratégie (clé `frame_strategy.mode`)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/frames",
        help="Répertoire racine de sortie (frames sous <output-dir>/<video_stem>/)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Chemin de sortie du manifest JSON (sinon affichage console uniquement)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Réutiliser les frames déjà présentes au lieu de les réécrire",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Largeur max optionnelle (redimensionnement FFmpeg, désactivé par défaut)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Hauteur max optionnelle (redimensionnement FFmpeg, désactivé par défaut)",
    )
    parser.add_argument(
        "--min-frames-per-segment",
        type=int,
        default=2,
        help="Frames minimum par segment (garantie mouvement, défaut 2 ; fallback 1 si trop court)",
    )

    args = parser.parse_args()

    video = Path(args.video_path)
    if not video.exists() or not video.is_file():
        print(f"Erreur : vidéo introuvable : {video}", file=sys.stderr)
        return 1

    try:
        segments = _load_json(args.segments, "segments")
        strategy = _load_json(args.strategy, "strategy")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    try:
        manifest = extract_frames(
            video_path=str(video),
            segments=segments,
            strategy=strategy,
            output_dir=args.output_dir,
            overwrite=not args.no_overwrite,
            max_width=args.max_width,
            max_height=args.max_height,
            min_frames_per_segment=args.min_frames_per_segment,
        )
    except FrameExtractionError as exc:
        print(f"Erreur d'extraction : {exc}", file=sys.stderr)
        return 1

    output_text = json.dumps(manifest, indent=2, ensure_ascii=False)

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(output_text, encoding="utf-8")
        print(f"Manifest écrit : {manifest_path}", file=sys.stderr)

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
