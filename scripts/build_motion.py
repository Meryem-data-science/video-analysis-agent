import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.motion_representation import (
    MotionRepresentationError,
    build_motion_representations,
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
        description="Construire les représentations de mouvement (grille + composite) avec Pillow."
    )
    parser.add_argument(
        "--frames-manifest",
        required=True,
        help="Chemin du manifest de frames du Niveau 4 (clé `frames`)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/motion",
        help="Répertoire racine de sortie (images sous <output-dir>/<video_stem>/)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Chemin de sortie du manifest motion JSON (sinon affichage console uniquement)",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=360,
        help="Largeur cible des vignettes de la grille (défaut 360)",
    )
    parser.add_argument(
        "--composite-long-side",
        type=int,
        default=1080,
        help="Côté long max des frames pour le composite (défaut 1080)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Réutiliser les images déjà présentes au lieu de les réécrire",
    )

    args = parser.parse_args()

    try:
        frames_manifest = _load_json(args.frames_manifest, "frames-manifest")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    try:
        manifest = build_motion_representations(
            frames_manifest=frames_manifest,
            output_dir=args.output_dir,
            thumbnail_width=args.thumbnail_width,
            composite_long_side=args.composite_long_side,
            overwrite=not args.no_overwrite,
            source_frames_manifest_path=args.frames_manifest,
        )
    except MotionRepresentationError as exc:
        print(f"Erreur de construction du mouvement : {exc}", file=sys.stderr)
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
