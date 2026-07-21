import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.vlm_pack_builder import VlmPackError, build_vlm_pack_from_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construire le pack VLM (par segment) à partir d'un manifest de fusion."
    )
    parser.add_argument(
        "--fusion", required=True, help="Manifest de fusion audio-visuelle (Niveau 7)"
    )
    parser.add_argument(
        "--output", required=True, help="Chemin de sortie du pack VLM JSON"
    )
    parser.add_argument(
        "--max-frames-per-segment",
        type=int,
        default=6,
        help="Nombre max de frames sélectionnées par segment (défaut 6)",
    )
    parser.add_argument(
        "--skip-not-ready",
        action="store_true",
        help="Exclure les segments dont ready_for_vlm est False",
    )

    args = parser.parse_args()

    try:
        pack = build_vlm_pack_from_file(
            fusion_manifest_path=args.fusion,
            max_frames=args.max_frames_per_segment,
            skip_not_ready=args.skip_not_ready,
        )
    except VlmPackError as exc:
        print(f"Erreur de construction du pack VLM : {exc}", file=sys.stderr)
        return 1

    output_text = json.dumps(pack, indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(output_text, encoding="utf-8")
    except OSError as exc:
        print(f"Erreur : écriture du pack impossible ({output_path}) : {exc}", file=sys.stderr)
        return 1

    summary = pack.get("summary", {})
    print(f"Pack VLM écrit : {output_path}", file=sys.stderr)
    print(f"  segment_count          = {summary.get('segment_count')}", file=sys.stderr)
    print(f"  total_selected_frames  = {summary.get('total_selected_frames')}", file=sys.stderr)
    print(f"  ready_for_vlm_count    = {summary.get('ready_for_vlm_count')}", file=sys.stderr)
    print(f"  max_frames_per_segment = {summary.get('max_frames_per_segment')}", file=sys.stderr)
    print(f"  next_step              = {pack.get('next_step')}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
