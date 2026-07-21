import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.vlm_result_validation import (
    VlmResultValidationError,
    load_json,
    validate_results_dir,
)
from video_agent.run_identity import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valider strictement les réponses JSON d'un VLM vis-à-vis du pack VLM."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Dossier contenant les fichiers vlm_result_segment_XXX.json",
    )
    parser.add_argument(
        "--pack", required=True, help="Manifest du pack VLM (Niveau 8)"
    )
    parser.add_argument(
        "--output",
        default="vlm_results_summary.json",
        help="Chemin de sortie du résumé de validation JSON (défaut vlm_results_summary.json)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Erreur : dossier de résultats introuvable : {results_dir}", file=sys.stderr)
        return 1

    try:
        vlm_pack = load_json(args.pack, "pack VLM")
        summary = validate_results_dir(str(results_dir), vlm_pack)
    except VlmResultValidationError as exc:
        print(f"Erreur de validation : {exc}", file=sys.stderr)
        return 1

    output_text = json.dumps(summary, indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_json(output_path, summary)
    except OSError as exc:
        print(f"Erreur : écriture du résumé impossible ({output_path}) : {exc}", file=sys.stderr)
        return 1

    s = summary.get("summary", {})
    print(f"Résumé de validation écrit : {output_path}", file=sys.stderr)
    print(f"  segment_count   = {summary.get('segment_count')}", file=sys.stderr)
    print(f"  valid_count     = {summary.get('valid_count')}", file=sys.stderr)
    print(f"  invalid_count   = {summary.get('invalid_count')}", file=sys.stderr)
    print(f"  status          = {summary.get('status')}", file=sys.stderr)
    print(f"  missing_count   = {summary.get('missing_segment_count')}", file=sys.stderr)
    print(f"  extra_segments  = {summary.get('extra_segments')}", file=sys.stderr)
    print(f"  average_confidence       = {s.get('average_confidence')}", file=sys.stderr)
    print(f"  segments_missing_results = {s.get('segments_missing_results')}", file=sys.stderr)
    print(f"  segments_with_errors     = {s.get('segments_with_errors')}", file=sys.stderr)
    print(f"  segments_with_warnings   = {s.get('segments_with_warnings')}", file=sys.stderr)
    print(f"  next_step       = {summary.get('next_step')}", file=sys.stderr)

    return 0 if summary.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
