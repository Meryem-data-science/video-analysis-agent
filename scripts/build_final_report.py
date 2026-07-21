import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Console UTF-8 : évite un crash cp1252 lors de l'écho JSON sur stdout (Windows) même
# sans PYTHONUTF8=1 (le contenu non latin/accents reste imprimable).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from video_agent.final_report import FinalReportError, build_final_report_from_files
from video_agent.run_identity import atomic_write_json, atomic_write_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construire le rapport vidéo final (synthèse des analyses VLM validées)."
    )
    parser.add_argument("--results-dir", default="outputs/vlm_results/demo_darija",
                        help="Dossier des résultats VLM (vlm_result_segment_*.json)")
    parser.add_argument("--fusion", default="outputs/audio_visual_fusion_demo_darija.json",
                        help="Manifest de fusion audio-visuelle (timestamps, langue)")
    parser.add_argument("--pack", default=None,
                        help="Pack VLM attendu (sinon les segments de --fusion font autorité)")
    parser.add_argument("--summary", default=None,
                        help="Résumé de validation (défaut <results-dir>/vlm_results_summary.json)")
    parser.add_argument("--output", default="outputs/final_report_demo_darija.json",
                        help="Chemin de sortie du rapport JSON")
    parser.add_argument("--text-output", default="outputs/final_report_demo_darija.txt",
                        help="Chemin de sortie du résumé texte lisible")
    parser.add_argument("--no-arc", action="store_true",
                        help="Desactive l'arc narratif (rapport 100%% hors-ligne, aucun appel LLM)")
    parser.add_argument("--model", default=None,
                        help="Modele pour l'arc narratif (defaut : modele par defaut du CLI)")
    parser.add_argument("--inspection", default=None,
                        help="Inspection technique (defaut : outputs/inspection_<stem>.json si present)")
    parser.add_argument("--scene-audio", default=None,
                        help="Audio par scene (defaut : outputs/audio_events_<stem>.json si present)")
    parser.add_argument("--audio-features", default=None,
                        help="Features audio librosa (defaut : outputs/audio_features_<stem>.json si present)")

    args = parser.parse_args()

    summary_path = args.summary
    if summary_path is None:
        candidate = Path(args.results_dir) / "vlm_results_summary.json"
        summary_path = str(candidate) if candidate.exists() else None

    try:
        report = build_final_report_from_files(
            results_dir=args.results_dir,
            fusion_manifest_path=args.fusion,
            summary_path=summary_path,
            include_arc=not args.no_arc,
            model=args.model,
            inspection_path=args.inspection,
            scene_audio_path=args.scene_audio,
            audio_features_path=args.audio_features,
            vlm_pack_path=args.pack,
        )
    except FinalReportError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    # Écriture JSON (UTF-8, arabe préservé).
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, report)

    # Écriture du résumé texte lisible.
    text_path = Path(args.text_output)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(text_path, report["readable_summary"])

    print(f"Rapport écrit : {out_path}", file=sys.stderr)
    print(f"Résumé texte écrit : {text_path}", file=sys.stderr)
    print(f"  status={report['status']} scenes={report['scene_count']} langue={report.get('language')} "
          f"ton={report['aggregate']['tone']} "
          f"conf_moy={report['aggregate']['average_confidence']} "
          f"next_step={report['next_step']}", file=sys.stderr)

    # Le résumé lisible sur stderr (pour lecture humaine) + le JSON complet sur stdout.
    print("\n" + report["readable_summary"], file=sys.stderr)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
