import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.pipeline import STEP_ORDER, PipelineError, run_full_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline vidéo local, puis VLM selon le mode disabled/manual/automatic."
    )
    parser.add_argument("video", help="Nom du fichier vidéo sous VIDEO_ROOT (ex: demo_darija.mp4)")
    parser.add_argument("--config", default="config/vlm_config.example.json",
                        help="Config VLM (manual par défaut ; automatic exige un provider)")
    parser.add_argument("--model", default=None, help="Override de modèle (sinon config)")
    parser.add_argument("--overwrite-analysis", action="store_true",
                        help="Réanalyser les segments (sinon réutilise les résultats valides)")
    parser.add_argument("--step-timeout", type=float, default=900.0,
                        help="Délai max par étape (s)")
    parser.add_argument("--force", action="store_true",
                        help="Ré-exécute TOUT en ignorant (et purgeant) les checkpoints de reprise")
    parser.add_argument("--from-step", default=None, choices=STEP_ORDER, metavar="ÉTAPE",
                        help="Reprend à partir de cette étape (efface son checkpoint et ceux des "
                             "étapes suivantes). Étapes : " + ", ".join(STEP_ORDER))
    parser.add_argument("--no-audio", action="store_true",
                        help="Désactive l'étape audio par scène (musique + SFX via Gemini)")
    parser.add_argument("--audio-model", default=None,
                        help="Override du modèle Gemini pour l'audio (sinon $GEMINI_MODEL puis défaut)")
    parser.add_argument("--no-arc", action="store_true",
                        help="Désactive l'arc narratif du rapport (aucun appel Gemini texte)")
    parser.add_argument("--no-audio-features", action="store_true",
                        help="Désactive les features audio librosa par scène (local, sans réseau)")

    args = parser.parse_args()

    try:
        result = run_full_pipeline(
            video_name=args.video,
            config_path=args.config,
            model=args.model,
            overwrite_analysis=args.overwrite_analysis,
            step_timeout=args.step_timeout,
            enable_audio=not args.no_audio,
            audio_model=args.audio_model,
            enable_arc=not args.no_arc,
            enable_audio_features=not args.no_audio_features,
            force=args.force,
            from_step=args.from_step,
        )
    except PipelineError as exc:
        print(f"Erreur pipeline : {exc}", file=sys.stderr)
        return 1

    ok = result["ok"]
    print("", file=sys.stderr)
    print(f"[pipeline] === {'TERMINÉ' if ok else 'INTERROMPU'} — run_id={result['run_id']} "
          f"audio={result['audio_present']} status={result.get('status')} ===", file=sys.stderr)
    print(f"[pipeline] Run dir : {result['run_dir']}", file=sys.stderr)
    for st in result["steps"]:
        print(f"[pipeline]   {'✓' if st['ok'] else '✗'} {st['step']}", file=sys.stderr)
    if result["reports"]["json"]:
        print(f"[pipeline] Rapport JSON : {result['reports']['json']}", file=sys.stderr)
    if result["reports"]["txt"]:
        print(f"[pipeline] Rapport TXT  : {result['reports']['txt']}", file=sys.stderr)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
