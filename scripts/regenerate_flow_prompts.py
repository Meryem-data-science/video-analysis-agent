"""CLI — réécrit les flow_prompt par clip via Gemini vision (écrase regen_brief_<id>.json).

Clé API : GEMINI_API_KEY ou GOOGLE_API_KEY dans l'environnement.
Exemple :
    $env:GEMINI_API_KEY="..."; .\\.venv\\Scripts\\python.exe scripts\\regenerate_flow_prompts.py --video-id Video6
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from video_agent.regen_prompt_vision import (
    DEFAULT_VISION_MODEL, RegenVisionError, regenerate_flow_prompts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Réécrit les flow_prompt par clip depuis leurs propres images (Gemini vision)."
    )
    parser.add_argument("--video-id", default="Video6", help="Identifiant vidéo (défaut Video6)")
    parser.add_argument("--model", default=DEFAULT_VISION_MODEL, help="Modèle Gemini vision")
    args = parser.parse_args()

    try:
        report = regenerate_flow_prompts(args.video_id, model=args.model)
    except RegenVisionError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print("\n===== RAPPORT — réécriture flow_prompt (Gemini vision) =====", file=sys.stderr)
    print(f"  modèle  : {report['model']}", file=sys.stderr)
    print(f"  backup  : {report['backup']}", file=sys.stderr)
    print(f"  réécrits: {report['clips_rewritten']}", file=sys.stderr)
    print(f"  gardés  : {report['clips_kept_old']}", file=sys.stderr)
    if report["failures"]:
        for f in report["failures"]:
            print(f"  ÉCHEC {f['clip_id']} : {f['error']}", file=sys.stderr)
    print("", file=sys.stderr)
    for c in report["checks"]:
        mark = "OK " if c["ok"] else "ÉCHEC"
        detail = f"  [{c['detail']}]" if c["detail"] else ""
        print(f"  [{mark}] {c['name']}{detail}", file=sys.stderr)
    print(f"  ----> {report['passed']}/{report['total']} vérifications passées", file=sys.stderr)

    print("\n----- clip_01 AVANT -----", file=sys.stderr)
    print("  " + (report["clip01_before"] or "(vide)"), file=sys.stderr)
    print("----- clip_01 APRÈS -----", file=sys.stderr)
    print("  " + (report["clip01_after"] or "(vide)"), file=sys.stderr)
    print("----- clip_04 APRÈS -----", file=sys.stderr)
    print("  " + (report["clip04_after"] or "(vide)"), file=sys.stderr)

    print("\n  ➡️ Relance ensuite : scripts/build_flow_package.py --video-id "
          f"{args.video_id}  (pour rafraîchir Flow_regeneration/ avec les nouveaux prompts).",
          file=sys.stderr)

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("clip01_before", "clip01_after", "clip04_after")},
                     indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
