import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from video_agent.vlm_strategy import (
    VlmStrategyError,
    check_pack_exists,
    load_vlm_config,
    prepare_strategy,
)
from video_agent.run_identity import atomic_write_json, atomic_write_text


def _segment_ready(segment: dict) -> bool:
    """Un segment est prêt pour le VLM s'il a grille + composite + frames sélectionnées."""
    return bool(
        segment.get("motion_grid_path")
        and segment.get("motion_composite_path")
        and (segment.get("selected_frames") or [])
    )


def _build_prompt_md(segment: dict, language) -> str:
    """Construit le contenu Markdown du prompt VLM d'un segment."""
    seg_id = segment.get("segment_id")
    lines = []
    lines.append(f"# Prompt VLM — segment {seg_id}")
    lines.append("")
    lines.append(f"- segment_id : {seg_id}")
    lines.append(f"- start_s : {segment.get('start_s')}")
    lines.append(f"- end_s : {segment.get('end_s')}")
    lines.append(f"- langue : {language}")
    lines.append("")

    lines.append("## Images fournies")
    lines.append(f"- motion_grid : `{segment.get('motion_grid_path')}`")
    lines.append(f"- motion_composite : `{segment.get('motion_composite_path')}`")
    lines.append("- frames sélectionnées :")
    for f in segment.get("selected_frames", []) or []:
        lines.append(f"  - `{f.get('path')}` (t={f.get('timestamp_s')}s)")
    lines.append("")

    lines.append("## Audio / parole")
    lines.append(f"- combined_text : {segment.get('combined_text', '')!r}")
    lines.append("- transcript_segments :")
    for t in segment.get("transcript_segments", []) or []:
        lines.append(
            f"  - [{t.get('start_s')}-{t.get('end_s')}s] {t.get('text', '')!r} "
            f"(confiance={t.get('confidence')})"
        )
    lines.append("")

    lines.append("## Instructions")
    for instr in segment.get("instructions_for_vlm", []) or []:
        lines.append(f"- {instr}")
    lines.append("")

    lines.append("## Questions")
    for q in segment.get("questions_for_vlm", []) or []:
        lines.append(f"- {q}")
    lines.append("")

    lines.append("## Format de sortie attendu (JSON)")
    lines.append("```json")
    lines.append(json.dumps(segment.get("expected_output_schema", {}), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Consignes strictes")
    lines.append("- Réponds uniquement en JSON valide.")
    lines.append(
        "- N'invente rien. Si une information n'est pas visible ou audible, "
        "indique qu'elle est incertaine."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Préparer un dossier de run VLM (prompts + plan) sans appeler de modèle."
    )
    parser.add_argument("--pack", required=True, help="Manifest du pack VLM (Niveau 8)")
    parser.add_argument("--config", required=True, help="Config VLM (config/vlm_config.example.json)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie du run (défaut : outputs/vlm_runs/<video_stem>)",
    )

    args = parser.parse_args()

    # Validation config + pack + plan (rejette api/local, n'appelle aucun modèle).
    try:
        plan = prepare_strategy(args.pack, args.config)
        pack = check_pack_exists(args.pack)
        config = load_vlm_config(args.config)
    except VlmStrategyError as exc:
        print(f"Erreur de stratégie VLM : {exc}", file=sys.stderr)
        return 1

    # Dossier de sortie : explicite ou dérivé du video_stem.
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stem = Path(pack.get("video_path", "video")).stem or "video"
        out_dir = Path("outputs/vlm_runs") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    language = pack.get("language")
    segments = pack.get("segments", [])

    selected_segments = []
    ready_count = 0
    try:
        for seg in segments:
            seg_id = seg.get("segment_id")
            prompt_path = out_dir / f"prompt_segment_{int(seg_id):03d}.md"
            atomic_write_text(prompt_path, _build_prompt_md(seg, language))

            ready = _segment_ready(seg)
            if ready:
                ready_count += 1
            selected_segments.append(
                {
                    "segment_id": seg_id,
                    "start_s": seg.get("start_s"),
                    "end_s": seg.get("end_s"),
                    "prompt_path": prompt_path.as_posix(),
                    "motion_grid_path": seg.get("motion_grid_path"),
                    "motion_composite_path": seg.get("motion_composite_path"),
                    "selected_frames": seg.get("selected_frames", []),
                    "combined_text": seg.get("combined_text", ""),
                    "ready_for_vlm": ready,
                }
            )

        selected_inputs = {
            "pack_path": args.pack,
            "config_path": args.config,
            "mode": plan["mode"],
            "provider": plan["provider"],
            "model": plan["model"],
            "segments": selected_segments,
        }
        atomic_write_json(out_dir / "selected_inputs.json", selected_inputs)

        next_step = {
            "manual": "manual_vlm_analysis",
            "automatic": "automatic_vlm_analysis",
            "disabled": None,
        }[plan["mode"]]
        run_plan = {
            "plan_type": "vlm_run_plan",
            "executed": False,
            "mode": plan["mode"],
            "provider": plan["provider"],
            "model": plan["model"],
            "pack_path": args.pack,
            "config_path": args.config,
            "output_dir": out_dir.as_posix(),
            "segment_count": len(segments),
            "ready_for_vlm_count": ready_count,
            "next_step": next_step,
        }
        atomic_write_json(out_dir / "run_plan.json", run_plan)
    except OSError as exc:
        print(f"Erreur : écriture des fichiers de run impossible : {exc}", file=sys.stderr)
        return 1

    print(f"Run VLM préparé : {out_dir}", file=sys.stderr)
    print(f"  mode={run_plan['mode']} provider={run_plan['provider']} model='{run_plan['model']}'", file=sys.stderr)
    print(f"  segment_count={run_plan['segment_count']} ready_for_vlm_count={ready_count}", file=sys.stderr)
    print(f"  next_step={run_plan['next_step']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
