import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from video_agent.vlm_provider import (
    VlmProviderError,
    get_vlm_provider,
    resolve_vlm_settings,
)
from video_agent.run_identity import atomic_write_json
from video_agent.vlm_result_validation import (
    VlmResultValidationError,
    validate_results_dir,
    validate_single_result,
)


def _load_json(path: Path, label: str) -> dict:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Fichier {label} introuvable : {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON {label} invalide ({path}) : {exc}") from exc


def _load_config(config_path: Path) -> dict:
    """Charge la config VLM (ou {} si absente/illisible)."""
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _resolve_model(config: dict, cli_model):
    """Modèle VLM : --model > config.gemini_vlm_model > config.model > None.

    None laisse le provider Gemini résoudre via $GEMINI_VLM_MODEL puis son défaut.
    """
    if cli_model:
        return cli_model
    for key in ("gemini_vlm_model", "model"):
        val = config.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _segment_images(seg: dict) -> list:
    paths = []
    if seg.get("motion_grid_path"):
        paths.append(seg["motion_grid_path"])
    if seg.get("motion_composite_path"):
        paths.append(seg["motion_composite_path"])
    paths += [f.get("path") for f in seg.get("selected_frames", []) if f.get("path")]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyser TOUS les segments en mode automatic via un provider explicite."
    )
    parser.add_argument("--run-dir", default="outputs/vlm_runs/demo_darija",
                        help="Dossier de run VLM (selected_inputs.json + prompts)")
    parser.add_argument("--pack", default="outputs/vlm_pack_demo_darija.json",
                        help="Pack VLM (cohérence du validateur)")
    parser.add_argument("--config", default="config/vlm_config.example.json",
                        help="Config VLM (source du modèle, jamais codé en dur)")
    parser.add_argument("--results-dir", default="outputs/vlm_results/demo_darija",
                        help="Dossier de sortie des résultats VLM")
    parser.add_argument("--summary-output", default=None,
                        help="Chemin du résumé (défaut <results-dir>/vlm_results_summary.json)")
    parser.add_argument("--model", default=None, help="Override de modèle (sinon config/env)")
    parser.add_argument("--provider", default=None,
                        help="Provider automatique : gemini. Sinon $VLM_PROVIDER ou config ; aucun fallback.")
    parser.add_argument("--only", type=int, default=None, help="Limiter à un seul segment_id")
    parser.add_argument("--overwrite", action="store_true",
                        help="Réanalyser même si un résultat valide existe déjà")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Délai max par segment (s)")

    args = parser.parse_args()

    try:
        selected = _load_json(Path(args.run_dir) / "selected_inputs.json", "selected_inputs")
        pack = _load_json(Path(args.pack), "pack VLM")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(Path(args.config))
    model = _resolve_model(config, args.model)
    try:
        settings = resolve_vlm_settings(config, cli_provider=args.provider)
    except VlmProviderError as exc:
        print(f"Erreur de configuration VLM : {exc}", file=sys.stderr)
        return 1
    if settings["mode"] != "automatic":
        print(
            f"Erreur : run_vlm_agent exige mode='automatic' ; mode actuel={settings['mode']!r}. "
            "Aucun provider n'a été initialisé.",
            file=sys.stderr,
        )
        return 1
    provider_name = str(settings["provider"])
    provider = get_vlm_provider(provider_name, model=model)
    print(f"[*] provider VLM = {provider_name} (modèle={model or '<défaut env>'})", file=sys.stderr)

    segments = selected.get("segments", [])
    if args.only is not None:
        segments = [s for s in segments if int(s.get("segment_id")) == args.only]
        if not segments:
            print(f"Erreur : segment {args.only} absent de selected_inputs.json", file=sys.stderr)
            return 1

    pack_by_id = {int(s.get("segment_id")): s for s in pack.get("segments", [])}
    statuses = []  # (segment_id, status, detail)

    for seg in segments:
        seg_id = int(seg.get("segment_id"))
        out_path = results_dir / f"vlm_result_segment_{seg_id:03d}.json"

        # Réutilisation si un résultat valide existe déjà (sauf --overwrite).
        if not args.overwrite and out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                existing_report = validate_single_result(
                    existing, seg_id, segment_pack=pack_by_id.get(seg_id)
                )
                if existing_report["valid"]:
                    statuses.append((seg_id, "skipped", "résultat existant valide réutilisé"))
                    print(f"[*] segment {seg_id}: réutilisé (valide)", file=sys.stderr)
                    continue
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass

        prompt_path = Path(seg["prompt_path"])
        if not prompt_path.exists():
            statuses.append((seg_id, "error", f"prompt introuvable : {prompt_path}"))
            print(f"[!] segment {seg_id}: prompt introuvable", file=sys.stderr)
            continue
        prompt_text = prompt_path.read_text(encoding="utf-8")
        image_paths = _segment_images(seg)

        print(f"[*] segment {seg_id}: appel VLM {provider_name} ({len(image_paths)} images, "
              f"model={model or '<défaut env>'}) …", file=sys.stderr)
        try:
            result = provider.analyze_segment(
                prompt_text=prompt_text,
                image_paths=image_paths,
                segment_id=seg_id,
                start_s=seg.get("start_s"),
                end_s=seg.get("end_s"),
                combined_text=seg.get("combined_text", ""),
                model=model,
                timeout_s=args.timeout,
            )
        except Exception as exc:  # un échec de segment ne bloque pas les autres
            statuses.append((seg_id, "error", f"{type(exc).__name__}: {exc}"))
            print(f"[!] segment {seg_id}: échec — {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        report = validate_single_result(result, seg_id, segment_pack=pack_by_id.get(seg_id))
        atomic_write_json(out_path, result)
        statuses.append((seg_id, "ok" if report["valid"] else "invalid",
                         f"errors={len(report['errors'])} warnings={len(report['warnings'])}"))
        print(f"[*] segment {seg_id}: écrit, valid={report['valid']} "
              f"({len(report['errors'])} err, {len(report['warnings'])} warn)", file=sys.stderr)

    # Résumé global via le validateur de dossier.
    try:
        summary = validate_results_dir(str(results_dir), pack)
    except VlmResultValidationError as exc:
        print(f"Erreur de résumé : {exc}", file=sys.stderr)
        return 1

    summary_path = Path(args.summary_output) if args.summary_output else results_dir / "vlm_results_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)

    print(f"Résumé écrit : {summary_path}", file=sys.stderr)
    print(f"  segments traités : {[(sid, st) for sid, st, _ in statuses]}", file=sys.stderr)
    print(f"  valid_count={summary['valid_count']} invalid_count={summary['invalid_count']} "
          f"avg_conf={summary['summary']['average_confidence']} next_step={summary['next_step']}",
          file=sys.stderr)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Code retour 0 si tous valides, 2 sinon (au moins un invalide/manquant).
    return 0 if summary["invalid_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
