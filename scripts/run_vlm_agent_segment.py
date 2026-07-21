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
from video_agent.vlm_result_validation import validate_single_result
from video_agent.run_identity import atomic_write_json


def _load_json(path: Path, label: str) -> dict:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Fichier {label} introuvable : {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON {label} invalide ({path}) : {exc}") from exc


def _load_config(config_path: Path) -> dict:
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _resolve_model(config: dict, cli_model: str | None) -> str | None:
    """Modèle VLM : --model > config.gemini_vlm_model > config.model > None."""
    if cli_model:
        return cli_model
    for key in ("gemini_vlm_model", "model"):
        val = config.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyser UN segment en mode automatic via un provider explicite."
    )
    parser.add_argument("--run-dir", default="outputs/vlm_runs/demo_darija",
                        help="Dossier de run VLM (selected_inputs.json + prompts)")
    parser.add_argument("--segment", type=int, default=1, help="segment_id à analyser (défaut 1)")
    parser.add_argument("--pack", default="outputs/vlm_pack_demo_darija.json",
                        help="Pack VLM (pour la cohérence du validateur)")
    parser.add_argument("--config", default="config/vlm_config.example.json",
                        help="Config VLM (source du modèle, jamais codé en dur)")
    parser.add_argument("--results-dir", default="outputs/vlm_results/demo_darija",
                        help="Dossier de sortie des résultats VLM")
    parser.add_argument("--model", default=None, help="Override de modèle (sinon config/env)")
    parser.add_argument("--provider", default=None,
                        help="Provider automatique : gemini. Sinon $VLM_PROVIDER ou config ; aucun fallback.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Délai max de l'appel (s)")

    args = parser.parse_args()

    try:
        selected = _load_json(Path(args.run_dir) / "selected_inputs.json", "selected_inputs")
        pack = _load_json(Path(args.pack), "pack VLM")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    # Segment du run.
    seg = next((s for s in selected.get("segments", []) if int(s.get("segment_id")) == args.segment), None)
    if seg is None:
        print(f"Erreur : segment {args.segment} absent de selected_inputs.json", file=sys.stderr)
        return 1

    prompt_path = Path(seg["prompt_path"])
    if not prompt_path.exists():
        print(f"Erreur : prompt introuvable : {prompt_path}", file=sys.stderr)
        return 1
    prompt_text = prompt_path.read_text(encoding="utf-8")

    # Images : motion grid + composite + frames sélectionnées.
    image_paths = []
    if seg.get("motion_grid_path"):
        image_paths.append(seg["motion_grid_path"])
    if seg.get("motion_composite_path"):
        image_paths.append(seg["motion_composite_path"])
    image_paths += [f.get("path") for f in seg.get("selected_frames", []) if f.get("path")]

    pack_seg = next((s for s in pack.get("segments", []) if int(s.get("segment_id")) == args.segment), None)
    config = _load_config(Path(args.config))
    model = _resolve_model(config, args.model)
    try:
        settings = resolve_vlm_settings(config, cli_provider=args.provider)
    except VlmProviderError as exc:
        print(f"Erreur de configuration VLM : {exc}", file=sys.stderr)
        return 1
    if settings["mode"] != "automatic":
        print(
            f"Erreur : run_vlm_agent_segment exige mode='automatic' ; "
            f"mode actuel={settings['mode']!r}. Aucun provider n'a été initialisé.",
            file=sys.stderr,
        )
        return 1
    provider_name = str(settings["provider"])
    provider = get_vlm_provider(provider_name, model=model)

    print(f"[*] segment={args.segment} images={len(image_paths)} provider={provider_name} "
          f"model={model or '<défaut env>'}", file=sys.stderr)

    try:
        result = provider.analyze_segment(
            prompt_text=prompt_text,
            image_paths=image_paths,
            segment_id=args.segment,
            start_s=seg.get("start_s"),
            end_s=seg.get("end_s"),
            combined_text=seg.get("combined_text", ""),
            model=model,
            timeout_s=args.timeout,
        )
    except Exception as exc:  # noqa: BLE001 — un échec affiche une erreur claire
        print(f"Erreur VLM ({provider_name}) : {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # Validation avec le même validateur que le mode manual.
    report = validate_single_result(result, args.segment, segment_pack=pack_seg)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"vlm_result_segment_{args.segment:03d}.json"
    atomic_write_json(out_path, result)

    print(f"Résultat écrit : {out_path}", file=sys.stderr)
    print(f"  valid={report['valid']} errors={len(report['errors'])} "
          f"warnings={len(report['warnings'])} confidence={report['confidence']}", file=sys.stderr)
    if report["errors"]:
        print(f"  errors: {report['errors']}", file=sys.stderr)
    if report["warnings"]:
        print(f"  warnings: {report['warnings']}", file=sys.stderr)

    # Affiche le résultat + le rapport de validation sur stdout (UTF-8).
    print(json.dumps({"result": result, "validation": report}, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
