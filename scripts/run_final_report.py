r"""Commande de démonstration pour produire et contrôler un rapport complet.

Lance le pipeline en forçant les composants de rapport à ON. La configuration VLM
doit toutefois être en mode ``automatic`` avec un provider explicite ; en mode
``manual``, la commande rend un échec contrôlé avec le statut ``awaiting_vlm``.

À la fin, affiche un CONTRÔLE DE COMPLÉTUDE clair + un ⚠️ par composant incomplet.

Usage :
    $env:PYTHONUTF8=1; .\.venv\Scripts\python.exe scripts\run_final_report.py demo_scene1.mp4
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Console UTF-8 : évite un crash cp1252 sur les ⚠️ / texte non latin (Windows).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from video_agent.pipeline import PipelineError, run_full_pipeline

# Tous les composants FORCÉS à ON — aucun moyen de les désactiver ici.
FORCED_FLAGS: Dict[str, bool] = {
    "enable_arc": True,
    "enable_audio": True,
    "enable_audio_features": True,
}


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    """Charge un JSON ; renvoie {} si absent/illisible (jamais d'exception)."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _count(scenes: List[Dict[str, Any]], pred: Callable[[Dict[str, Any]], bool]) -> int:
    return sum(1 for sc in scenes if pred(sc))


def build_completeness(
    result: Dict[str, Any],
    report: Dict[str, Any],
    summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Calcule le contrôle de complétude + la liste des avertissements. Pur (testable)."""
    steps = result.get("steps") or []
    steps_total = len(steps)
    steps_ok = sum(1 for s in steps if isinstance(s, dict) and s.get("ok"))

    scenes = report.get("scenes") or []
    total = len(scenes)

    screen_text_n = _count(scenes, lambda sc: bool(sc.get("texte_ecran")))
    gem_n = _count(scenes, lambda sc: (sc.get("audio_scene") or {}).get("status") == "ok")
    lib_n = _count(
        scenes,
        lambda sc: ((sc.get("audio_scene") or {}).get("measured_features") or {}).get("status") == "ok",
    )

    arc = report.get("arc_narratif") if isinstance(report.get("arc_narratif"), dict) else {}
    arc_sub_filled = any(isinstance(v, str) and v.strip() for v in arc.values())
    resume = report.get("resume_humain")
    arc_filled = bool(arc_sub_filled and isinstance(resume, str) and resume.strip())

    audio_present = bool(result.get("audio_present"))

    pipeline_status = result.get("status")
    pipeline_complete = pipeline_status in (None, "complete")
    check = {
        "ok": bool(result.get("ok")) and pipeline_complete,
        "pipeline_status": pipeline_status,
        "steps_ok": steps_ok,
        "steps_total": steps_total,
        "scene_count": report.get("scene_count"),
        "valid_count": summary.get("valid_count"),
        "segment_count": summary.get("segment_count"),
        "arc_filled": arc_filled,
        "screen_text_scenes": screen_text_n,
        "gemini_audio_ok": gem_n,
        "librosa_features_ok": lib_n,
        "scenes_total": total,
        "audio_present": audio_present,
    }

    warnings: List[str] = []
    if pipeline_status == "awaiting_vlm":
        warnings.append("⚠️ pipeline en attente des résultats VLM manuels")
    elif not check["ok"]:
        warnings.append("⚠️ pipeline ok=False (rapport possiblement incomplet)")
    if steps_total and steps_ok < steps_total:
        warnings.append(f"⚠️ étapes partielles {steps_ok}/{steps_total}")
    if not arc_filled:
        warnings.append("⚠️ arc vide (arc_narratif / resume_humain non remplis)")
    if total == 0:
        warnings.append("⚠️ aucune scène dans le rapport")
    else:
        if screen_text_n == 0:
            warnings.append(f"⚠️ texte écran (VLM) sur 0/{total} scène(s)")
        # Avertissements audio uniquement si la vidéo a effectivement de l'audio.
        if audio_present:
            if gem_n == 0:
                warnings.append(f"⚠️ audio Gemini sur 0/{total} scène(s)")
            elif gem_n < total:
                warnings.append(f"⚠️ audio Gemini partiel {gem_n}/{total}")
            if lib_n == 0:
                warnings.append(f"⚠️ librosa measured_features sur 0/{total} scène(s)")
            elif lib_n < total:
                warnings.append(f"⚠️ librosa measured_features partiel {lib_n}/{total}")
    return check, warnings


def format_completeness(check: Dict[str, Any], warnings: List[str]) -> str:
    """Met en forme le contrôle de complétude (texte lisible)."""
    total = check["scenes_total"]
    lines = [
        "===== CONTRÔLE DE COMPLÉTUDE =====",
        f"  ok                     : {check['ok']}",
        f"  statut pipeline        : {check.get('pipeline_status') or 'legacy/non renseigné'}",
        f"  étapes réussies        : {check['steps_ok']}/{check['steps_total']}",
        f"  scene_count            : {check['scene_count']}",
        f"  valid_count            : {check['valid_count']}/{check['segment_count']}",
        f"  audio_present          : {check['audio_present']}",
        f"  arc rempli             : {'O' if check['arc_filled'] else 'N'}",
        f"  texte écran (VLM)      : {check['screen_text_scenes']}/{total} scène(s)",
        f"  audio Gemini 'ok'      : {check['gemini_audio_ok']}/{total} scène(s)",
        f"  librosa features 'ok'  : {check['librosa_features_ok']}/{total} scène(s)",
        "==================================",
    ]
    if warnings:
        lines.append("AVERTISSEMENTS :")
        lines.extend(f"  {w}" for w in warnings)
    else:
        lines.append("✅ Rapport complet : aucun composant manquant.")
    return "\n".join(lines)


def run_demo(
    video_name: str,
    *,
    config_path: str = "config/vlm_config.example.json",
    runner: Callable[..., Dict[str, Any]] = run_full_pipeline,
    load_json: Callable[[Optional[str]], Dict[str, Any]] = _load_json,
) -> int:
    """Exécute le pipeline (tout forcé ON) puis affiche le contrôle de complétude.

    Le texte à l'écran est lu nativement par le VLM (plus d'étape OCR ni de langue OCR).
    """
    result = runner(video_name, config_path=config_path, **FORCED_FLAGS)

    report = load_json((result.get("reports") or {}).get("json"))
    summary = load_json(result.get("summary"))

    check, warnings = build_completeness(result, report, summary)

    report_json = (result.get("reports") or {}).get("json")
    report_txt = (result.get("reports") or {}).get("txt")
    print("", file=sys.stderr)
    print(f"[run_final_report] Rapport JSON : {report_json}", file=sys.stderr)
    print(f"[run_final_report] Rapport TXT  : {report_txt}", file=sys.stderr)
    print("\n" + format_completeness(check, warnings))

    # Exit 0 si le pipeline a abouti (un rapport a été produit), sinon 1.
    return 0 if check["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Produire un rapport complet (requiert mode VLM automatic + provider explicite).")
    parser.add_argument("video", help="Nom/chemin de la vidéo (sous VIDEO_ROOT)")
    parser.add_argument("--config", default="config/vlm_config.example.json",
                        help="Config VLM ; doit utiliser mode=automatic pour produire le rapport")
    args = parser.parse_args()
    try:
        return run_demo(args.video, config_path=args.config)
    except PipelineError as exc:
        print(f"Erreur pipeline : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
