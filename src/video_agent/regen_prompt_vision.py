"""Réécrit les flow_prompt par clip à partir de SES propres images (Gemini vision).

Problème corrigé : les flow_prompt des sous-clips d'une même scène décrivaient toute la
scène (ex. le clip 1 mentionnait un ordinateur/fondu qui n'apparaissent qu'à la fin de
la scène). Ici, on envoie à Gemini les 3 images RÉELLES du clip (start/mid/end, ou la
card pour un static_card) et on fait réécrire un flow_prompt visuel décrivant uniquement
ce qui est visible. On ÉCRASE clip["flow_prompt"] dans regen_brief_<id>.json.

Non destructif (sauf le champ flow_prompt) :
- backup une seule fois de outputs/regen_brief_<id>.json (horodaté si un backup existe) ;
- ne modifie QUE regen_brief_<id>.json (flow_prompt + bloc de traçabilité) ;
- lecture seule sur images/transcript/frames/final_report/worksheet/Flow_regeneration ;
- CONNEXION Gemini IDENTIQUE à l'audio : réutilise ``scene_audio._init_gemini``
  (GEMINI_API_KEY, même ``genai.Client(api_key=..., http_options=...)``). Clé
  absente/invalide -> erreur claire AVANT toute écriture.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"

DEFAULT_VISION_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_MS = 120_000

# Consigne négative RENFORCÉE (exactement celle demandée), placée en fin de prompt.
NEGATIVE_DIRECTIVE = (
    "No on-screen text, no captions, no subtitles, no logos, no UI. "
    "No spoken words, no voiceover, no dialogue, no narration, no music, no sound. "
    "Completely silent footage. Do not add any people, objects, or text not described above."
)

_sleep = time.sleep  # isolé pour testabilité


class RegenVisionError(Exception):
    """Erreur de réécriture des flow_prompt via Gemini vision."""


def _init_client():
    """Initialise le client Gemini via EXACTEMENT le même helper que l'analyse audio.

    On délègue à ``scene_audio._init_gemini`` (source unique de vérité pour la connexion
    Gemini du projet) : même variable d'environnement (GEMINI_API_KEY), même
    ``genai.Client(api_key=..., http_options=HttpOptions(timeout=...))``. Ainsi la
    connexion est identique à celle de l'audio et tout correctif d'auth appliqué à
    l'audio profite automatiquement à ce script.
    """
    from .scene_audio import _init_gemini as _audio_init_gemini

    client, types = _audio_init_gemini()
    if client is None:
        raise RegenVisionError(
            "Connexion Gemini impossible (même méthode que l'audio) : GEMINI_API_KEY "
            "absente/invalide ou 'google-genai' non installé. Aucun fichier modifié."
        )
    return client, types


_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _image_part(types, path: Path):
    if not path.exists() or not path.is_file():
        raise RegenVisionError(f"Image introuvable : {path}")
    mt = _MEDIA.get(path.suffix.lower(), "image/jpeg")
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mt)


def _clip_image_paths(clip: Dict[str, Any], frames_dir: Path) -> List[Tuple[str, Path]]:
    """Retourne [(label, path)] des images réelles du clip (start/mid/end ou card)."""
    ref = clip.get("reference_images", {}) or {}
    keys = (("start_frame", "start"), ("mid_frame", "mid"), ("end_frame", "end")) \
        if clip.get("type") != "static_card" else (("card_frame", "card"),)
    out: List[Tuple[str, Path]] = []
    for refkey, label in keys:
        rel = ref.get(refkey)
        if rel:
            out.append((label, OUTPUTS / Path(rel)))
    return out


def _subjects_context(clip: Dict[str, Any], registry_by_id: Dict[str, Dict[str, Any]]) -> str:
    descs = [registry_by_id[a]["descriptor"] for a in clip.get("subjects", []) if a in registry_by_id]
    return "; ".join(descs) if descs else "(no specific subjects expected — describe what you see)"


def _build_instruction(clip: Dict[str, Any], brief: Dict[str, Any],
                       registry_by_id: Dict[str, Dict[str, Any]]) -> str:
    aspect = brief.get("video", {}).get("aspect_ratio", "9:16")
    secs = int(round(clip.get("target_duration_s", 0) or 0))
    rendering = brief.get("global_style", {}).get(
        "rendering", "clean 3D motion design, glossy objects, generous negative space")
    subjects = _subjects_context(clip, registry_by_id)
    n_imgs = "the single frame" if clip.get("type") == "static_card" else \
        "the 3 frames (start, middle, end of the clip)"
    return (
        "You are a visual director for an image-only video generator (Google Flow).\n"
        f"Look ONLY at {n_imgs} provided and write ONE visual generation prompt that "
        "describes ONLY what is actually visible in these frames.\n\n"
        f"Aspect ratio: {aspect} vertical. Duration: about {secs} seconds.\n"
        f"Global rendering style: {rendering}.\n"
        f"Expected subjects (only mention them IF visible): {subjects}.\n\n"
        "Requirements:\n"
        "- Describe only objects actually visible: their colors, shapes and layout, and how "
        "they evolve from start to end.\n"
        "- Describe the motion implied by the start -> middle -> end evolution.\n"
        "- Describe the camera/framing and the background/lighting you observe.\n"
        "- Do NOT invent anything that is not in these frames. In particular, do NOT add a "
        "laptop, a computer, a logo, any text, or any object from another part of the video "
        "if it is not visible here.\n"
        "- Exclude any on-screen text, any voice, any music, any sound effect.\n"
        f"- Include the format ({aspect} vertical) and the duration (about {secs} seconds).\n"
        f"- End your prompt with EXACTLY this sentence: {NEGATIVE_DIRECTIVE}\n\n"
        "Respond with PLAIN TEXT ONLY: the final prompt itself, no preamble, no Markdown, "
        "no surrounding quotes."
    )


_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z]*)?\s*|\s*```\s*$")


def _clean_response(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t)
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    t = t.strip()
    if len(t) >= 2 and t[0] in "\"'«" and t[-1] in "\"'»":
        t = t[1:-1].strip()
    return t


def _ensure_constraints(text: str, aspect: str, secs: int) -> str:
    """Garantit présence de la consigne négative + format + durée (ajoute si absent)."""
    out = text.strip()
    low = out.lower()
    if aspect not in out or "second" not in low:
        out += f" {aspect} vertical, about {secs} seconds."
    if "completely silent footage" not in out.lower():
        out += " " + NEGATIVE_DIRECTIVE
    return out.strip()


def _is_transient(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(k in s for k in ("503", "429", "unavailable", "overloaded", "deadline", "timeout"))


def _rewrite_one_clip(client, types, clip: Dict[str, Any], brief: Dict[str, Any],
                      registry_by_id: Dict[str, Dict[str, Any]], frames_dir: Path,
                      model: str, *, max_retries: int = 2, base_delay: float = 2.0) -> str:
    imgs = _clip_image_paths(clip, frames_dir)
    if not imgs:
        raise RegenVisionError(f"{clip['clip_id']} : aucune image de référence.")
    parts: List[Any] = [_build_instruction(clip, brief, registry_by_id)]
    for _label, p in imgs:
        parts.append(_image_part(types, p))
    attempt = 0
    while True:
        try:
            resp = client.models.generate_content(model=model, contents=parts)
            raw = getattr(resp, "text", None)
            if not raw or not raw.strip():
                raise RegenVisionError("réponse vide de Gemini")
            aspect = brief.get("video", {}).get("aspect_ratio", "9:16")
            secs = int(round(clip.get("target_duration_s", 0) or 0))
            return _ensure_constraints(_clean_response(raw), aspect, secs)
        except Exception as exc:  # noqa: BLE001
            if _is_transient(exc) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                if delay > 0:
                    _sleep(delay)
                attempt += 1
                continue
            if isinstance(exc, RegenVisionError):
                raise
            raise RegenVisionError(f"{clip['clip_id']} : appel Gemini échoué : {exc}") from exc


def _make_backup(brief_path: Path, video_id: str) -> Path:
    """Sauvegarde unique. Si un backup existe déjà, suffixe horodaté (ne pas l'écraser)."""
    backup = brief_path.with_name(f"regen_brief_{video_id}.backup.json")
    if backup.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = brief_path.with_name(f"regen_brief_{video_id}.backup.{ts}.json")
    shutil.copyfile(brief_path, backup)
    return backup


def regenerate_flow_prompts(
    video_id: str, model: str = DEFAULT_VISION_MODEL, outputs_dir: Optional[Path] = None,
    *, now_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Réécrit les flow_prompt via Gemini vision et écrase regen_brief_<id>.json."""
    out_dir = Path(outputs_dir) if outputs_dir else OUTPUTS
    brief_path = out_dir / f"regen_brief_{video_id}.json"
    if not brief_path.exists():
        raise RegenVisionError(
            f"Brief introuvable : {brief_path}. Lance d'abord "
            f"scripts/build_regen_brief.py --video-id {video_id}."
        )
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    frames_dir = out_dir / f"regen_frames_{video_id}"

    # Snapshot mtimes des SOURCES à NE PAS toucher (preuve non-destructif).
    watched: List[Path] = [
        out_dir / f"transcript_{video_id}.json",
        out_dir / f"final_report_{video_id}.json",
        out_dir / f"regen_worksheet_{video_id}.md",
    ]
    watched += sorted(frames_dir.glob("*")) if frames_dir.exists() else []
    mtimes_before = {str(p): p.stat().st_mtime_ns for p in watched if p.exists()}

    # Init client AVANT toute écriture : si pas de clé -> erreur claire, rien n'est touché.
    client, types = _init_client()

    # Backup unique (avant overwrite).
    backup_path = _make_backup(brief_path, video_id)

    registry_by_id = {a["asset_id"]: a for a in brief.get("asset_registry", [])}
    clips = brief.get("clips", []) or []
    old_prompts = {c["clip_id"]: c.get("flow_prompt", "") for c in clips}

    rewritten: List[str] = []
    kept_old: List[str] = []
    failures: List[Dict[str, str]] = []
    for c in clips:
        try:
            new_prompt = _rewrite_one_clip(client, types, c, brief, registry_by_id, frames_dir, model)
            c["flow_prompt"] = new_prompt
            rewritten.append(c["clip_id"])
        except Exception as exc:  # noqa: BLE001 — un échec isolé ne casse pas le run
            kept_old.append(c["clip_id"])
            failures.append({"clip_id": c["clip_id"], "error": str(exc)[-300:]})

    brief["flow_prompt_source"] = {
        "method": "gemini_vision_per_clip",
        "model": model,
        "regenerated_at": now_iso or datetime.now(timezone.utc).isoformat(),
        "clips_rewritten": rewritten,
        "clips_kept_old": kept_old,
        "backup": backup_path.name,
    }
    brief_path.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    # Snapshot APRÈS — sources hors regen_brief inchangées.
    sources_untouched = all(
        p.stat().st_mtime_ns == mtimes_before[str(p)] for p in watched if str(p) in mtimes_before
    )
    return _self_test(brief, brief_path, backup_path, old_prompts, rewritten, kept_old,
                      failures, sources_untouched, model)


def _self_test(
    brief: Dict[str, Any], brief_path: Path, backup_path: Path, old_prompts: Dict[str, str],
    rewritten: List[str], kept_old: List[str], failures: List[Dict[str, str]],
    sources_untouched: bool, model: str,
) -> Dict[str, Any]:
    checks: List[Tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    clips = brief.get("clips", []) or []
    by_id = {c["clip_id"]: c for c in clips}
    aspect = brief.get("video", {}).get("aspect_ratio", "9:16")

    chk("backup créé", backup_path.exists(), backup_path.name)
    chk("sources hors regen_brief inchangées (mtimes)", sources_untouched)
    chk("≥ 1 clip réécrit", len(rewritten) >= 1, f"{len(rewritten)} réécrits, {len(kept_old)} gardés")

    banned = ("says:", "voiceover says", "on-screen text appears", "background:")
    fmt_ok, neg_ok, clean_ok = True, True, True
    for cid in rewritten:
        fp = by_id[cid].get("flow_prompt", "")
        low = fp.lower()
        if not fp or aspect not in fp or "second" not in low:
            fmt_ok = False
        if "completely silent footage" not in low:
            neg_ok = False
        if any(b in low for b in banned):
            clean_ok = False
    chk("nouveaux prompts : format + durée", fmt_ok)
    chk("nouveaux prompts : consigne négative (Completely silent footage)", neg_ok)
    chk("nouveaux prompts : sans voix/texte (says:/voiceover/On-screen text/Background:)", clean_ok)

    # Anti-régression : clip_01 ne doit plus mentionner laptop/computer/fading.
    c1 = by_id.get("clip_01", {})
    c1_low = (c1.get("flow_prompt", "") or "").lower()
    c1_clean = ("clip_01" not in rewritten) or not any(
        w in c1_low for w in ("laptop", "computer", "fading", "fade")
    )
    chk("anti-régression clip_01 : pas de laptop/computer/fading", c1_clean)

    passed = sum(1 for _, ok, _ in checks if ok)
    return {
        "ok": passed == len(checks),
        "passed": passed, "total": len(checks),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "model": model,
        "backup": backup_path.as_posix(),
        "clips_rewritten": rewritten,
        "clips_kept_old": kept_old,
        "failures": failures,
        "clip01_before": old_prompts.get("clip_01", ""),
        "clip01_after": by_id.get("clip_01", {}).get("flow_prompt", ""),
        "clip04_after": by_id.get("clip_04", {}).get("flow_prompt", ""),
        "brief_path": brief_path.as_posix(),
    }
