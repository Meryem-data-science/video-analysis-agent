"""Assemble un dossier autonome Flow_regeneration/<video_id>/ (zippable tel quel).

Rassemble, par COPIE, tout ce qu'il faut pour régénérer la vidéo dans Google Flow puis
la remonter, à partir de ``outputs/regen_brief_<id>.json`` (+ frames + worksheet).

Garanties :
- COPIE uniquement : les fichiers de ``outputs/`` restent intacts (vérif mtimes).
- Idempotent : si le dossier cible existe, il est vidé puis recréé (sans toucher outputs/).
- Aucun appel API/réseau ; ffmpeg non requis (les frames sont déjà extraites).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"
PACKAGE_ROOT = PROJECT_ROOT / "Flow_regeneration"


class FlowPackageError(Exception):
    """Erreur d'assemblage du package Flow."""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def _snapshot_mtimes(root: Path) -> Dict[str, int]:
    """mtimes (ns) de tous les fichiers sous root — pour prouver outputs/ inchangé."""
    snap: Dict[str, int] = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                snap[str(p)] = p.stat().st_mtime_ns
    return snap


def build_flow_package(
    video_id: str, outputs_dir: Optional[Path] = None,
    package_root: Optional[Path] = None, include_frames_source: bool = True,
) -> Dict[str, Any]:
    """Construit Flow_regeneration/<video_id>/ et retourne le rapport d'auto-test."""
    out_dir = Path(outputs_dir) if outputs_dir else OUTPUTS
    pkg_root = Path(package_root) if package_root else PACKAGE_ROOT

    brief_path = out_dir / f"regen_brief_{video_id}.json"
    if not brief_path.exists():
        raise FlowPackageError(
            f"Brief introuvable : {brief_path}. Lance d'abord "
            f"`scripts/build_regen_brief.py --video-id {video_id}`."
        )
    try:
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FlowPackageError(f"Brief JSON invalide ({brief_path}) : {exc}") from exc

    worksheet_path = out_dir / f"regen_worksheet_{video_id}.md"
    frames_src_dir = out_dir / f"regen_frames_{video_id}"

    # Snapshot mtimes AVANT (preuve non-destructif) sur tout outputs/.
    mtimes_before = _snapshot_mtimes(out_dir)

    # Destination — idempotent : on vide puis recrée (UNIQUEMENT sous Flow_regeneration/).
    dest = pkg_root / video_id
    if dest.resolve().parent != pkg_root.resolve():
        raise FlowPackageError("Chemin de destination inattendu (sécurité).")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    missing_images: List[str] = []

    # Racine : brief + worksheet.
    _write_json(dest / "regen_brief.json", brief)
    if worksheet_path.exists():
        shutil.copyfile(worksheet_path, dest / "worksheet.md")
    else:
        missing_images.append(f"(worksheet absente : {worksheet_path.name})")

    # Clips.
    clips = brief.get("clips", []) or []
    rename_map = {"start_frame": "start.jpg", "mid_frame": "mid.jpg",
                  "end_frame": "end.jpg", "card_frame": "card.jpg"}
    for c in clips:
        cdir = dest / "clips" / c["clip_id"]
        cdir.mkdir(parents=True, exist_ok=True)
        _write_text(cdir / "flow_prompt.txt", c.get("flow_prompt", "") or "")
        _write_json(cdir / "post_production.json", c.get("post_production", {}) or {})
        ref = c.get("reference_images", {}) or {}
        for key, newname in rename_map.items():
            rel = ref.get(key)
            if not rel:
                continue
            src = out_dir / Path(rel)  # rel = "regen_frames_<id>/clip_XX_*.jpg"
            if src.exists():
                shutil.copyfile(src, cdir / newname)
            else:
                missing_images.append(f"{c['clip_id']}/{newname} (source {rel} absente)")

    # Montage (post_production_global).
    pg = brief.get("post_production_global", {}) or {}
    mdir = dest / "montage"
    mdir.mkdir(parents=True, exist_ok=True)
    _write_text(mdir / "full_voiceover_script.txt", pg.get("full_voiceover_script", "") or "")
    _write_json(mdir / "voice_profile.json", pg.get("voice_profile", {}) or {})
    _write_json(mdir / "music_track.json", pg.get("music_track", {}) or {})
    _write_json(mdir / "on_screen_text_timeline.json", pg.get("on_screen_text_timeline", []) or [])

    # Archive optionnelle des frames sources.
    if include_frames_source and frames_src_dir.exists():
        shutil.copytree(frames_src_dir, dest / "frames_source")

    # README.
    _write_text(dest / "README.md", _render_readme(brief, missing_images))

    # Snapshot APRÈS — outputs/ doit être identique.
    mtimes_after = _snapshot_mtimes(out_dir)
    outputs_untouched = mtimes_before == mtimes_after

    return _self_test(brief, dest, out_dir, outputs_untouched, missing_images, mtimes_before)


def _render_readme(brief: Dict[str, Any], missing_images: List[str]) -> str:
    v = brief.get("video", {})
    clips = brief.get("clips", []) or []
    order = brief.get("reassembly", {}).get("order", [c["clip_id"] for c in clips])
    total = brief.get("reassembly", {}).get("total_duration_s", v.get("duration_s"))
    L: List[str] = []
    L.append(f"# Régénération Flow — {brief.get('video_id')}")
    L.append("")
    L.append(f"- Vidéo : **{brief.get('video_id')}** | Durée à retrouver : **{total} s** "
             f"| Clips : **{len(clips)}** | Format : **{v.get('aspect_ratio', '?')}** "
             f"| Langue : {v.get('language', '?')}")
    L.append("")
    L.append("## ⚠️ Important")
    L.append("Google Flow génère **uniquement l'image MUETTE**. La **voix**, la **musique**, "
             "les **effets sonores** et le **texte à l'écran** sont ajoutés **AU MONTAGE** "
             "(pas dans Flow).")
    L.append("")
    L.append("## Générer chaque clip dans Flow")
    L.append("1. Ouvrir `clips/clip_XX/flow_prompt.txt` et **copier le texte** dans Flow.")
    L.append("2. Uploader `clips/clip_XX/start.jpg` comme **image de départ** "
             "(+ `end.jpg` / `mid.jpg` en référence).")
    L.append("3. Régler la **durée Flow** du clip sur la durée indiquée dans `worksheet.md`.")
    L.append("4. Générer.")
    L.append("5. Vérifier que le clip ressemble au segment original correspondant.")
    L.append("")
    L.append("## Raccords (clips « continuation »)")
    L.append("Pour les clips marqués **continuation** dans `worksheet.md`, le fichier "
             "`start.jpg` est la **COPIE de la fin du clip précédent** : utiliser cette image "
             "comme image de départ Flow pour un **raccord fluide**.")
    L.append("")
    L.append("## Montage final")
    L.append("1. Remettre les clips **dans l'ordre** (voir « Ordre de remontage » ci-dessous "
             "et dans `worksheet.md`).")
    L.append("2. Poser la **voix off unique** : `montage/full_voiceover_script.txt` "
             "(profil voix : `montage/voice_profile.json`).")
    L.append("3. Ajouter la **musique** : `montage/music_track.json` (fond continu).")
    L.append("4. Ajouter les **textes à l'écran** aux bons moments : "
             "`montage/on_screen_text_timeline.json` (+ position dans chaque "
             "`clips/clip_XX/post_production.json`).")
    L.append("")
    L.append("## Ordre de remontage")
    L.append("")
    L.append(" → ".join(order))
    L.append("")
    L.append(f"**Durée totale à retrouver : {total} s.**")
    L.append("")
    L.append("## Contenu du dossier")
    L.append("- `regen_brief.json` — brief complet (source de vérité).")
    L.append("- `worksheet.md` — fiche clip par clip (copier-coller).")
    L.append("- `clips/clip_XX/` — `flow_prompt.txt`, images (`start/mid/end` ou `card`), "
             "`post_production.json`.")
    L.append("- `montage/` — voix off, profil voix, musique, timeline texte écran.")
    L.append("- `frames_source/` — archive des frames extraites (optionnel).")
    if missing_images:
        L.append("")
        L.append("## ⚠️ Images manquantes")
        for m in missing_images:
            L.append(f"- {m}")
    L.append("")
    return "\n".join(L)


def _build_tree(dest: Path, max_files_per_dir: int = 8) -> str:
    """Arborescence lisible du dossier créé (frames_source résumé)."""
    lines: List[str] = [dest.name + "/"]

    def walk(d: Path, prefix: str) -> None:
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        # Résume les gros dossiers de frames.
        if d.name == "frames_source":
            n = len(list(d.glob("*")))
            lines.append(prefix + f"... {n} fichiers (archive frames)")
            return
        files = [e for e in entries if e.is_file()]
        dirs = [e for e in entries if e.is_dir()]
        shown_files = files[:max_files_per_dir]
        for i, e in enumerate(shown_files):
            lines.append(prefix + "├── " + e.name)
        if len(files) > max_files_per_dir:
            lines.append(prefix + f"├── … (+{len(files) - max_files_per_dir} fichiers)")
        for j, e in enumerate(dirs):
            last = j == len(dirs) - 1
            lines.append(prefix + ("└── " if last else "├── ") + e.name + "/")
            walk(e, prefix + ("    " if last else "│   "))

    walk(dest, "")
    return "\n".join(lines)


def _self_test(
    brief: Dict[str, Any], dest: Path, out_dir: Path, outputs_untouched: bool,
    missing_images: List[str], mtimes_before: Dict[str, int],
) -> Dict[str, Any]:
    checks: List[Tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    clips = brief.get("clips", []) or []
    chk("dossier package créé", dest.is_dir())
    chk("outputs/ inchangé (mtimes)", outputs_untouched, f"{len(mtimes_before)} fichiers")
    chk("sources d'origine toujours présentes",
        (out_dir / f"regen_brief_{brief.get('video_id')}.json").exists())

    clip_dirs = sorted((dest / "clips").glob("clip_*")) if (dest / "clips").exists() else []
    chk("nb dossiers clips == nb clips du brief", len(clip_dirs) == len(clips),
        f"{len(clip_dirs)} vs {len(clips)}")

    clips_ok = True
    for c in clips:
        cdir = dest / "clips" / c["clip_id"]
        fp = cdir / "flow_prompt.txt"
        if not (fp.exists() and fp.read_text(encoding="utf-8").strip()):
            clips_ok = False
        if not (cdir / "post_production.json").exists():
            clips_ok = False
        if c.get("type") == "static_card":
            if not (cdir / "card.jpg").exists():
                clips_ok = False
        else:
            if not ((cdir / "start.jpg").exists() and (cdir / "end.jpg").exists()):
                clips_ok = False
    chk("chaque clip : flow_prompt.txt + post_production.json + images attendues", clips_ok)

    vo = dest / "montage" / "full_voiceover_script.txt"
    full = brief.get("post_production_global", {}).get("full_voiceover_script", "")
    vo_ok = vo.exists() and vo.read_text(encoding="utf-8").strip() != "" and (
        not full or vo.read_text(encoding="utf-8").strip() == full.strip()
    )
    # « contient le dernier segment » : le dernier mot/segment du script complet est présent.
    last_tail = full.strip().split(".")[-2].strip() if full.strip().count(".") >= 1 else full.strip()
    if vo.exists() and last_tail:
        vo_ok = vo_ok and last_tail in vo.read_text(encoding="utf-8")
    chk("montage/full_voiceover_script.txt non vide + complet (contient le dernier segment)",
        vo_ok, f"tail='{last_tail[:30]}'")

    chk("montage : voice_profile + music_track + timeline présents",
        all((dest / "montage" / f).exists() for f in
            ("voice_profile.json", "music_track.json", "on_screen_text_timeline.json")))
    chk("README.md présent", (dest / "README.md").exists())

    passed = sum(1 for _, ok, _ in checks if ok)
    return {
        "ok": passed == len(checks),
        "passed": passed, "total": len(checks),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "dest": dest.as_posix(),
        "clip_count": len(clips),
        "missing_images": missing_images,
        "tree": _build_tree(dest),
    }
