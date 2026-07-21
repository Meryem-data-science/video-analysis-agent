"""Niveau 13 — Export PDF du rapport vidéo final.

Met en forme le rapport final (`final_report_<stem>.json`) en PDF lisible. Le
rapport contient de l'**arabe (Darija)** : on garantit un rendu correct (pas de
carrés vides, sens correct) via une police Unicode arabe-capable (Tahoma/Arial) +
**arabic-reshaper** (formes contextuelles) + **python-bidi** (réordonnancement RTL).

Backend : reportlab (pur Python — pas de dépendance native type GTK/Pango).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape


class ReportPdfError(Exception):
    """Erreur de l'export PDF du rapport."""


# Polices candidates (Latin + arabe, avec formes de présentation arabes).
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/trado.ttf",
]
_FONT_NAME = "ReportArabic"


def _find_font(font_path: Optional[str]) -> str:
    """Retourne le chemin d'une police arabe-capable (override ou auto-détection)."""
    if font_path:
        p = Path(font_path)
        if not p.exists():
            raise ReportPdfError(f"Police introuvable : {p}")
        return str(p)
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise ReportPdfError(
        "Aucune police Unicode arabe-capable trouvée (Tahoma/Arial). "
        "Fournissez --font vers un .ttf compatible (ex. Noto Sans Arabic)."
    )


def _shape(text: Any) -> str:
    """Reshape l'arabe (formes contextuelles) + réordonne RTL.

    No-op sur du texte purement latin. Tout caractère arabe est rendu lisible
    dans le bon sens.
    """
    if text is None:
        return ""
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(str(text)))


def _para(text: Any, style) -> Any:
    """Construit un Paragraph reportlab à partir d'un texte mis en forme arabe + échappé."""
    from reportlab.platypus import Paragraph

    return Paragraph(escape(_shape(text)), style)


def _join(items: Any) -> str:
    if not items:
        return "—"
    return " ; ".join(str(x) for x in items)


def build_report_pdf(
    report: Dict[str, Any],
    output_path: str,
    font_path: Optional[str] = None,
) -> str:
    """Génère le PDF du rapport final et retourne le chemin de sortie.

    Args:
        report: rapport vidéo final (dict, Niveau 12).
        output_path: chemin du PDF à écrire.
        font_path: police TTF arabe-capable (sinon auto-détection système).
    """
    if not isinstance(report, dict):
        raise ReportPdfError("`report` doit être un dictionnaire.")

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Spacer

    font_file = _find_font(font_path)
    try:
        pdfmetrics.registerFont(TTFont(_FONT_NAME, font_file))
    except Exception as exc:  # police illisible / non TTF
        raise ReportPdfError(f"Police inutilisable ({font_file}) : {exc}") from exc

    stem = Path(report.get("video_path", "video")).stem or "video"
    agg = report.get("aggregate", {}) or {}

    base = ParagraphStyle("base", fontName=_FONT_NAME, fontSize=10, leading=14, alignment=TA_LEFT)
    title = ParagraphStyle("title", parent=base, fontSize=18, leading=22, spaceAfter=4,
                           textColor=colors.HexColor("#1a3a5a"))
    meta = ParagraphStyle("meta", parent=base, fontSize=10, leading=14, textColor=colors.HexColor("#444444"))
    h2 = ParagraphStyle("h2", parent=base, fontSize=12, leading=16, spaceBefore=10, spaceAfter=2,
                        textColor=colors.HexColor("#1a3a5a"))
    label = ParagraphStyle("label", parent=base, fontSize=9.5, leading=13, textColor=colors.HexColor("#333333"))
    body = ParagraphStyle("body", parent=base, fontSize=10, leading=14)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Rapport vidéo final — {stem}",
    )

    story: List[Any] = []
    story.append(_para(f"Rapport vidéo final — {stem}", title))
    story.append(_para(
        f"Vidéo : {report.get('video_path', '')}   |   Langue : {report.get('language') or '?'}",
        meta))
    story.append(_para(
        f"Durée : {report.get('total_duration_s')} s   |   "
        f"Scènes : {report.get('scene_count')}   |   "
        f"Confiance moyenne : {agg.get('average_confidence')}",
        meta))
    story.append(_para(f"Ton global : {_join(agg.get('tone'))}", meta))
    story.append(Spacer(1, 6 * mm))

    for sc in report.get("scenes", []) or []:
        sid = sc.get("segment_id")
        story.append(_para(
            f"Scène {sid} — [{sc.get('start_s')}–{sc.get('end_s')} s] "
            f"(confiance {sc.get('confidence')})", h2))
        story.append(_para(sc.get("scene_description", ""), body))
        story.append(_para(f"Personnages / objets : {_join(sc.get('people_objects'))}", label))
        story.append(_para(f"Actions : {_join(sc.get('actions'))}", label))
        story.append(_para(f"Relation audio-visuel : {sc.get('audio_visual_relation', '—')}", label))
        if sc.get("combined_text"):
            story.append(_para(f"Parole : {sc.get('combined_text')}", label))
        story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 4 * mm))
    story.append(_para("Résumé global", h2))
    for line in (report.get("readable_summary", "") or "").splitlines():
        if line.strip():
            story.append(_para(line, body))
        else:
            story.append(Spacer(1, 2 * mm))

    try:
        doc.build(story)
    except Exception as exc:
        raise ReportPdfError(f"Échec de génération du PDF : {exc}") from exc

    logging.info("build_report_pdf done -> %s", out)
    return str(out)


def build_report_pdf_from_file(
    report_json_path: str,
    output_path: Optional[str] = None,
    font_path: Optional[str] = None,
) -> str:
    """Charge un rapport JSON et génère le PDF (sortie dérivée du stem si non fournie)."""
    import json

    p = Path(report_json_path)
    if not p.exists() or not p.is_file():
        raise ReportPdfError(f"Rapport JSON introuvable : {p}")
    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportPdfError(f"Rapport JSON invalide ({p}) : {exc}") from exc

    if output_path is None:
        stem = Path(report.get("video_path", "video")).stem or "video"
        output_path = str(p.parent / f"final_report_{stem}.pdf")

    return build_report_pdf(report, output_path, font_path=font_path)
