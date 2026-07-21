import json
from pathlib import Path

import pytest

# Dépendances de rendu : sauter proprement si absentes.
pytest.importorskip("reportlab")
pytest.importorskip("arabic_reshaper")
pytest.importorskip("bidi")

from src.video_agent import report_pdf as rp
from src.video_agent.report_pdf import ReportPdfError


ARABIC = "زنقة السعين"


def _report():
    return {
        "report_type": "final_video_report",
        "video_path": "vidéos/demo_darija.mp4",
        "language": "ar",
        "scene_count": 2,
        "total_duration_s": 16.0,
        "scenes": [
            {
                "segment_id": 1, "start_s": 0.0, "end_s": 10.0, "confidence": 0.6,
                "scene_description": f"Des enfants devant une porte. Légende {ARABIC}.",
                "people_objects": ["enfants", "porte"],
                "actions": ["sonner", "rire"],
                "audio_visual_relation": "parole approximative",
                "combined_text": "مرحبا",
            },
            {
                "segment_id": 2, "start_s": 10.0, "end_s": 16.0, "confidence": 0.78,
                "scene_description": "Une femme en colère menace.",
                "people_objects": ["femme"], "actions": ["fuir"],
                "audio_visual_relation": "réprimande", "combined_text": "سلام",
            },
        ],
        "aggregate": {
            "people_objects": ["enfants", "porte", "femme"],
            "actions": ["sonner", "rire", "fuir"],
            "tone": ["humoristique", "tension/colère"],
            "average_confidence": 0.69,
        },
        "readable_summary": f"Rapport vidéo final — demo_darija\nLangue: ar\nScène avec {ARABIC}.\n",
    }


# --- _shape ---------------------------------------------------------------------
def test_shape_latin_identity():
    assert rp._shape("abc 123") == "abc 123"
    assert rp._shape(None) == ""


def test_shape_arabic_reorders():
    out = rp._shape(ARABIC)
    assert isinstance(out, str) and out
    # Le réordonnancement RTL / reshaping change la séquence par rapport au logique.
    assert out != ARABIC


# --- _find_font -----------------------------------------------------------------
def test_find_font_autodetect():
    # Au moins une police candidate doit exister (Windows : Tahoma/Arial).
    try:
        path = rp._find_font(None)
    except ReportPdfError:
        pytest.skip("aucune police arabe-capable sur ce système")
    assert path.lower().endswith(".ttf")


def test_find_font_missing_override(tmp_path):
    with pytest.raises(ReportPdfError):
        rp._find_font(str(tmp_path / "nope.ttf"))


# --- _join ----------------------------------------------------------------------
def test_join():
    assert rp._join([]) == "—"
    assert rp._join(["a", "b"]) == "a ; b"


# --- build_report_pdf -----------------------------------------------------------
def test_build_report_pdf_writes_pdf(tmp_path):
    try:
        rp._find_font(None)
    except ReportPdfError:
        pytest.skip("aucune police arabe-capable")
    out = tmp_path / "report.pdf"
    path = rp.build_report_pdf(_report(), str(out))
    assert Path(path) == out
    assert out.exists()
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:5] == b"%PDF-"  # en-tête PDF valide


def test_build_report_pdf_not_dict(tmp_path):
    with pytest.raises(ReportPdfError):
        rp.build_report_pdf(["x"], str(tmp_path / "r.pdf"))


def test_build_report_pdf_from_file(tmp_path):
    try:
        rp._find_font(None)
    except ReportPdfError:
        pytest.skip("aucune police arabe-capable")
    rj = tmp_path / "final_report_demo_darija.json"
    rj.write_text(json.dumps(_report(), ensure_ascii=False), encoding="utf-8")
    out = rp.build_report_pdf_from_file(str(rj))
    p = Path(out)
    assert p.exists() and p.suffix == ".pdf"
    assert p.name == "final_report_demo_darija.pdf"  # stem dérivé


def test_build_report_pdf_from_file_missing(tmp_path):
    with pytest.raises(ReportPdfError):
        rp.build_report_pdf_from_file(str(tmp_path / "nope.json"))
