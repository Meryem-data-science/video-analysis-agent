import json

import pytest

from src.video_agent import final_report as fr
from src.video_agent.final_report import FinalReportError


# --- Helpers --------------------------------------------------------------------
def _vlm(seg_id, conf, people, actions, uses_speech=True, scene="scène", relation="rel",
         uncertainties=None):
    bounds = {1: (0.0, 10.0, "مرحبا"), 2: (10.0, 16.0, "سلام")}
    start_s, end_s, combined_text = bounds.get(seg_id, (0.0, 1.0, ""))
    return {
        "segment_id": seg_id,
        "start_s": start_s,
        "end_s": end_s,
        "combined_text": combined_text,
        "scene_description": scene,
        "people_objects": people,
        "actions": actions,
        "motion_description": "m",
        "audio_visual_relation": relation,
        "uses_speech": uses_speech,
        "mood": [],
        "texte_ecran": [],
        "confidence": conf,
        "uncertainties": uncertainties or [],
        "evidence": {"visual": ["v"], "audio": ["a"]},
    }


def _fusion():
    return {
        "video_path": "vidéos/demo_darija.mp4",
        "language": "ar",
        "summary": {"audio_activity_mode": "active_audio"},
        "segments": [
            {"segment_id": 1, "start_s": 0.0, "end_s": 10.0, "duration_s": 10.0, "combined_text": "مرحبا"},
            {"segment_id": 2, "start_s": 10.0, "end_s": 16.0, "duration_s": 6.0, "combined_text": "سلام"},
        ],
    }


def _results():
    return {
        1: _vlm(1, 0.6, ["enfants", "porte"], ["sonner"], scene="Une farce humoristique d'enfants",
                uncertainties=["transcription incertaine"]),
        2: _vlm(2, 0.78, ["vieille femme", "enfants"], ["fuir"],
                scene="Une femme en colère menace", relation="ton de réprimande",
                uncertainties=["transcription incertaine", "nombre d'enfants"]),
    }


# --- helpers purs ---------------------------------------------------------------
def test_dedup_preserve():
    assert fr._dedup_preserve(["a", "b", "a", " a ", "", "c"]) == ["a", "b", "c"]


def test_detect_tone_hints():
    assert "humoristique" in fr._detect_tone_hints(["c'est une farce, rires"])
    assert "tension/colère" in fr._detect_tone_hints(["la femme est en colère"])
    assert fr._detect_tone_hints(["rien de notable"]) == ["indéterminé"]


def test_truncate():
    assert fr._truncate("abc", 10) == "abc"
    out = fr._truncate("x" * 50, 10)
    assert out.endswith("…") and len(out) == 10


# --- build_final_report ---------------------------------------------------------
def test_build_final_report_structure():
    report = fr.build_final_report(_results(), _fusion(), summary=None,
                                   sources={"results_dir": "x"})
    assert report["report_type"] == "final_video_report"
    assert report["status"] == "complete"
    assert report["next_step"] is None
    assert report["language"] == "ar"
    assert report["scene_count"] == 2
    assert report["total_duration_s"] == 16.0
    # scènes horodatées, triées par segment_id
    assert [s["segment_id"] for s in report["scenes"]] == [1, 2]
    assert report["scenes"][0]["start_s"] == 0.0 and report["scenes"][0]["end_s"] == 10.0
    assert report["scenes"][0]["combined_text"] == "مرحبا"


def test_build_final_report_aggregate():
    report = fr.build_final_report(_results(), _fusion())
    agg = report["aggregate"]
    # union dédupliquée
    assert agg["people_objects"] == ["enfants", "porte", "vieille femme"]
    assert agg["actions"] == ["sonner", "fuir"]
    assert "humoristique" in agg["tone"] or "tension/colère" in agg["tone"]
    assert agg["languages"] == ["ar"]
    assert agg["uses_speech_any"] is True
    assert agg["segments_with_speech"] == 2
    assert agg["audio_activity_mode"] == "active_audio"
    # confiance moyenne calculée si pas de summary
    assert agg["average_confidence"] == pytest.approx(0.69)
    assert agg["uncertainties"] == ["transcription incertaine", "nombre d'enfants"]


def test_build_final_report_uses_summary_confidence():
    summary = {"summary": {"average_confidence": 0.42}}
    report = fr.build_final_report(_results(), _fusion(), summary=summary)
    assert report["aggregate"]["average_confidence"] == 0.42


def test_readable_summary_content():
    report = fr.build_final_report(_results(), _fusion())
    text = report["readable_summary"]
    assert "Rapport vidéo final" in text
    assert "Scènes :" in text
    assert "ar" in text
    assert "مرحبا" in text or "farce" in text.lower()  # contenu présent
    assert "Ton global" in text


def test_global_summary_present():
    report = fr.build_final_report(_results(), _fusion())
    assert report["global_summary"].startswith("Vidéo en ar")


# --- corrections Niveau "audio dynamique" --------------------------------------
def _fusion_en_lowconf():
    return {
        "video_path": "vidéos/clip.mp4",
        "language": "en",
        "summary": {"audio_activity_mode": "active_audio"},
        "segments": [
            {"segment_id": 1, "start_s": 0.0, "end_s": 10.0, "duration_s": 10.0,
             "combined_text": "Thank you",
             "transcript_segments": [{"start_s": 0.0, "end_s": 2.0, "text": "Thank you", "confidence": 0.5}]},
        ],
    }


def test_audio_line_dynamic_language_no_darija():
    results = {1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=True)}
    report = fr.build_final_report(results, _fusion_en_lowconf())
    text = report["readable_summary"]
    assert "langue détectée : en" in text
    assert "Darija" not in text  # plus de libellé statique
    # confiance transcription faible -> mention "approximative"
    assert "approximative" in text.lower()
    assert report["aggregate"]["average_transcript_confidence"] == 0.5


def test_audio_line_no_speech():
    results = {1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=False)}
    fusion = {"video_path": "v.mp4", "language": None, "summary": {}, "segments": [
        {"segment_id": 1, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0, "combined_text": ""}]}
    report = fr.build_final_report(results, fusion)
    assert "aucune parole exploitable" in report["readable_summary"]
    assert "Darija" not in report["readable_summary"]


def test_audio_line_high_conf_no_approx():
    results = {1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=True)}
    fusion = {"video_path": "v.mp4", "language": "fr", "summary": {"audio_activity_mode": "active_audio"},
              "segments": [{"segment_id": 1, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0,
                            "transcript_segments": [{"start_s": 0, "end_s": 2, "text": "bonjour", "confidence": 0.92}]}]}
    report = fr.build_final_report(results, fusion)
    assert "langue détectée : fr" in report["readable_summary"]
    assert "approximative" not in report["readable_summary"].lower()


# --- correction : incertitudes "pas d'audio" exclues du global -----------------
def test_no_audio_uncertainty_excluded_when_speech():
    # Correction #2 : audio ACTIF -> l'incertitude « absence d'audio » est retirée au
    # niveau SCÈNE *et* agrégat, et remplacée (segment sans parole) par l'énoncé honnête.
    results = {
        1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=True,
                uncertainties=["correspondance audio/visuel incertaine"]),
        2: _vlm(2, 0.7, ["z"], ["w"], uses_speech=False,
                uncertainties=["Absence d'audio/parole : impossible de confirmer un contexte sonore",
                               "détails flous"]),
    }
    report = fr.build_final_report(results, _fusion())  # fusion ar, active_audio
    uncert = report["aggregate"]["uncertainties"]
    assert any("correspondance audio" in u for u in uncert)
    assert any("détails flous" in u for u in uncert)
    # l'incertitude « pas d'audio » est exclue de l'agrégat
    assert not any("absence d'audio" in u.lower() for u in uncert)
    # ET retirée au niveau de la scène (audio actif), remplacée par l'énoncé honnête
    sc2 = report["scenes"][1]["uncertainties"]
    assert not any("absence d'audio" in u.lower() for u in sc2)
    assert any("non verbal actif" in u.lower() for u in sc2)
    assert report["aggregate"]["non_speech_audio_present"] is True


def test_no_audio_uncertainty_kept_when_no_speech():
    results = {1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=False,
                       uncertainties=["No audio available"])}
    fusion = {"video_path": "v.mp4", "language": None, "summary": {}, "segments": [
        {"segment_id": 1, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0}]}
    report = fr.build_final_report(results, fusion)
    # vidéo sans parole -> on garde l'incertitude (pas de contradiction)
    assert any("no audio" in u.lower() for u in report["aggregate"]["uncertainties"])


# --- correction : dédup insensible casse/ponctuation ---------------------------
def test_dedup_case_and_punct_insensitive():
    out = fr._dedup_preserve(["Courir", "courir.", "COURIR", "Sauter"])
    assert out == ["Courir", "Sauter"]


# --- contrat de données (schema_version, champs toujours présents) -------------
_SCENE_KEYS = [
    "segment_id", "start_s", "end_s", "duration_s", "scene_description",
    "people_objects", "actions", "motion_description", "audio_visual_relation",
    "uses_speech", "mood", "combined_text", "confidence", "uncertainties",
    "texte_ecran", "evidence", "audio_scene", "audio_visual_coherence",
]


def test_schema_version_present():
    report = fr.build_final_report(_results(), _fusion())
    assert report["schema_version"] == "1.11"
    # schema_version en première clé du JSON
    assert list(report.keys())[0] == "schema_version"


# --- texte à l'écran : lu NATIVEMENT par le VLM (champ texte_ecran) -------------
def _vlm_min(seg_id, scene="", people=None, actions=None, uncertainties=None,
             texte_ecran=None):
    return {"segment_id": seg_id, "scene_description": scene,
            "people_objects": people or [], "actions": actions or [],
            "motion_description": "", "audio_visual_relation": "", "uses_speech": True,
            "confidence": 0.7, "uncertainties": uncertainties or [],
            "texte_ecran": texte_ecran or [],
            "evidence": {"visual": [], "audio": []}}


def _se(text, lang="en", location="bas"):
    """Une entrée texte_ecran VLM {text, lang, location}."""
    return {"text": text, "lang": lang, "location": location}


def test_screen_text_injected_per_scene():
    results = {1: _vlm_min(1, texte_ecran=[_se("beko")]), 2: _vlm_min(2)}
    report = fr.build_final_report(results, _fusion())
    sc1 = next(s for s in report["scenes"] if s["segment_id"] == 1)
    sc2 = next(s for s in report["scenes"] if s["segment_id"] == 2)
    assert [d["text"] for d in sc1["texte_ecran"]] == ["beko"]
    assert sc1["texte_ecran"][0]["lang"] == "en"
    assert sc1["texte_ecran"][0]["location"] == "bas"
    assert "confidence" not in sc1["texte_ecran"][0]  # plus de confidence (lecture VLM)
    assert sc2["texte_ecran"] == []
    keys = list(sc1.keys())
    assert keys[keys.index("uncertainties") + 1] == "texte_ecran"


def test_screen_text_cleaning():
    results = {1: _vlm_min(1, texte_ecran=[_se("="), _se("x"), _se("90"), _se("beko")])}
    report = fr.build_final_report(results, _fusion())
    texts = [d["text"] for d in next(s for s in report["scenes"] if s["segment_id"] == 1)["texte_ecran"]]
    assert "=" not in texts and "x" not in texts   # < 2 car. / non alphanumérique jetés
    assert "90" in texts and "beko" in texts


def test_screen_text_fuzzy_dedup():
    results = {1: _vlm_min(1, texte_ecran=[_se("زنقة", lang="ar"), _se("زنفة", lang="ar")])}
    te = next(s for s in fr.build_final_report(results, _fusion())["scenes"]
              if s["segment_id"] == 1)["texte_ecran"]
    assert len(te) == 1 and te[0]["text"] == "زنقة"  # 1re occurrence gardée


def test_aggregate_texte_ecran_detecte():
    results = {1: _vlm_min(1, texte_ecran=[_se("beko")]),
               2: _vlm_min(2, texte_ecran=[_se("beko"), _se("liga")])}
    agg = fr.build_final_report(results, _fusion())["aggregate"]["texte_ecran_detecte"]
    beko = next(e for e in agg if e["text"] == "beko")
    assert beko["scenes"] == [1, 2]
    assert "confidence" not in beko  # agrégat VLM : pas de confidence
    assert next(e for e in agg if e["text"] == "liga")["scenes"] == [2]


def test_screen_text_empty_when_vlm_has_none():
    # _results() ne fournit pas de texte_ecran -> [] partout (jamais d'invention).
    report = fr.build_final_report(_results(), _fusion())
    assert all(s["texte_ecran"] == [] for s in report["scenes"])
    assert report["aggregate"]["texte_ecran_detecte"] == []


# --- étape 5a : consommation de l'audio par scène (scene_audio) -----------------
def _audio_events(mood="énergique"):
    return {
        "music": {"present": True, "confidence": 0.8, "mood": mood, "mood_confidence": 0.7,
                  "energy": "moyenne", "tempo_feel": "modéré"},
        "sound_effects": [{"label": "whoosh", "confidence": 0.6}],
        "speech_overlap": False,
        "evidence": "musique de fond continue",
        "uncertainties": ["incertain sur le genre"],
    }


def _scene_audio(events_map, model="gemini-3-flash-preview", status_map=None):
    """Artefact scene_audio factice ; events_map: {scene_index: events|None}."""
    scenes = []
    for k, ev in events_map.items():
        st = (status_map or {}).get(k, "ok" if ev is not None else "échec_gracieux")
        scenes.append({"scene_index": k, "start_s": 0.0, "end_s": 1.0,
                       "audio_events": ev or {}, "audio_status": st})
    return {"schema": "scene_audio/1.0", "video_path": "v.mp4",
            "model_used": model, "status": "ok", "scenes": scenes}


def test_audio_events_merged_per_scene():
    sa = _scene_audio({1: _audio_events("calme"), 2: _audio_events("tendu")})
    report = fr.build_final_report(_results(), _fusion(), scene_audio=sa)
    sc1 = next(s for s in report["scenes"] if s["segment_id"] == 1)
    sc2 = next(s for s in report["scenes"] if s["segment_id"] == 2)
    assert sc1["audio_scene"]["events"]["music"]["mood"] == "calme"
    assert sc2["audio_scene"]["events"]["music"]["mood"] == "tendu"
    assert sc1["audio_scene"]["status"] == "ok"
    # bloc présent juste après evidence (ordre stable)
    keys = list(sc1.keys())
    assert keys[keys.index("evidence") + 1] == "audio_scene"


def test_audio_provenance_present():
    sa = _scene_audio({1: _audio_events(), 2: _audio_events()}, model="gemini-2.5-flash")
    report = fr.build_final_report(_results(), _fusion(), scene_audio=sa)
    for sc in report["scenes"]:
        assert sc["audio_scene"]["source"] == "gemini-audio"
        assert sc["audio_scene"]["model"] == "gemini-2.5-flash"


def test_audio_failed_scene_marked_but_no_invention():
    # scène 2 en échec gracieux côté audio -> events vide, statut reporté, pas d'invention.
    sa = _scene_audio({1: _audio_events(), 2: None})
    report = fr.build_final_report(_results(), _fusion(), scene_audio=sa)
    sc2 = next(s for s in report["scenes"] if s["segment_id"] == 2)
    assert sc2["audio_scene"]["source"] == "gemini-audio"
    assert sc2["audio_scene"]["status"] == "échec_gracieux"
    assert sc2["audio_scene"]["events"] == {}


def test_audio_degradation_none():
    report = fr.build_final_report(_results(), _fusion(), scene_audio=None)
    for sc in report["scenes"]:
        av = sc["audio_scene"]
        assert av["source"] is None and av["model"] is None
        assert av["status"] == "absent" and av["events"] == {}
        # measured_features (librosa) toujours présent, défaut « absent ».
        assert av["measured_features"] == fr._default_measured_features()


def test_audio_degradation_invalid_artifact():
    # artefact mal formé (scenes non liste) -> rapport construit sans audio, pas de crash.
    report = fr.build_final_report(_results(), _fusion(), scene_audio={"scenes": "oops"})
    for sc in report["scenes"]:
        assert sc["audio_scene"]["source"] is None
        assert sc["audio_scene"]["events"] == {}


def test_audio_partial_coverage_unmatched_scene_stays_absent():
    # seule la scène 1 a de l'audio ; la scène 2 garde le bloc « absent ».
    sa = _scene_audio({1: _audio_events()})
    report = fr.build_final_report(_results(), _fusion(), scene_audio=sa)
    sc2 = next(s for s in report["scenes"] if s["segment_id"] == 2)
    assert sc2["audio_scene"]["source"] is None and sc2["audio_scene"]["status"] == "absent"


# --- étape 5b : croisement déterministe son↔visuel (audio_visual_coherence) ------
def _vlm_mood(seg_id, mood, conf=0.8):
    """VLM minimal AVEC humeur visuelle de scène (champ `mood`)."""
    v = _vlm(seg_id, conf, ["x"], ["y"])
    v["mood"] = mood if isinstance(mood, list) else [mood]
    return v


def _events_mood(mood, conf=0.8):
    ev = _audio_events(mood)
    ev["music"]["mood_confidence"] = conf
    return ev


def _one_scene_fusion():
    return {"video_path": "v.mp4", "language": "fr",
            "summary": {"audio_activity_mode": "active_audio"},
            "segments": [{"segment_id": 1, "start_s": 0.0, "end_s": 5.0,
                          "duration_s": 5.0, "combined_text": ""}]}


def _coh(report):
    return report["scenes"][0]["audio_visual_coherence"]


def test_av_coherence_coherent():
    # (a) audio présent + même polarité (positif/positif) -> cohérent.
    results = {1: _vlm_mood(1, ["ludique"], conf=0.8)}
    sa = _scene_audio({1: _events_mood("joyeux", 0.9)})
    report = fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa)
    coh = _coh(report)
    assert coh["verdict"] == "cohérent"
    assert coh["audio_mood"] == "joyeux" and coh["visual_tone"] == "ludique"
    assert coh["audio_polarity"] == "positif" and coh["visual_polarity"] == "positif"
    assert coh["visual_tone_source"] == "vlm-mood"
    assert coh["confidence"] == 0.8  # min(0.9, 0.8)
    audit = report["sound_design_audit"]
    assert audit["cohérent"] == 1 and audit["scenes_croisées"] == 1
    assert audit["couverture"] == 1.0
    # cohérent -> AUCUNE note d'incertitude ajoutée.
    assert not any("contraste" in u.lower() for u in report["scenes"][0]["uncertainties"])


def test_av_coherence_contraste_enriches_uncertainties():
    # (b) audio présent + polarités opposées (positif vs tension) -> contraste + note.
    results = {1: _vlm_mood(1, ["tendu"], conf=0.8)}
    sa = _scene_audio({1: _events_mood("joyeux", 0.8)})
    report = fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa)
    coh = _coh(report)
    assert coh["verdict"] == "contraste"
    assert coh["visual_polarity"] == "tension" and coh["audio_polarity"] == "positif"
    assert report["sound_design_audit"]["contraste"] == 1
    notes = report["scenes"][0]["uncertainties"]
    assert any("contraste audio↔visuel" in u.lower() for u in notes)


def test_av_audio_non_evalue_when_no_audio_keeps_visual():
    # (c) audio absent -> audio_non_évalué MAIS mood visuel conservé comme socle.
    results = {1: _vlm_mood(1, ["joie"], conf=0.8)}
    report = fr.build_final_report(results, _one_scene_fusion(), scene_audio=None)
    coh = _coh(report)
    assert coh["verdict"] == "audio_non_évalué"
    assert coh["visual_tone"] == "joie" and coh["visual_polarity"] == "positif"
    assert coh["audio_mood"] is None
    # le mood VISUEL riche reste intact dans la scène (socle non appauvri).
    assert report["scenes"][0]["mood"] == ["joie"]
    # aucune note criarde ajoutée.
    assert not any("contraste" in u.lower() for u in report["scenes"][0]["uncertainties"])
    audit = report["sound_design_audit"]
    assert audit["audio_non_évalué"] == 1 and audit["scenes_croisées"] == 0
    assert audit["couverture"] == 0.0


def test_av_audio_non_evalue_when_gemini_failed():
    # audio Gemini en échec_gracieux (ex. 503) -> audio_non_évalué, visuel conservé.
    results = {1: _vlm_mood(1, ["colère"], conf=0.8)}
    sa = _scene_audio({1: None})  # ev None -> audio_status échec_gracieux
    coh = _coh(fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa))
    assert coh["verdict"] == "audio_non_évalué"
    assert coh["visual_tone"] == "colère" and coh["visual_polarity"] == "tension"


def test_av_audio_non_evalue_when_mood_unexploitable():
    # audio OK mais ambiance « incertain » -> audio_non_évalué (jamais un contraste).
    results = {1: _vlm_mood(1, ["joie"], conf=0.8)}
    sa = _scene_audio({1: _events_mood("incertain", 0.8)})
    coh = _coh(fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa))
    assert coh["verdict"] == "audio_non_évalué"
    assert coh["visual_tone"] == "joie"


def test_av_negatif_vs_tension_is_coherent():
    # négatif (visuel) vs tension (audio) -> même signe négatif -> cohérent (pas contraste).
    results = {1: _vlm_mood(1, ["tristesse"], conf=0.8)}
    sa = _scene_audio({1: _events_mood("menaçant", 0.8)})
    coh = _coh(fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa))
    assert coh["visual_polarity"] == "négatif" and coh["audio_polarity"] == "tension"
    assert coh["verdict"] == "cohérent"


def test_av_neutre_never_contrasts():
    # neutre (audio inconnu) vs positif (visuel) -> jamais de contraste -> cohérent.
    results = {1: _vlm_mood(1, ["joie"], conf=0.8)}
    sa = _scene_audio({1: _events_mood("énergique et synthétique", 0.8)})  # inconnu -> neutre
    coh = _coh(fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa))
    assert coh["audio_polarity"] == "neutre"
    assert coh["verdict"] == "cohérent"


def test_av_missing_visual_no_crash_no_indetermine():
    # VLM sans mood + audio positif -> pas de crash, jamais « indéterminé ».
    results = {1: _vlm_mood(1, [], conf=0.8)}  # mood vide
    sa = _scene_audio({1: _events_mood("joyeux", 0.9)})
    coh = _coh(fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa))
    assert coh["visual_tone"] is None and coh["visual_polarity"] == "neutre"
    assert coh["verdict"] != "indéterminé"
    assert coh["verdict"] == "cohérent"  # neutre vs positif -> pas de contraste


def test_av_audit_counters_and_coverage():
    # absence totale d'audio -> tout en audio_non_évalué, couverture 0, plus de clé « indéterminé ».
    results = {1: _vlm_mood(1, ["joie"], conf=0.8), 2: _vlm_mood(2, ["calme"], conf=0.8)}
    fusion = {"video_path": "v.mp4", "language": "fr",
              "summary": {"audio_activity_mode": "active_audio"},
              "segments": [
                  {"segment_id": 1, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0, "combined_text": ""},
                  {"segment_id": 2, "start_s": 5.0, "end_s": 9.0, "duration_s": 4.0, "combined_text": ""}]}
    report = fr.build_final_report(results, fusion, scene_audio=None)
    audit = report["sound_design_audit"]
    assert audit["total_scenes"] == 2 and audit["audio_non_évalué"] == 2
    assert audit["cohérent"] == 0 and audit["contraste"] == 0 and audit["scenes_croisées"] == 0
    assert audit["couverture"] == 0.0
    assert "audio non évalué" in audit["summary"]
    assert "indéterminé" not in audit  # la catégorie n'existe plus


def test_av_coherence_block_always_present_and_ordered():
    report = fr.build_final_report(_results(), _fusion())  # sans audio ni mood
    for sc in report["scenes"]:
        keys = list(sc.keys())
        assert keys[keys.index("audio_scene") + 1] == "audio_visual_coherence"
        assert sc["audio_visual_coherence"]["verdict"] == "audio_non_évalué"


# --- polarité canonique : couverture lexicale du vocabulaire + tolérance ------------
def test_polarite_required_vocabulary():
    pol = fr._polarite
    for w in ("détresse", "compassion", "tristesse", "supplication", "déni", "désarroi",
              "gêne", "déception", "inquiétude", "doute", "scepticisme", "anxiété",
              "hésitation"):
        assert pol(w) == "négatif", w
    for w in ("joie", "enthousiasme", "fierté", "soulagement", "gratitude", "satisfaction",
              "confiance", "espoir", "optimisme", "calme", "légèreté", "tendresse",
              "humour", "ironie"):
        assert pol(w) == "positif", w
    for w in ("tension", "menace", "colère", "agacement", "exaspération", "indignation"):
        assert pol(w) == "tension", w
    for w in ("surprise", "incrédulité"):
        assert pol(w) == "neutre", w


def test_polarite_collision_resolved_by_order():
    # « déception légère » contient 'leger' (positif) ET 'decept' (négatif) -> négatif gagne.
    assert fr._polarite("déception légère") == "négatif"
    assert fr._polarite("déception amusée") == "négatif"


def test_polarite_unknown_falls_back_to_neutre():
    # TOLÉRANCE TOTALE : terme inconnu / pur timbre / vide -> neutre (jamais d'erreur).
    for w in ("énergique", "dynamique", "vif", "synthétique", "atmosphérique",
              "mystérieux", "groovy", "xyzzy", "", None):
        assert fr._polarite(w) == "neutre", w


def test_no_mood_ever_produces_indetermine():
    # PREUVE : aucune combinaison (mood visuel × mood audio), connue OU inconnue, ne
    # produit « indéterminé » ; le mood visuel reste toujours le socle de la scène.
    vocab = ["détresse", "tristesse", "désarroi", "déception", "inquiétude", "doute",
             "scepticisme", "anxiété", "joie", "enthousiasme", "fierté", "calme",
             "légèreté", "tendresse", "tension", "menace", "colère", "agacement",
             "surprise", "incrédulité", "humour", "groovy", "énergique", "blarg",
             "incertain", "aucune"]
    autorises = {"cohérent", "contraste", "pas_de_musique", "audio_non_évalué"}
    for vm in vocab:
        for am in vocab:
            results = {1: _vlm_mood(1, [vm], conf=0.8)}
            sa = _scene_audio({1: _events_mood(am, 0.8)})
            coh = _coh(fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa))
            assert coh["verdict"] in autorises, (vm, am, coh["verdict"])
            assert coh["verdict"] != "indéterminé", (vm, am)
            assert coh["visual_tone"] == vm  # socle visuel toujours présent


def _events_no_music():
    ev = _audio_events("aucune")
    ev["music"]["present"] = False
    ev["music"]["mood"] = "aucune"
    return ev


def test_av_pas_de_musique_keeps_visual():
    results = {1: _vlm_mood(1, ["joie"], conf=0.8)}
    sa = _scene_audio({1: _events_no_music()})
    report = fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa)
    coh = _coh(report)
    assert coh["verdict"] == "pas_de_musique"
    assert coh["visual_tone"] == "joie"  # mood visuel (socle) conservé
    audit = report["sound_design_audit"]
    assert audit["pas_de_musique"] == 1 and audit["scenes_croisées"] == 0
    assert "sans musique" in audit["summary"]
    # pas de note d'incertitude parasite pour « pas de musique »
    assert not any("contraste" in u.lower() for u in report["scenes"][0]["uncertainties"])


def test_av_contraste_positif_vs_tension():
    results = {1: _vlm_mood(1, ["colère"], conf=0.8)}                  # tension
    sa = _scene_audio({1: _events_mood("épique et triomphant", 0.8)})  # positif
    report = fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa)
    assert _coh(report)["verdict"] == "contraste"
    assert report["sound_design_audit"]["contraste"] == 1


def test_av_emerveillement_visual_makes_coherent():
    results = {1: _vlm_mood(1, ["émerveillement"], conf=0.8)}          # positif
    sa = _scene_audio({1: _events_mood("joyeux et enthousiaste", 0.8)})  # positif
    assert _coh(fr.build_final_report(results, _one_scene_fusion(), scene_audio=sa))["verdict"] == "cohérent"


# --- étape librosa-2 : consommation des features audio mesurées (measured_features) ---
def _feat(tempo=120.0, energy=0.1, activity="actif", beats=10, reason=None):
    return {"source": "librosa", "audio_activity": activity, "energy_rms": energy,
            "tempo_bpm": tempo, "tempo_reason": reason, "tempo_note": "note tempo (facteur 2)",
            "beat_count": beats, "beat_density_per_s": 2.0}


def _audio_features(features_map, status_map=None, method_notes=None):
    scenes = []
    for k, f in features_map.items():
        st = (status_map or {}).get(k, "ok" if f is not None else "features_échec")
        scenes.append({"scene_index": k, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0,
                       "features": f or {}, "audio_status": st})
    return {"schema": "audio_features/1.0", "video_path": "v.mp4", "status": "ok",
            "params": {"sr": 22050},
            "method_notes": method_notes or {"tempo": "note", "audio_activity": "énergétique"},
            "scenes": scenes}


def _mf(report, seg_id):
    sc = next(s for s in report["scenes"] if s["segment_id"] == seg_id)
    return sc["audio_scene"]["measured_features"]


def test_measured_features_merged_per_scene():
    af = _audio_features({1: _feat(128.0, 0.2, "actif"), 2: _feat(90.0, 0.04, "quasi_silence")})
    report = fr.build_final_report(_results(), _fusion(), audio_features=af)
    mf1, mf2 = _mf(report, 1), _mf(report, 2)
    assert mf1["source"] == "librosa" and mf1["status"] == "ok"
    assert mf1["tempo_bpm"] == 128.0 and mf1["energy_rms"] == 0.2
    assert mf1["audio_activity"] == "actif" and mf1["beat_count"] == 10
    assert mf1["beat_density_per_s"] == 2.0
    assert mf1["method_notes"]["tempo"] == "note"           # method_notes propagées
    assert mf2["audio_activity"] == "quasi_silence" and mf2["tempo_bpm"] == 90.0


def test_measured_features_default_absent():
    report = fr.build_final_report(_results(), _fusion(), audio_features=None)
    for sc in report["scenes"]:
        assert sc["audio_scene"]["measured_features"] == fr._default_measured_features()


def test_measured_features_short_scene_tempo_null_preserved():
    # tempo_bpm null + raison conservés tels quels (pas d'invention).
    af = _audio_features({1: _feat(tempo=None, reason="scène trop courte (< 4 s)")})
    report = fr.build_final_report(_results(), _fusion(), audio_features=af)
    mf1 = _mf(report, 1)
    assert mf1["status"] == "ok" and mf1["tempo_bpm"] is None
    assert mf1["tempo_reason"] == "scène trop courte (< 4 s)"


def test_measured_features_coexists_with_scene_audio():
    sa = _scene_audio({1: _audio_events()})
    af = _audio_features({1: _feat(140.0)})
    report = fr.build_final_report(_results(), _fusion(), scene_audio=sa, audio_features=af)
    av = next(s for s in report["scenes"] if s["segment_id"] == 1)["audio_scene"]
    assert av["source"] == "gemini-audio"                   # bloc Gemini préservé
    assert av["measured_features"]["source"] == "librosa"   # + features librosa
    assert av["measured_features"]["tempo_bpm"] == 140.0


def test_measured_features_failed_scene_marked_no_invention():
    af = _audio_features({1: None}, status_map={1: "features_échec"})
    report = fr.build_final_report(_results(), _fusion(), audio_features=af)
    mf1 = _mf(report, 1)
    assert mf1["status"] == "features_échec"
    assert mf1["source"] is None and mf1["tempo_bpm"] is None and mf1["energy_rms"] is None


def test_measured_features_invalid_artifact():
    report = fr.build_final_report(_results(), _fusion(), audio_features={"scenes": "oops"})
    for sc in report["scenes"]:
        assert sc["audio_scene"]["measured_features"] == fr._default_measured_features()


# --- traçabilité : bloc provenance + vlm_mode propagé -------------------------------
def test_build_provenance_backfill():
    prov = fr._build_provenance(
        vlm_mode="gemini", provider="gemini",
        scene_audio={"model_used": "gemini-3-flash-preview", "status": "partiel",
                     "scenes": [{"scene_index": 16, "audio_status": "échec_gracieux"}]},
        audio_features={"params": {"sr": 22050, "tempo_min_duration_s": 4.0}},
        summary={"results": [{"valid": False, "segment_id": 5, "errors": ["x"]}]},
        frames_manifest={"frames": [{"frame_id": 1}, {"frame_id": 2}, {"frame_id": 3}]},
        motion_manifest={"segments": [{"segment_id": 1, "frame_count": 1},
                                      {"segment_id": 2, "frame_count": 6}]},
        pipeline_status={"steps": [{"step": "motion", "ok": False, "returncode": 1, "detail": "boom"}]},
        frames_manifest_path="outputs/frames_manifest_x.json",
        motion_manifest_path="outputs/motion_manifest_x.json",
    )
    assert prov["visual"]["mode"] == "gemini" and prov["visual"]["provider"] == "gemini"
    assert prov["visual"]["model"] is None and "non enregistré" in prov["visual"]["model_note"]
    assert prov["audio_semantic"] == {"engine": "gemini", "model": "gemini-3-flash-preview",
                                      "status": "partiel"}
    assert prov["audio_measured"]["engine"] == "librosa" and prov["audio_measured"]["params"]["sr"] == 22050
    assert "ocr" not in prov  # plus de provenance OCR (texte écran lu par le VLM)
    assert prov["frames"]["total"] == 3 and prov["frames"]["per_scene"] == {"1": 1, "2": 6}
    assert prov["frames"]["manifest"].endswith("frames_manifest_x.json")
    assert prov["composites"]["manifest"].endswith("motion_manifest_x.json")
    assert prov["composites"]["available"] is True
    # erreurs CONNUES agrégées : step KO + segment VLM invalide + scène audio en échec (sc16)
    srcs = {e["source"] for e in prov["errors"]}
    assert srcs == {"pipeline_step", "vlm_validation", "audio_semantic"}
    assert any(e.get("scene_index") == 16 for e in prov["errors"])
    # champs PERDUS : null + raison (jamais inventés)
    for k in ("model_visuel_exact", "prompt_version", "temps_execution", "retries"):
        assert prov["lost"][k]["value"] is None and prov["lost"][k]["reason"]


def test_build_provenance_empty_inputs_all_null():
    prov = fr._build_provenance()
    assert prov["visual"]["mode"] is None and prov["audio_semantic"]["engine"] is None
    assert prov["audio_measured"]["engine"] is None and "ocr" not in prov
    assert prov["frames"]["total"] is None and prov["composites"]["available"] is None
    assert prov["errors"] == []
    assert all(prov["lost"][k]["value"] is None for k in prov["lost"])


def test_provenance_present_and_placed_after_metadata():
    report = fr.build_final_report(_results(), _fusion())
    prov = report["provenance"]
    assert set(prov["lost"]) == {"model_visuel_exact", "prompt_version", "temps_execution", "retries"}
    keys = list(report.keys())
    assert keys[keys.index("metadata") + 1] == "provenance"


def test_vlm_mode_propagated_to_metadata_and_provenance():
    report = fr.build_final_report(_results(), _fusion(), vlm_mode="gemini")
    assert report["metadata"]["generation"]["vlm_mode"] == "gemini"
    assert report["provenance"]["visual"]["mode"] == "gemini"


# --- correction #4 : metadata techniques (lecture pure de l'inspection) ---------
def _inspection():
    return {
        "file": {"duration_s": 10.0, "size_mb": 1.6, "bitrate_kbps": 1341},
        "video": {"present": True, "codec": "h264", "width": 478, "height": 850,
                  "orientation": "vertical", "fps_used": 24.0, "avg_fps": 24.0},
        "audio": {"present": True, "codec": "aac", "sample_rate": 44100, "channels": 2},
        "derived": {"platform_hint": "mp4/mov - compatible web/mobile"},
    }


def test_metadata_complete_from_inspection():
    report = fr.build_final_report(_results(), _fusion(), inspection=_inspection())
    keys = list(report.keys())
    assert keys[keys.index("sources") + 1] == "metadata"  # position fixe après sources
    m = report["metadata"]
    assert m["resolution"] == {"width": 478, "height": 850}
    assert m["orientation"] == "vertical"
    assert m["duration_s"] == 10.0           # priorité à l'inspection
    assert m["fps"] == 24.0
    assert m["video_codec"] == "h264"
    assert m["audio_present"] is True
    assert m["audio_codec"] == "aac"
    assert m["sample_rate"] == 44100 and m["channels"] == 2
    assert m["bitrate_kbps"] == 1341 and m["size_mb"] == 1.6
    assert m["platform_hint"] == "mp4/mov - compatible web/mobile"
    assert m["generation"]["schema_version"] == "1.11"
    assert isinstance(m["generation"]["generated_at"], str) and m["generation"]["generated_at"]
    assert m["generation"]["vlm_mode"] is None


def test_metadata_audio_absent():
    insp = _inspection(); insp["audio"] = {"present": False}
    m = fr.build_final_report(_results(), _fusion(), inspection=insp)["metadata"]
    assert m["audio_present"] is False
    assert m["audio_codec"] is None and m["sample_rate"] is None and m["channels"] is None


def test_metadata_none_inspection_all_null_but_generation():
    m = fr.build_final_report(_results(), _fusion(), inspection=None)["metadata"]
    assert m["resolution"] == {"width": None, "height": None}
    assert m["orientation"] is None and m["video_codec"] is None and m["fps"] is None
    assert m["audio_present"] is False and m["audio_codec"] is None
    assert m["bitrate_kbps"] is None and m["size_mb"] is None and m["platform_hint"] is None
    assert m["format_hint"] is None  # résolution inconnue -> pas de format inféré
    assert m["generation"]["schema_version"] == "1.11"
    assert m["generation"]["generated_at"]
    assert m["duration_s"] == 16.0  # repli sur total_duration (depuis scenes)


# --- format_hint (déterministe, description du format sans affirmer d'origine) ---
def test_format_hint_vertical_court():
    m = fr.build_final_report(_results(), _fusion(), inspection=_inspection())["metadata"]
    assert "vertical" in m["format_hint"] and "court" in m["format_hint"]


def test_format_hint_vertical_long():
    insp = _inspection(); insp["file"]["duration_s"] = 120.0
    m = fr.build_final_report(_results(), _fusion(), inspection=insp)["metadata"]
    assert m["format_hint"] == "vertical long"


def test_format_hint_horizontal():
    insp = _inspection(); insp["video"]["width"] = 1920; insp["video"]["height"] = 1080
    m = fr.build_final_report(_results(), _fusion(), inspection=insp)["metadata"]
    assert m["format_hint"] == "paysage (format standard/web)"


def test_format_hint_carre():
    insp = _inspection(); insp["video"]["width"] = 800; insp["video"]["height"] = 800
    m = fr.build_final_report(_results(), _fusion(), inspection=insp)["metadata"]
    assert m["format_hint"] == "carré (format typique fil Instagram)"


def test_format_hint_none_when_no_resolution():
    insp = _inspection(); insp["video"]["width"] = None; insp["video"]["height"] = None
    m = fr.build_final_report(_results(), _fusion(), inspection=insp)["metadata"]
    assert m["format_hint"] is None


# --- correction #3 : montage_rythme (déterministe) + genre (fondu dans l'arc) ----
def _fusion_n(durations):
    segs, t = [], 0.0
    for i, d in enumerate(durations, start=1):
        segs.append({"segment_id": i, "start_s": round(t, 3), "end_s": round(t + d, 3),
                     "duration_s": d, "combined_text": ""})
        t += d
    return {"video_path": "v.mp4", "language": "en",
            "summary": {"audio_activity_mode": "active_audio"}, "segments": segs}


def _results_n(n):
    return {i: _vlm(i, 0.7, ["x"], ["y"]) for i in range(1, n + 1)}


def test_montage_rythme_plan_unique():
    mr = fr.build_final_report(_results_n(1), _fusion_n([8.0]))["aggregate"]["montage_rythme"]
    assert mr["label"] == "plan_unique" and mr["scene_count"] == 1 and mr["cuts"] == 0


def test_montage_rythme_rapide():
    mr = fr.build_final_report(_results_n(6), _fusion_n([1.5] * 6))["aggregate"]["montage_rythme"]
    assert mr["label"] == "rapide" and mr["cuts"] == 5
    assert mr["average_scene_duration_s"] == pytest.approx(1.5)
    assert mr["cuts_per_minute"] == pytest.approx(5 * 60 / 9, rel=1e-3)


def test_montage_rythme_lent_tres_lent_modere():
    assert fr.build_final_report(_results_n(2), _fusion_n([3.0, 3.0]))["aggregate"]["montage_rythme"]["label"] == "modere"
    assert fr.build_final_report(_results_n(2), _fusion_n([7.0, 7.0]))["aggregate"]["montage_rythme"]["label"] == "lent"
    assert fr.build_final_report(_results_n(2), _fusion_n([12.0, 12.0]))["aggregate"]["montage_rythme"]["label"] == "tres_lent"


def test_montage_rythme_indetermine_when_no_durations():
    # n>1 mais aucune durée exploitable (segments sans timestamps) -> label indetermine,
    # métriques null (on n'affirme jamais un rythme sans preuve).
    fusion = {"video_path": "v.mp4", "language": None, "summary": {},
              "segments": [{"segment_id": 1}, {"segment_id": 2}]}
    mr = fr.build_final_report(_results_n(2), fusion)["aggregate"]["montage_rythme"]
    assert mr["label"] == "indetermine"
    assert mr["scene_count"] == 2 and mr["cuts"] == 1
    assert mr["average_scene_duration_s"] is None
    assert mr["cuts_per_minute"] is None


def test_genre_empty_without_generator():
    assert fr.build_final_report(_results(), _fusion())["aggregate"]["genre"] == []


def test_genre_from_generator_deduped():
    def gen(v, f):
        arc = _good_arc(); arc["genre"] = ["humour", "Humour", "famille", "sport"]; return arc
    assert fr.build_final_report(_results(), _fusion(), arc_generator=gen)["aggregate"]["genre"] == ["humour", "famille", "sport"]


def test_genre_malformed_yields_empty():
    def gen(v, f):
        return {"genre": "humour"}  # str au lieu de liste -> _as_list("humour") == []
    assert fr.build_final_report(_results(), _fusion(), arc_generator=gen)["aggregate"]["genre"] == []


# --- correction #2 : audio actif ne doit jamais affirmer une absence d'audio ----
def test_matcher_catches_total_absence_and_relation():
    assert fr._is_no_audio_uncertainty(
        "Absence totale d'audio/parole : aucune relation audio-visuelle vérifiable")
    assert fr._is_no_audio_uncertainty("aucune relation audio-visuelle")
    # l'énoncé honnête (audio présent non verbal) n'est PAS une absence
    assert not fr._is_no_audio_uncertainty(fr._NON_VERBAL_AUDIO_NOTE)


def test_audio_active_no_speech_segment_gets_honest_note():
    results = {
        1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=True),
        2: _vlm(2, 0.7, ["z"], ["w"], uses_speech=False,
                uncertainties=["Absence totale d'audio/parole : aucune relation audio-visuelle vérifiable",
                               "arrière-plan flou"]),
    }
    report = fr.build_final_report(results, _fusion())  # active_audio
    sc2 = report["scenes"][1]["uncertainties"]
    assert not any("absence" in u.lower() for u in sc2)
    assert not any("aucune relation audio" in u.lower() for u in sc2)
    assert any("non verbal actif" in u.lower() for u in sc2)
    assert any("arrière-plan flou" in u for u in sc2)  # autres incertitudes conservées
    assert report["aggregate"]["non_speech_audio_present"] is True
    # plus aucune « absence d'audio » dans l'agrégat
    assert not any("absence totale" in u.lower()
                   for u in report["aggregate"]["uncertainties"])


def test_no_audio_video_unchanged_no_fabricated_note():
    # Vidéo sans audio (pas d'active_audio) : aucun énoncé non-verbal fabriqué.
    results = {1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=False,
                       uncertainties=["No audio available"])}
    fusion = {"video_path": "v.mp4", "language": None, "summary": {}, "segments": [
        {"segment_id": 1, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0}]}
    report = fr.build_final_report(results, fusion)
    assert report["aggregate"]["non_speech_audio_present"] is False
    assert any("no audio" in u.lower() for u in report["aggregate"]["uncertainties"])
    assert not any("non verbal actif" in u.lower()
                   for sc in report["scenes"] for u in sc["uncertainties"])


def test_non_speech_audio_present_flag_false_when_all_speech():
    # _results() : 2 segments avec parole -> pas de segment non-verbal
    report = fr.build_final_report(_results(), _fusion())
    assert report["aggregate"]["non_speech_audio_present"] is False


# --- correction #1 : arc narratif ----------------------------------------------
_ARC_KEYS = ["arc_narratif", "arc_emotionnel", "resume_humain", "arc_confidence", "arc_evidence"]
_ARC_SUBKEYS = ["situation_initiale", "element_declencheur", "developpement", "climax", "denouement"]


def test_arc_fields_present_and_empty_without_generator():
    # Sans arc_generator -> 5 champs présents et vides (aucun appel LLM).
    report = fr.build_final_report(_results(), _fusion())
    for k in _ARC_KEYS:
        assert k in report
    assert report["arc_narratif"] == {k: "" for k in _ARC_SUBKEYS}
    assert report["arc_emotionnel"] == ""
    assert report["resume_humain"] == ""
    assert report["arc_confidence"] == 0.0
    assert report["arc_evidence"] == []
    # ordre : arc juste après aggregate et avant next_step
    keys = list(report.keys())
    assert keys.index("aggregate") < keys.index("arc_narratif")
    assert keys.index("arc_evidence") < keys.index("next_step")


def _good_arc():
    return {
        "arc_narratif": {
            "situation_initiale": "deux enfants devant une porte",
            "element_declencheur": "ils sonnent",
            "developpement": "ils s'enfuient",
            "climax": "la femme sort, furieuse",
            "denouement": "les enfants ont disparu",
        },
        "arc_emotionnel": "de l'amusement à la tension",
        "resume_humain": "Des enfants font une farce puis fuient une femme en colère.",
        "arc_confidence": 0.7,
        "arc_evidence": [1, 2],
    }


def test_arc_fields_from_generator_normalized():
    report = fr.build_final_report(_results(), _fusion(), arc_generator=lambda v, f: _good_arc())
    assert report["arc_narratif"]["climax"] == "la femme sort, furieuse"
    assert report["arc_confidence"] == 0.7
    assert report["arc_evidence"] == [1, 2]
    assert report["resume_humain"].startswith("Des enfants")
    assert report["arc_emotionnel"] == "de l'amusement à la tension"


def test_arc_generator_failure_yields_empty():
    def gen(v, f):
        raise RuntimeError("SDK indisponible")
    report = fr.build_final_report(_results(), _fusion(), arc_generator=gen)
    assert report["arc_confidence"] == 0.0
    assert report["resume_humain"] == ""
    assert report["arc_evidence"] == []
    assert report["arc_narratif"] == {k: "" for k in _ARC_SUBKEYS}


def test_arc_partial_generator_normalized():
    # Champs manquants comblés ; arc_evidence : str/bool exclus, 2.0 -> 2.
    def gen(v, f):
        return {"resume_humain": "ok", "arc_evidence": [1, "x", 2.0, True]}
    report = fr.build_final_report(_results(), _fusion(), arc_generator=gen)
    assert report["resume_humain"] == "ok"
    assert report["arc_narratif"]["situation_initiale"] == ""
    assert report["arc_confidence"] == 0.0
    assert report["arc_evidence"] == [1, 2]


def test_arc_confidence_clamped_and_evidence_filtered():
    # confiance hors bornes -> clamp [0,1] ; evidence hors scènes -> filtrée.
    def gen(v, f):
        arc = _good_arc()
        arc["arc_confidence"] = 1.8           # > 1 -> 1.0
        arc["arc_evidence"] = [1, 2, 99, -3]  # 99 et -3 absents de vlm_results
        return arc
    report = fr.build_final_report(_results(), _fusion(), arc_generator=gen)
    assert report["arc_confidence"] == 1.0
    assert report["arc_evidence"] == [1, 2]  # seuls les segment_id réels conservés


def test_arc_confidence_negative_clamped():
    def gen(v, f):
        arc = _good_arc()
        arc["arc_confidence"] = -0.5
        return arc
    report = fr.build_final_report(_results(), _fusion(), arc_generator=gen)
    assert report["arc_confidence"] == 0.0


def test_scene_keys_always_present_and_ordered():
    # seg1 complet ; seg2 minimal (uniquement segment_id) -> tous les champs présents.
    results = {
        1: _vlm(1, 0.6, ["a"], ["b"], uncertainties=["u"]),
        2: {"segment_id": 2},  # minimal, sans aucun champ descriptif
    }
    report = fr.build_final_report(results, _fusion())
    for sc in report["scenes"]:
        assert list(sc.keys()) == _SCENE_KEYS  # ordre constant
    sc2 = report["scenes"][1]
    # champs vides mais présents et typés
    assert sc2["scene_description"] == "" and isinstance(sc2["scene_description"], str)
    assert sc2["people_objects"] == [] and isinstance(sc2["people_objects"], list)
    assert sc2["actions"] == []
    assert sc2["uncertainties"] == []
    assert sc2["uses_speech"] is False
    assert sc2["mood"] == [] and isinstance(sc2["mood"], list)  # toujours présent, [] si vide
    assert sc2["combined_text"] == "سلام"  # depuis la fusion (seg 2)
    assert sc2["evidence"] == {"visual": [], "audio": []}


def test_evidence_always_present():
    r = _vlm(1, 0.6, ["a"], ["b"])
    del r["evidence"]  # résultat sans evidence -> doit être complété par défaut
    report = fr.build_final_report({1: r}, _fusion())
    ev = report["scenes"][0]["evidence"]
    assert ev == {"visual": [], "audio": []}
    assert list(ev.keys()) == ["visual", "audio"]


def test_type_coercion_bad_types():
    # people_objects en chaîne, uncertainties en None, confidence en chaîne.
    bad = {"segment_id": 1, "scene_description": 123, "people_objects": "pas une liste",
           "actions": None, "uncertainties": None, "confidence": "x", "uses_speech": 1,
           "evidence": "nope"}
    report = fr.build_final_report({1: bad}, _fusion())
    sc = report["scenes"][0]
    assert sc["people_objects"] == []   # chaîne -> [] (pas de split caractère)
    assert sc["actions"] == []
    assert sc["uncertainties"] == []
    assert sc["scene_description"] == "123"
    assert sc["confidence"] is None
    assert sc["uses_speech"] is True
    assert sc["evidence"] == {"visual": [], "audio": []}


def test_scene_mood_from_results_and_dominant_moods():
    results = {
        1: _vlm(1, 0.6, ["a"], ["b"]),
        2: _vlm(2, 0.7, ["c"], ["d"]),
    }
    results[1]["mood"] = ["joie", "tendresse"]
    results[2]["mood"] = ["tendresse", "surprise"]  # "tendresse" en doublon inter-scènes
    report = fr.build_final_report(results, _fusion())
    # mood repris par scène
    assert report["scenes"][0]["mood"] == ["joie", "tendresse"]
    assert report["scenes"][1]["mood"] == ["tendresse", "surprise"]
    # agrégat dédupliqué, ordre préservé
    assert report["aggregate"]["dominant_moods"] == ["joie", "tendresse", "surprise"]


def test_dominant_moods_always_present_even_empty():
    # _vlm n'inclut pas mood -> scènes mood [] -> dominant_moods []
    report = fr.build_final_report(_results(), _fusion())
    assert "dominant_moods" in report["aggregate"]
    assert report["aggregate"]["dominant_moods"] == []
    for sc in report["scenes"]:
        assert sc["mood"] == []


def test_scene_keys_constant_across_videos():
    # même jeu de clés quelles que soient les données (audio/sans audio).
    r1 = fr.build_final_report(_results(), _fusion())
    no_audio = {1: _vlm(1, 0.6, ["x"], ["y"], uses_speech=False)}
    fusion_na = {"video_path": "v.mp4", "language": None, "summary": {},
                 "segments": [{"segment_id": 1, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0}]}
    r2 = fr.build_final_report(no_audio, fusion_na)
    assert list(r1["scenes"][0].keys()) == list(r2["scenes"][0].keys())


# --- erreurs --------------------------------------------------------------------
def test_empty_results_raises():
    with pytest.raises(FinalReportError):
        fr.build_final_report({}, _fusion())


def test_fusion_not_dict_raises():
    with pytest.raises(FinalReportError):
        fr.build_final_report(_results(), ["pas un dict"])


def test_scene_without_fusion_timestamps():
    # segment 3 absent de la fusion -> timestamps None mais scène construite
    results = {3: _vlm(3, 0.5, ["x"], ["y"])}
    report = fr.build_final_report(results, _fusion())
    assert report["scenes"][0]["start_s"] is None
    assert report["scene_count"] == 1
    assert report["total_duration_s"] is None


# --- from_files (tmp_path) ------------------------------------------------------
def test_build_from_files(tmp_path):
    results_dir = tmp_path / "res"
    results_dir.mkdir()
    (results_dir / "vlm_result_segment_001.json").write_text(
        json.dumps(_results()[1], ensure_ascii=False), encoding="utf-8")
    (results_dir / "vlm_result_segment_002.json").write_text(
        json.dumps(_results()[2], ensure_ascii=False), encoding="utf-8")
    fusion_path = tmp_path / "fusion.json"
    fusion_path.write_text(json.dumps(_fusion(), ensure_ascii=False), encoding="utf-8")

    report = fr.build_final_report_from_files(str(results_dir), str(fusion_path))
    assert report["scene_count"] == 2
    assert report["status"] == "complete"
    assert report["validation"]["valid_segment_count"] == 2
    assert report["sources"]["fusion_manifest"] == str(fusion_path)


def test_build_from_files_partial_uses_only_valid_results(tmp_path):
    results_dir = tmp_path / "res"; results_dir.mkdir()
    (results_dir / "vlm_result_segment_001.json").write_text(
        json.dumps(_results()[1], ensure_ascii=False), encoding="utf-8")
    fusion_path = tmp_path / "fusion.json"
    fusion_path.write_text(json.dumps(_fusion(), ensure_ascii=False), encoding="utf-8")
    report = fr.build_final_report_from_files(str(results_dir), str(fusion_path), include_arc=False)
    assert report["status"] == "partial"
    assert report["scene_count"] == 1
    assert [scene["segment_id"] for scene in report["scenes"]] == [1]
    assert report["validation"]["missing_segments"] == [2]
    assert report["next_step"] == {"action": "correct_vlm_results", "segment_ids": [2]}


def test_build_from_files_never_complete_with_invalid_result(tmp_path):
    results_dir = tmp_path / "res"; results_dir.mkdir()
    good = _results()[1]
    bad = _results()[2].copy(); bad["actions"] = [123]
    (results_dir / "vlm_result_segment_001.json").write_text(json.dumps(good), encoding="utf-8")
    (results_dir / "vlm_result_segment_002.json").write_text(json.dumps(bad), encoding="utf-8")
    fusion_path = tmp_path / "fusion.json"
    fusion_path.write_text(json.dumps(_fusion()), encoding="utf-8")
    report = fr.build_final_report_from_files(str(results_dir), str(fusion_path), include_arc=False)
    assert report["status"] == "partial"
    assert report["validation"]["invalid_segments"] == [2]
    assert [scene["segment_id"] for scene in report["scenes"]] == [1]


def test_build_from_files_all_invalid_produces_failed_report(tmp_path):
    results_dir = tmp_path / "res"; results_dir.mkdir()
    for sid, result in _results().items():
        bad = result.copy(); bad["scene_description"] = ""
        (results_dir / f"vlm_result_segment_{sid:03d}.json").write_text(
            json.dumps(bad), encoding="utf-8")
    fusion_path = tmp_path / "fusion.json"
    fusion_path.write_text(json.dumps(_fusion()), encoding="utf-8")
    report = fr.build_final_report_from_files(str(results_dir), str(fusion_path), include_arc=False)
    assert report["status"] == "failed"
    assert report["scene_count"] == 0
    assert report["validation"]["invalid_segment_count"] == 2
    assert report["next_step"]["action"] == "rerun_vlm_analysis"


def test_old_validation_summary_is_not_trusted(tmp_path):
    results_dir = tmp_path / "res"; results_dir.mkdir()
    for sid, result in _results().items():
        (results_dir / f"vlm_result_segment_{sid:03d}.json").write_text(
            json.dumps(result), encoding="utf-8")
    old_summary = tmp_path / "summary.json"
    old_summary.write_text(json.dumps({"valid_count": 2, "segment_count": 2}), encoding="utf-8")
    fusion_path = tmp_path / "fusion.json"
    fusion_path.write_text(json.dumps(_fusion()), encoding="utf-8")
    report = fr.build_final_report_from_files(
        str(results_dir), str(fusion_path), summary_path=str(old_summary), include_arc=False
    )
    compatibility = report["validation"]["summary_compatibility"]
    assert compatibility["compatible"] is False
    assert any("ancien" in warning for warning in compatibility["warnings"])


def test_build_from_files_missing_dir(tmp_path):
    with pytest.raises(FinalReportError):
        fr.build_final_report_from_files(str(tmp_path / "nope"), str(tmp_path / "f.json"))
