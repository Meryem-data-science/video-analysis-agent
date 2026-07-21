# Rapport de nettoyage des anciennes analyses

> **Archive historique.** Ce rapport décrit une opération locale antérieure et
> ne constitue pas la documentation normative de `v0.1.0`. Voir
> [ARCHITECTURE.md](ARCHITECTURE.md) pour l’état publié.

- **Date du nettoyage** : 2026-06-17
- **Périmètre** : `outputs/` uniquement. Aucun fichier de code, script, test, doc, config,
  venv ou vidéo n'a été touché.
- **Volume libéré** : ~380 Mo (1902 → 376 fichiers restants dans `outputs/`, tous dans des
  dossiers conservés non liés à la pipeline d'analyse vidéo).

## Méthode (sécurisée)
1. Inspection complète de `outputs/` et de la racine, sans suppression.
2. Construction de la liste exacte des suppressions (219 entrées de 1er niveau).
3. Garde-fou : vérification qu'aucune entrée n'est `src/ scripts/ tests/ docs/ config/
   vidéos/ .venv/ .git/` ni hors de `outputs/` → **aucune violation**.
4. Suppression ciblée, par lots explicites (les motions de suppression en masse ont été
   refusées par le bac à sable ; aucune suppression partielle n'a eu lieu avant cela).
5. Recréation des sous-dossiers vides attendus par la pipeline.

## Dossiers supprimés (puis recréés vides)
Générés par la pipeline (`pipeline.py`) :
- `outputs/.checkpoints/` (checkpoints de reprise)
- `outputs/audio/` (WAV extraits)
- `outputs/frames/` (frames extraites)
- `outputs/motion/` (grilles + composites de mouvement)
- `outputs/vlm_results/` (résultats VLM par segment + résumés)
- `outputs/vlm_runs/` (prompts + plans de run VLM)

Ces 6 dossiers ont été **recréés vides** car la pipeline les attend.

## Fichiers supprimés (215 au total)
Tous les JSON/TXT/PDF produits par les étapes de la pipeline, par préfixe (toutes vidéos
confondues : `demo*`, `Girl`, `gross`, `jeune_fille`, `Video6`, `Auto`, …) :
- `inspection_*.json`, `strategy_*.json`, `segments_*.json`
- `frames_manifest_*.json`, `motion_manifest_*.json`
- `audio_manifest_*.json`, `audio_light_*.json`, `transcript_*.json`
- `audio_visual_fusion_*.json`, `vlm_pack_*.json`
- `audio_events_*.json`, `audio_features_*.json`, `scene_ocr_*.json`
- `pipeline_status_*.json`
- `final_report_*.json`, `final_report_*.txt`, `final_report_demo_darija.pdf`

Plus 2 fichiers de capture de run créés par l'agent :
- `_pipeline_scene1_progress.log`, `_pipeline_scene2_stdout.json`

## Dossiers / fichiers CONSERVÉS dans `outputs/` (non clairement générés par la pipeline)
Par prudence (règle « seulement si clairement généré par la pipeline d'analyse vidéo »),
**non supprimés** — aucune étape de `pipeline.py` ne les produit :
- Dossiers : `outputs/deep_analysis/`, `outputs/promo_build/`, `outputs/video_generation/`
- Fichiers : `bg_music.mp3`, `test_voiceover.mp3`, `clip3_frame7.jpg`, `coordinates.txt`,
  `daily_ai_watch_2026-06-21.md`, `ocr_results.json`, `ocr_results.txt`

> Si tu confirmes que ces éléments sont aussi d'anciens résultats à jeter, je peux les
> supprimer dans une seconde passe (ils ne ressemblent pas aux sorties de la pipeline,
> d'où leur conservation par défaut).

## Confirmations de sécurité
- **Vidéos sources NON supprimées** ✅ — `vidéos/` intact (17 vidéos) :
  `Auto.mp4, demo.mp4, demo_anime.mp4, demo_darija.mp4, demo_home.mp4, demo_messi.mp4,
  demo_panda.mp4, demo_pittza.mp4, demo_scene1.mp4, demo_scene2.mp4, demo_short.mp4,
  demo_ytb.mp4, demo_ytb_original.mp4, Girl.mp4, gross.mp4, jeune_fille.mp4, Video6.mp4`.
- **Code et ressources NON supprimés** ✅ — vérifié existant après nettoyage :
  `src/`=True, `scripts/`=True, `tests/`=True, `config/`=True, `docs/`=True,
  `vidéos/`=True, `.venv/`=True, `.git/`=True.
- `outputs/` existe et ne contient plus aucun artefact de pipeline (sous-dossiers pipeline
  vides + éléments non-pipeline conservés).

## Relancer une nouvelle analyse avec Gemini
```powershell
$env:GEMINI_API_KEY = "votre_cle"     # obligatoire (VLM visuel + arc + audio Gemini)
# VLM_PROVIDER=gemini est le défaut — rien à régler.

.\.venv\Scripts\python.exe scripts\run_full_pipeline.py demo_scene1.mp4
```
(remplacer `demo_scene1.mp4` par la vidéo voulue, présente sous `vidéos/`).
