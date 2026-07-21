# Plan de migration full-Gemini (sans Claude au runtime)

> **Archive historique, non normative.** Ce plan précède l’architecture publiée
> en `v0.1.0`. Voir [ARCHITECTURE.md](ARCHITECTURE.md) et le README pour le
> comportement actuel.

> **Note Git** : le projet est un dépôt git (`.git/` présent) **mais le binaire `git`
> n'est pas installé** sur cette machine (introuvable sur le PATH et aux emplacements
> usuels). Impossible donc d'exécuter `git status` ni de créer la branche
> `migration/full-gemini-no-claude-runtime`. Ce document tient lieu de plan, comme prévu
> par le fallback « si ce n'est pas un repo git ». **À faire côté humain** : créer la
> branche puis committer.

## Fichiers CRÉÉS
- `src/video_agent/vlm_schema.py` — schéma VLM canonique + normalisation (hors Claude).
- `src/video_agent/vlm_gemini_runner.py` — `GeminiVLMProvider` (VLM visuel via google-genai).
- `src/video_agent/vlm_provider.py` — abstraction + sélection du provider (défaut **gemini**).
- `src/video_agent/gemini_text_runner.py` — génération texte structurée via Gemini (arc).
- `tests/test_vlm_gemini_runner.py`, `tests/test_gemini_text_runner.py`, `tests/test_vlm_provider.py`.
- `docs/FULL_GEMINI_NO_CLAUDE_MIGRATION_REPORT.md` (rapport final).

## Fichiers MODIFIÉS
- `scripts/run_vlm_agent.py`, `scripts/run_vlm_agent_segment.py` — appellent le provider Gemini.
- `src/video_agent/pipeline.py` — étape `claude_agent_analysis` → `vlm_analysis` (+ compat checkpoint).
- `src/video_agent/vlm_strategy.py` — modes/providers : `gemini` implémenté, défaut gemini.
- `src/video_agent/final_report.py` — `vlm_mode="gemini"`.
- `src/video_agent/mcp_server.py` — libellés/descriptions (texte) Claude → Gemini.
- `config/vlm_config.example.json` — provider/vlm_provider/gemini_vlm_model/gemini_text_model/audio_provider.
- `src/video_agent/narrative_arc.py` — utilise `gemini_text_runner` au lieu du SDK Claude.
- `tests/test_narrative_arc.py`, `tests/test_final_report.py`, `tests/test_pipeline.py`, `tests/test_vlm_strategy.py`.

## Fichiers SUPPRIMÉS
- `src/video_agent/vlm_agent_runner.py` (seul fichier important `claude_agent_sdk`).
- `tests/test_vlm_agent_runner.py` (mockait `claude_agent_sdk`).

## Intouchés (par exigence)
- `src/video_agent/scene_audio.py` (audio Gemini), chemins/JSON de sortie, contrat JSON VLM.
