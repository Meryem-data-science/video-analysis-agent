# Changelog

Les changements notables suivent l’esprit de
[Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le projet utilise
le versionnage sémantique.

## [Unreleased]

Aucun changement documenté.

## [0.1.0] - 2026-07-19

### Added

- Pipeline local-first d’inspection, segmentation, extraction de frames, mouvement,
  audio, transcription et fusion audio-visuelle.
- Modes VLM explicites `disabled`, `manual` et `automatic`, avec Gemini optionnel.
- Validation stricte par segment et statuts `complete`, `partial`, `failed`,
  `awaiting_vlm` et `local_complete`.
- Runs déterministes isolés sous `outputs/runs/<run_id>/`, checkpoints hashés et
  écritures atomiques.
- CLI installable, serveur MCP et API FastAPI synchrone.
- Jobs asynchrones locaux avec progression, nettoyage des uploads et annulation
  des travaux encore en attente.
- Interface React, Vite et TypeScript pour lancer et consulter les analyses.
- Vidéo synthétique déterministe, fixtures VLM simulées et dataset versionné.
- Framework d’évaluation JSON/HTML et benchmark reproductible.
- Packaging Python avec extras `audio`, `gemini`, `pdf`, `web`, `dev` et `all`.
- CI Windows/Linux, contrôle Ruff et vérification d’hygiène du dépôt.

### Security

- Validation des uploads, extensions, tailles et chemins d’artefacts.
- Refus du path traversal et de l’exposition de `.env` ou chemins arbitraires.
- Aucun chargement du SDK Gemini dans les modes `manual` et `disabled`.
- Contrôle automatisé des secrets, chemins personnels, médias et outputs suivis.
