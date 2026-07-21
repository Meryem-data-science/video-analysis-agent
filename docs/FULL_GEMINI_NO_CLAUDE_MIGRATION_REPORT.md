# Migration full-Gemini — suppression de Claude au runtime

> **Archive historique.** Les décisions publiques de `v0.1.0` sont : mode
> `manual` par défaut, `disabled` local et `automatic` avec Gemini explicite.
> Voir [ARCHITECTURE.md](ARCHITECTURE.md).

> **Objectif atteint** : l'agent d'analyse vidéo ne dépend plus de Claude à l'exécution.
> Le VLM **visuel** et l'arc **narratif** passent par **Gemini** (google-genai) ; l'audio
> Gemini existant est inchangé. Le contrat JSON VLM, les chemins de sortie et les noms de
> fichiers sont préservés. **382 tests passent** (hors-ligne, mocks ; aucun appel API réel).

## 0. Git
Le dépôt possède un `.git/` mais le binaire **`git` n'est pas installé** sur la machine
(introuvable sur le PATH et aux emplacements usuels) : impossible d'exécuter `git status`
ni de créer la branche `migration/full-gemini-no-claude-runtime`. Le plan a été consigné
dans [FULL_GEMINI_TOUCH_PLAN.md](FULL_GEMINI_TOUCH_PLAN.md) (fallback prévu). **À faire
côté humain** : créer la branche puis committer.

## 1. Fichiers modifiés
| Fichier | Changement |
|---|---|
| `scripts/run_vlm_agent.py` | N'appelle plus Claude ; sélectionne le provider (`get_vlm_provider`, défaut **gemini**) ; modèle résolu via `gemini_vlm_model`/env ; flag `--provider`. |
| `scripts/run_vlm_agent_segment.py` | Idem (un seul segment). |
| `src/video_agent/pipeline.py` | Étape `claude_agent_analysis` → **`vlm_analysis`** ; tolérance rc==2 sur `vlm_analysis` ; `STEP_ORDER` mis à jour ; **compat checkpoint** (`_LEGACY_MARKER_ALIASES`) ; docstring/commentaires Gemini. |
| `src/video_agent/vlm_strategy.py` | Modes : `gemini`+`manual_ui` implémentés (plus aucun mode Claude) ; défaut `gemini` ; providers : `gemini` ajouté, `claude_code`/`anthropic_api` retirés. |
| `src/video_agent/final_report.py` | `vlm_mode = "gemini"` (était `claude_agent`) ; provenance `provider` défaut `gemini` ; docstrings. |
| `src/video_agent/narrative_arc.py` | Importe `gemini_text_runner.analyze_text_with_gemini` au lieu du SDK Claude ; fallback arc vide conservé. |
| `src/video_agent/mcp_server.py` | Descriptions/erreurs (texte) Claude → Gemini. Aucun changement de logique ni de stdio. |
| `src/video_agent/mcp_artifacts_server.py` | Docstring (Gemini visuel + audio). |
| `config/vlm_config.example.json` | `provider`/`vlm_provider`/`audio_provider`=`gemini`, `gemini_vlm_model`, `gemini_text_model`. |
| Tests | `test_vlm_strategy.py`, `test_pipeline.py`, `test_final_report.py`, `test_narrative_arc.py` adaptés à Gemini. |

## 2. Fichiers créés / supprimés
**Créés :**
- `src/video_agent/vlm_schema.py` — contrat VLM (build_output_schema + `to_gemini_schema` + `normalize_vlm_result` + directives de prompt), **indépendant de tout fournisseur**.
- `src/video_agent/vlm_gemini_runner.py` — `GeminiVLMProvider` + `analyze_segment()` (VLM visuel google-genai).
- `src/video_agent/vlm_provider.py` — abstraction + `get_vlm_provider`/`resolve_provider_name` (défaut **gemini**, `manual_ui` supporté, legacy `claude*` → repli gemini).
- `src/video_agent/gemini_text_runner.py` — `analyze_text_with_gemini()` (texte structuré pour l'arc).
- `tests/test_vlm_gemini_runner.py`, `tests/test_gemini_text_runner.py`, `tests/test_vlm_provider.py`.
- `docs/FULL_GEMINI_TOUCH_PLAN.md`, `docs/FULL_GEMINI_NO_CLAUDE_MIGRATION_REPORT.md`.

**Supprimés :**
- `src/video_agent/vlm_agent_runner.py` (**seul fichier important `claude_agent_sdk`**).
- `tests/test_vlm_agent_runner.py` (mockait `claude_agent_sdk`, devenu inutile).

## 3. Nouvelle architecture Gemini
```
run_vlm_agent.py ─► vlm_provider.get_vlm_provider(VLM_PROVIDER|config, défaut gemini)
                       └─ GeminiVLMProvider ─► vlm_gemini_runner.analyze_segment()
                                                  ├─ google-genai (images + prompt)
                                                  ├─ response_mime_type=application/json
                                                  ├─ response_schema = to_gemini_schema(build_output_schema)
                                                  ├─ strip ```json, parse strict
                                                  └─ normalize_vlm_result()  (segment_id réinjecté,
                                                     confidence∈[0,1], tous champs garantis)
                       └─ vlm_result_validation.validate_single_result()  (inchangé)
                       └─ écrit outputs/vlm_results/<stem>/vlm_result_segment_NNN.json  (inchangé)
                                + vlm_results_summary.json                              (inchangé)

final_report.py ─► narrative_arc.generate_narrative_arc()
                       └─ gemini_text_runner.analyze_text_with_gemini()  (texte seul, JSON)
```
Le schéma de sortie VLM (12 champs) est **identique** à l'ancien : `segment_id`,
`scene_description`, `people_objects`, `actions`, `motion_description`,
`audio_visual_relation`, `uses_speech`, `mood`, `texte_ecran`, `confidence`,
`uncertainties`, `evidence{visual,audio}`.

## 4. Lancer la pipeline avec Gemini
```powershell
$env:GEMINI_API_KEY = "votre_cle"            # OBLIGATOIRE
# (optionnel) $env:GEMINI_VLM_MODEL = "gemini-2.5-flash"
# (optionnel) $env:GEMINI_TEXT_MODEL = "gemini-2.5-flash"
# VLM_PROVIDER=gemini est la valeur par défaut (rien à régler)

.\.venv\Scripts\python.exe scripts\run_full_pipeline.py demo_scene1.mp4
# un seul segment (debug) :
.\.venv\Scripts\python.exe scripts\run_vlm_agent_segment.py --segment 1
# mode manuel (aucun appel modèle) :
$env:VLM_PROVIDER = "manual_ui"
```

## 5. Variables d'environnement
| Variable | Rôle | Obligatoire |
|---|---|---|
| `GEMINI_API_KEY` | clé API Gemini (VLM visuel + arc + audio) | **Oui** (pour toute analyse Gemini) |
| `VLM_PROVIDER` | `gemini` (défaut) \| `manual_ui` | Non (défaut gemini) |
| `GEMINI_VLM_MODEL` | modèle VLM visuel | Non (sinon `GEMINI_MODEL` puis `gemini-2.5-flash`) |
| `GEMINI_TEXT_MODEL` | modèle texte (arc) | Non (sinon `GEMINI_VLM_MODEL` puis défaut) |
| `GEMINI_MODEL` | modèle **audio** existant (scene_audio.py) ; **fallback documenté** pour le VLM | Non |

## 6. Ce qui est devenu indépendant de Claude
- **VLM visuel** : 100% Gemini (`vlm_gemini_runner`).
- **Arc narratif** : 100% Gemini texte (`gemini_text_runner`) — Solution A retenue (la moins risquée : on garde la qualité de l'arc tout en gardant le fallback « arc vide » si Gemini indisponible).
- **Génération structurée** : schéma commun `vlm_schema` (plus dans un fichier Claude).
- **Sélection de provider** : aucune valeur `claude` ; toute config legacy `claude*` retombe sur gemini.

## 7. État de `narrative_arc.py`
Migré en **Gemini texte** (Solution A). Robustesse conservée : si `GEMINI_API_KEY`
manque ou si Gemini échoue, `generate_narrative_arc` capte l'exception et renvoie un
**arc vide** structurellement complet → la fiche finale est toujours produite, **sans
plantage** et **sans Claude**.

## 8. État de l'audio Gemini existant
`scene_audio.py` **inchangé** (modèle `GEMINI_MODEL`, défaut `gemini-2.5-flash`, cache
par scène, retry 503). Sa suite `tests/test_scene_audio.py` reste **verte**. Aucune
régression introduite.

## 9. Résultats des tests
`.\.venv\Scripts\python.exe -m pytest -q` → **382 passed, 3 warnings** (warnings =
`DeprecationWarning` Pillow préexistant, sans rapport). Nouveaux tests mockés :
- `test_vlm_gemini_runner.py` : JSON valide → contrat respecté, `segment_id` réinjecté, `confidence` clampée, champs garantis, fences ```json nettoyés, validation OK, clé absente → erreur claire.
- `test_gemini_text_runner.py` : parsing, fences, erreurs, priorité de modèle.
- `test_vlm_provider.py` : défaut gemini, priorité CLI>env>config, legacy claude→gemini, manual_ui.
- `test_narrative_arc.py` : réécrit (mock Gemini texte), fallback arc vide si Gemini KO.
Import sanity : tous les modules (dont `mcp_server`) s'importent **sans rien écrire sur stdout**.

## 10. Recherche anti-Claude (src/ scripts/ tests/ config/)
| Motif | Occurrences runtime |
|---|---|
| `claude_agent_sdk` | **0** |
| `ClaudeAgentOptions` | **0** |
| `ResultMessage` | **0** |
| `analyze_segment_with_agent` | **0** |
| `analyze_text_with_agent` | **0** |
| `VLM_PROVIDER=claude` | **0** |
| `claude_agent` | 4 — **toutes légitimes** : 3 dans `pipeline.py` (alias de checkpoint *legacy* documenté ; moteur = Gemini) + 1 dans `test_vlm_provider.py` (test prouvant le repli legacy→gemini). |

Les autres occurrences du mot « claude » (~26) sont uniquement des **commentaires/docstrings**
qui disent explicitement « plus de Claude », « remplace Claude », « indépendant de Claude »
(aucun import ni appel). Aucune dépendance Claude au runtime, aucune config/valeur par défaut Claude.

## 11. Limites restantes
- **Compatibilité checkpoint** : un ancien marqueur `claude_agent_analysis.done` est encore
  reconnu pour ne pas relancer l'analyse VLM (désormais Gemini) ; c'est volontaire et documenté
  (`_LEGACY_MARKER_ALIASES` dans `pipeline.py`).
- **Schéma `response_schema` Gemini** : adapté via `to_gemini_schema` (retrait de `const`/
  `additionalProperties`) ; en cas de version SDK ne l'acceptant pas, repli automatique sur
  `response_mime_type=application/json` + parsing/normalisation (le contrat reste garanti).
- **`claude-agent-sdk`** peut rester installé dans le venv mais **n'est plus jamais importé** ;
  il peut être désinstallé sans impact.
- **Git indisponible** : branche/commit à réaliser côté humain.
- Tests **mockés** uniquement : aucun appel réseau réel à Gemini n'a été exécuté (par exigence).

## Critères de réussite — état
- [x] Fonctionne sans abonnement Claude (clé Gemini suffit).
- [x] Le code principal n'importe pas `claude_agent_sdk`.
- [x] VLM visuel = Gemini.
- [x] Arc narratif indépendant de Claude (Gemini texte + fallback).
- [x] `final_report` indique `gemini`.
- [x] `scene_audio.py` fonctionnel (tests verts).
- [x] Schéma JSON VLM inchangé + chemins/fichiers inchangés.
- [x] Pipeline complète produit toujours la fiche finale.
- [x] Tests passent (382).
