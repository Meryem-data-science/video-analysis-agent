# Audit — Migration du VLM visuel (Claude → Gemini)

> **Archive historique, non normative.** Cette proposition précède les modes
> publics `manual`, `disabled` et `automatic` de `v0.1.0`. Voir
> [ARCHITECTURE.md](ARCHITECTURE.md) pour l’implémentation actuelle.

> **Statut : AUDIT SEULEMENT.** Ce document ne modifie aucun code. Il cartographie
> l'existant et propose une stratégie de migration sûre, additive et réversible.
> Objectif : pouvoir choisir le VLM **visuel** (Claude / Gemini / manual_ui) sans
> rien casser, **sans toucher** à l'analyse **audio** Gemini déjà en place, ni aux
> niveaux validés (inspection, segmentation, frames, motion, audio, fusion, fiche finale).

---

## 1. Résumé de la pipeline actuelle

**Point d'entrée principal** : `run_full_pipeline()` dans
[src/video_agent/pipeline.py](../src/video_agent/pipeline.py), exposé via la CLI
[scripts/run_full_pipeline.py](../scripts/run_full_pipeline.py) et via le tool MCP
`run_full_pipeline_file` ([src/video_agent/mcp_server.py](../src/video_agent/mcp_server.py)).

Chaque étape « lourde » est déléguée à **son CLI** (`scripts/<x>.py`) lancé en
**sous-processus** (stdout→DEVNULL, stderr→fichier, UTF-8 forcé). L'ordre réel,
déduit des appels `step(...)` + des étapes en-process et de la constante `STEP_ORDER` :

| # | Étape (nom checkpoint) | Module / script | Sortie JSON (sous `outputs/`) |
|---|---|---|---|
| 1 | `inspection` | `inspection.py` / `inspect_video.py` | `inspection_<stem>.json` |
| 2 | `strategy` | `strategy.py` / `decide_strategy.py` | `strategy_<stem>.json` |
| 3 | `segmentation` | `segmentation.py` / `segment_video.py` | `segments_<stem>.json` |
| 4 | `frames` | `frame_extraction.py` / `extract_frames.py` | `frames_manifest_<stem>.json` (+ `frames/<stem>/…`) |
| 5 | `motion` | `motion_representation.py` / `build_motion.py` | `motion_manifest_<stem>.json` (+ `motion/<stem>/…`) |
| 6 | `audio_extraction` *(si audio)* | `audio_extraction.py` / `extract_audio.py` | `audio_manifest_<stem>.json` (+ `audio/<stem>/audio.wav`) |
| 7 | `audio_light_analysis` *(si audio)* | `audio_light_analysis.py` / `analyze_audio_light.py` | `audio_light_<stem>.json` |
| 8 | `transcription` *(si recommandé)* | `transcription.py` / `transcribe_audio.py` (faster-whisper) | `transcript_<stem>.json` |
| 9 | `fusion` | `audio_visual_fusion.py` / `fuse_audio_visual.py` | `audio_visual_fusion_<stem>.json` |
| 10 | `vlm_pack` | `vlm_pack_builder.py` / `build_vlm_pack.py` | `vlm_pack_<stem>.json` |
| 11 | `prepare_vlm_run` | `vlm_strategy.py` / `prepare_vlm_run.py` | `vlm_runs/<stem>/{run_plan,selected_inputs}.json` + `prompt_segment_NNN.md` |
| 12 | **`claude_agent_analysis`** | **`vlm_agent_runner.py` / `run_vlm_agent.py`** | `vlm_results/<stem>/vlm_result_segment_NNN.json` + `vlm_results_summary.json` |
| 13a | `scene_audio` *(en-process, non-fatal)* | **`scene_audio.py` (Gemini audio)** | `audio_events_<stem>.json` |
| 13b | `scene_audio_features` *(en-process, librosa)* | `scene_audio_features.py` | `audio_features_<stem>.json` |
| 14 | `final_report` | `final_report.py` / `build_final_report.py` (+ `narrative_arc.py`) | `final_report_<stem>.json` + `.txt` |

**Dossiers** : input = `vidéos/` (`$VIDEO_ROOT`) ; work+output = `outputs/`
(checkpoints de reprise sous `outputs/.checkpoints/<stem>_<empreinte>/`). La sécurité
des chemins (MCP + pipeline) impose de rester sous `PROJECT_ROOT` / `VIDEO_ROOT`.

**Orchestration** : `run_full_pipeline()` (séquentiel, reprenable par checkpoints) ;
`step()`/`_run_step()` pour les étapes-sous-processus ; `scene_audio` / `scene_audio_features`
appelées **en-process** (try/except non-fatal). Étape 12 = **seul** appel VLM **visuel**.

---

## 2. Carte des fichiers importants

**VLM visuel (Claude) — cœur de la migration :**
- [src/video_agent/vlm_agent_runner.py](../src/video_agent/vlm_agent_runner.py) — appelle le **Claude Agent SDK** ; `build_output_schema()`, `analyze_segment_with_agent()`, `analyze_text_with_agent()`, `_execute_query()`, directives FR/mood/texte_ecran.
- [scripts/run_vlm_agent.py](../scripts/run_vlm_agent.py) — CLI étape 12 (boucle TOUS les segments, écrit les résultats + résumé).
- [scripts/run_vlm_agent_segment.py](../scripts/run_vlm_agent_segment.py) — CLI un seul segment (debug).
- [src/video_agent/vlm_result_validation.py](../src/video_agent/vlm_result_validation.py) — **valide le schéma** des résultats VLM (indépendant du provider).
- [src/video_agent/vlm_pack_builder.py](../src/video_agent/vlm_pack_builder.py) / [src/video_agent/vlm_strategy.py](../src/video_agent/vlm_strategy.py) — préparent les entrées + le plan/prompts (sans appel modèle).
- [config/vlm_config.example.json](../config/vlm_config.example.json) — `mode`/`provider`/`model`/`input_policy`/`output_policy`.

**Audio Gemini (à NE PAS toucher) :**
- [src/video_agent/scene_audio.py](../src/video_agent/scene_audio.py) — Gemini `google-genai` (musique/SFX/émotion audio).
- Prototypes hors pipeline : [scripts/gemini_audio_per_scene.py](../scripts/gemini_audio_per_scene.py), [scripts/gemini_audio_prototype.py](../scripts/gemini_audio_prototype.py), [scripts/gemini_smoke_test.py](../scripts/gemini_smoke_test.py).

**Consommateurs du résultat VLM :**
- [src/video_agent/final_report.py](../src/video_agent/final_report.py) — agrège les `vlm_result_segment_*.json`, fixe `vlm_mode="claude_agent"`, bloc `provenance.visual.mode`.
- [src/video_agent/narrative_arc.py](../src/video_agent/narrative_arc.py) — arc narratif via **Claude texte** (`analyze_text_with_agent`).
- [src/video_agent/mcp_server.py](../src/video_agent/mcp_server.py) — tool `run_vlm_agent_analysis` (+ chaîne complète).

**Tests liés** : `tests/test_vlm_agent_runner.py`, `tests/test_vlm_strategy.py`,
`tests/test_vlm_pack_builder.py`, `tests/test_final_report.py`, `tests/test_pipeline.py`,
`tests/test_narrative_arc.py`, `tests/test_scene_audio.py`.

---

## 3. Dépendances Claude actuelles (où Claude est le VLM)

| Emplacement | Usage Claude |
|---|---|
| [vlm_agent_runner.py](../src/video_agent/vlm_agent_runner.py) | `from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query` (import différé). Options : `output_format={"type":"json_schema","schema":…}`, `allowed_tools=[]`, `permission_mode="bypassPermissions"`, `setting_sources=[]`, `max_turns`. **Auth = abonnement** (CLI Claude Code embarqué, *pas* de clé API). Images en blocs base64. |
| [run_vlm_agent.py](../scripts/run_vlm_agent.py) / [run_vlm_agent_segment.py](../scripts/run_vlm_agent_segment.py) | Importent `analyze_segment_with_agent` (couplage direct au provider Claude). |
| [narrative_arc.py](../src/video_agent/narrative_arc.py) | `analyze_text_with_agent` (Claude **texte**, pour l'arc — pas d'image). |
| [pipeline.py](../src/video_agent/pipeline.py) | Étape nommée `claude_agent_analysis` ; tolérance `rc==2`. |
| [final_report.py](../src/video_agent/final_report.py) | `vlm_mode = "claude_agent"` (codé en dur dans `build_final_report_from_files`) → `metadata.generation.vlm_mode` + `provenance.visual.mode`. |
| [vlm_strategy.py](../src/video_agent/vlm_strategy.py) | `_KNOWN_MODES`/`_IMPLEMENTED_MODES` incluent `claude_agent` ; `_KNOWN_PROVIDERS` inclut déjà `gemini_ui`, `google_api`. |
| [mcp_server.py](../src/video_agent/mcp_server.py) | Tool `run_vlm_agent_analysis` (« mode claude_agent … via le Claude Agent SDK »). |
| Tests | `test_vlm_agent_runner.py` (mock du SDK), assertions `"claude_agent"` dans `test_final_report.py`/`test_pipeline.py`/`test_narrative_arc.py`. |

> Pas d'occurrence de `claude -p`, ni de dossier `visual_claude`, ni de fonction
> `save_claude_visual_analysis`. Le couplage Claude est **concentré** dans
> `vlm_agent_runner.py` (+ ses 2 CLI). La validation et la fiche finale sont, elles,
> **agnostiques** au provider (elles lisent du JSON).

---

## 4. Dépendances Gemini actuelles (déjà en place — NE PAS toucher)

| Aspect | Détail ([scene_audio.py](../src/video_agent/scene_audio.py)) |
|---|---|
| Domaine | **Audio uniquement** : musique, effets sonores, émotion/ambiance audio par scène. **Aucune analyse visuelle.** |
| SDK | `from google import genai` ; `from google.genai import types` (import différé). |
| Auth | Variable d'env **`GEMINI_API_KEY`** (clé API). Absente → dégradation gracieuse (aucune analyse). |
| Modèle | `DEFAULT_MODEL = "gemini-2.5-flash"`, surchargé par **`GEMINI_MODEL`** ou argument `model`. |
| Appel | `client.files.upload(...)` → `client.models.generate_content(model, contents=[PROMPT, audio], config=GenerateContentConfig(response_mime_type="application/json", response_schema=RESPONSE_SCHEMA))` ; `HttpOptions(timeout=45 s)`. |
| Robustesse | Retry backoff exponentiel + jitter sur 503/UNAVAILABLE ; cache **par scène** (`scene_NNN.json`) ; ne lève jamais. |
| Sortie | `audio_events_<stem>.json` (`schema "scene_audio/1.0"`), par scène : `music{present,confidence,mood,mood_confidence,energy,tempo_feel}`, `sound_effects[]`, `speech_overlap`, `evidence`, `uncertainties`, `audio_status`. |

→ **Le SDK `google-genai` et `GEMINI_API_KEY`/`GEMINI_MODEL` sont déjà installés et
utilisés.** Un VLM **visuel** Gemini réutiliserait le même SDK/la même clé, mais via un
**module séparé** (ne pas réutiliser ni modifier `scene_audio.py`).

---

## 5. Contrat de sortie VLM visuel à PRÉSERVER (par segment)

Source de vérité : `build_output_schema()` ([vlm_agent_runner.py](../src/video_agent/vlm_agent_runner.py))
**et** `validate_result_schema()` ([vlm_result_validation.py](../src/video_agent/vlm_result_validation.py)).
Tout provider (Claude, Gemini, manual_ui) **doit** produire EXACTEMENT ce JSON,
fichier par segment : `outputs/vlm_results/<stem>/vlm_result_segment_NNN.json`.

```jsonc
{
  "segment_id": 3,                       // entier, == segment du run (Claude: const ; Gemini: à réinjecter)
  "scene_description": "…",              // string (obligatoire)
  "people_objects": ["…"],               // array<string> (obligatoire)
  "actions": ["…"],                      // array<string> (obligatoire)
  "motion_description": "…",             // string (obligatoire) — mouvement via composite
  "audio_visual_relation": "…",          // string (obligatoire)
  "uses_speech": true,                   // bool (obligatoire)
  "mood": ["…"],                         // array<string> (obligatoire) — émotion visuelle
  "texte_ecran": [                        // array<object> (obligatoire) — texte à l'écran lu nativement
    { "text": "…", "lang": "ar|fr|en|inconnue", "location": "haut|bas|…" }
  ],
  "confidence": 0.0,                     // number ∈ [0,1] (obligatoire ; validé)
  "uncertainties": ["…"],                // array<string> (obligatoire)
  "evidence": {                          // object (obligatoire)
    "visual": ["…"],                     //   array<string> (obligatoire)
    "audio":  ["…"]                      //   array<string> (obligatoire)
  }
}
```

**Champs obligatoires** : les 12 ci-dessus (`additionalProperties:false` côté schéma SDK).
**Décor** : porté par `scene_description` + `people_objects` (pas de champ « décor » dédié).
**Timestamps** : **non produits par le VLM** — ils viennent de la fusion ; `final_report`
réinjecte `start_s`/`end_s`/`duration_s` par `segment_id`. **Texte visuel** : `texte_ecran`
(lu nativement par le VLM, pas d'OCR séparé). **Confidences** : un seul `confidence` global
par segment (les blocs `mood`/`texte_ecran` n'ont pas de score). **Fallback/erreurs** :
- côté runner : exception `VlmAgentError` si réponse non structurée → l'orchestrateur (`run_vlm_agent.py`) marque le segment en erreur **sans bloquer** les autres ;
- côté validation : `validate_single_result` renvoie `valid:false` + `errors[]` (schéma/segment_id/confidence) et `warnings[]` (anti-hallucination) ;
- `segment_id` est imposé par nous (Claude via `const` ; un provider Gemini devra l'écraser après parsing).

Champs **dérivés** (ajoutés plus tard par `final_report`, **pas** par le VLM) :
`audio_scene`, `audio_visual_coherence`, `measured_features` — hors contrat VLM.

---

## 6. Risques de casse (si on s'y prend mal — notamment en « renommant » Claude→Gemini)

| Zone | Risque | Mitigation |
|---|---|---|
| **Fusion audio-visuelle** | Faible : la fusion produit le **pack** (entrée VLM) et ne lit aucun résultat VLM. Aucun couplage Claude/Gemini. | Ne rien changer. |
| **Validation** (`vlm_result_validation.py`) | Cassure si Gemini omet un champ (`mood`, `texte_ecran`, `evidence.audio`, …) ou met `confidence` hors [0,1]. | Mapper `build_output_schema` → `response_schema` Gemini à l'**identique** ; réinjecter `segment_id` ; clamworking `confidence`. |
| **Fiche finale** (`final_report.py`) | `vlm_mode="claude_agent"` **codé en dur** → `provenance.visual.mode` deviendrait faux pour Gemini ; tests l'asseront. | Rendre `vlm_mode` paramétrable (déduit du provider) **sans** changer le schéma 1.10. |
| **Arc narratif** (`narrative_arc.py`) | Couplé à Claude **texte** (`analyze_text_with_agent`). Si on « débranche » Claude globalement, l'arc casse. | **Hors périmètre VLM visuel** : laisser l'arc sur Claude (ou le rendre configurable plus tard). Ne pas supprimer le provider Claude. |
| **Tests** | `test_vlm_agent_runner.py` (mock SDK Claude), assertions `"claude_agent"` dans `test_final_report.py`/`test_pipeline.py`/`test_narrative_arc.py`. | Garder ClaudeProvider et ses tests ; ajouter des tests Gemini **mockés** ; adapter les assertions de `vlm_mode` au provider choisi. |
| **Scripts MCP** | `run_vlm_agent_analysis` délègue à `run_vlm_agent.py` (sous-processus, stdout→DEVNULL). Si un provider Gemini **imprime sur stdout**, et qu'un jour on l'appelle in-process, ça pollue le stdio MCP. | Le provider Gemini **ne doit jamais** écrire sur stdout (logs → `logging`/stderr/fichier), comme `scene_audio.py`. Conserver le pattern sous-processus. |
| **Audio Gemini** | Confusion possible entre « Gemini audio » (`scene_audio.py`) et « Gemini visuel » (nouveau). Réutiliser/éditer `scene_audio.py` casserait l'audio validé. | **Module séparé** (`vlm_gemini_runner.py`), zéro import depuis `scene_audio.py`. |
| **Chemins/JSON** | Renommer dossiers (`vlm_results/`, fichiers `vlm_result_segment_NNN.json`) casserait fusion/validation/fiche. | **Ne renommer aucun chemin ni JSON** — l'abstraction est interne au runner. |
| **Config** | `vlm_strategy.validate_config` rejette les modes inconnus. | Ajouter `gemini` aux modes implémentés **avant** de l'utiliser. |

---

## 7. Plan de migration en petites étapes (additif, réversible)

> Principe : **abstraction `VLMProvider`** au seul point de couplage (le runner), sélection
> par `VLM_PROVIDER` / config, **schéma de sortie identique**, **chemins/JSON inchangés**,
> **Claude conservé par défaut**, **audio Gemini intact**.

1. **Geler le contrat** : extraire `build_output_schema()` comme **source unique** réutilisée par tous les providers (déjà le cas pour Claude). Aucun changement de schéma.
2. **Définir l'interface** (nouveau fichier, ex. `src/video_agent/vlm_provider.py`) :
   ```python
   class VLMProvider(Protocol):
       name: str
       def analyze_segment(self, prompt_text: str, image_paths: list[str],
                           segment_id: int, model: str | None, timeout_s: float) -> dict: ...
   ```
3. **ClaudeProvider** (wrapper mince) : délègue à `analyze_segment_with_agent` existant. **Aucune modification** de `vlm_agent_runner.py` côté logique.
4. **GeminiVLMProvider** (nouveau fichier `src/video_agent/vlm_gemini_runner.py`, **séparé** de `scene_audio.py`) :
   - SDK `google-genai`, `GEMINI_API_KEY`, modèle via config/`GEMINI_MODEL` (défaut multimodal type `gemini-2.5-flash`/`-pro`) ;
   - images en `Part` (base64/bytes) + prompt + `response_mime_type="application/json"` + `response_schema` = miroir de `build_output_schema` ;
   - **réinjecter `segment_id`** après parsing (Gemini ne gère pas `const`), clamp `confidence` ∈ [0,1], garantir les listes par défaut → schéma identique ;
   - logs sur `logging`/stderr uniquement, **jamais stdout**.
5. **ManualUIProvider** : conserve le mode `manual_ui` existant (aucun appel modèle ; l'utilisateur dépose les JSON, on valide).
6. **Sélecteur** `get_vlm_provider()` : priorité `--provider` CLI > `$VLM_PROVIDER` (`claude|gemini|manual_ui`) > `provider`/`vlm_provider` de la config > **défaut `claude`**. Étendre `vlm_strategy._IMPLEMENTED_MODES` avec `gemini`.
7. **Brancher au point unique** : `run_vlm_agent.py` (+ `run_vlm_agent_segment.py`) appellent `provider.analyze_segment(...)` au lieu de `analyze_segment_with_agent` en dur. La boucle, l'écriture `vlm_result_segment_NNN.json`, le résumé et la **validation** restent identiques.
8. **Provenance** : rendre `vlm_mode` dynamique dans `build_final_report_from_files` (= nom du provider), sans changer le schéma 1.10 (`provenance.visual.mode` reflète `claude`/`gemini`/`manual_ui`).
9. **MCP** : `run_vlm_agent_analysis` inchangé (le provider est choisi dans le sous-processus via env/config) → **stdio MCP propre garanti**.
10. **Compat chemins** : aucun renommage ; si la fusion/fiche lisaient d'anciens chemins, ils restent valides (rien ne bouge).
11. **Arc & audio** : `narrative_arc.py` (Claude texte) et `scene_audio.py` (Gemini audio) **non modifiés**.

**Rollback** : `VLM_PROVIDER=claude` (ou défaut) restaure exactement le comportement actuel.

---

## 8. Tests à lancer AVANT et APRÈS

**Commande** : `.\.venv\Scripts\python.exe -m pytest -q` (hors-ligne ; aucun appel API).

**Baseline AVANT migration** (doit être vert, sert de référence) :
- toute la suite — état connu : **369 passed**.

**Ciblés / à AJOUTER pour la migration** (tous mockés, zéro API) :
- `tests/test_vlm_agent_runner.py` — ClaudeProvider toujours conforme au schéma (existant).
- `tests/test_vlm_provider.py` *(nouveau)* — sélecteur `claude|gemini|manual_ui` (env > config > défaut) ; défaut = claude.
- `tests/test_vlm_gemini_runner.py` *(nouveau)* — **mock `google.genai`** : la sortie Gemini respecte `build_output_schema` (12 champs, `segment_id` réinjecté, `confidence` clampée, listes par défaut) et passe `validate_single_result`.
- `tests/test_vlm_result_validation.py` *(à créer si absent)* — un résultat Gemini valide et un invalide.
- `tests/test_final_report.py` — `provenance.visual.mode` reflète le provider (paramétrer l'assertion `"claude_agent"`).
- `tests/test_pipeline.py` — l'étape 12 (mockée) fonctionne quel que soit le provider ; pas de régression de l'ordre/checkpoints.
- `tests/test_scene_audio.py` — **inchangé** : prouve que l'audio Gemini n'est pas affecté.

**Vérifs non-test (manuelles, hors API)** :
- `python -c "import src.video_agent.mcp_server"` → s'importe sans print sur stdout.
- Lancer le serveur MCP et confirmer **stdout vide** (logs uniquement sur stderr/fichier).
- `VLM_PROVIDER=claude` reproduit la baseline (rollback OK).

---

### Annexe — points de couplage à modifier (récap minimal)

| Fichier | Modif attendue (phase d'implémentation) |
|---|---|
| `src/video_agent/vlm_provider.py` *(nouveau)* | interface + ClaudeProvider + ManualUIProvider + `get_vlm_provider()` |
| `src/video_agent/vlm_gemini_runner.py` *(nouveau)* | GeminiVLMProvider (séparé de `scene_audio.py`) |
| `scripts/run_vlm_agent.py` / `run_vlm_agent_segment.py` | appeler le provider sélectionné (au lieu de Claude en dur) |
| `src/video_agent/vlm_strategy.py` | `gemini` dans `_IMPLEMENTED_MODES` |
| `src/video_agent/final_report.py` | `vlm_mode` dynamique (= provider), schéma 1.10 inchangé |
| `config/vlm_config.example.json` | documenter `provider: claude|gemini|manual_ui` (clé déjà présente) |
| **NON touchés** | `scene_audio.py`, `narrative_arc.py`, fusion, validation (schéma), chemins/JSON, tools MCP |
