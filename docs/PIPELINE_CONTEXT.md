# Agent d'analyse vidéo — Contexte complet du pipeline

> **Archive historique, non normative.** Certains noms de modes et versions de
> schémas ci-dessous décrivent un prototype antérieur. L’état de `v0.1.0` est
> documenté dans [ARCHITECTURE.md](ARCHITECTURE.md).

> Document factuel produit en **lisant le code réel** du dépôt
> `<racine-du-depot>/`. Les points non vérifiables dans le
> code sont marqués **(à confirmer)**. Aucun secret n'est exposé.

---

## 1. Vue d'ensemble

L'agent transforme une vidéo (fichier sous `VIDEO_ROOT`, par défaut `vidéos/`) en une
**analyse structurée scène par scène**. Le pipeline est *local-first* et déterministe
jusqu'au pack VLM, puis appelle un VLM (mode `claude_agent`, via le Claude Agent SDK
authentifié par abonnement) pour décrire chaque scène, valide les réponses contre un
schéma JSON, et agrège le tout en un **rapport final JSON** (`final_report_<stem>.json`)
+ un résumé texte lisible (`.txt`) et un export **PDF** optionnel. Chaque niveau existe
en **CLI** et comme **tool MCP** miroir (16 tools). L'objectif actuel est de produire ce
JSON d'analyse détaillé, fiable et stable comme contrat de données.

---

## 2. Arborescence (hors `.venv`, `__pycache__`, `.git`, vidéos)

```
Agent_Analyse_Vedio/
  CLAUDE.md
  README.md
  inspection.py                 # fichier à la racine (doublon ? non importé par src) (à confirmer)
  mcp_server.log                # log résiduel
  config/
    vlm_config.example.json
  docs/
    PIPELINE_CONTEXT.md          # ce document
  src/video_agent/
    __init__.py
    inspection.py                # N1 inspection ffprobe
    strategy.py                  # N2 décision de stratégie
    segmentation.py              # N3 segmentation (PySceneDetect)
    frame_extraction.py          # N4 extraction frames (FFmpeg)
    motion_representation.py     # N5 grille + composite (Pillow)
    audio_extraction.py          # N6B extraction audio (FFmpeg)
    audio_light_analysis.py      # N6C analyse audio légère (stdlib wave)
    transcription.py             # N6D transcription (faster-whisper)
    audio_visual_fusion.py       # N7 fusion alignée par timestamps
    vlm_pack_builder.py          # N8 pack VLM par segment
    vlm_strategy.py              # N9 modes/config + plan de run
    vlm_result_validation.py     # N10 validation des résultats VLM
    vlm_agent_runner.py          # N11 mode claude_agent (Claude Agent SDK)
    final_report.py              # N12 rapport final (agrégation)
    report_pdf.py                # N13 export PDF
    pipeline.py                  # N14 orchestrateur complet
    mcp_server.py                # serveur MCP (FastMCP) — 16 tools, 1711 lignes
  scripts/                       # 1 CLI par niveau (17 scripts)
    inspect_video.py  decide_strategy.py  segment_video.py  extract_frames.py
    build_motion.py  extract_audio.py  analyze_audio_light.py  transcribe_audio.py
    fuse_audio_visual.py  build_vlm_pack.py  prepare_vlm_run.py  validate_vlm_results.py
    run_vlm_agent.py  run_vlm_agent_segment.py  build_final_report.py
    export_report_pdf.py  run_full_pipeline.py
  tests/                         # 14 fichiers pytest
    test_segmentation.py  test_frame_extraction.py  test_motion_representation.py
    test_audio_extraction.py  test_audio_light_analysis.py  test_transcription.py
    test_audio_visual_fusion.py  test_vlm_pack_builder.py  test_vlm_strategy.py
    test_vlm_result_validation.py  test_vlm_agent_runner.py  test_final_report.py
    test_report_pdf.py  test_pipeline.py
  outputs/                       # artefacts générés (par <video_stem>)
    inspection_<stem>.json  strategy_<stem>.json  segments_<stem>.json
    frames_manifest_<stem>.json  motion_manifest_<stem>.json
    audio_manifest_<stem>.json  audio_light_<stem>.json  transcript_<stem>.json
    audio_visual_fusion_<stem>.json  vlm_pack_<stem>.json
    final_report_<stem>.{json,txt,pdf}  pipeline_status_<stem>.json
    frames/<stem>/segment_NNN/frame_NNN.jpg
    motion/<stem>/segment_NNN/{motion_grid,motion_composite}.jpg
    audio/<stem>/audio.wav
    vlm_runs/<stem>/{run_plan,selected_inputs}.json + prompt_segment_NNN.md
    vlm_results/<stem>/vlm_result_segment_NNN.json + vlm_results_summary.json
```

Stems présents dans `outputs/` : `demo`, `demo_darija`, `demo_home`, `demo_messi`.

---

## 3. Environnement & dépendances

- **Python** : 3.13.13 (venv local `.\.venv\Scripts\python.exe`).
- **Pas de `requirements.txt` ni `pyproject.toml`** : les dépendances sont installées
  directement dans le venv (à figer dans un requirements si besoin).

Versions clés (`pip freeze`) et rôle :

| Lib | Version | Rôle dans le pipeline |
|---|---|---|
| `fastmcp` | 3.4.2 | Serveur MCP (expose les 16 tools en stdio). |
| `mcp` | 1.27.2 | Protocole MCP sous-jacent. |
| `scenedetect` | 0.7 | N3 segmentation en scènes (ContentDetector). |
| `opencv-python` | 4.13.0.92 | Dépendance de PySceneDetect (décodage vidéo). |
| `pillow` | 12.2.0 | N5 grille temporelle + composite de mouvement. |
| `faster-whisper` | 1.2.1 | N6D transcription audio. |
| `ctranslate2` | 4.8.0 | Moteur de faster-whisper (CPU int8). |
| `av` | 17.1.0 | Décodage audio/vidéo (faster-whisper). |
| `onnxruntime` | 1.26.0 | VAD/dep de faster-whisper. |
| `claude-agent-sdk` | 0.2.101 | N11 mode `claude_agent` (appel VLM via abonnement). |
| `reportlab` | 4.5.1 | N13 génération PDF. |
| `arabic-reshaper` | 3.0.1 | N13 mise en forme des glyphes arabes. |
| `python-bidi` | 0.6.10 | N13 réordonnancement RTL de l'arabe. |
| `pypdfium2` | 5.10.1 | Rasterisation PDF (vérification visuelle ; pas dans le flux). |
| `numpy` | 2.4.6 | Calculs (présent ; utilisé indirectement). |
| `pytest` | 9.0.3 | Tests. |

FFmpeg/ffprobe sont requis sur le PATH (N1 inspection, N4 frames, N6B audio).
Les `anthropic` API key ne sont **pas** utilisées (mode `claude_agent` = abonnement CLI).

---

## 4. Pipeline par niveaux

| N° | Nom | But | Entrée | Sortie | Fichier(s) | Outils/libs |
|---|---|---|---|---|---|---|
| 1 | Inspection | Métadonnées techniques | vidéo | `inspection_<stem>.json` (audio.present, durée, fps, dims…) | `inspection.py` / `scripts/inspect_video.py` | ffprobe |
| 2 | Stratégie | Choix d'échantillonnage/audio | inspection | `strategy_<stem>.json` (`audio_strategy`, `frame_strategy`…) | `strategy.py` / `decide_strategy.py` | — |
| 3 | Segmentation | Découpage en scènes | vidéo + strategy | `segments_<stem>.json` (segments start/end) | `segmentation.py` / `segment_video.py` | PySceneDetect |
| 4 | Frames | Extraction d'images | vidéo + segments + strategy | `frames_manifest_<stem>.json` + `frames/<stem>/…jpg` | `frame_extraction.py` / `extract_frames.py` | FFmpeg |
| 5 | Motion | Grille + composite | frames_manifest | `motion_manifest_<stem>.json` + `motion/<stem>/…jpg` | `motion_representation.py` / `build_motion.py` | Pillow |
| 6B | Audio | WAV mono 16 kHz | vidéo (si audio) | `audio_manifest_<stem>.json` + `audio/<stem>/audio.wav` | `audio_extraction.py` / `extract_audio.py` | FFmpeg |
| 6C | Audio léger | Énergie/silence, reco Whisper | audio.wav | `audio_light_<stem>.json` (`recommendation.run_whisper`) | `audio_light_analysis.py` / `analyze_audio_light.py` | stdlib `wave` |
| 6D | Transcription | Parole horodatée | audio.wav (si run_whisper) | `transcript_<stem>.json` | `transcription.py` / `transcribe_audio.py` | faster-whisper |
| 7 | Fusion | Aligner audio/visuel par timestamps | segments+frames+motion (+audio_light+transcript) | `audio_visual_fusion_<stem>.json` | `audio_visual_fusion.py` / `fuse_audio_visual.py` | — |
| 8 | Pack VLM | Entrées par segment pour le VLM | fusion | `vlm_pack_<stem>.json` (frames sélectionnées, prompts) | `vlm_pack_builder.py` / `build_vlm_pack.py` | — |
| 9 | Stratégie/Run VLM | Config + prompts du run | pack + config | `vlm_runs/<stem>/{run_plan,selected_inputs}.json + prompt_segment_NNN.md` | `vlm_strategy.py` + `prepare_vlm_run.py` | — |
| 11 | Analyse claude_agent | Décrire chaque scène (VLM) | run + pack + config | `vlm_results/<stem>/vlm_result_segment_NNN.json` | `vlm_agent_runner.py` / `run_vlm_agent.py` | Claude Agent SDK |
| 10 | Validation | Valider les résultats VLM | results_dir + pack | `vlm_results/<stem>/vlm_results_summary.json` | `vlm_result_validation.py` / `validate_vlm_results.py` | — |
| 12 | Rapport final | Synthèse globale | results + fusion (+summary) | `final_report_<stem>.json` + `.txt` | `final_report.py` / `build_final_report.py` | — |
| 13 | Export PDF | PDF mis en forme (arabe OK) | final_report JSON | `final_report_<stem>.pdf` | `report_pdf.py` / `export_report_pdf.py` | reportlab + arabic-reshaper + python-bidi |
| 14 | Orchestrateur | Tout enchaîner depuis un nom de vidéo | nom de vidéo | tous les artefacts + `pipeline_status_<stem>.json` | `pipeline.py` / `run_full_pipeline.py` | (appelle les CLI ci-dessus) |

> Ordre d'exécution réel dans l'orchestrateur (N9 « prepare_vlm_run » et N11 « analyse »
> précèdent N10 « validation ») — voir §8.

---

## 5. Pattern CLI + MCP en miroir

**Principe.** Chaque niveau a : un **module** (`src/video_agent/<x>.py`) avec la logique
pure, un **script CLI** (`scripts/<x>.py`) qui parse les arguments, appelle le module et
écrit les fichiers en **UTF-8 `ensure_ascii=False`**, et un **tool MCP** qui **n'implémente
pas** la logique : il **délègue au CLI dans un sous-processus isolé** (stdout/stderr vers
fichiers, attente noyau bornée par `timeout`, `PYTHONUTF8=1`), puis relit le fichier de
sortie et le renvoie en dict. C'est le « miroir ».

**Exemple — Niveau 12 (rapport final).**

CLI `scripts/build_final_report.py` (extrait verbatim) :
```python
from video_agent.final_report import FinalReportError, build_final_report_from_files
...
    report = build_final_report_from_files(
        results_dir=args.results_dir,
        fusion_manifest_path=args.fusion,
        summary_path=summary_path,
    )
    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path = Path(args.text_output)
    text_path.write_text(report["readable_summary"], encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

Tool MCP `build_final_report_file` (extrait verbatim) :
```python
@app.tool(
    name="build_final_report_file",
    description=("Construire le rapport vidéo final : synthèse globale ... Retourne le rapport JSON."),
    run_in_thread=True,
)
def build_final_report_file(
    results_dir: str = "outputs/vlm_results/demo_darija",
    fusion_manifest_path: str = "outputs/audio_visual_fusion_demo_darija.json",
    summary_path: str | None = None,
    output_path: str = "outputs/final_report_demo_darija.json",
    text_output_path: str = "outputs/final_report_demo_darija.txt",
) -> dict:
    """Retourne le rapport vidéo final (dict JSON sérialisable)."""
    results_resolved = _resolve_existing_project_dir(results_dir, "Le dossier de résultats")
    fusion_file = _resolve_existing_project_file(fusion_manifest_path, "Le manifest de fusion")
    ...
    result = _run_final_report_subprocess(results_resolved, fusion_file, output_out, text_out, summary_file)
    return result
```

Helper de délégation `_run_final_report_subprocess` (extrait verbatim) :
```python
def _run_final_report_subprocess(results_dir, fusion_file, output_out, text_out, summary_file, timeout=120.0) -> dict:
    """Construit le rapport vidéo final dans un sous-processus isolé et retourne le JSON."""
    script = PROJECT_ROOT / "scripts" / "build_final_report.py"
    cmd = [sys.executable, str(script), "--results-dir", str(results_dir),
           "--fusion", str(fusion_file), "--output", str(output_out), "--text-output", str(text_out)]
    if summary_file is not None:
        cmd += ["--summary", str(summary_file)]
    env = os.environ.copy(); env["PYTHONUTF8"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
    with open(err_path, "w", encoding="utf-8") as errf:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=errf, timeout=timeout, cwd=str(PROJECT_ROOT), env=env)
    if proc.returncode != 0 or not output_out.exists():
        raise InspectError(...)
    return json.loads(output_out.read_text(encoding="utf-8"))
```

> Le tool MCP et la CLI partagent donc **le même code** (`build_final_report_from_files`),
> appelé via la CLI ; le tool ne réimplémente rien.

---

## 6. Intégration MCP (section centrale)

### 6.1 Définition & lancement du serveur

Le serveur est défini dans `src/video_agent/mcp_server.py` avec **FastMCP** (extrait verbatim) :
```python
app = FastMCP(
    name="Agent Analyse Vidéo MCP",
    instructions=("Expose inspect_video_file ... et run_full_pipeline_file pour exécuter tout le pipeline ..."),
    version="0.1",
)
```
Lancement (stdio), en fin de fichier (extrait verbatim) :
```python
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.info(f"Démarrage MCP stdio. VIDEO_ROOT={VIDEO_ROOT}")
    asyncio.run(app.run_stdio_async())
```
Chaque tool est enregistré par le décorateur `@app.tool(name=..., description=..., run_in_thread=True)`.
Sécurité des chemins : helpers `_resolve_video_path`, `_resolve_existing_project_file`,
`_resolve_existing_project_dir`, `_resolve_output_path` (tout doit rester sous `PROJECT_ROOT`,
les vidéos sous `VIDEO_ROOT`). `VIDEO_ROOT` = `$env:VIDEO_ROOT` sinon `vidéos/` sinon `videos/`.

### 6.2 Liste des 16 tools (nom, signature, rôle)

> `run_in_thread=True` sur tous **sauf** `inspect_video_file` et `decide_video_strategy`
> (légers, exécutés en ligne). Les tools « lourds » délèguent à un CLI en sous-processus.

| # | Tool | Signature (params : types = défauts) | Rôle |
|---|---|---|---|
| 1 | `inspect_video_file` | `(path: str) -> dict` | Inspecte la vidéo (ffprobe) → JSON normalisé. |
| 2 | `decide_video_strategy` | `(inspection: Any) -> dict` | Stratégie depuis l'inspection (dict ou chemin). |
| 3 | `segment_video_file` | `(path: str, strategy_path: str|None="outputs/strategy_demo.json") -> dict` | Segmentation en scènes (PySceneDetect, sous-processus). |
| 4 | `extract_video_frames` | `(path: str, segments_path="outputs/segments_demo.json", strategy_path="outputs/strategy_demo.json", output_dir="outputs/frames", manifest_path: str|None=None, overwrite=True, max_width: int|None=None, max_height: int|None=None) -> dict` | Extrait des frames (FFmpeg). |
| 5 | `build_motion_representation` | `(frames_manifest_path="outputs/frames_manifest_demo.json", output_dir="outputs/motion", manifest_path: str|None=None, thumbnail_width=360, composite_long_side=1080, overwrite=True) -> dict` | Grille + composite de mouvement (Pillow). |
| 6 | `extract_audio_track` | `(path: str, output_dir="outputs/audio", manifest_path: str|None=None, sample_rate=16000, channels=1, overwrite=True) -> dict` | Extrait l'audio WAV mono 16 kHz (FFmpeg). |
| 7 | `analyze_audio_light_file` | `(audio_path: str, audio_manifest_path: str|None=None, output_path: str|None=None, window_ms=500) -> dict` | Analyse énergie/silence + reco Whisper. |
| 8 | `transcribe_audio_file` | `(audio_path: str, audio_manifest_path: str|None=None, audio_light_analysis_path: str|None=None, output_path: str|None=None, model="small", device="cpu", compute_type="int8", language: str|None=None, task="transcribe", beam_size=5) -> dict` | Transcription faster-whisper. |
| 9 | `fuse_audio_visual_file` | `(segments_manifest_path: str, frames_manifest_path: str, motion_manifest_path: str, audio_light_path: str|None=None, transcript_path: str|None=None, output_path: str|None=None) -> dict` | Fusion alignée par timestamps. |
| 10 | `build_vlm_pack_file` | `(fusion_manifest_path: str, output_path: str|None=None, max_frames_per_segment=6, skip_not_ready=False) -> dict` | Pack VLM par segment. |
| 11 | `prepare_vlm_run_file` | `(pack_path: str, config_path="config/vlm_config.example.json", output_dir: str|None=None) -> dict` | Génère prompts + plan de run (sans modèle). |
| 12 | `validate_vlm_results_file` | `(results_dir: str, pack_path: str, output_path: str|None=None) -> dict` | Valide les résultats VLM vs pack → résumé. |
| 13 | `run_vlm_agent_analysis` | `(run_dir="outputs/vlm_runs/demo_darija", pack_path="outputs/vlm_pack_demo_darija.json", config_path="config/vlm_config.example.json", results_dir="outputs/vlm_results/demo_darija", model: str|None=None, overwrite=False, only: int|None=None) -> dict` | Analyse claude_agent de tous les segments + validation + résumé. |
| 14 | `build_final_report_file` | `(results_dir="outputs/vlm_results/demo_darija", fusion_manifest_path="outputs/audio_visual_fusion_demo_darija.json", summary_path: str|None=None, output_path="outputs/final_report_demo_darija.json", text_output_path="outputs/final_report_demo_darija.txt") -> dict` | Rapport final (synthèse globale). |
| 15 | `export_report_pdf_file` | `(report_path="outputs/final_report_demo_darija.json", output_path: str|None=None, font_path: str|None=None) -> dict` | Export PDF (arabe rendu correctement) ; renvoie `{pdf_path, size_bytes, report_path}`. |
| 16 | `run_full_pipeline_file` | `(video_name: str, config_path="config/vlm_config.example.json", model: str|None=None, overwrite_analysis=False) -> dict` | Pipeline complet depuis le nom d'une vidéo ; renvoie le statut. |

### 6.3 Comment un tool « miroir » réutilise la logique CLI

Pour les tools lourds : `tool → _resolve_*` (sécurité chemins) → `_run_<x>_subprocess` →
`subprocess.run([python, scripts/<x>.py, …], env=PYTHONUTF8=1, stdout/stderr→fichiers,
timeout)` → relecture du fichier de sortie JSON → renvoi du dict. La logique métier vit
dans le module `src/video_agent/<x>.py`, appelée par le CLI ; le tool ne fait que résoudre
les chemins et déléguer. (Voir l'exemple complet §5.)

**Raison du pattern sous-processus** : documenté dans le code/CLAUDE.md — sous le serveur
MCP stdio (Windows), le thread lecteur stdin affame le GIL, ce qui bloque le travail
CPU lourd (ex. PySceneDetect) et `subprocess capture_output` ; déléguer à un **processus
séparé** avec stdout/stderr vers **fichiers** (pas de pipes lus par le parent) résout le
blocage. `PYTHONUTF8=1` évite le crash d'encodage (texte arabe imprimé en cp1252).

---

## 7. Modes VLM (`manual_ui` vs `claude_agent`)

Définis dans `src/video_agent/vlm_strategy.py` (extrait verbatim) :
```python
_KNOWN_MODES = {"manual_ui", "claude_code", "claude_agent", "api", "local"}
_IMPLEMENTED_MODES = {"manual_ui", "claude_code", "claude_agent"}
_NOT_IMPLEMENTED_MODES = {"api", "local"}
```
`validate_config()` (docstring verbatim) : « Rejette explicitement les modes non
implémentés (`api`, `local`) avec le message « mode non implémenté à ce stade ». »
Le mode vient du fichier `config/vlm_config.example.json` (champ `mode`, défaut
`manual_ui` (à confirmer la valeur exacte du fichier)). Le champ `model` est laissé vide
dans la config et **jamais codé en dur** dans le code.

- **`manual_ui`** : `prepare_vlm_run.py` (N9) génère, par segment, un `prompt_segment_NNN.md`
  + `selected_inputs.json` ; l'utilisateur fournit lui-même prompt+images à un VLM externe.
  Aucun appel modèle automatique.
- **`claude_agent`** (branché dans `src/video_agent/vlm_agent_runner.py`) : automatise
  l'analyse via le **Claude Agent SDK** (`claude-agent-sdk`), authentifié par
  l'**abonnement** (CLI Claude Code embarqué, **pas de clé API**). `analyze_segment_with_agent()`
  envoie images + texte du prompt via `query(..., ClaudeAgentOptions(output_format={"type":
  "json_schema","schema": build_output_schema(segment_id)}, model=..., allowed_tools=[],
  permission_mode="bypassPermissions", setting_sources=[]))` et récupère
  `ResultMessage.structured_output` (JSON garanti par schéma). Le `model` vient de la config
  (None ⇒ modèle par défaut du CLI). Le prompt envoyé inclut deux consignes ajoutées par le
  runner : `_FRENCH_DIRECTIVE` (toutes les descriptions en français, sans traduire paroles /
  texte à l'écran / noms propres) et `_MOOD_DIRECTIVE` (champ `mood`, liste courte en français).
- **Bascule** : via le champ `mode` de la config. **L'orchestrateur (N14) utilise
  `claude_agent`** (appelle `run_vlm_agent.py`). `manual_ui` est le flux « prompts + humain ».
  `claude_code` est listé comme implémenté mais n'a pas de runner dédié distinct (à confirmer
  son usage exact).
- **Pourquoi `api`/`local` rejetés** : commentaire/docstring du code → non implémentés à ce
  stade, pour éviter tout appel API accidentel et coût inattendu (ils pourront être ajoutés).

---

## 8. Orchestrateur `run_full_pipeline.py` (séquence exacte)

`scripts/run_full_pipeline.py` → `pipeline.run_full_pipeline(video_name, …)`. Entrée : le
**nom** d'une vidéo sous `VIDEO_ROOT`. Chaque étape est un CLI lancé en sous-processus
(`_run_step`, `PYTHONUTF8=1`), statut collecté, erreurs isolées. Ordre réel des `step(...)` :

1. `inspection`  → `inspect_video.py` → `inspection_<stem>.json` (lit `audio.present`)
2. `strategy`    → `decide_strategy.py`
3. `segmentation`→ `segment_video.py`
4. `frames`      → `extract_frames.py`
5. `motion`      → `build_motion.py`
6. *(si audio)* `audio_extraction` → `extract_audio.py`
7. *(si audio)* `audio_light_analysis` → `analyze_audio_light.py` (lit `recommendation.run_whisper`)
8. *(si audio & run_whisper)* `transcription` → `transcribe_audio.py --model small`
9. `fusion`      → `fuse_audio_visual.py` (ajoute `--audio-light`/`--transcript` seulement si produits)
10. `vlm_pack`   → `build_vlm_pack.py`
11. `prepare_vlm_run` → `prepare_vlm_run.py` (prompts + selected_inputs)
12. `claude_agent_analysis` → `run_vlm_agent.py` (analyse + validation + `vlm_results_summary.json`) — code retour 2 toléré
13. `final_report` → `build_final_report.py` → `final_report_<stem>.json` + `.txt`

Étapes critiques : un échec arrête proprement (statut partiel). **Branche audio
optionnelle** : si `audio.present=false`, les étapes 6–8 sont sautées et la fusion se fait
sans audio (`uses_speech=false`, `evidence.audio` vide accepté par le validateur). Le statut
final est écrit dans `outputs/pipeline_status_<stem>.json` :
`{pipeline_type:"full_video_pipeline", video_path, video_stem, audio_present, steps[], ok,
reports:{json, txt}, summary}`.

---

## 9. Schéma du `final_report` JSON (`schema_version = "1.0"`)

Construit par `final_report.build_final_report()`. **Clés de premier niveau, dans l'ordre :**

| Clé | Type | Portée |
|---|---|---|
| `schema_version` | str (`"1.0"`) | global |
| `report_type` | str (`"final_video_report"`) | global |
| `video_path` | str | global |
| `language` | str \| null | global (langue détectée, depuis la fusion) |
| `sources` | object `{results_dir, fusion_manifest, validation_summary}` | global |
| `scene_count` | int | global |
| `total_duration_s` | number \| null | global |
| `scenes` | list[object] | **par scène** (voir ci-dessous) |
| `aggregate` | object | **agrégat global** (voir ci-dessous) |
| `next_step` | str (`"complete"`) | global |
| `global_summary` | str | global |
| `readable_summary` | str | global (résumé texte multi-ligne ; aussi écrit en `.txt`) |

**Objet `scenes[]` — 15 clés par scène, ordre constant :**

| Clé | Type | Source |
|---|---|---|
| `segment_id` | int | — |
| `start_s` / `end_s` / `duration_s` | number \| null | fusion |
| `scene_description` | str | VLM |
| `people_objects` | list[str] | VLM |
| `actions` | list[str] | VLM |
| `motion_description` | str | VLM |
| `audio_visual_relation` | str | VLM |
| `uses_speech` | bool | VLM |
| `mood` | list[str] | VLM |
| `combined_text` | str | fusion (paroles transcrites) |
| `confidence` | number \| null | VLM |
| `uncertainties` | list[str] | VLM |
| `evidence` | object `{visual: list[str], audio: list[str]}` | VLM |

**Objet `aggregate` (global) :**

| Clé | Type |
|---|---|
| `people_objects` | list[str] (dédupliqué) |
| `actions` | list[str] (dédupliqué) |
| `tone` | list[str] (heuristique mots-clés) |
| `dominant_moods` | list[str] (union dédupliquée des `mood`) |
| `languages` | list[str] |
| `uses_speech_any` | bool |
| `segments_with_speech` | int |
| `audio_activity_mode` | str \| null |
| `average_confidence` | number |
| `average_transcript_confidence` | number \| null |
| `uncertainties` | list[str] (dédupliqué ; incertitudes « pas d'audio » exclues si `uses_speech_any`) |

**Contrat de structure** : tous les champs sont **toujours présents** avec un type cohérent
(listes → `[]`, chaînes → `""`, nombres → nombre ou `null`), ordre des clés constant d'une
scène et d'une vidéo à l'autre (normaliseurs `_as_str` / `_as_list` / `_as_num`).

---

## 10. Tests

- **Organisation** : 14 fichiers `tests/test_<module>.py`, un par module (logique pure ;
  PySceneDetect/FFmpeg/Whisper/SDK mockés ou helpers factices ; le SDK Claude est mocké dans
  `test_vlm_agent_runner.py` via `sys.modules`).
- **Commande** : `\.venv\Scripts\python.exe -m pytest tests/`
  *(`pytest -q` sans cible fonctionne aussi ; lancer `tests/` reste recommandé).*
- **Total** : **255 tests** (3 warnings inoffensifs — `Image.getdata` déprécié dans Pillow,
  retrait prévu 2027).

---

## 11. Signatures des fichiers clés

### `final_report.py`
```python
class FinalReportError(Exception): ...
SCHEMA_VERSION = "1.0"
def load_json(path: str, label: str) -> Dict[str, Any]            # lecture JSON robuste
def build_final_report(vlm_results: Dict[int, Dict], fusion: Dict,
                       summary: Optional[Dict] = None,
                       sources: Optional[Dict] = None) -> Dict     # agrège VLM + fusion → rapport
def build_final_report_from_files(results_dir: str, fusion_manifest_path: str,
                                  summary_path: Optional[str] = None) -> Dict  # charge fichiers puis agrège
# internes : _as_str/_as_list/_as_num (normaliseurs de type), _dedup_preserve,
#            _is_no_audio_uncertainty, _detect_tone_hints, _truncate, _build_scene,
#            _fmt_ts, _build_readable_summary, _load_vlm_results
```

### `vlm_agent_runner.py`
```python
class VlmAgentError(Exception): ...
def build_output_schema(segment_id: int) -> Dict[str, Any]
    # json_schema de sortie ; segment_id figé via const ; required inclut mood
def analyze_segment_with_agent(prompt_text: str, image_paths: List[str], segment_id: int,
                               model: Optional[str] = None, timeout_s: float = 300.0) -> Dict[str, Any]
    # Analyse un segment via le Claude Agent SDK et retourne un dict JSON validé par schéma
# internes : _image_block, _build_user_message (ajoute _FRENCH_DIRECTIVE + _MOOD_DIRECTIVE),
#            _run_query (import différé du SDK, asyncio.wait_for)
```

### `vlm_result_validation.py`
```python
class VlmResultValidationError(Exception): ...
def load_json(path: str, label: str) -> Dict[str, Any]
def validate_result_schema(result: Dict[str, Any]) -> List[str]
    # Valide le schéma d'un résultat VLM. Retourne la liste des erreurs bloquantes.
    # Champs requis : segment_id, scene_description, people_objects, actions,
    # motion_description, audio_visual_relation, uses_speech, mood (liste),
    # confidence (nombre, [0,1]), uncertainties (liste), evidence.{visual,audio} (listes)
def check_segment_id(result, expected_segment_id: int) -> List[str]   # concordance segment_id (bloquant)
def check_confidence_range(result) -> List[str]                       # confidence ∈ [0,1]
def check_evidence(result, segment_pack=None) -> List[str]            # warnings sur evidence vide
def check_anti_hallucination_flags(result, segment_pack=None) -> List[str]  # warnings anti-hallucination
def validate_single_result(result, expected_segment_id: int, segment_pack=None) -> Dict
    # → {valid, segment_id, errors, warnings, confidence}
def validate_results_dir(results_dir: str, vlm_pack: Dict) -> Dict    # valide un dossier → résumé
```

---

## 12. État des corrections

**Déjà corrigé / en place (dans le code) :**
- **Français uniforme** des champs descriptifs en mode `claude_agent` (`_FRENCH_DIRECTIVE`),
  sans traduire paroles (`combined_text`), texte à l'écran cité, ni noms propres.
- **Contradiction audio** : la ligne « Audio » du résumé est dynamique (langue détectée,
  « approximative » si confiance transcription < 0.7, plus de libellé « Darija » statique) ;
  les incertitudes « pas d'audio » sont exclues de l'**agrégat global** quand
  `uses_speech_any=true` (mais conservées au niveau de la scène concernée).
- **`schema_version` + structure stable** : `schema_version="1.0"`, tous les champs toujours
  présents et typés, ordre des clés constant (scène et agrégat), `evidence` toujours présent.
- **`mood` par scène** (liste française, requis dans le validateur et le `output_format`) +
  **`aggregate.dominant_moods`** (émotions dominantes, dédupliquées, ordre préservé).
- **Déduplication** insensible à la casse/ponctuation finale pour les agrégats.

**Prévu / pas encore dans le code (à confirmer la priorité) :**
- **Arc narratif** (relation/évolution entre scènes, progression de la vidéo).
- **Champs structurés** supplémentaires (ex. entités normalisées, rôles, objets typés).
- **OCR** du texte à l'écran (sous-titres incrustés) — actuellement non effectué (le VLM lit
  visuellement, signalé dans les incertitudes).
- **Métadonnées** enrichies (provenance, modèle utilisé, horodatage de génération, coûts).

---

*Fin du document — généré par lecture du code réel ; sections incertaines marquées « (à confirmer) ».*
