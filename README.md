# Video Analysis Agent

**Explainable multimodal video analysis agent for short-form content.**

[![Release](https://img.shields.io/badge/release-v0.1.0-159779)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-481%20passed-159779)](tests/)
[![License](https://img.shields.io/badge/license-MIT-0c2233)](LICENSE)
[![VLM default](https://img.shields.io/badge/VLM-manual%20%7C%20offline-d89a35)](#modes-vlm)

Pipeline Python local-first qui transforme une vidéo en artefacts structurés et
auditables : scènes, frames, audio, transcription, packs VLM, validation stricte
et rapport final. Chaque analyse est isolée par un `run_id` déterministe.

Le mode `manual` est le mode par défaut : il prépare les packs et prompts sans
SDK Gemini, clé API ou appel réseau. Gemini n'est ni gratuit par hypothèse, ni
obligatoire ; il intervient uniquement en mode `automatic` explicitement configuré.

> **Démonstration visuelle :** l’emplacement prévu est
> `docs/demo/video-agent-demo.gif`. Aucun faux écran n’est publié : consultez le
> [guide d’enregistrement](docs/demo/README.md) pour produire le GIF depuis
> l’application réelle et le dataset synthétique.

## Démarrage rapide

```console
python -m pip install ".[web]"
video-agent generate-demo
video-agent web --ui --host 127.0.0.1 --port 8000
```

Dans l’interface, cliquer sur **Lancer la démonstration**. Ce parcours est local,
reproductible, sans vidéo personnelle et sans fournisseur cloud. Si le frontend
n’est pas encore compilé, exécuter d’abord `npm ci` puis `npm run build` dans
`frontend/`.

Benchmark de référence du scénario synthétique : **3/3 scènes**, **100 % de
couverture**, **F1 des limites à ±500 ms = 1,000**, médiane locale **19,204 s**
sur deux exécutions. Ces temps décrivent une machine et une configuration
précises ; voir [le benchmark](evaluation/BENCHMARK.md) et ses limites.

Pour une présentation technique plus détaillée, voir le
[portfolio du projet](docs/PORTFOLIO.md) et le
[diagramme d’architecture](docs/ARCHITECTURE.md).

> Le projet est en version alpha `0.1.0`. Les formats principaux sont testés, mais
> les interfaces peuvent encore évoluer avant la version 1.0.

## Fonctionnalités

- inspection FFprobe et stratégie d'analyse ;
- segmentation visuelle avec PySceneDetect ;
- extraction FFmpeg de frames et de pistes audio ;
- grilles/composites de mouvement avec Pillow ;
- analyse audio locale et transcription faster-whisper optionnelle ;
- fusion par chevauchement temporel ;
- packs VLM et prompts par segment ;
- modes VLM `disabled`, `manual` et `automatic` ;
- fournisseur Gemini optionnel, chargé uniquement en mode automatique ;
- validation stricte des résultats VLM et rapport final structuré ;
- runs déterministes, isolés et reprenables ;
- CLI locale, serveur MCP et API FastAPI ;
- interface React responsive pour l'upload, le suivi et la consultation.

## Architecture

```text
React / CLI / MCP → pipeline Python → outputs/runs/<run_id>/
                           ├─ disabled  → local_complete
                           ├─ manual    → awaiting_vlm → validation → rapport
                           └─ automatic → Gemini explicite → validation → rapport
```

Le code applicatif se trouve dans `src/video_agent/`. Les scripts historiques de
`scripts/` restent la source des commandes détaillées et sont réutilisés par les
points d'entrée installables.

Le pipeline couvre inspection, segmentation, frames, mouvement, audio,
transcription, fusion et préparation VLM. FastAPI l’expose sans dupliquer la
logique métier ; React consomme uniquement l’API. Le diagramme complet, les
checkpoints et les statuts sont documentés dans
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

L'interface se trouve dans `frontend/`. Elle ne contient aucune logique métier :
elle appelle l'API, affiche ses statuts et ne déclare jamais un rapport `complete`
si le backend ne retourne pas ce statut.

## Prérequis

- Python 3.12 ou 3.13 ;
- FFmpeg et FFprobe accessibles dans le `PATH`.

Vérification :

```console
ffmpeg -version
ffprobe -version
```

Sous Windows, FFmpeg peut par exemple être installé via `winget`, Chocolatey ou
Scoop. Sous Debian/Ubuntu : `sudo apt install ffmpeg`.

## Installation

Installation minimale, suffisante pour l'inspection, la segmentation, l'extraction
de frames, les traitements visuels, la préparation manuelle et MCP :

```console
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install .
```

Pour une **analyse locale complète d'une vidéo avec audio**, incluant la
transcription faster-whisper et les features librosa, l'installation recommandée
est :

```console
python -m pip install ".[audio]"
```

Une installation minimale peut traiter une vidéo sans parole, mais elle ne peut
pas exécuter une transcription demandée. Le pipeline renvoie alors une erreur
explicite invitant à installer l'extra `audio`.

Installation éditable pour contribuer :

```console
python -m pip install -e ".[all,dev]"
```

### Extras

| Extra | Fonctionnalités ajoutées |
|---|---|
| `audio` | faster-whisper, librosa, NumPy et lecture audio scientifique |
| `gemini` | SDK `google-genai` pour le mode `automatic` |
| `pdf` | export PDF avec ReportLab |
| `web` | API FastAPI, serveur Uvicorn et gestion des formulaires multipart |
| `dev` | pytest, Ruff et construction de distributions |
| `all` | toutes les fonctionnalités d'exécution (`audio`, `gemini`, `pdf`, `web`) |

Exemples :

```console
python -m pip install ".[audio]"
python -m pip install ".[audio,pdf]"
python -m pip install ".[all,dev]"
```

Le SDK Gemini ne fait pas partie des dépendances minimales.

## Commandes

```console
video-agent --help
video-agent run --help
video-agent-run --help
video-agent-validate --help
video-agent mcp
video-agent web --help
```

`video-agent-run` et `video-agent run` exécutent la même fonction historique.

Par défaut, les vidéos sont recherchées dans `videos/` ou `vidéos/`. La racine
peut être redéfinie avec `VIDEO_ROOT`.

## API locale

Installation de l'API seule, ou installation recommandée avec toute la chaîne
audio locale :

```console
python -m pip install ".[web]"
python -m pip install ".[audio,web]"
```

Lancement sur la boucle locale uniquement :

```console
video-agent web --host 127.0.0.1 --port 8000
```

La documentation Swagger est disponible sur <http://127.0.0.1:8000/docs>.
L'option `--reload` est réservée au développement. Le serveur n'écoute jamais
automatiquement sur toutes les interfaces réseau.

### Exécution synchrone

`POST /api/runs` conserve le comportement historique : la connexion HTTP reste
ouverte pendant toutes les étapes locales.

```console
curl -X POST http://127.0.0.1:8000/api/runs \
  -F "video=@clip.mp4" \
  -F "mode=manual"
```

La requête attend la fin des étapes locales puis retourne le `run_id`, le statut
`awaiting_vlm`, le dossier logique et les principaux artefacts. Aucun appel Gemini
n'est effectué. Les prompts sont consultables via l'API ; les JSON produits
manuellement peuvent ensuite être importés avec
`POST /api/runs/{run_id}/vlm-results`, validés et agrégés en rapport.

### Jobs asynchrones locaux

`POST /api/jobs` applique les mêmes validations et protections d'upload, mais
retourne immédiatement `202 Accepted`. L'analyse est exécutée dans une file locale
en mémoire par un `ThreadPoolExecutor` borné :

```console
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F "video=@clip.mp4" \
  -F "mode=manual"
```

Réponse initiale :

```json
{
  "job_id": "identifiant-aleatoire",
  "status": "queued",
  "progress_percent": 0,
  "status_url": "/api/jobs/identifiant-aleatoire"
}
```

Polling :

```console
curl http://127.0.0.1:8000/api/jobs/identifiant-aleatoire
```

Le pipeline publie sa progression après l'achèvement réel de ses étapes
(inspection, segmentation, frames, audio, transcription, fusion, préparation et
analyse VLM éventuelle, validation et rapport). Le pourcentage n'augmente pas
pendant une opération longue sans nouvelle étape terminée.

Le nombre de jobs simultanés est contrôlé par `VIDEO_AGENT_MAX_WORKERS` :

```powershell
$env:VIDEO_AGENT_MAX_WORKERS = "1"
video-agent web
```

La valeur par défaut est `1` et les valeurs acceptées vont de `1` à `8`. Une
valeur invalide empêche le démarrage avec une erreur de configuration claire.

La file et l'historique des jobs sont uniquement en mémoire : ils ne survivent
pas au redémarrage du serveur. À l'arrêt normal, l'API refuse les nouveaux jobs,
annule ceux qui n'ont pas démarré, attend la fin des jobs déjà en cours, puis
nettoie leurs uploads. Après un arrêt brutal, les uploads abandonnés sont supprimés
au prochain démarrage lorsqu'ils dépassent
`VIDEO_AGENT_UPLOAD_TTL_SECONDS` (24 heures par défaut).

L'annulation via `POST /api/jobs/{job_id}/cancel` est fiable uniquement pour un
job encore `queued`. Un job déjà `running` retourne `409` : FFmpeg, Whisper et les
sous-processus actuels ne sont pas interrompus de force.

Endpoints principaux :

- `GET /api/health` : dépendances système et extras disponibles ;
- `POST /api/runs` : upload contrôlé et exécution du pipeline existant ;
- `POST /api/jobs` : création asynchrone avec réponse immédiate ;
- `GET /api/jobs` et `/api/jobs/{job_id}` : liste et suivi de progression ;
- `POST /api/jobs/{job_id}/cancel` : annulation d'un job encore en attente ;
- `GET /api/runs/{run_id}` : manifest, statut et diagnostics ;
- `GET /api/runs/{run_id}/scenes` et `/report` : résultats structurés ;
- `GET /api/runs/{run_id}/artifacts` : catalogue sécurisé du manifest ;
- `POST /api/runs/{run_id}/validate` et `/build-report` : validation et rapport.

## Interface web

L'interface permet à un utilisateur non technique de déposer une vidéo, choisir
le mode VLM, suivre le job, parcourir les scènes et compléter un run manuel.
L'architecture reste séparée :

```text
Navigateur React/Vite
  → API FastAPI /api/*
    → file locale de jobs
      → pipeline Python existant
        → outputs/runs/<run_id>/
```

Capture de démonstration : emplacement prévu dans
`docs/screenshots/frontend-dashboard.png`.
GIF de démonstration : emplacement prévu dans `docs/demo/video-agent-demo.gif`.

### Développement

Prérequis frontend : Node.js 20.19+ ou 22.12+. Installer d'abord le backend
recommandé :

```console
python -m pip install ".[audio,web]"
```

Puis lancer deux terminaux :

```console
# Terminal 1 — racine du dépôt
video-agent web --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
npm install
npm run dev
```

Ouvrir <http://127.0.0.1:5173>. La variable `VITE_API_BASE_URL` permet de
changer l'URL de l'API ; sa valeur de développement par défaut est
`http://127.0.0.1:8000`.

### Production locale

```console
cd frontend
npm install
npm run build
cd ..
video-agent web --ui --host 127.0.0.1 --port 8000
```

Ouvrir ensuite <http://127.0.0.1:8000>. FastAPI sert `frontend/dist` lorsqu'il
est présent, garde les routes `/api/*` prioritaires et renvoie `index.html` lors
du rechargement d'une route frontend. Sans build, l'API démarre normalement :
Node.js n'est donc pas requis pour un usage API seul.

### Parcours manual

Le mode `manual`, sélectionné par défaut, envoie la vidéo à `POST /api/jobs`,
affiche les étapes réellement terminées puis présente le run en
`awaiting_vlm`. L'utilisateur peut télécharger les packs et prompts déclarés,
importer un ou plusieurs JSON, relancer la validation stricte et construire le
rapport. Un résultat partiel reste affiché `partial`.

L'interface ne survit pas davantage que l'API à un redémarrage pour le suivi des
jobs en mémoire ; les artefacts des runs sont, eux, conservés. L'annulation est
disponible uniquement avant le démarrage effectif du job. Il n'y a actuellement
ni authentification, ni WebSocket, ni base de données.

## Démonstration reproductible

La démonstration ne requiert ni vidéo personnelle, ni clé API, ni téléchargement
de média. Elle génère localement une vidéo verticale CC0 de 18 secondes avec
trois scènes, des formes animées et du texte à l'écran :

```console
video-agent generate-demo
```

Les sorties sont `outputs/demo/video_agent_demo.mp4` et
`outputs/demo/demo_manifest.json`.

Dans l'interface, cliquer sur **Lancer la démonstration**. Le backend génère ou
réutilise la vidéo, crée un job `manual`, puis affiche la progression habituelle.
Le bandeau « Mode démonstration local » confirme qu'aucune donnée n'est envoyée
à un service externe.

Pour présenter le rapport complet sans appeler un modèle, utiliser ensuite
**Charger les résultats de démonstration**. Ces résultats sont explicitement
étiquetés comme fixtures simulées. Leur chargement est refusé si le hash du run
ne correspond pas exactement à la vidéo synthétique générée.

## Évaluation et benchmark

Le dataset de référence se trouve dans `evaluation/datasets/demo/`. La vidéo
n'est pas versionnée : elle est régénérée automatiquement.

```console
video-agent evaluate --dataset evaluation/datasets/demo \
  --include-performance \
  --scene-tolerance-ms 500

video-agent evaluate \
  --run-id <run_id> \
  --annotations evaluation/datasets/demo/annotations/demo_annotations.json \
  --format html

video-agent benchmark \
  --dataset evaluation/datasets/demo \
  --repeat 3 \
  --output outputs/demo/benchmark_report.json
```

L'évaluation écrit systématiquement `evaluation_report.json` et
`evaluation_report.html`. Le benchmark écrit un JSON et un tableau Markdown.

Exemple mesuré sur la vidéo synthétique :

```json
{
  "status": "local_complete",
  "expected_scenes": 3,
  "detected_scenes": 3,
  "boundary_f1": 1.0,
  "temporal_coverage_ratio": 1.0,
  "frame_count": 24,
  "error_count": 0,
  "warning_count": 0
}
```

Le [benchmark local généré](evaluation/BENCHMARK.md) donne, sur la machine de
référence et deux répétitions, une médiane de 19,204 s avec un écart-type
population de 0,054 s. Ces temps ne doivent pas être comparés directement à
ceux d'une autre machine.

### Interprétation honnête des métriques

Mesures fiables :

- durée et timestamps extraits ;
- nombre de scènes et de frames ;
- trous, chevauchements, couverture et taille des artefacts ;
- temps muraux sur la machine et la configuration enregistrées ;
- validité structurelle des résultats et cohérence des identifiants.

Mesures approximatives :

- précision/rappel des limites, car ils dépendent de la tolérance choisie ;
- objets, actions et texte à l'écran, comparés lexicalement après normalisation ;
- similarité textuelle `SequenceMatcher`, qui compare des caractères et ne
  mesure pas scientifiquement la qualité sémantique.

La taille maximale d'upload vaut 2 Gio par défaut et peut être ajustée avec
`VIDEO_AGENT_MAX_UPLOAD_BYTES`. Seules les extensions vidéo explicitement
autorisées sont acceptées ; le nom envoyé par le navigateur n'est jamais utilisé
comme chemin de stockage.

## Modes VLM

La configuration de référence se trouve dans `config/vlm_config.example.json`.

### Manual, mode par défaut

```json
{
  "mode": "manual",
  "provider": null,
  "model": ""
}
```

```console
video-agent-run clip.mp4 --config config/vlm_config.example.json
```

Le pipeline termine avec `awaiting_vlm` et indique les packs/prompts à analyser.
Aucune clé API ni dépendance Gemini n'est nécessaire.

### Disabled

Créez une configuration locale non suivie contenant :

```json
{
  "mode": "disabled",
  "provider": null,
  "model": ""
}
```

Puis exécutez :

```console
video-agent-run clip.mp4 --config config/vlm_config.disabled.json
```

Le statut final est `local_complete`. Les artefacts locaux et le pack sont produits,
mais aucune analyse VLM n'est effectuée.

### Automatic avec Gemini

```console
python -m pip install ".[gemini]"
```

Configurez la clé dans l'environnement, jamais dans un fichier suivi par Git :

```powershell
$env:GEMINI_API_KEY = "votre-cle-locale"
```

Configuration :

```json
{
  "mode": "automatic",
  "provider": "gemini",
  "gemini_vlm_model": "gemini-2.5-flash",
  "gemini_text_model": "gemini-2.5-flash"
}
```

```console
video-agent-run clip.mp4 --config config/vlm_config.automatic.json
```

Un mode ou provider inconnu provoque une erreur explicite. Claude n'est pas encore
implémenté.

## Isolation des runs

L'identité d'un run dépend du SHA-256 du contenu vidéo, de la configuration, des
paramètres importants, des schémas et des prompts. Le nom du fichier n'est pas une
identité.

```text
outputs/runs/<run_id>/
├── run_manifest.json
├── checkpoints/
├── inspection/
├── scenes/
├── frames/
├── audio/
├── transcription/
├── fusion/
├── vlm_pack/
├── vlm_results/
└── reports/
```

Les checkpoints vérifient les hashes des entrées et sorties avant toute reprise.
Les anciens artefacts basés sur le nom de fichier sont signalés comme legacy mais
ne sont pas réutilisés automatiquement.

## Confidentialité

Les modes `manual` et `disabled` n'effectuent aucun appel VLM réseau. Les outils
système locaux restent responsables de leurs propres mises à jour ou téléchargements
de modèles.

En mode `automatic`, des frames, transcriptions, textes et éventuellement extraits
audio peuvent être transmis à Gemini. Avant utilisation, vérifiez les consentements,
la base légale, les obligations de confidentialité et les conditions du fournisseur.

## Développement

```console
python -m ruff check .
python scripts/check_repository_hygiene.py
python -m pytest
python -m pip check
python -m build
```

La CI exécute ces contrôles sous Windows 3.12/3.13 et Linux 3.12.

## Current limitations

- FFmpeg est une dépendance système, non installée par pip.
- Les modèles Whisper sont téléchargés au premier usage selon le backend.
- Gemini est le seul provider automatique implémenté.
- Aucun verrou interprocessus ne sérialise encore deux exécutions simultanées du
  même `run_id`.
- Les anciens dossiers `outputs/` ne sont pas migrés.
- `/api/runs` est synchrone ; `/api/jobs` fournit la file locale asynchrone.
- Les jobs asynchrones et leur historique sont perdus à chaque redémarrage.
- L'annulation ne peut pas interrompre un job qui exécute déjà FFmpeg ou Whisper.
- L'API ne fournit encore ni authentification, ni file distribuée, ni persistance
  des jobs.
- Le benchmark local n'est pas une comparaison normalisée entre machines.
- Les métriques VLM lexicales ne remplacent pas une évaluation humaine.

## Licence

MIT, voir [LICENSE](LICENSE).
