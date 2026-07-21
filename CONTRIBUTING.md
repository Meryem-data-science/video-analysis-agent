# Contribuer

Merci de contribuer à Video Analysis Agent. Le projet vise un pipeline local-first,
déterministe et testable sous Windows et Linux.

## Environnement de développement

Prérequis : Python 3.12 ou 3.13 et FFmpeg/ffprobe disponibles dans le `PATH`.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all,dev]"
```

Sous Linux, remplacez `.\.venv\Scripts\python.exe` par `.venv/bin/python`.

Avant une pull request :

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/check_repository_hygiene.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

## Règles

- Ne commitez aucune vidéo utilisateur, sortie de run, clé API ou donnée privée.
- Ajoutez des tests pour toute correction ou nouvelle fonctionnalité.
- Préservez les modes `manual`, `disabled` et `automatic`.
- Documentez les changements visibles dans `CHANGELOG.md`.
- Gardez les changements métier séparés des changements de packaging.
