# Interface web

Interface React, Vite et TypeScript de Video Analysis Agent. Elle pilote
exclusivement l'API FastAPI : aucune inspection, segmentation, transcription ou
validation VLM n'est réimplémentée côté navigateur.

## Prérequis

- Node.js 20.19+ ou 22.12+ ;
- l'API locale installée avec `.[web]`, idéalement `.[audio,web]`.

## Développement

Terminal 1, à la racine du dépôt :

```console
video-agent web --host 127.0.0.1 --port 8000
```

Terminal 2 :

```console
cd frontend
npm install
npm run dev
```

Ouvrir <http://127.0.0.1:5173>. L'URL de l'API vaut
`http://127.0.0.1:8000` par défaut. Pour la changer, créer un fichier local
`frontend/.env.local` :

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Ne placez aucun secret dans une variable `VITE_*` : Vite les expose au bundle
navigateur.

## Tests et build

```console
npm test
npm run typecheck
npm run build
```

Le build est produit dans `frontend/dist/`. FastAPI le détecte automatiquement
depuis le dépôt, ou peut être lancé explicitement avec :

```console
video-agent web --ui --host 127.0.0.1 --port 8000
```

L'API reste disponible sous `/api/*` et le fallback SPA permet de recharger une
route frontend directement.

## Principes

- `manual` est le mode initial et n'appelle aucun fournisseur cloud ;
- le polling d'un job est séquentiel et s'arrête sur `completed`, `failed` ou
  `cancelled` ;
- seuls les jobs `queued` proposent une annulation ;
- les résultats VLM importés sont validés par le backend ;
- les erreurs affichées excluent les stacks et chemins locaux.

La file de jobs reste locale et volatile : un redémarrage du serveur efface son
historique. Les runs déjà écrits dans `outputs/runs/` restent disponibles.

## Mode démonstration

Le bouton **Lancer la démonstration** appelle `POST /api/demo/jobs`. Le serveur
génère une vidéo synthétique verticale, puis lance un job `manual` sans audio et
sans fournisseur VLM.

Après le traitement :

- la section Évaluation distingue mesures calculées et annotations de référence ;
- **Charger les résultats de démonstration** importe des fixtures simulées ;
- ces fixtures sont refusées si le hash du run ne correspond pas à la démo ;
- aucun appel Gemini ou téléchargement de média n'est effectué.
