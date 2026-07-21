# Enregistrer la démonstration

Le GIF final est prévu à cet emplacement :

```text
docs/demo/video-agent-demo.gif
```

Il doit être enregistré depuis l’application réelle. Aucun écran ou résultat ne
doit être fabriqué lorsque le navigateur visuel n’est pas disponible.

## Préparation

```console
python -m pip install ".[web]"
video-agent generate-demo
cd frontend
npm ci
npm run build
cd ..
video-agent web --ui --host 127.0.0.1 --port 8000
```

Ouvrir `http://127.0.0.1:8000`, placer le navigateur à **1440 × 900 px** et
utiliser un zoom de 100 %. Pour LinkedIn, exporter idéalement en MP4 1080p puis
créer un GIF de prévisualisation de 1280 px de large, 12 à 15 images/s, inférieur
à 15 Mo si la plateforme de destination l’exige.

## Scénario de 45 à 60 secondes

| Temps | Écran et action | Message à faire passer |
|---:|---|---|
| 0–5 s | Terminal : lancement local, puis ouverture de l’accueil | Application locale, aucun compte requis |
| 5–11 s | Clic sur **Lancer la démonstration** | Vidéo synthétique, aucune donnée personnelle |
| 11–22 s | Carte de progression | Étapes réelles : inspection, scènes, frames, fusion et pack VLM |
| 22–30 s | Timeline des trois scènes et artefacts | Sorties locales structurées et consultables |
| 30–37 s | Clic sur **Charger les résultats de démonstration** | Fixtures simulées, jamais présentées comme un modèle réel |
| 37–43 s | Validation : segments valides/manquants/invalides | Validation stricte avant tout statut `complete` |
| 43–50 s | Construction et ouverture du rapport complet | Rapport produit uniquement depuis les résultats validés |
| 50–58 s | Section Évaluation et liens JSON/HTML | Métriques locales, benchmark et limites documentées |

## Script voix ou sous-titres LinkedIn

> Je lance Video Analysis Agent entièrement en local. La démonstration génère sa
> propre vidéo verticale : aucune donnée personnelle et aucune clé API. Le job
> expose la progression réelle du pipeline, puis les scènes, frames et packs VLM.
> Pour montrer le parcours complet sans appeler un modèle, je charge des fixtures
> clairement marquées comme simulées et liées au hash exact de la vidéo. La
> validation contrôle chaque segment avant de construire le rapport. Enfin, le
> framework d’évaluation compare le run aux annotations synthétiques et publie
> couverture, limites de scènes, temps et erreurs avec leurs limites
> méthodologiques.

## Ordre des écrans

1. en-tête et zone de dépôt ;
2. bandeau « Mode démonstration local » ;
3. progression `queued` puis `running` ;
4. résumé du run et timeline ;
5. panneau VLM avec badge « simulé » ;
6. résumé de validation ;
7. rapport `complete` ;
8. panneau d’évaluation et rapport HTML.

## Informations à masquer

Avant l’enregistrement, vérifier l’absence de :

- nom de session, chemin de profil Windows ou nom de machine ;
- autres onglets, favoris, notifications ou avatar de navigateur ;
- adresse IP autre que `127.0.0.1` ;
- variables d’environnement, clés API ou contenu de `.env` ;
- noms de vidéos personnelles ou anciens `run_id` sensibles ;
- logs contenant des chemins locaux ;
- adresse e-mail, dépôt privé ou terminal affichant l’historique.

Recadrer la capture sur le navigateur. Les fixtures doivent rester visiblement
étiquetées « simulées ».
