# Video Analysis Agent — Portfolio technique

## Le problème

L'analyse d'une vidéo courte mélange des traitements coûteux et difficiles à
auditer : inspection, segmentation, extraction d'images, audio, transcription,
analyse multimodale et synthèse. Un prototype qui enchaîne ces étapes sans
isolation peut mélanger les artefacts de deux vidéos, masquer des résultats
invalides ou déclencher un appel cloud involontaire.

Video Analysis Agent transforme ce flux en pipeline local-first, observable et
reproductible. Chaque run possède une identité de contenu et des artefacts
vérifiables. L'analyse VLM reste optionnelle.

## Architecture

```text
React / Vite / TypeScript
        │ REST + polling
        ▼
FastAPI ── file locale bornée
        │
        ▼
Pipeline Python
inspection → scènes → frames → mouvement → audio → fusion → pack VLM
        │
        ├─ disabled → local_complete
        ├─ manual   → awaiting_vlm → import → validation → rapport
        └─ automatic / Gemini, uniquement sur configuration explicite
```

Les artefacts sont isolés sous `outputs/runs/<run_id>/`. Le `run_id` dépend du
hash vidéo, de la configuration, des versions de schémas et de prompts.

## Défis techniques

### Fiabilité des reprises

Un simple fichier `.done` ne prouve pas qu'un artefact existe encore. Les
checkpoints enregistrent les empreintes d'entrées et de sorties. Une corruption
ou modification invalide l'étape concernée et ses dépendances.

### Validation VLM

Un rapport ne devient `complete` que si chaque segment attendu possède un JSON
strictement valide, avec le bon identifiant et les bons timestamps. Les
résultats supplémentaires, dupliqués, illisibles ou incomplets sont signalés.

### Exécution locale asynchrone

FastAPI utilise une file en mémoire et un `ThreadPoolExecutor` borné. La
progression reflète uniquement des étapes réellement terminées. Le polling
React est séquentiel, annulable et nettoyé lors du démontage.

### Démonstration sans données personnelles

Une vidéo verticale de 18 secondes est générée par Pillow et FFmpeg avec trois
scènes synthétiques. Aucun média externe n'est téléchargé. Les fixtures VLM sont
étiquetées « simulées » et liées au hash exact de cette vidéo au chargement.

## Décisions

- `manual` par défaut : aucun SDK ou appel Gemini nécessaire ;
- fournisseurs derrière une abstraction explicite ;
- JSON atomiques pour les manifests, checkpoints et rapports ;
- FastAPI réutilise le pipeline existant au lieu de le réimplémenter ;
- dataset et évaluateur séparés de la production ;
- métriques lexicales présentées comme approximatives ;
- React sans bibliothèque UI lourde.

## Résultats mesurés

Dataset synthétique v1, Windows 11, Python 3.13.13, FFmpeg 8.1.1, 8 CPU
logiques, audio désactivé, VLM `disabled` :

| Mesure | Résultat |
|---|---:|
| Durée vidéo | 18,0 s |
| Scènes attendues / détectées | 3 / 3 |
| F1 des limites à ±500 ms | 1,000 |
| Couverture temporelle | 100 % |
| Frames extraites | 24 |
| Trous / chevauchements | 0 / 0 |
| Médiane pipeline, 2 runs | 19,204 s |
| Écart-type population | 0,054 s |

Ces chiffres décrivent cette machine et cette configuration. Ils ne constituent
pas un classement général du pipeline.

## Limites

- la file de jobs ne survit pas au redémarrage ;
- l'annulation forcée d'un FFmpeg ou Whisper déjà lancé n'est pas disponible ;
- FFmpeg reste une dépendance système ;
- Gemini est le seul fournisseur automatique implémenté ;
- l'évaluation d'objets/actions est lexicale, pas sémantique ;
- `SequenceMatcher` ne remplace ni un juge humain ni une étude scientifique ;
- deux machines ne sont pas comparées comme si leurs temps étaient équivalents.

## Pistes futures

- persistance locale optionnelle des jobs ;
- interruption coopérative des sous-processus ;
- corpus synthétique plus varié et annotations humaines indépendantes ;
- métriques sémantiques validées avec protocole humain ;
- fournisseur Claude derrière l'interface existante ;
- packaging optionnel du frontend compilé.

## Proposition de publication LinkedIn

> J'ai transformé un prototype d'analyse vidéo en application local-first
> reproductible : runs isolés par hash, validation VLM stricte, API FastAPI,
> jobs asynchrones et interface React. Pour rendre la démo partageable sans
> données personnelles, le projet génère sa propre vidéo CC0 et mesure
> segmentation, couverture, temps et artefacts. Sur le scénario synthétique :
> 3/3 scènes, 100 % de couverture et 19,2 s de médiane locale — avec les limites
> méthodologiques publiées à côté des chiffres.

## Éléments CV

- Conception d'un pipeline Python local-first d'analyse vidéo avec FFmpeg,
  PySceneDetect, Pillow et faster-whisper optionnel.
- Mise en place d'identités SHA-256, checkpoints vérifiables et écritures
  atomiques pour empêcher le mélange d'artefacts.
- Développement d'une API FastAPI asynchrone et d'une interface React/TypeScript
  responsive avec suivi de progression.
- Création d'un framework d'évaluation reproductible, rapports JSON/HTML et
  benchmark documentant environnement, dispersion et limites.
- Sécurisation des uploads, chemins d'artefacts, erreurs API et intégrations VLM
  optionnelles.
