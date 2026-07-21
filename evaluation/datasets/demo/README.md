# Dataset synthétique de démonstration

Ce dataset ne contient aucune vidéo personnelle. `videos/video_agent_demo.mp4`
est généré à la demande par `video-agent generate-demo` ou automatiquement par
`video-agent evaluate --dataset evaluation/datasets/demo`.

- provenance : génération locale déterministe ;
- licence : CC0-1.0 ;
- durée : 18 secondes ;
- format : vertical 360×640, 6 fps ;
- audio : absent ;
- scènes de référence : 3, avec coupures à 6 s et 12 s ;
- réseau : aucun téléchargement.

`annotations/demo_annotations.json` contient la vérité terrain. Son hash MP4
correspond à la génération de référence. Une autre version de FFmpeg peut
éventuellement produire un conteneur binaire différent malgré des frames
identiques ; l'évaluateur le signale explicitement.

`vlm_fixtures/demo_vlm_results.json` est une fixture simulée. Elle n'est jamais
présentée comme une sortie de modèle et ne peut être liée qu'à un run dont le
hash correspond à la vidéo de démonstration générée localement.
