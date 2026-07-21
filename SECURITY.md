# Politique de sécurité

## Signaler une vulnérabilité

N'ouvrez pas d'issue publique pour une vulnérabilité exploitable ou une fuite de
données. Utilisez l'onglet **Security** du dépôt GitHub pour envoyer un signalement
privé, avec les étapes de reproduction et l'impact estimé.

La branche principale et la dernière version publiée reçoivent les correctifs de
sécurité.

## Données et fournisseurs cloud

Les modes `manual` et `disabled` n'appellent aucun fournisseur VLM. Le mode
`automatic` avec Gemini peut envoyer des frames, du texte, des transcriptions et
des extraits audio au fournisseur configuré. N'utilisez ce mode qu'après avoir
obtenu les autorisations nécessaires et examiné les conditions du fournisseur.

Les clés API doivent rester dans l'environnement ou dans un gestionnaire de
secrets. Elles ne doivent jamais être placées dans une configuration suivie par Git.

