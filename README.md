# Reconnaissance faciale

Projet Python de reconnaissance faciale sur image, webcam et flux RTSP, réalisé dans le cadre du cours de chef-d'œuvre.

> Ce projet a été commencé à mes débuts en programmation, pendant l'année 2023-2024. Sa première version était volontairement simple mais peu structurée: la majorité de la logique se trouvait dans `main.py`, avec une seconde version spécifique dans `rstp.py`. La structure actuelle est une réorganisation progressive de cette base, sans changer les fonctionnalités principales.

## Fonctionnalités

- chargement des visages connus depuis un dossier d'images;
- reconnaissance sur une image `.jpeg`;
- reconnaissance en temps réel avec une webcam;
- reconnaissance sur un flux caméra RTSP;
- affichage du nom et du cadre autour des visages détectés;
- sauvegarde d'une image analysée dans `output/`.

## Installation

Le projet utilise Python 3, OpenCV et la bibliothèque `face-recognition`. Cette dernière dépend notamment de `dlib`, dont l'installation peut nécessiter CMake et les outils de compilation C/C++.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sur Windows, l'activation du virtualenv est différente:

```powershell
venv\Scripts\activate
```

## Préparer les données

Placez les images de référence dans `known_people_face/`. Le nom du fichier devient le nom reconnu. Par exemple, `alice.jpg` sera affichée comme `alice`.

Pour le mode image, placez une image `.jpeg` dans `image_test/`. Les images produites par ce mode sont enregistrées dans `output/`.

Ces dossiers contiennent des données locales et sont ignorés par Git. Aucune image personnelle n'est fournie dans ce dépôt.

## Lancer l'application

Depuis la racine du projet:

```bash
python main.py
```

Choisissez ensuite un mode:

1. `webcam` ou `1` pour la caméra locale;
2. `rtsp` ou `2` pour le flux RTSP configuré;
3. `image` ou `3` pour analyser une image.

La touche `q` permet de quitter les fenêtres vidéo.

La variante RTSP threadée historique reste disponible avec:

```bash
python rstp.py
```

## Structure du projet

```text
.
├── config/
│   └── settings.py       # URL RTSP, chemins, seuil et paramètres vidéo
├── core/
│   ├── camera.py         # Webcam et flux RTSP
│   ├── database.py       # Chargement et encodage des visages connus
│   ├── detector.py       # Détection et extraction des visages
│   └── recognizer.py     # Comparaison des encodages
├── models/
│   └── face.py           # Structures de données liées aux visages
├── ui/
│   ├── cli.py            # Menu en ligne de commande
│   └── gui.py            # Affichage OpenCV et interactions vidéo
├── main.py               # Point d'entrée principal
├── rstp.py               # Point d'entrée RTSP threadé historique
├── requirements.txt      # Dépendances Python
└── README.md
```

Les réglages principaux se trouvent dans `config/settings.py`, notamment le seuil de reconnaissance (`tolerance`) et l'adresse du flux RTSP.

## Prochaines évolutions

Une fonctionnalité de reconnaissance des plaques d'immatriculation est prévue prochainement. Elle aura pour objectif de compléter la reconnaissance faciale en identifiant les véhicules autorisés à partir de leur plaque.

Le projet pourra ensuite être relié à un serveur domotique afin d'automatiser l'ouverture d'un portail lorsqu'une personne ou un véhicule autorisé est reconnu. Cette intégration pourra notamment permettre de:

- transmettre le résultat de la reconnaissance au serveur domotique;
- vérifier les autorisations avant toute action;
- commander l'ouverture du portail;
- conserver un historique des accès.

Ces fonctionnalités ne sont pas encore implémentées dans la version actuelle.

## Limites et contexte

Ce projet est un projet d'apprentissage réalisé en 2023-2024. Il n'est pas conçu comme un système de sécurité prêt pour la production. La précision dépend de la qualité des images, de l'éclairage et du seuil choisi. Les données de visages restent locales et doivent être utilisées avec l'accord des personnes concernées.
