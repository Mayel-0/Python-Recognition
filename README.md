# Reconnaissance faciale et de plaques

Projet Python de reconnaissance faciale et de plaques sur image, webcam et flux RTSP, réalisé dans le cadre du cours de chef-d'œuvre.

> Ce projet a été commencé à mes débuts en programmation, pendant l'année 2023-2024. Sa première version était volontairement simple mais peu structurée: la majorité de la logique se trouvait dans `main.py`, avec une seconde version spécifique dans `rstp.py`. La structure actuelle est une réorganisation progressive de cette base, sans changer les fonctionnalités principales.

## Fonctionnalités

- chargement des visages connus depuis un dossier d'images;
- reconnaissance sur une image `.jpeg`;
- reconnaissance faciale en temps réel avec une webcam ou un flux RTSP;
- détection des plaques avec un modèle YOLO spécialisé;
- lecture des plaques avec EasyOCR et prétraitement OpenCV;
- comparaison des plaques avec une liste blanche;
- modes webcam et RTSP séparés pour les visages et les plaques;
- affichage du nom, du texte et de la couleur de reconnaissance;
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

Pour la reconnaissance des plaques:

- placez le modèle YOLO spécialisé dans `models/license_plate.pt`;
- ajoutez les plaques autorisées dans `known_plates/whitelist.txt`, une par ligne;
- les espaces, tirets et différences de casse sont normalisés lors de la comparaison.

Le fichier `models/license_plate.pt` est obligatoire pour les modes plaques. Le modèle générique `yolov8n.pt` n'est pas utilisé pour cette détection.

Ces dossiers contiennent des données locales et sont ignorés par Git. Aucune image personnelle n'est fournie dans ce dépôt.

## Lancer l'application

Depuis la racine du projet:

```bash
python main.py
```

Choisissez ensuite un mode:

1. `webcam-visages` ou `1` pour la reconnaissance faciale avec la caméra locale;
2. `webcam-plaques` ou `2` pour la reconnaissance de plaques avec la caméra locale;
3. `rtsp-visages` ou `3` pour la reconnaissance faciale sur le flux RTSP;
4. `rtsp-plaques` ou `4` pour la reconnaissance de plaques sur le flux RTSP;
5. `image` ou `5` pour analyser une image avec la reconnaissance faciale.

Les traitements visage et plaque sont séparés. En mode plaques, la base de visages n'est pas chargée. La touche `q` permet de quitter les fenêtres vidéo.

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
│   ├── recognizer.py     # Comparaison des encodages
│   ├── plate_detector.py # Détection YOLO et lecture EasyOCR des plaques
│   └── plate_checker.py  # Vérification avec la liste blanche
├── models/
│   ├── face.py           # Structures de données liées aux visages
│   ├── plate.py          # Structure PlateResult, immutable
│   └── license_plate.pt  # Modèle YOLO spécialisé
├── ui/
│   ├── cli.py            # Menu en ligne de commande
│   └── gui.py            # Affichage OpenCV et interactions vidéo
├── main.py               # Point d'entrée principal
├── rstp.py               # Point d'entrée RTSP threadé historique
├── requirements.txt      # Dépendances Python
└── README.md
```

Les réglages principaux se trouvent dans `config/settings.py`, notamment le seuil de reconnaissance (`tolerance`) et l'adresse du flux RTSP.

## Fonctionnement de la reconnaissance des plaques

Le modèle YOLO localise les plaques dans chaque image. Le crop est ensuite agrandi, converti en niveaux de gris, amélioré avec CLAHE et traité avec une variante binarisée OTSU. EasyOCR lit les deux variantes et conserve le résultat le plus fiable.

Une plaque reconnue dans `known_plates/whitelist.txt` est affichée en vert. Une plaque absente de la liste est affichée en rouge.

## Prochaines évolutions

Le projet pourra ensuite être relié à un serveur domotique afin d'automatiser l'ouverture d'un portail lorsqu'une personne ou un véhicule autorisé est reconnu. Cette intégration pourra notamment permettre de:

- transmettre le résultat de la reconnaissance au serveur domotique;
- vérifier les autorisations avant toute action;
- commander l'ouverture du portail;
- conserver un historique des accès.

Ces fonctionnalités domotiques ne sont pas encore implémentées dans la version actuelle.

## Limites et contexte

Ce projet est un projet d'apprentissage réalisé en 2023-2024. Il n'est pas conçu comme un système de sécurité prêt pour la production. La précision dépend de la qualité des images, de l'éclairage et du seuil choisi. Les données de visages restent locales et doivent être utilisées avec l'accord des personnes concernées.
