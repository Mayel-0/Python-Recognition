import os
from pathlib import Path

import face_recognition
from tqdm import tqdm


def get_images(path: Path) -> list[str]:
    if not path.exists():
        print(f"Dossier introuvable : {path}")
        raise SystemExit(1)

    images = [
        str(file)
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in (".jpeg", ".jpg", ".png")
    ]
    if not images:
        print(f"Aucune image trouvée dans {path}")
        raise SystemExit(1)
    return images


def encode_face(image_path: str) -> tuple | None:
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"Aucun visage trouvé dans : {image_path}")
            return None
        face_name = Path(image_path).stem
        return encodings[0], face_name
    except Exception as error:
        print(f"Erreur encodage {image_path} : {error}")
        return None


def load_known_faces(path: Path) -> tuple[list, list]:
    known_people = get_images(path)

    encodings = []
    names = []
    for image_path in tqdm(known_people, desc="Chargement des visages"):
        result = encode_face(image_path)
        if result:
            encodings.append(result[0])
            names.append(result[1])

    print(f"{len(encodings)} visage(s) chargé(s)")
    return encodings, names
