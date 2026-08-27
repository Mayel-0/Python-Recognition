import os
import concurrent.futures
import face_recognition
import cv2
import numpy as np
from tqdm import tqdm
import uuid

RTSP_URL = "rtsp://192.168.1.154:554/main_ch"
KNOWN_FACES_DIR = "known_people_face"
IMAGE_TEST_DIR = "image_test"
OUTPUT_DIR = "output"
TOLERANCE = 0.55
PROCESS_EVERY_N_FRAMES = 0.50


def get_images(path: str) -> list[str]:
    if not os.path.exists(path):
        print(f"Dossier introuvable : {path}")
        exit(1)
    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".jpeg", ".jpg", ".png")):
                images.append(os.path.join(root, file))
    if not images:
        print(f"Aucune image trouvée dans {path}")
        exit(1)
    return images


def encode_face(image_path: str) -> tuple | None:
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"Aucun visage trouvé dans : {image_path}")
            return None
        face_name = os.path.splitext(os.path.basename(image_path))[0]
        return (encodings[0], face_name)
    except Exception as e:
        print(f"Erreur encodage {image_path} : {e}")
        return None


def face_encoding() -> tuple[list, list]:
    known_people = get_images(KNOWN_FACES_DIR)
    known_face_encodings = []
    known_face_names = []

    core = max(1, os.cpu_count() // 2)

    with concurrent.futures.ProcessPoolExecutor(max_workers=core) as executor:
        results = list(tqdm(
            executor.map(encode_face, known_people),
            total=len(known_people),
            desc="Chargement des visages"
        ))

    for result in results:
        if result:
            known_face_encodings.append(result[0])
            known_face_names.append(result[1])

    print(f"{len(known_face_encodings)} visage(s) chargé(s)")
    return known_face_encodings, known_face_names


def open_stream(source) -> cv2.VideoCapture:
    """Ouvre un flux vidéo (webcam ou RTSP) avec TCP et options optimisées."""
    if isinstance(source, str):
        # Force TCP + buffer réduit pour moins de latence
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|max_delay;500000"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la source : {source}")
        exit(1)
    return cap


def process_frame(frame, known_face_encodings, known_face_names):
    """Détecte et identifie les visages dans une frame."""
    # Réduit encore plus la frame pour accélérer la détection (0.25 au lieu de 0.5)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        name = "Inconnu"
        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < TOLERANCE:
                name = known_face_names[best_match_index]
        face_names.append(name)

    # Remet les coordonnées à l'échelle originale (x4 car on a réduit à 0.25)
    face_locations = [(top * 4, right * 4, bottom * 4, left * 4)
                      for (top, right, bottom, left) in face_locations]

    return face_locations, face_names


def draw_faces(frame, face_locations, face_names):
    """Dessine les rectangles et noms sur la frame."""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    return frame


def stream_recognition(source, known_face_encodings, known_face_names):
    """Reconnaissance en temps réel sur webcam ou flux RTSP."""
    cap = open_stream(source)
    frame_count = 0
    face_locations = []
    face_names = []

    label = "RTSP Camera" if isinstance(source, str) else "Webcam"
    print(f"Flux ouvert : {label} — appuie sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame perdue, reconnexion...")
            cap.release()
            cap = open_stream(source)
            continue

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            face_locations, face_names = process_frame(
                frame, known_face_encodings, known_face_names
            )

        frame = draw_faces(frame, face_locations, face_names)

        cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Visages : {len(face_locations)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Reconnaissance faciale", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_face_location(image_path: str) -> tuple:
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        print(f"Aucun visage trouvé dans : {image_path}")
        exit(1)
    return face_locations[0]


def draw_rectangle(image_path: str, output_filename: str, coordinates: tuple):
    image = cv2.imread(image_path)
    top, right, bottom, left = coordinates
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 10)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, image)
    print(f"Image sauvegardée : {output_path}")


def start_image(known_face_encodings, known_face_names):
    name_file = input("Nom de l'image à analyser (sans extension) : ").strip()
    image_path = os.path.join(IMAGE_TEST_DIR, f"{name_file}.jpeg")

    if not os.path.exists(image_path):
        print(f"Image introuvable : {image_path}")
        exit(1)

    unknown_image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(unknown_image)

    if not encodings:
        print("Aucun visage détecté dans l'image.")
        exit(1)

    unknown_face_encoding = encodings[0]
    results = face_recognition.compare_faces(
        known_face_encodings, unknown_face_encoding, tolerance=TOLERANCE
    )

    print("Analyse en cours...")

    location = get_face_location(image_path)
    output_filename = f"{name_file}_result_{uuid.uuid4()}.jpeg"
    draw_rectangle(image_path, output_filename, location)

    matched = [known_face_names[i] for i, match in enumerate(results) if match]
    if matched:
        print(f"Personne(s) reconnue(s) : {', '.join(matched)}")
    else:
        print("Aucune correspondance trouvée.")


def main():
    known_face_encodings, known_face_names = face_encoding()

    print("\nModes disponibles :")
    print("  1. webcam   — flux webcam locale")
    print("  2. rtsp     — flux caméra PTZ (192.168.1.154)")
    print("  3. image    — analyse d'une image")

    state = input("\nChoix : ").strip().lower()

    if state in ("webcam", "1"):
        stream_recognition(0, known_face_encodings, known_face_names)
    elif state in ("rtsp", "2"):
        stream_recognition(RTSP_URL, known_face_encodings, known_face_names)
    elif state in ("image", "3"):
        start_image(known_face_encodings, known_face_names)
    else:
        print("Choix invalide.")
        exit(1)


if __name__ == "__main__":
    main()
