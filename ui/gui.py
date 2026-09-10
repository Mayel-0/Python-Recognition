import uuid
from pathlib import Path

import cv2
import face_recognition

from config.settings import settings
from core.camera import open_stream, RTSPStreamThread
from core.recognizer import recognize_frame, recognize_image


def draw_faces(frame, face_locations, face_names, text_color=(255, 0, 0)):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            text_color,
            1,
        )
    return frame


def run_stream(source, known_face_encodings, known_face_names):
    capture = open_stream(source)
    frame_count = 0
    face_locations = []
    face_names = []
    label = "RTSP Camera" if isinstance(source, str) else "Webcam"
    print(f"Flux ouvert : {label} — appuie sur 'q' pour quitter")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Frame perdue, reconnexion...")
                capture.release()
                capture = open_stream(source)
                continue

            if frame_count % settings.process_every_n_frames == 0:
                face_results = recognize_frame(
                    frame, known_face_encodings, known_face_names
                )
                face_locations = [result.location for result in face_results]
                face_names = [result.name for result in face_results]

            draw_faces(frame, face_locations, face_names)
            cv2.putText(
                frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2,
            )
            cv2.putText(
                frame, f"Visages : {len(face_locations)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
            cv2.imshow("Reconnaissance faciale", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_rtsp_thread(known_face_encodings, known_face_names):
    stream = RTSPStreamThread(settings.rtsp_url)
    if not stream.start():
        return

    frame_count = 0
    face_locations = []
    face_names = []
    print(f"Lancement du flux RTSP ({settings.rtsp_url}) — Appuie sur 'q' pour quitter")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None or frame.size == 0:
                continue

            if frame_count % settings.process_every_n_frames == 0:
                face_results = recognize_frame(
                    frame, known_face_encodings, known_face_names
                )
                face_locations = [result.location for result in face_results]
                face_names = [result.name for result in face_results]

            draw_faces(frame, face_locations, face_names, text_color=(255, 255, 255))
            cv2.putText(
                frame, "RTSP Camera", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2,
            )
            cv2.putText(
                frame, f"Visages : {len(face_locations)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
            cv2.imshow("Reconnaissance Faciale RTSP", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


def get_face_location(image_path: Path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    if not locations:
        print(f"Aucun visage trouvé dans : {image_path}")
        raise SystemExit(1)
    return locations[0]


def draw_rectangle(image_path: Path, output_filename: str, coordinates) -> None:
    image = cv2.imread(str(image_path))
    top, right, bottom, left = coordinates
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 10)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / output_filename
    cv2.imwrite(str(output_path), image)
    print(f"Image sauvegardée : {output_path}")


def run_image(known_face_encodings, known_face_names):
    name_file = input("Nom de l'image à analyser (sans extension) : ").strip()
    image_path = settings.image_test_dir / f"{name_file}.jpeg"
    if not image_path.exists():
        print(f"Image introuvable : {image_path}")
        raise SystemExit(1)

    names = recognize_image(image_path, known_face_encodings, known_face_names)
    if not names:
        print("Aucun visage détecté dans l'image.")
        raise SystemExit(1)

    print("Analyse en cours...")
    location = get_face_location(image_path)
    output_filename = f"{name_file}_result_{uuid.uuid4()}.jpeg"
    draw_rectangle(image_path, output_filename, location)
    matched = [name for name in names if name != "Inconnu"]
    if matched:
        print(f"Personne(s) reconnue(s) : {', '.join(matched)}")
    else:
        print("Aucune correspondance trouvée.")
