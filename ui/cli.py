from config.settings import settings
from core.database import load_known_faces
from ui.gui import (
    run_face_stream,
    run_image,
    run_plate_stream,
    run_rtsp_thread,
)


def main() -> None:
    print("\nModes disponibles :")
    print("  1. webcam-visages   — reconnaissance faciale")
    print("  2. webcam-plaques   — reconnaissance de plaques")
    print("  3. rtsp-visages     — reconnaissance faciale")
    print("  4. rtsp-plaques     — reconnaissance de plaques")
    print("  5. image             — analyse faciale d'une image")
    state = input("\nChoix : ").strip().lower()

    if state in ("webcam-visages", "webcam-visage", "1"):
        known_face_encodings, known_face_names = load_known_faces(
            settings.known_faces_dir
        )
        run_face_stream(0, known_face_encodings, known_face_names)
    elif state in ("webcam-plaques", "webcam-plaque", "2"):
        run_plate_stream(0)
    elif state in ("rtsp-visages", "rtsp-visage", "3"):
        known_face_encodings, known_face_names = load_known_faces(
            settings.known_faces_dir
        )
        run_face_stream(settings.rtsp_url, known_face_encodings, known_face_names)
    elif state in ("rtsp-plaques", "rtsp-plaque", "4"):
        run_plate_stream(settings.rtsp_url)
    elif state in ("image", "5"):
        known_face_encodings, known_face_names = load_known_faces(
            settings.known_faces_dir
        )
        run_image(known_face_encodings, known_face_names)
    else:
        print("Choix invalide.")
        raise SystemExit(1)


def run_threaded_rtsp() -> None:
    known_face_encodings, known_face_names = load_known_faces(
        settings.known_faces_dir
    )
    run_rtsp_thread(known_face_encodings, known_face_names)
