from config.settings import settings
from core.database import load_known_faces
from ui.gui import run_image, run_rtsp_thread, run_stream


def main() -> None:
    known_face_encodings, known_face_names = load_known_faces(
        settings.known_faces_dir
    )

    print("\nModes disponibles :")
    print("  1. webcam   — flux webcam locale")
    print("  2. rtsp     — flux caméra PTZ (192.168.1.154)")
    print("  3. image    — analyse d'une image")
    state = input("\nChoix : ").strip().lower()

    if state in ("webcam", "1"):
        run_stream(0, known_face_encodings, known_face_names)
    elif state in ("rtsp", "2"):
        run_stream(settings.rtsp_url, known_face_encodings, known_face_names)
    elif state in ("image", "3"):
        run_image(known_face_encodings, known_face_names)
    else:
        print("Choix invalide.")
        raise SystemExit(1)


def run_threaded_rtsp() -> None:
    known_face_encodings, known_face_names = load_known_faces(
        settings.known_faces_dir
    )
    run_rtsp_thread(known_face_encodings, known_face_names)
