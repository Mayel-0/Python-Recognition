import numpy as np
import face_recognition

from config.settings import settings
from core.detector import detect_faces
from models.face import FaceResult


def recognize_frame(frame, known_face_encodings, known_face_names):
    locations, encodings = detect_faces(frame)
    names = []
    distances = []

    for encoding in encodings:
        name = "Inconnu"
        distance = None
        if known_face_encodings:
            face_distances = face_recognition.face_distance(
                known_face_encodings, encoding
            )
            best_match_index = int(np.argmin(face_distances))
            distance = float(face_distances[best_match_index])
            if distance < settings.tolerance:
                name = known_face_names[best_match_index]
        names.append(name)
        distances.append(distance)

    scale = 1 / settings.frame_scale
    scaled_locations = [
        (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))
        for top, right, bottom, left in locations
    ]
    return [
        FaceResult(name=name, location=location, confidence=distance)
        for location, name, distance in zip(scaled_locations, names, distances)
    ]


def recognize_image(image_path, known_face_encodings, known_face_names):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return []

    results = []
    for encoding in encodings:
        name = "Inconnu"
        if known_face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, encoding, tolerance=settings.tolerance
            )
            matched = [
                known_face_names[index]
                for index, match in enumerate(matches)
                if match
            ]
            if matched:
                name = matched[0]
        results.append(name)
    return results
