import cv2
import face_recognition

from config.settings import settings


def detect_faces(frame):
    small_frame = cv2.resize(
        frame, (0, 0), fx=settings.frame_scale, fy=settings.frame_scale
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    encodings = face_recognition.face_encodings(rgb_small_frame, locations)
    return locations, encodings
