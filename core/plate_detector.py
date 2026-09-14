from ultralytics import YOLO
import easyocr
import cv2
from pathlib import Path

from models.plate import PlateResult

class PlateDetector:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "license_plate.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle spécialisé introuvable: {model_path}. "
                "Téléchargez un checkpoint YOLO de détection de plaques "
                "et placez-le sous models/license_plate.pt."
            )

        self.model = YOLO(str(model_path))
        self.reader = easyocr.Reader(["fr", "en"])  # chargé une fois

    def preprocess(self, crop):
        crop = cv2.copyMakeBorder(
            crop, 4, 4, 4, 4, cv2.BORDER_REPLICATE
        )
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    @staticmethod
    def clean_text(text):
        return "".join(character for character in text.upper() if character.isalnum())

    def read_plate(self, crop):
        gray = self.preprocess(crop)
        thresholded = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        variants = (gray, thresholded)
        candidates = []
        for image in variants:
            candidates.extend(
                self.reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                )
            )

        candidates = [
            (self.clean_text(text), confidence)
            for (_, text, confidence) in candidates
            if self.clean_text(text)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate[1])

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        plates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height, width = frame.shape[:2]
                padding_x = max(1, int((x2 - x1) * 0.05))
                padding_y = max(1, int((y2 - y1) * 0.15))
                x1_crop = max(0, x1 - padding_x)
                y1_crop = max(0, y1 - padding_y)
                x2_crop = min(width, x2 + padding_x)
                y2_crop = min(height, y2 + padding_y)
                crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                if crop.size == 0:
                    continue
                plate = self.read_plate(crop)
                if plate is None:
                    continue
                text, confidence = plate
                plates.append(
                    PlateResult(
                        text=text,
                        location=(x1, y1, x2, y2),
                        confidence=confidence,
                    )
                )
        return plates
