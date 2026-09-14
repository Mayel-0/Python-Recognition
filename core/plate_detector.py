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
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        return gray

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        plates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = self.preprocess(crop)
                ocr_results = self.reader.readtext(crop)
                for (_, text, ocr_conf) in ocr_results:
                    plates.append(
                        PlateResult(
                            text=text.upper().strip(),
                            location=(x1, y1, x2, y2),
                            confidence=ocr_conf,
                        )
                    )
        return plates
