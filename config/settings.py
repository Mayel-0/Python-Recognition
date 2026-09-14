from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    rtsp_url: str = "rtsp://192.168.1.154:554/main_ch"
    known_faces_dir: Path = Path("known_people_face")
    image_test_dir: Path = Path("image_test")
    output_dir: Path = Path("output")
    tolerance: float = 0.55
    frame_scale: float = 0.25
    process_every_n_frames: int = 1
    known_plates_dir: Path = Path("known_plates")


settings = Settings()
