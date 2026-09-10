from dataclasses import dataclass


FaceLocation = tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceResult:
    name: str
    location: FaceLocation
    confidence: float | None = None
