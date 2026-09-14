from dataclasses import dataclass


PlateLocation = tuple[int, int, int, int]


@dataclass(frozen=True)
class PlateResult:
    text: str
    location: PlateLocation
    confidence: float | None = None
    reconnue: bool = False
