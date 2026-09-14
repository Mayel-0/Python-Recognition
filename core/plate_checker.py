from config.settings import settings

def normalize_plate(plate_text: str) -> str:
  return "".join(character for character in plate_text.upper() if character.isalnum())

def load_whitelist(path: str) -> set[str]:
    with open(path) as f:
        return {
      normalize_plate(ligne)
          for ligne in f.readlines()
          if ligne.strip() and not ligne.startswith("#")
        }

def is_plate_allowed(plate_text: str, whitelist: set[str]) -> bool:
  return normalize_plate(plate_text) in whitelist
