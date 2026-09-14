from config.settings import settings

def load_whitelist(path: str) -> set[str]:
    with open(path) as f:
        return {
          ligne.strip()
          for ligne in f.readlines()
          if ligne.strip() and not ligne.startswith("#")
        }

def is_plate_allowed(plate_text: str, whitelist: set[str]) -> bool:
    return plate_text in whitelist
