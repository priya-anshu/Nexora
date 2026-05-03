from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
UI_DIR = PROJECT_ROOT / "ui"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MONITORING_DIR.mkdir(parents=True, exist_ok=True)
