"""Repository paths used by scripts, the app, and tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "forecast_bundle.joblib"
