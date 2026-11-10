"""Project identity and paths; importing settings has no side effects."""

from pathlib import Path

APP_NAME = "SilkSight"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_SEED_PATH = PROJECT_ROOT / "catalog/landmarks.seed.json"
CATALOG_PATH = PROJECT_ROOT / "catalog/landmarks.json"
UNKNOWN_SOURCES_PATH = PROJECT_ROOT / "catalog/unknown_sources.json"

# Images are local only (git-ignored); the manifest and manual exclusions are versioned.
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
EXCLUSIONS_PATH = DATA_DIR / "exclusions.csv"
CONTACT_SHEETS_DIR = PROJECT_ROOT / "outputs/contact_sheets"

HOTELIA_BASE_URL = "https://hotelia.uz"
# Hotelia matches destinations by exact English city name (``Samarqand`` returns nothing),
# so only cities from Hotelia-next ``libs/search/destinations.ts`` are allowed.
HOTELIA_DESTINATIONS = (
    "Tashkent",
    "Samarkand",
    "Bukhara",
    "Almaty",
    "Astana",
    "Shymkent",
    "Bishkek",
    "Osh",
    "Cholpon-Ata",
)

COUNTRY_CODES = ("UZ", "KZ", "KG", "TJ", "TM")
CATEGORIES = ("architecture", "monument", "nature")

# Model weights (Hugging Face cache), embeddings and trained models are local, git-ignored.
MODELS_DIR = PROJECT_ROOT / ".models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
MODEL_A_DIR = ARTIFACTS_DIR / "model_a"
