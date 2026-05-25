import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# =========================
# Database / Auth
# =========================
DATABASE_URL = f"sqlite:///{BASE_DIR / 'backend' / 'wheatiq.db'}"
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# =========================
# LLM / Chatbot
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# =========================
# Wheat detector paths
# Stage 0: wheat / not_wheat
# =========================
WHEAT_DETECTOR_MODEL_PATH = BASE_DIR / "models" / "wheat_detector" / "best_wheat_detector_model.keras"
WHEAT_DETECTOR_INDEX_TO_LABEL_PATH = BASE_DIR / "models" / "wheat_detector" / "wheat_detector_index_to_label.json"

# =========================
# Binary classifier paths
# Stage 1: healthy / diseased
# =========================
BINARY_MODEL_PATH = BASE_DIR / "models" / "binary" / "best_binary_model.keras"
BINARY_INDEX_TO_LABEL_PATH = BASE_DIR / "models" / "binary" / "binary_index_to_label.json"

# =========================
# Disease classifier paths
# Stage 2: disease classification
# =========================
KERAS_MODEL_PATH = BASE_DIR / "models" / "keras" / "best_keras_model.keras"
INDEX_TO_LABEL_PATH = BASE_DIR / "models" / "keras" / "index_to_label.json"

# =========================
# Knowledge base paths
# =========================
KB_DISEASES_PATH = BASE_DIR / "knowledge_base" / "diseases.json"
KB_TRANSLATIONS_PATH = BASE_DIR / "knowledge_base" / "translations.json"

# =========================
# Sample images
# =========================
SAMPLE_IMAGES_DIR = BASE_DIR / "data" / "sample_images"

# =========================
# Upload directory
# =========================
UPLOAD_DIR = BASE_DIR / "backend" / "app" / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)