import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "keras" / "best_keras_model.keras"
INDEX_TO_LABEL_PATH = PROJECT_ROOT / "models" / "keras" / "index_to_label.json"

# Change this path to any image you want to test
IMAGE_PATH = PROJECT_ROOT / "data" / "sample_images" / "rust.png"

# IMPORTANT:
# This must match the image size used during training
IMAGE_SIZE = (300, 300)


# =========================
# Basic checks
# =========================
print("Checking files...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not INDEX_TO_LABEL_PATH.exists():
    raise FileNotFoundError(f"Label mapping file not found: {INDEX_TO_LABEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")


# =========================
# Load model and labels
# =========================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading label mapping...")
with open(INDEX_TO_LABEL_PATH, "r", encoding="utf-8") as f:
    index_to_label = {int(k): v for k, v in json.load(f).items()}

print("\nLoaded classes:")
for idx, label in index_to_label.items():
    print(f"{idx}: {label}")


# =========================
# Load and preprocess image
# =========================
print(f"\nLoading image from: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize(IMAGE_SIZE)

image_array = np.array(image).astype("float32")
image_array = np.expand_dims(image_array, axis=0)


# =========================
# Predict
# =========================
print("Running prediction...")
probs = model.predict(image_array, verbose=0)[0]

top_indices = np.argsort(probs)[::-1][:3]
predicted_index = int(top_indices[0])
predicted_label = index_to_label[predicted_index]
predicted_confidence = float(probs[predicted_index])

print("\n===== Prediction Result =====")
print(f"Predicted disease: {predicted_label}")
print(f"Confidence: {predicted_confidence * 100:.2f}%")

print("\nTop 3 predictions:")
for idx in top_indices:
    idx = int(idx)
    print(f"- {index_to_label[idx]}: {probs[idx] * 100:.2f}%")