
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "wheat_detector" / "final_wheat_detector_model.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "wheat_detector" / "wheat_detector_index_to_label.json"

IMAGE_SIZE = (224, 224)


MODEL = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    INDEX_TO_LABEL = {int(k): v for k, v in json.load(f).items()}


def preprocess_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(image_path: Path):
    arr = preprocess_image(image_path)
    raw_prob = float(MODEL.predict(arr, verbose=0)[0][0])

    label_0 = INDEX_TO_LABEL[0]
    label_1 = INDEX_TO_LABEL[1]

    probs = {
        label_0: 1.0 - raw_prob,
        label_1: raw_prob,
    }

    predicted_label = max(probs, key=probs.get)
    confidence = probs[predicted_label]

    print(f"\nImage: {image_path.name}")
    print("Raw sigmoid output:", raw_prob)
    print("Probabilities:", probs)
    print("Predicted:", predicted_label)
    print("Confidence:", f"{confidence:.4f}")


if __name__ == "__main__":
    test_images = [
        PROJECT_ROOT / "data" / "sample_images" / "healthy.png",
        PROJECT_ROOT / "data" / "sample_images" / "fusarium_head_blight.png",
        
         PROJECT_ROOT / "data" / "sample_images" / "monkey.png",
         PROJECT_ROOT / "data" / "sample_images" / "flower1.png",
    ]

    for image_path in test_images:
        if image_path.exists():
            predict_image(image_path)
        else:
            print(f"File not found: {image_path}")