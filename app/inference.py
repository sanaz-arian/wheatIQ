import json
import numpy as np
from PIL import Image
import tensorflow as tf

from app.config import (
    KERAS_MODEL_PATH,
    INDEX_TO_LABEL_PATH,
    BINARY_MODEL_PATH,
    BINARY_INDEX_TO_LABEL_PATH,
    WHEAT_DETECTOR_MODEL_PATH,
    WHEAT_DETECTOR_INDEX_TO_LABEL_PATH,
)

# =========================
# Thresholds
# =========================
WHEAT_THRESHOLD = 0.80
DISEASE_THRESHOLD = 0.50


# =========================
# Load disease classifier
# =========================
DISEASE_MODEL = tf.keras.models.load_model(KERAS_MODEL_PATH)

with open(INDEX_TO_LABEL_PATH, "r", encoding="utf-8") as f:
    DISEASE_INDEX_TO_LABEL = {int(k): v for k, v in json.load(f).items()}


# =========================
# Load binary classifier
# =========================
BINARY_MODEL = tf.keras.models.load_model(BINARY_MODEL_PATH)

with open(BINARY_INDEX_TO_LABEL_PATH, "r", encoding="utf-8") as f:
    BINARY_INDEX_TO_LABEL = {int(k): v for k, v in json.load(f).items()}


# =========================
# Load wheat detector
# =========================
WHEAT_DETECTOR_MODEL = tf.keras.models.load_model(WHEAT_DETECTOR_MODEL_PATH)

with open(WHEAT_DETECTOR_INDEX_TO_LABEL_PATH, "r", encoding="utf-8") as f:
    WHEAT_DETECTOR_INDEX_TO_LABEL = {int(k): v for k, v in json.load(f).items()}


# =========================
# Image preprocessing
# =========================
def preprocess_image(image_path: str, image_size=(224, 224)):
    """
    Load an image from disk, convert it to RGB,
    resize it to the required size, and return
    a batch tensor with shape (1, H, W, 3).
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    arr = np.array(image).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr


# =========================
# Helpers
# =========================
def get_confidence_level(confidence: float) -> str:
    if confidence < 0.50:
        return "low"
    elif confidence < 0.70:
        return "moderate"
    return "high"


def sigmoid_probs_from_mapping(raw_prob: float, index_to_label: dict[int, str]) -> dict[str, float]:
    """
    Convert sigmoid output into a label->probability dictionary
    using the saved class mapping.

    IMPORTANT:
    For a sigmoid model with one output neuron:
    - raw_prob is the probability of class index 1
    - (1 - raw_prob) is the probability of class index 0

    This function safely maps those probabilities using the JSON mapping.
    """
    if 0 not in index_to_label or 1 not in index_to_label:
        raise ValueError(
            f"Sigmoid model mapping must contain indices 0 and 1, got: {index_to_label}"
        )

    label_0 = index_to_label[0]
    label_1 = index_to_label[1]

    return {
        label_0: 1.0 - raw_prob,
        label_1: raw_prob,
    }


def sorted_predictions(probs_dict: dict[str, float]) -> list[dict]:
    """
    Convert a label->probability dictionary to a sorted list of dicts.
    """
    return [
        {"label": label, "confidence": float(conf)}
        for label, conf in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    ]


# =========================
# Wheat / Not Wheat classifier
# =========================
def predict_wheat_detector(image_path: str):
    """
    Predict whether the uploaded image is wheat or not_wheat.

    This function DOES NOT assume:
    - class 0 = not_wheat
    - class 1 = wheat

    Instead, it reads the mapping from:
    WHEAT_DETECTOR_INDEX_TO_LABEL
    """
    arr = preprocess_image(image_path, image_size=(224, 224))

    raw_prob = float(WHEAT_DETECTOR_MODEL.predict(arr, verbose=0)[0][0])
    probs_dict = sigmoid_probs_from_mapping(raw_prob, WHEAT_DETECTOR_INDEX_TO_LABEL)

    predicted_label = max(probs_dict, key=probs_dict.get)
    confidence = float(probs_dict[predicted_label])

    print("=== Wheat Detector Debug ===")
    print("Raw sigmoid output (probability of class index 1):", raw_prob)
    print("Wheat detector mapping:", WHEAT_DETECTOR_INDEX_TO_LABEL)
    print("Wheat detector probabilities:", probs_dict)
    print("Predicted wheat label:", predicted_label)
    print("============================")

    return predicted_label, confidence, probs_dict


# =========================
# Binary classifier
# =========================
def predict_binary(image_path: str):
    """
    Predict whether the image is healthy or diseased.

    This function DOES NOT assume a fixed class order.
    It reads the mapping from:
    BINARY_INDEX_TO_LABEL

    Example:
    - if index 0 = diseased and index 1 = healthy
      then raw sigmoid output = probability of healthy
    - if the mapping is reversed, the function still works
    """
    arr = preprocess_image(image_path, image_size=(224, 224))

    raw_prob = float(BINARY_MODEL.predict(arr, verbose=0)[0][0])
    probs_dict = sigmoid_probs_from_mapping(raw_prob, BINARY_INDEX_TO_LABEL)

    predicted_label = max(probs_dict, key=probs_dict.get)
    confidence = float(probs_dict[predicted_label])

    print("=== Binary Debug ===")
    print("Raw sigmoid output (probability of class index 1):", raw_prob)
    print("Binary mapping:", BINARY_INDEX_TO_LABEL)
    print("Binary probabilities:", probs_dict)
    print("Predicted binary label:", predicted_label)
    print("====================")

    return predicted_label, confidence, probs_dict


# =========================
# Disease classifier
# =========================
def predict_disease(image_path: str):
    """
    Predict the disease class for a diseased wheat image.

    Returns:
        predicted_label: str
        confidence: float
        top_predictions: list[dict]
    """
    arr = preprocess_image(image_path, image_size=(300, 300))
    probs = DISEASE_MODEL.predict(arr, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:3]
    predicted_idx = int(top_indices[0])

    predicted_label = DISEASE_INDEX_TO_LABEL[predicted_idx]
    confidence = float(probs[predicted_idx])

    top_predictions = [
        {
            "label": DISEASE_INDEX_TO_LABEL[int(idx)],
            "confidence": float(probs[int(idx)])
        }
        for idx in top_indices
    ]

    return predicted_label, confidence, top_predictions


# =========================
# Full pipeline
# =========================
def predict_image_pipeline(image_path: str):
    """
    Full three-stage pipeline:
    0. Wheat / not_wheat
    1. Healthy / diseased
    2. If diseased -> disease classifier

    Returns a dictionary with the final result.
    """

    # ---------------------------------
    # Stage 0: Wheat detector
    # ---------------------------------
    wheat_label, wheat_confidence, wheat_probs = predict_wheat_detector(image_path)
    prob_wheat = float(wheat_probs.get("wheat", 0.0))
    wheat_confidence_level = get_confidence_level(prob_wheat)

    if prob_wheat < WHEAT_THRESHOLD:
        return {
            "status": "rejected",
            "reason": "not_wheat",
            "wheat_label": wheat_label,
            "wheat_confidence": prob_wheat,
            "wheat_confidence_level": wheat_confidence_level,
            "binary_label": None,
            "binary_confidence": None,
            "binary_confidence_level": None,
            "predicted_label": "not_wheat",
            "confidence": prob_wheat,
            "confidence_level": wheat_confidence_level,
            "top_predictions": sorted_predictions(wheat_probs),
            "is_healthy": False,
            "is_wheat": False,
        }

    # ---------------------------------
    # Stage 1: Healthy / Diseased
    # ---------------------------------
    binary_label, binary_confidence, binary_probs = predict_binary(image_path)
    binary_confidence_level = get_confidence_level(binary_confidence)

    if binary_label == "healthy":
        return {
            "status": "ok",
            "reason": None,
            "wheat_label": "wheat",
            "wheat_confidence": prob_wheat,
            "wheat_confidence_level": wheat_confidence_level,
            "binary_label": "healthy",
            "binary_confidence": binary_confidence,
            "binary_confidence_level": binary_confidence_level,
            "predicted_label": "healthy",
            "confidence": binary_confidence,
            "confidence_level": binary_confidence_level,
            "top_predictions": [
                {
                    "label": "healthy",
                    "confidence": binary_confidence
                }
            ],
            "is_healthy": True,
            "is_wheat": True,
        }

    # ---------------------------------
    # Stage 2: Disease classifier
    # ---------------------------------
    predicted_label, confidence, top_predictions = predict_disease(image_path)
    disease_confidence_level = get_confidence_level(confidence)

    if confidence < DISEASE_THRESHOLD:
        return {
            "status": "rejected",
            "reason": "uncertain_disease",
            "wheat_label": "wheat",
            "wheat_confidence": prob_wheat,
            "wheat_confidence_level": wheat_confidence_level,
            "binary_label": binary_label,
            "binary_confidence": binary_confidence,
            "binary_confidence_level": binary_confidence_level,
            "predicted_label": "uncertain",
            "confidence": confidence,
            "confidence_level": disease_confidence_level,
            "top_predictions": top_predictions,
            "is_healthy": False,
            "is_wheat": True,
        }

    return {
        "status": "ok",
        "reason": None,
        "wheat_label": "wheat",
        "wheat_confidence": prob_wheat,
        "wheat_confidence_level": wheat_confidence_level,
        "binary_label": binary_label,
        "binary_confidence": binary_confidence,
        "binary_confidence_level": binary_confidence_level,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "confidence_level": disease_confidence_level,
        "top_predictions": top_predictions,
        "is_healthy": False,
        "is_wheat": True,
    }