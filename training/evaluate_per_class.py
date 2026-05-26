
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "keras" / "best_keras_model.keras"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "keras_ready" / "test"
INDEX_TO_LABEL_PATH = PROJECT_ROOT / "models" / "keras" / "index_to_label.json"

# IMPORTANT:
# This must match the image size used during training
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16


# =========================
# Load model and labels
# =========================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading label mapping...")
with open(INDEX_TO_LABEL_PATH, "r", encoding="utf-8") as f:
    index_to_label = {int(k): v for k, v in json.load(f).items()}

class_names = [index_to_label[i] for i in sorted(index_to_label.keys())]

print("\nModel classes:")
for idx, label in index_to_label.items():
    print(f"{idx}: {label}")


# =========================
# Load test dataset
# =========================
print(f"\nLoading test dataset from: {TEST_DIR}")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTensorFlow detected classes:")
print(test_ds.class_names)


# =========================
# Collect true labels
# =========================
y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)

# =========================
# Predict all test images
# =========================
print("\nRunning predictions on test folder...")
y_prob = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# =========================
# Overall accuracy
# =========================
overall_acc = np.mean(y_true == y_pred)
print("\n===== Overall Accuracy =====")
print(f"Overall test accuracy: {overall_acc:.4f}")
print(f"Overall test accuracy (%): {overall_acc * 100:.2f}%")

# =========================
# Classification report
# =========================
print("\n===== Classification Report =====")
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)
print(report)

# Save report to file
report_path = PROJECT_ROOT / "training" / "classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nSaved classification report to: {report_path}")

# =========================
# Confusion matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

import matplotlib.pyplot as plt  
import seaborn as sns  

plt.figure(figsize=(10,8))  
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  
plt.xlabel("Predicted")  
plt.ylabel("True")  
plt.title("Confusion Matrix")  
plt.savefig("confusion_matrix.png")   # مهم!
plt.show()

print("\n===== Confusion Matrix =====")
print(cm)

cm_path = PROJECT_ROOT / "training" / "confusion_matrix.csv"
np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

print(f"Saved confusion matrix to: {cm_path}")

# =========================
# Per-class accuracy
# =========================
print("\n===== Per-Class Accuracy =====")
per_class_lines = []

for i, class_name in enumerate(class_names):
    class_mask = (y_true == i)
    total_count = np.sum(class_mask)
    correct_count = np.sum((y_true == i) & (y_pred == i))

    if total_count == 0:
        acc = 0.0
    else:
        acc = correct_count / total_count

    line = (
        f"{class_name}: "
        f"{correct_count}/{total_count} correct "
        f"-> accuracy = {acc:.4f} ({acc * 100:.2f}%)"
    )
    print(line)
    per_class_lines.append(line)

per_class_path = PROJECT_ROOT / "training" / "per_class_accuracy.txt"
with open(per_class_path, "w", encoding="utf-8") as f:
    f.write("\n".join(per_class_lines))

print(f"\nSaved per-class accuracy to: {per_class_path}")

# =========================
# Low-confidence analysis
# =========================
print("\n===== Low-Confidence Analysis =====")
top_confidences = np.max(y_prob, axis=1)

low_conf_threshold = 0.50
moderate_conf_threshold = 0.70

low_conf_count = np.sum(top_confidences < low_conf_threshold)
moderate_conf_count = np.sum((top_confidences >= low_conf_threshold) & (top_confidences < moderate_conf_threshold))
high_conf_count = np.sum(top_confidences >= moderate_conf_threshold)

print(f"Low confidence (<50%): {low_conf_count}")
print(f"Moderate confidence (50% to <70%): {moderate_conf_count}")
print(f"High confidence (>=70%): {high_conf_count}")

confidence_lines = [
    f"Low confidence (<50%): {low_conf_count}",
    f"Moderate confidence (50% to <70%): {moderate_conf_count}",
    f"High confidence (>=70%): {high_conf_count}",
]

confidence_path = PROJECT_ROOT / "training" / "confidence_summary.txt"
with open(confidence_path, "w", encoding="utf-8") as f:
    f.write("\n".join(confidence_lines))

print(f"Saved confidence summary to: {confidence_path}")