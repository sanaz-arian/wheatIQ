import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "binary_dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "binary" / "best_binary_model.keras"

OUTPUT_DIR = PROJECT_ROOT / "models" / "binary" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8


def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return test_ds, test_ds.class_names


def plot_confusion_matrix_percentage(cm, class_names, acc, precision, recall, f1):
    # نرمال‌سازی به درصد (row-wise)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(right=0.78)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")

    ax.set_title("Stage 1: Wheat vs Non-Wheat (Normalized %)")

    metrics_text = (
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-score:  {f1:.4f}\n\n"
        f"n = {cm.sum()}"
    )

    fig.text(
        0.82, 0.50, metrics_text,
        fontsize=11,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
    )

    plt.savefig(OUTPUT_DIR / "stage1_confusion_matrix_percent.png", dpi=300, bbox_inches="tight")
    plt.show()

def print_binary_confusion_explanation(cm, class_names):
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        print("\n===== Confusion Matrix Details =====")
        print(f"True Negative  ({class_names[0]} predicted as {class_names[0]}): {tn}")
        print(f"False Positive ({class_names[0]} predicted as {class_names[1]}): {fp}")
        print(f"False Negative ({class_names[1]} predicted as {class_names[0]}): {fn}")
        print(f"True Positive  ({class_names[1]} predicted as {class_names[1]}): {tp}")

        print("\nRecall formula:")
        print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {tp / (tp + fn):.4f}")


def main():
    print("Loading saved model...")
    model = load_model(MODEL_PATH)

    print("Loading test dataset...")
    test_ds, class_names = load_test_dataset()
    print("Class names:", class_names)

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = (preds > 0.5).astype(int).flatten()

        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    print("\n===== Test Metrics =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Accuracy (sklearn): {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    print("\n===== Confusion Matrix =====")
    print(cm)

    np.savetxt(OUTPUT_DIR / "stage2_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print_binary_confusion_explanation(cm, class_names)

    plot_confusion_matrix_percentage(cm, class_names, acc, precision, recall, f1)


if __name__ == "__main__":
    main()