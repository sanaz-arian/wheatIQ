import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "keras_ready"
MODEL_PATH = PROJECT_ROOT / "models" / "keras" / "final_keras_model.keras"

OUTPUT_DIR = PROJECT_ROOT / "models" / "keras" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (300, 300)
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


def plot_confusion_matrix_percent(cm, class_names, acc, precision, recall, f1):
    # row-wise normalization
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    cm_percent = np.nan_to_num(cm_percent)

    fig, ax = plt.subplots(figsize=(15, 10))

    # فضای بیشتری در سمت راست برای colorbar و metrics box
    fig.subplots_adjust(right=0.78, bottom=0.22)

    im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Percentage (%)", rotation=90, va="bottom")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Stage 3: Multi-Class Disease Confusion Matrix (%)"
    )

    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    threshold = cm_percent.max() / 2.0
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            value = cm_percent[i, j]
            ax.text(
                j, i, f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9
            )

    metrics_text = (
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1-score:  {f1:.4f}"
    )

    # باکس را بیشتر به راست بردیم
    fig.text(
        0.85, 0.50, metrics_text,
        fontsize=11,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
    )

    plt.savefig(OUTPUT_DIR / "stage3_confusion_matrix_percent.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_multiclass_confusion_note():
    print("\n===== Confusion Matrix Note =====")
    print("This confusion matrix is row-normalized.")
    print("Each row sums to 100%, showing how each true class was distributed across predicted classes.")
    print("Diagonal values show the percentage of correct predictions for each class.")
    print("Off-diagonal values show the percentage of confusion with other classes.")


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
        pred_labels = np.argmax(preds, axis=1)

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

    print("\n===== Confusion Matrix (Counts) =====")
    print(cm)

    np.savetxt(OUTPUT_DIR / "stage3_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print_multiclass_confusion_note()

    plot_confusion_matrix_percent(cm, class_names, acc, precision, recall, f1)


if __name__ == "__main__":
    main()