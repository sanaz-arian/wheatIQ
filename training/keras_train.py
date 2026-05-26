
import os
import json
import shutil
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utils import normalize_label_name, DISPLAY_NAME_MAP

# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw_dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "keras_ready"
MODEL_DIR = PROJECT_ROOT / "models" / "keras"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 8
SEED = 42
EPOCHS_FROZEN = 40
EPOCHS_FINE_TUNE = 0
AUTOTUNE = tf.data.AUTOTUNE


# =========================
# Reproducibility
# =========================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_split(split_name: str):
    """
    Copy images from noisy folder names into clean canonical class folders.
    Example:
        valid/aphid_valid -> processed/valid/aphid
    """
    source_split = RAW_DATASET_DIR / split_name
    target_split = PROCESSED_DIR / split_name

    target_split.mkdir(parents=True, exist_ok=True)

    for folder in source_split.iterdir():
        if not folder.is_dir():
            continue

        canonical = normalize_label_name(folder.name)
        target_class_dir = target_split / canonical
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_file in folder.iterdir():
            if img_file.is_file():
                shutil.copy2(img_file, target_class_dir / img_file.name)


def build_processed_dataset():
    ensure_clean_dir(PROCESSED_DIR)
    for split in ["train", "valid", "test"]:
        prepare_split(split)


def save_label_mapping(class_names):
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_label = {idx: name for name, idx in label_to_index.items()}

    with open(MODEL_DIR / "label_to_index.json", "w", encoding="utf-8") as f:
        json.dump(label_to_index, f, indent=2)

    with open(MODEL_DIR / "index_to_label.json", "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2)

    display_name_map = {
        name: DISPLAY_NAME_MAP.get(name, name.replace("_", " ").title())
        for name in class_names
    }
    with open(MODEL_DIR / "display_names.json", "w", encoding="utf-8") as f:
        json.dump(display_name_map, f, indent=2)


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "valid",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    save_label_mapping(class_names)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds, test_ds, class_names


def build_model(num_classes: int):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ])

    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(300, 300, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(300, 300, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model

class TestEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.test_losses = []
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_losses.append(loss)
        self.test_accuracies.append(acc)

        print(f"\nTest Loss: {loss:.4f} - Test Accuracy: {acc:.4f}")

def train():
    print("Preparing processed dataset...")
    build_processed_dataset()

    print("Loading datasets...")
    train_ds, valid_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)

    print("Building model...")
    model, base_model = build_model(num_classes)
    model.summary()
    test_callback = TestEvaluationCallback(test_ds)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_keras_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    print("Training frozen base model...")
    history_frozen = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks + [test_callback]
    )

    print("Fine-tuning top layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        train_ds,
        validation_data=valid_ds,
        initial_epoch=EPOCHS_FROZEN,
        epochs=EPOCHS_FROZEN + EPOCHS_FINE_TUNE,
        callbacks=callbacks + [test_callback]
    )


    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


    import matplotlib.pyplot as plt

    # Combine histories
    acc = history_frozen.history["accuracy"] #+ history_fine.history["accuracy"]
    val_acc = history_frozen.history["val_accuracy"] #+ history_fine.history["val_accuracy"]
    
    loss = history_frozen.history["loss"] #+ history_fine.history["loss"]
    val_loss = history_frozen.history["val_loss"] #+ history_fine.history["val_loss"]

    test_acc = test_callback.test_accuracies
    test_loss = test_callback.test_losses

    # Create explicit epoch numbers
    epochs = list(range(1, len(acc) + 1))

    # =====================
    # Accuracy Plot
    # =====================
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, acc, marker='o', label="Training Accuracy")
    plt.plot(epochs, val_acc, marker='o', label="Validation Accuracy")
    plt.plot(epochs, test_acc, marker='o', label="Test Accuracy")

    plt.title("Model Accuracy over Epochs")

    plt.xlabel("Epoch")          # 🔥 محور X
    plt.ylabel("Accuracy")       # 🔥 محور Y

    plt.xticks(epochs)           # نمایش دقیق شماره ایپاک‌ها
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================
    # Loss Plot
    # =====================
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, loss, marker='o', label="Training Loss")
    plt.plot(epochs, val_loss, marker='o', label="Validation Loss")
    plt.plot(epochs, test_loss, marker='o', label="Test Loss")

    plt.title("Model Loss over Epochs")

    plt.xlabel("Epoch")          #   محور X
    plt.ylabel("Loss")           #   محور Y

    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()

    model.save(MODEL_DIR / "final_keras_model.keras")
    print("Saved best Keras model.")


if __name__ == "__main__":
    train()


