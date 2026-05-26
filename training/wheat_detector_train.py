
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# =========================
# Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "wheat_detector_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "wheat_detector"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS_FROZEN = 4
EPOCHS_FINE_TUNE = 4
AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "valid",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names

    with open(MODEL_DIR / "wheat_detector_index_to_label.json", "w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds, test_ds, class_names


def build_model():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ])

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

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
    print("Loading wheat detector datasets...")
    train_ds, valid_ds, test_ds, class_names = load_datasets()
    print("Wheat detector classes:", class_names)
    model, base_model = build_model()
    model.summary()
    test_callback = TestEvaluationCallback(test_ds)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_wheat_detector_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    print("Training frozen wheat detector...")
    history_frozen = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks + [test_callback]
    )

    print("Fine-tuning wheat detector...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks + [test_callback]
    )

    print("Evaluating wheat detector on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Wheat Detector Test Loss: {test_loss:.4f}")
    print(f"Wheat Detector Test Accuracy: {test_acc:.4f}")

    import matplotlib.pyplot as plt

    # Combine histories
    acc = history_frozen.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history_frozen.history["val_accuracy"] + history_fine.history["val_accuracy"]
    
    loss = history_frozen.history["loss"] + history_fine.history["loss"]
    val_loss = history_frozen.history["val_loss"] + history_fine.history["val_loss"]

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
    
    model.save(MODEL_DIR / "final_wheat_detector_model.keras")
    print("Saved final wheat detector model.")


if __name__ == "__main__":
    train()