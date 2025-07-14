from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
)
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import MeanIoU
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf

SIZE = 512
INPUTS = Input((SIZE, SIZE, 1))

MODEL_NAME = "correct_split_all_augementations"


def create_model() -> Model:
    """
    This function returns a UNET as specified by the notebook.
    """
    s = INPUTS
    # Contraction path
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[INPUTS], outputs=[outputs])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", MeanIoU(num_classes=2)],
    )

    return model


def load_data(augmented_dir: Path):
    img_dataset = []
    mask_dataset = []
    imgs = sorted(list(augmented_dir.glob("*.jpg")))
    masks = sorted(list(augmented_dir.glob("*.png")))
    for img_path, mask_path in tqdm(zip(imgs, masks), desc="Loading Data"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # convert image to grayscale, just to be sure, bc i dont know how many channels the imgs have.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # make sure the masks are binary
        mask = (mask > 0).astype(np.uint8)

        img_dataset.append(img)
        mask_dataset.append(mask)

    # convert lists to numpy arrays and add channel dimension
    x = np.array(img_dataset).reshape(-1, SIZE, SIZE, 1)
    y = np.array(mask_dataset).reshape(-1, SIZE, SIZE, 1)

    return x, y


if __name__ == "__main__":
    # Load model
    model = create_model()
    # Load data
    print("Loading Data...")
    train_data_path = (
        Path(__file__).parents[2] / "data" / "training" / "train" / "augmented"
    )
    val_data_path = (
        Path(__file__).parents[2] / "data" / "training" / "val" / "augmented"
    )
    X_train, y_train = load_data(train_data_path)
    X_val, y_val = load_data(val_data_path)
    print("Done Loading Data.")
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set:  {X_val.shape}, {y_val.shape}")

    # cuda?
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs found:", gpus)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU available:", tf.config.list_physical_devices("GPU") != [])
    print("Legacy GPU check:", tf.test.is_gpu_available(cuda_only=True))

    # Train the model

    model_dir = Path(__file__).parents[2] / "models" / MODEL_NAME
    model_dir.mkdir(exist_ok=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=3)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(model_dir / "model_{epoch:02d}_val_loss={val_loss:.4f}.h5"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=8,
        epochs=20,
        validation_data=(X_val, y_val),
        shuffle=False,
        callbacks=[early_stopping, checkpoint_cb],
    )
    # Save the trained model
    model.save(model_dir / f"model.h5")

    # --- 1. Plot training & validation accuracy and loss over epochs ---
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val   Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val   Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(model_dir / f"{MODEL_NAME}_lossplot.png")
