from pathlib import Path
import sys
import os
import numpy as np
import keras
from sklearn.metrics import jaccard_score
from tensorflow.keras.layers import Conv2DTranspose as _Conv2DTranspose
import matplotlib.pyplot as plt

# Ensure project root is on path for data loading
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from train.train import load_data

MODEL_NAMES = [
    "reference",
    "correct_split_only_rotation",
    "correct_split_all_augmentations",
]


class Conv2DTranspose(_Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        # pop out groups if present, then call base constructor
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def evaluate_model(model_name, X_test, y_test):
    """
    Load a model, compute IoU on the test set, and save example masks.
    """
    # Paths
    base_dir = Path(__file__).parents[2]
    model_path = base_dir / "models" / model_name / "model.h5"
    results_dir = Path(__file__).parent / "results" / model_name
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    print(f"\n=== Evaluating model '{model_name}' ===")

    if model_name == "reference":  # the reference has a weird outdated groups attr.
        model = keras.models.load_model(
            model_path, custom_objects={"Conv2DTranspose": Conv2DTranspose}
        )
    else:
        model = keras.models.load_model(model_path)

    # model.summary()

    # Inference
    y_pred = model.predict(X_test)
    y_pred_thresh = (y_pred > 0.5).astype(np.uint8)

    # Compute IoU
    iou = jaccard_score(y_test.flatten(), y_pred_thresh.flatten(), average="binary")
    print(f"Model '{model_name}' IoU: {iou:.4f}")

    with open(results_dir / "iou.txt", "w") as f:
        f.write(str(iou))

    # Save example masks
    for idx in range(min(6, len(X_test))):
        img = X_test[idx]
        gt = y_test[idx].squeeze()
        pred = y_pred_thresh[idx].squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img.astype("uint8"))
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        fig.savefig(results_dir / f"example_{idx}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {min(6, len(X_test))} examples for '{model_name}' to {results_dir}")


def main():
    # Load test data once
    data_dir = Path(__file__).parents[2] / "data" / "training" / "test" / "resized"
    X_test, y_test = load_data(data_dir)

    # Evaluate each model in the list
    for name in MODEL_NAMES:
        evaluate_model(name, X_test, y_test)


if __name__ == "__main__":
    main()
