from pathlib import Path
import keras
import numpy as np
from sklearn.metrics import jaccard_score
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from train.train import load_data

model_name = "best_model_10_epochs.kers"
model_path = Path(__file__).parents[2] / "models" / model_name

model = keras.models.load_model(model_path)

model.summary()

# --- 2. Calculate Intersection over Union (IoU) on test set ---
# 2.1 Get predictions and threshold to binary masks
X_train, X_val, y_train, y_val = load_data()
y_pred = model.predict(X_val)
y_pred_thresh = (y_pred > 0.5).astype(np.uint8)

# 2.2 Flatten arrays for jaccard_score
y_true_flat = y_val.flatten()
y_pred_flat = y_pred_thresh.flatten()

# 2.3 Compute IoU (Jaccard Index)
iou_score = jaccard_score(y_true_flat, y_pred_flat, average="binary")

print("IoU score is:", iou_score)
