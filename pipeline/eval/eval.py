from pathlib import Path
import keras
import numpy as np
from sklearn.metrics import jaccard_score
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from train.train import load_data

model_name = "correct_split_only_rotation"
model_path = Path(__file__).parents[2] / "models" / model_name / "model.h5"

model = keras.models.load_model(model_path)

model.summary()

# --- 2. Calculate Intersection over Union (IoU) on test set ---
# 2.1 Get predictions and threshold to binary masks
data_path = Path(__file__).parents[2] / "data" / "training" / "test" / "resized"
X_test, y_test = load_data(data_path)
y_pred = model.predict(X_test)
y_pred_thresh = (y_pred > 0.5).astype(np.uint8)

# 2.2 Flatten arrays for jaccard_score
y_true_flat = y_test.flatten()
y_pred_flat = y_pred_thresh.flatten()

# 2.3 Compute IoU (Jaccard Index)
iou_score = jaccard_score(y_true_flat, y_pred_flat, average="binary")

print("IoU score is:", iou_score)
