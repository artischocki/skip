from keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

# Input and output directories
base_dir = Path(__file__).parents[2] / "data" / "inference" / "Images"
resized_dir = base_dir / "resized"
cropped_dir = base_dir / "cropped"
predictions_dir = base_dir / "predictions"
pred_overlay_dir = predictions_dir / "overlay"
cleaned_predictions_dir = base_dir / "cleaned_predictions"
cleaned_overlay_dir = cleaned_predictions_dir / "overlay"
predictions_dir.mkdir(exist_ok=True)
cleaned_predictions_dir.mkdir(exist_ok=True)
cleaned_overlay_dir.mkdir(exist_ok=True)
pred_overlay_dir.mkdir(exist_ok=True)


model_path = (
    Path(__file__).parents[2]
    / "models"
    / "correct_split_all_augmentations"
    / "model.h5"
)

# Load the model
model = load_model(model_path)
model.summary()

image_dataset = []
# Load data
print("Loading imgs:")
for img_path in tqdm(list(resized_dir.glob("*.[jp][pn]g"))):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    # convert image to grayscale, just to be sure, bc i dont know how many channels the tif imgs have.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_dataset.append(img_gray)

# convert lists to numpy arrays and add channel dimension
image_dataset = np.array(image_dataset).reshape(-1, 512, 512, 1)

print(f"Loaded {len(image_dataset)} samples.")
print(f"{image_dataset.shape=}")

# predict masks
y_pred = model.predict(image_dataset)
y_pred_thresh = (y_pred > 0.5).astype(np.uint8)


def resize_mask(mask, orig_size):
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.resize(orig_size, resample=Image.NEAREST)
    mask = np.array(mask_pil)
    # Wieder binärisieren
    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def create_overlay(img_path, mask):
    # Overlay-Bild erzeugen
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_bool = (mask > 127).astype(np.uint8)
    mask_color = np.zeros_like(img)
    mask_color[:, :, 2] = 255  # Blau

    alpha = 0.4
    overlay = np.where(
        mask_bool[..., None] == 1,
        (alpha * mask_color + (1 - alpha) * img).astype(np.uint8),
        img,
    )
    return Image.fromarray(overlay)


# # Convert to image and save
print("Saving predictions:")
for img_path, pred_mask in tqdm(zip(cropped_dir.glob("*.[jp][pn]g"), y_pred_thresh)):
    pred_mask = pred_mask.squeeze() * 255  # shape: (512, 512)
    orig_size = Image.open(img_path).size
    pred_mask = resize_mask(pred_mask, orig_size)
    pred_img = Image.fromarray(pred_mask.astype(np.uint8))  # ensure correct type
    pred_img.save(predictions_dir / (f"pred_{img_path.stem}.png"))
    # Overlay-Bild erzeugen
    overlay_img = create_overlay(img_path, pred_mask)
    overlay_path = pred_overlay_dir / f"overlay_{img_path.stem}.png"
    overlay_img.save(overlay_path)
print("Done.")


### CLEAN PREDICTIONS


def get_largest_connected_component(mask):
    """Behält das größte zusammenhängende Segment in einer binären Maske"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = (labels == largest_label).astype(np.uint8) * 255
    return cleaned


def preprocess_mask(mask_bin):
    """Wendet Closing und Opening auf die Maske an"""
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


print("Cleaning masks and creating overlays:")
mask_paths = sorted(predictions_dir.glob("pred_*.png"))
img_paths = sorted(cropped_dir.glob("*.[jp][pn]g"))

for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
    # Maske laden und binarisieren
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask_bin = (mask > 127).astype(np.uint8) * 255

    # Morphologie anwenden
    mask_cleaned = preprocess_mask(mask_bin)

    # Größtes zusammenhängendes Segment extrahieren
    cleaned_mask = get_largest_connected_component(mask_cleaned)

    # Wieder auf originale Größe bringen
    orig_size = Image.open(img_path).size
    cleaned_mask = resize_mask(cleaned_mask, orig_size)

    # Speichern bereinigter Maske
    cleaned_path = cleaned_predictions_dir / mask_path.name
    Image.fromarray(cleaned_mask).save(cleaned_path)

    # Overlay-Bild erzeugen
    overlay_img = create_overlay(img_path, cleaned_mask)
    overlay_path = cleaned_overlay_dir / f"overlay_{img_path.stem}.png"
    overlay_img.save(overlay_path)

print("Done. Cleaned masks and overlays saved.")
