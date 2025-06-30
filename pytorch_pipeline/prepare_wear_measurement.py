import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- CONFIG ---
base_path = Path(__file__).parents[1] / "test" / "Test_Dataset"
img_folder = base_path / "resized"
mask_folder = base_path / "predictions"
cleaned_folder = base_path / "cleaned_predictions"
overlay_folder = cleaned_folder / "overlay"

cleaned_folder.mkdir(exist_ok=True)
overlay_folder.mkdir(exist_ok=True)


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
mask_paths = sorted(mask_folder.glob("pred_*.png"))
img_paths = sorted(img_folder.glob("*.[jp][pn]g"))

for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
    # Maske laden und binarisieren
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask_bin = (mask > 127).astype(np.uint8) * 255

    # Morphologie anwenden
    mask_cleaned = preprocess_mask(mask_bin)

    # Größtes zusammenhängendes Segment extrahieren
    cleaned_mask = get_largest_connected_component(mask_cleaned)

    # Speichern bereinigter Maske
    cleaned_path = cleaned_folder / mask_path.name
    Image.fromarray(cleaned_mask).save(cleaned_path)

    # Overlay-Bild erzeugen
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_bool = (cleaned_mask > 127).astype(np.uint8)
    mask_color = np.zeros_like(img)
    mask_color[:, :, 2] = 255  # Blau

    alpha = 0.4
    overlay = np.where(
        mask_bool[..., None] == 1,
        (alpha * mask_color + (1 - alpha) * img).astype(np.uint8),
        img,
    )

    overlay_path = overlay_folder / f"overlay_{img_path.stem}.png"
    Image.fromarray(overlay).save(overlay_path)

print("Done. Cleaned masks and overlays saved.")
