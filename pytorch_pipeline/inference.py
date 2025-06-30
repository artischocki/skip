from pathlib import Path

from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch

from train import SimpleUNet


model = SimpleUNet()

model_path = (
    Path(__file__).parents[1]
    / "best_segmentation_model_all_augmentations_200_epochs.pt"
)

# Modellparameter laden
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# Modell in den Evaluierungsmodus setzen
model.eval()


# Pfade
int_folder = Path(__file__).parents[1] / "test" / "Test_Dataset" / "resized"
out_folder = Path(__file__).parents[1] / "test" / "Test_Dataset" / "predictions"
out_folder.mkdir(exist_ok=True)

# Bilddaten laden
image_dataset = []
orig_size = [1400, 1840]

print("Loading imgs:")
image_paths = list(int_folder.glob("*.[jp][pn]g"))
for img_path in tqdm(image_paths):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # in RGB konvertieren
    img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    image_dataset.append(img_resized)

image_dataset = np.array(image_dataset).astype(np.float32) / 255.0  # (N, 512, 512, 3)
image_dataset = np.transpose(image_dataset, (0, 3, 1, 2))  # → (N, 3, 512, 512)

print(f"Loaded {len(image_dataset)} samples.")
print(f"{image_dataset.shape=}")

# Vorhersage
y_pred = []
print("Predicting:")
with torch.no_grad():
    for img in tqdm(image_dataset):
        input_tensor = torch.from_numpy(img).unsqueeze(0)
        output = model(input_tensor)
        pred = (output.squeeze().numpy() > 0.5).astype(np.uint8)
        y_pred.append(pred)

# Speichern
print("Saving predictions:")
print(len(image_paths))
print(len(y_pred))
for img_path, pred_mask in tqdm(zip(image_paths, y_pred)):
    pred_mask = (pred_mask * 255).astype(np.uint8)
    pred_img = Image.fromarray(pred_mask)
    pred_img = pred_img.resize((orig_size[0], orig_size[1]), Image.NEAREST)
    pred_img.save(out_folder / f"pred_{img_path.stem}.png")
print("Done.")

overlay_folder = out_folder / "overlay"
overlay_folder.mkdir(exist_ok=True)

# Zusätzliches speichern von Maske über originalem Bild zu schnellen auswerten
print("Saving overlay predictions:")
base_path = Path(__file__).parents[1] / "test" / "Test_Dataset"
img_folder = base_path / "cropped"
mask_folder = base_path / "predictions"
overlay_folder = mask_folder / "overlay"
overlay_folder.mkdir(exist_ok=True)

# Bild- und Maskenpfade
img_paths = sorted(img_folder.glob("*.[jp][pn]g"))
mask_paths = sorted(mask_folder.glob("pred_*.png"))

for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
    # Originalbild laden
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Maske laden
    mask = Image.open(mask_path).convert("L")  # Graustufen
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)  # binär

    # Maske einfärben (blau)
    mask_color = np.zeros_like(img)
    mask_color[:, :, 2] = 255  # Blaukanal

    # Alpha-Blending
    alpha = 0.4
    overlay = np.where(
        mask[..., None] == 1,
        (alpha * mask_color + (1 - alpha) * img).astype(np.uint8),
        img,
    )

    # Speichern
    result = Image.fromarray(overlay)
    result.save(overlay_folder / f"overlay_{img_path.stem}.png")

print("Overlay export done.")
