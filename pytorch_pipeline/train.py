import os
from pathlib import Path
import random
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from tqdm import tqdm

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 1000
PATIENCE = 50
LR = 1e-4
DATA_ROOT = Path(__file__).parents[1] / "all_images_and_labels_cropped"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")


# === DATASET + AUGMENTATION ===
class SegmentationTransform:
    def __init__(self, img_size=256):
        self.img_size = img_size

    def __call__(self, image, mask):
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # FLIP
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # ROTATE
        angle = random.uniform(-4, 4)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # SCALE
        # if random.random() > 0.5:
        #     translate = (
        #         random.uniform(-0.05, 0.05) * self.img_size,
        #         random.uniform(-0.05, 0.05) * self.img_size,
        #     )
        #     scale = random.uniform(0.95, 1.05)
        #     image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0)
        #     mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0)

        # BRIGHTNESS, CONTRAST
        # if random.random() > 0.5:
        #     image = TF.adjust_brightness(
        #         image, brightness_factor=random.uniform(0.8, 1.2)
        #     )
        #     image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))

        # TO TENSOR
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # NOISE
        # if random.random() < 0.5:
        #     noise = torch.randn_like(image) * 0.05
        #     image = torch.clamp(image + noise, 0.0, 1.0)

        mask = (mask > 0.5).float()
        return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.root_dir = root_dir
        self.image_paths = sorted(
            [
                f
                for f in os.listdir(root_dir)
                if f.endswith(".jpg")
                or (f.endswith(".png") and not f.endswith("_label.png"))
            ]
        )
        self.augment = augment
        self.transform = SegmentationTransform() if augment else self.no_aug

    def no_aug(self, image, mask):
        image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = TF.resize(
            mask, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.NEAREST
        )
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, (mask > 0.5).float()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label_path = os.path.join(
            self.root_dir, img_name.rsplit(".", 1)[0] + "_label.png"
        )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("L")
        return self.transform(image, mask)


# === MODEL ===
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.dec1 = CBR(128, 64)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.up(x2)
        x4 = self.dec1(x3)
        out = torch.sigmoid(self.final(x4))
        return out


# === EARLY STOPPING ===
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# === LOAD DATA ===
full_ds = SegmentationDataset(DATA_ROOT, augment=True)
val_ds = SegmentationDataset(DATA_ROOT, augment=False)

train_ds, _ = random_split(
    full_ds, [int(0.8 * len(full_ds)), len(full_ds) - int(0.8 * len(full_ds))]
)
_, val_ds = random_split(
    val_ds, [int(0.8 * len(val_ds)), len(val_ds) - int(0.8 * len(val_ds))]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    # === TKINTER TRACKING ===
    # Tkinter live plot setup
    root = tk.Tk()
    root.title("Training Progress")

    fig, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    (train_line,) = ax.plot([], [], label="Train Loss")
    (val_line,) = ax.plot([], [], label="Val Loss")
    ax.legend()
    canvas.draw()

    # === TRAIN ===
    model = SimpleUNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    early_stopper = EarlyStopping(patience=PATIENCE)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = criterion(preds, masks)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}"
        )

        # === UPDATE LIVE PLOT ===
        train_line.set_data(range(len(train_losses)), train_losses)
        val_line.set_data(range(len(val_losses)), val_losses)
        ax.relim()
        ax.autoscale_view()
        canvas.draw()
        root.update_idletasks()
        root.update()

        # === EARLY STOPPING ===
        if early_stopper.step(avg_val_loss, model):
            print("Early stopping triggered.")
            break

    # === SAVE BEST MODEL ===
    model_name = "best_segmentation_model_hflip_vflip_4_degree_rotate.pt"
    torch.save(early_stopper.best_model, model_name)
    print(f"Best model saved as '{model_name}'.")

    # === SAVE LOSS PLOT ===
    fig.savefig(f"{model_name}_loss_plot.png")
    print(f"Loss plot saved as '{model_name}_loss_plot.png'.")

    # === CLOSE TKINTER WINDOW ===
    root.destroy()
