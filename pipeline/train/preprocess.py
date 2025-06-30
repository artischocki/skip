from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import rotate

# Input and output directories
base_dir = Path(__file__).parents[2] / "data" / "train"
source_dir = base_dir / "all_images_and_labels"
raw_imgs = list(source_dir.glob("*.jpg"))
raw_labels = list(source_dir.glob("*.png"))

cropped_dir = base_dir / "all_images_and_labels_cropped"
overlay_dir = base_dir / "all_images_and_labels_overlays"
resized_dir = base_dir / "all_images_and_labels_resized"
augment_dir = base_dir / "all_images_and_labels_augmented"
cropped_dir.mkdir(exist_ok=True)
overlay_dir.mkdir(exist_ok=True)
resized_dir.mkdir(exist_ok=True)
augment_dir.mkdir(exist_ok=True)


def crop_center(
    img: Image.Image, crop_w: int = 1400, crop_h: int = 1840
) -> Image.Image:
    w, h = img.size
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h))


def crop_and_save(paths, out_dir):
    for p in tqdm(paths, desc=f"Cropping to {out_dir.name}"):
        with Image.open(p) as img:
            cropped = crop_center(img)
            cropped.save(out_dir / p.name)


# 1) Crop images and labels
print("→ Cropping images...")
crop_and_save(raw_imgs, cropped_dir)
print("→ Cropping labels...")
crop_and_save(raw_labels, cropped_dir)

# 2) Generate overlays
print("→ Generating overlays...")
for img_path in tqdm(list(cropped_dir.glob("*.jpg")), desc="Overlaying"):
    label_path = cropped_dir / f"{img_path.stem}_label.png"
    if not label_path.exists():
        continue

    with Image.open(img_path) as img, Image.open(label_path) as lbl:
        # 1) convert to RGBA
        base = img.convert("RGBA")
        # 2) get mask (white where label)
        mask = lbl.convert("L")
        # 3) build an all-blue image with alpha=128
        blue_img = Image.new("RGBA", base.size, (0, 0, 255, 128))
        # 4) create an empty transparent image
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        # 5) paste only the blue_img where mask==255
        overlay.paste(blue_img, (0, 0), mask)
        # 6) alpha-composite to blend
        combined = Image.alpha_composite(base, overlay)

        combined.save(overlay_dir / f"{img_path.stem}_overlay.png")

print("Done. You can now review overlays in:\n", overlay_dir)


# 3) Resize
print("→ Resizing...")
SIZE = 512
image_dataset = []
mask_dataset = []

images = sorted(list(cropped_dir.glob("*.jpg")))
masks = sorted(list(cropped_dir.glob("*.png")))
for i, (img_path, mask_path) in tqdm(
    enumerate(zip(images, masks)), desc=f"Resizing to: {SIZE}x{SIZE}"
):

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # convert image to grayscale, just to be sure, bc i dont know how many channels the imgs have.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize both
    img_resized = cv2.resize(img_gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    msk_resized = cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    # binarize mask: 0 = parasitized, 1 = uninfected (just to make sure again that its binary)
    _, msk_resized = cv2.threshold(msk_resized, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(resized_dir / img_path.name), img_resized)
    cv2.imwrite(str(resized_dir / mask_path.name), msk_resized)


# 4) Augment
print("→ Augmenting Images...")


def no_trafo(image):
    return image


def rotation(image, angle):
    return rotate(image, angle, reshape=False, mode="reflect")


def h_flip(image):
    return np.fliplr(image)


def v_flip(image):
    return np.flipud(image)


def save_img(img, path):
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(path)


transformations = {
    "": [no_trafo],
    "_hflip": [h_flip],
    "_vflip": [v_flip],
    "_hflip_vflip": [h_flip, v_flip],
}

imgs = sorted(list(resized_dir.glob("*.jpg")))
masks = sorted(list(resized_dir.glob("*.png")))

for img_path, mask_path in tqdm(list(zip(imgs, masks)), desc="Augmenting..."):
    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))

    # rotate between -3 and 3 Degrees
    for angle in range(-3, 4):
        aug_img = rotation(img, angle)
        aug_mask = rotation(mask, angle)
        for t_name, t_funcs in transformations.items():
            for t_func in t_funcs:
                aug_img = t_func(aug_img)
                aug_mask = t_func(aug_mask)

            if angle == 0:
                rot_str = ""
            else:
                rot_str = f"_{angle}deg"

            aug_img_name = f"{img_path.stem}{t_name}{rot_str}.jpg"
            aug_mask_name = (
                f"label_{(mask_path.stem).split('_')[0]}{t_name}{rot_str}.png"
            )
            aug_img_path = augment_dir / aug_img_name
            aug_mask_path = augment_dir / aug_mask_name

            save_img(aug_img, aug_img_path)
            save_img(aug_mask, aug_mask_path)


print("All Done.")
