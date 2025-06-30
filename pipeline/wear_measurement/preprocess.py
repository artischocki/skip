from pathlib import Path
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Folder with the images
base_dir = Path(__file__).parents[2] / "data" / "test" / "Images"
raw_imgs_dir = base_dir / "images"
cropped_dir = base_dir / "cropped"
aligned_dir = base_dir / "aligned"
resized_dir = base_dir / "resized"
cropped_dir.mkdir(exist_ok=True)
aligned_dir.mkdir(exist_ok=True)
resized_dir.mkdir(exist_ok=True)


def align_imgs(int_dir, out_dir):
    # Load image filenames
    image_files = sorted(
        [
            filename
            for filename in os.listdir(int_dir)
            if filename.endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    print(len(image_files))
    image_files_source = [image_files[0]]

    # Load reference image
    img_path = os.path.join(int_dir, image_files_source[0])
    imgRef = cv2.imread(img_path)
    imgRef_gray = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
    sz = imgRef.shape

    # Use Euclidean motion model: rotation + translation only
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Initialize 2x3 warp matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # ECC optimization parameters
    number_of_iterations = 1000
    termination_eps = 1e-6
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    for filename in tqdm(image_files):
        # if already present in aligned folder skip. accidentally keyboard interupted :P
        if "aligned_" + filename in os.listdir(aligned_dir):
            print(
                f"Skipping {filename} because already aligned. Delete from aligned dir if you want to realign."
            )
            continue
        img_path = os.path.join(int_dir, filename)
        imgTest = cv2.imread(img_path)
        imgTest_gray = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)

        # Estimate warp matrix using ECC
        try:
            cc, warp_matrix_est = cv2.findTransformECC(
                imgRef_gray, imgTest_gray, warp_matrix.copy(), warp_mode, criteria
            )

            # Warp the image
            aligned_img = cv2.warpAffine(
                imgTest,
                warp_matrix_est,
                (sz[1], sz[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )

            # Save aligned image
            aligned_filename = out_dir / f"aligned_{filename}"
            cv2.imwrite(str(aligned_filename), aligned_img)

        except cv2.error as e:
            print(f"Failed to align {filename}: {e}")


def crop_imgs(input_dir: Path, output_dir) -> None:
    for img_path in tqdm(list(input_dir.glob("*.jpg"))):
        # crop the center section ~> 1400W * 1840H
        with Image.open(img_path) as img:
            width, height = img.size
            left = (width - 1400) // 2
            top = (height - 1840) // 2
            right = left + 1400
            bottom = top + 1840
            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_dir / img_path.name)
    return


def resize_imgs(input_dir: Path, output_dir) -> None:
    for img_path in tqdm(list(input_dir.glob("*.jpg"))):
        # resize to 512 x 512
        with Image.open(img_path) as img:
            resized = img.resize((512, 512), Image.LANCZOS)
            output_path = output_dir / img_path.name
            resized.save(output_path)
    return


print("Aligning...")
align_imgs(raw_imgs_dir, aligned_dir)
print("Done.")
print("Cropping...")
crop_imgs(aligned_dir, cropped_dir)
print("Done.")
print("Resizing...")
resize_imgs(cropped_dir, resized_dir)
print("Done.")
