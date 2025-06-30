import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from pathlib import Path

# Konfiguration
PIXEL_SIZE_UM = 1.725
input_dir = Path(__file__).parents[1] / "test" / "Test_Dataset" / "cleaned_predictions"
all_results = []


# --- Hilfsfunktionen ---
def get_main_angle(mask):
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
    if lines is None:
        return 0
    angles = []
    for rho, theta in lines[:, 0]:
        angle_deg = np.rad2deg(theta)
        if angle_deg > 90:
            angle_deg -= 180
        angles.append(angle_deg)
    return np.median(angles)


def rotate_image(image, angle_deg):
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, M


def inverse_rotate_point(x, y, M):
    M_inv = cv2.invertAffineTransform(M)
    pt = np.dot(M_inv, np.array([x, y, 1]))
    return int(pt[0]), int(pt[1])


def measure_max_thickness(rotated_mask):
    h, w = rotated_mask.shape
    max_thickness = 0
    max_x = 0
    max_y_range = (0, 0)
    for x in range(w):
        column = rotated_mask[:, x]
        ys = np.where(column > 0)[0]
        if len(ys) > 0:
            thickness = ys[-1] - ys[0]
            if thickness > max_thickness:
                max_thickness = thickness
                max_x = x
                max_y_range = (ys[0], ys[-1])
    return max_thickness, max_x, max_y_range


# --- 1. Alle Bildpfade sammeln ---
files = sorted(
    [f for f in input_dir.iterdir() if f.suffix in [".png", ".jpg", ".jpeg"]]
)

# --- 2. Globalen Schneidenwinkel bestimmen ---
angles = []
for file in files:
    mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, 0)
    angle = get_main_angle(binary)
    angles.append(angle)

global_angle = np.median(angles)
print(f"Globaler Schneidenwinkel (Median): {global_angle:.2f}°")

# --- 3. Messung mit konstantem Winkel ---
all_results = []
for file in files:
    mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, 0)

    rotated_mask, M = rotate_image(binary, -global_angle)
    thickness_px, max_x, (y1, y2) = measure_max_thickness(rotated_mask)
    thickness_um = thickness_px * PIXEL_SIZE_UM

    pt1 = inverse_rotate_point(max_x, y1, M)
    pt2 = inverse_rotate_point(max_x, y2, M)

    all_results.append(
        {
            "filename": file.name,
            "thickness_um": thickness_um,
            "pt1": pt1,
            "pt2": pt2,
            "mask": mask.copy(),
        }
    )

# --- 4. Visualisierung von 5 zufälligen Beispielen ---
sampled = random.sample(all_results, 5)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, res in zip(axes, sampled):
    img = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2RGB)
    cv2.line(img, res["pt1"], res["pt2"], (255, 0, 0), 2)
    ax.imshow(img)
    ax.set_title(f"{res['filename']}\nVBmax = {res['thickness_um']:.1f} μm")
    ax.axis("off")
plt.tight_layout()
plt.show()


# Farbzyklus für Schneiden
colors = ["red", "orange", "green", "blue"]


# Cut-Nummer aus Dateinamen extrahieren
def extract_cut_number(filename, base=2074, step=8):
    match = re.search(r"(\d+)", filename)
    if match:
        return ((int(match.group(1)) - base) // step) * step
    return -1


# Cut-Nummern und VBmax-Werte sammeln
cut_numbers = [extract_cut_number(res["filename"]) for res in all_results]
vbmax_values = [res["thickness_um"] for res in all_results]

# Plot erstellen
plt.figure(figsize=(10, 6))
for i, (vb, cut) in enumerate(zip(vbmax_values, cut_numbers)):
    color = colors[i % 4]
    plt.scatter(cut, vb, color=color)

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


plt.xlabel("Cut-Nummer")
plt.ylabel("VBmax [μm]")
plt.title("VBmax orthogonal zur Schneide (fester Winkel)")
plt.grid(True)
plt.tight_layout()
plt.show()
