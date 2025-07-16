import os
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# ----------------------------------------
# Configuration
# ----------------------------------------
# Pfad zur Input-Ordner mit bereinigten Masken
base_dir = Path(__file__).parents[2] / "data" / "inference" / "Video" / "video0000031"
cleaned_pred_dir = base_dir / "cleaned_predictions_only_rot"
results_dir = base_dir / "wear_measurement_only_rot"
results_dir.mkdir(exist_ok=True)
for i in range(1, 5):
    (results_dir / f"{i}").mkdir(exist_ok=True)
PIXEL_SIZE_UM = 1.725

# Wie viele Beispielbilder mit eingezeichneter Maximalbreite anzeigen?


# ----------------------------------------
# Hilfsfunktionen
# ----------------------------------------
def extract_cut_number(filename, base=2074, step=8):
    """Extrahiert aus dem Dateinamen die Cut-Nummer."""
    match = re.search(r"(\d+)", filename)
    if match:
        return ((int(match.group(1)) - base) // step) * step
    return -1


def get_main_angle(mask):
    """Berechnet den dominanten Linienwinkel (in Grad) einer Binär-Maske."""
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        deg = np.rad2deg(theta)
        # bring into [-90, +90]
        if deg > 90:
            deg -= 180
        angles.append(deg)
    return float(np.median(angles))


def rotate_image_and_mask(img, angle_deg):
    """Rotiere Bild um angle_deg um sein Zentrum (nearest-neighbor, border=0)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w, h / 2), angle_deg, 1.0)
    w *= 2
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def measure_max_thickness_with_coords(rot_mask):
    """
    Misst in einem rotieren Binärbild die maximale
    vertikale Dicke und gibt (px, x, y0, y1).
    """
    h, w = rot_mask.shape
    max_t = 0
    best = (0, 0, 0)
    for x in range(w):
        ys = np.where(rot_mask[:, x] > 0)[0]
        if ys.size:
            t = ys[-1] - ys[0]
            if t > max_t:
                max_t = t
                best = (x, ys[0], ys[-1])
    return max_t, best  # px, (x, y0, y1)


# ----------------------------------------
# 1) Über alle Bilder den globalen Winkel finden
# ----------------------------------------
angles = []
for fn in sorted(os.listdir(cleaned_pred_dir))[-5:]:
    if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        continue
    path = cleaned_pred_dir / fn
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    angles.append(get_main_angle(binary))

# main_angle = float(np.median(angles))
main_angle = 55.78  # der ausgemessene echte wert über die alignten bilder. einfach den verwenden, anstatt auf krampf iein winkel zu detektieren!
print(f"Verwendeter globaler Drehwinkel: {main_angle:.2f}°")

# ----------------------------------------
# 2) Alle Bilder messen, anzeigen und scatter plot
# ----------------------------------------
VBmax = []
cut_numbers = []


for i, fn in tqdm(enumerate(sorted(os.listdir(cleaned_pred_dir)))):
    if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        continue
    edge_id = i % 4

    # a) Cut-Nummer extrahieren
    cut_num = extract_cut_number(fn)
    cut_numbers.append(cut_num)

    # b) Maske laden und in Binär umwandeln
    path = cleaned_pred_dir / fn
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # c) global rotieren
    rotated = rotate_image_and_mask(binary, -main_angle)

    # d) Dicke messen
    thickness_px, (x_max, y0, y1) = measure_max_thickness_with_coords(rotated)
    VBmax.append(thickness_px * PIXEL_SIZE_UM)

    # e) Visualisierung
    vis = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (x_max, y0), (x_max, y1), (255, 0, 0), 2)
    plt.figure(figsize=(4, 6))
    plt.imshow(vis)
    plt.title(f"{fn}\nVBmax = {thickness_px*PIXEL_SIZE_UM:.1f} μm")
    plt.axis("off")
    out_path = results_dir / f"{edge_id + 1}" / f"{str(cut_num).zfill(3)}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()

# f) Scatter-Plot der VBmax-Werte
plt.figure(figsize=(10, 6))
colors = ["red", "orange", "green", "blue"]
for i, (vb, cut) in enumerate(zip(VBmax, cut_numbers)):
    plt.scatter(cut, vb, color=colors[i % len(colors)])
plt.xlabel("Cut-Nummer")
plt.ylabel("VBmax [μm]")
plt.title("VBmax orthogonal zur Schneide\n(globaler Drehwinkel korrigiert)")
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "VBMax_plot_all.png")


# ----------------------------------------
# 3) Outlier-Erkennung mit DBSCAN
# ----------------------------------------

# 2D-Daten: (Cut-Nummer, VBmax)
data = np.column_stack((cut_numbers, VBmax))

# DBSCAN-Parameter eps und min_samples ggf. anpassen
db = DBSCAN(eps=15, min_samples=1).fit(data)
labels = db.labels_

# Indizes der Ausreißer (label == -1)
outlier_idx = np.where(labels == -1)[0]
print("Gefundene Ausreißer-Indizes:", outlier_idx)

# ----------------------------------------
# 4) Separate Plots für die 4 Schneidkanten
# ----------------------------------------
for edge_id in range(4):
    # alle Indizes für diese Kante
    idxs = [i for i in range(len(cut_numbers)) if i % 4 == edge_id]

    # inliers vs. outliers
    inliers = [i for i in idxs if i not in outlier_idx]
    outliers = [i for i in idxs if i in outlier_idx]

    plt.figure(figsize=(8, 4))
    # Inlier in Blau
    if inliers:
        cuts_in = [cut_numbers[i] for i in inliers]
        vb_in = [VBmax[i] for i in inliers]
        plt.scatter(cuts_in, vb_in, color="blue", label="Inlier")

    # Outlier in Rot
    if outliers:
        cuts_out = [cut_numbers[i] for i in outliers]
        vb_out = [VBmax[i] for i in outliers]
        plt.scatter(cuts_out, vb_out, color="red", label="Outlier")

    plt.xlabel("Cut-Nummer")
    plt.ylabel("VBmax [μm]")
    plt.title(f"VBmax für Schneidkante {edge_id+1}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"VBMax_plot_edge_{edge_id + 1}.png")
    plt.close()
