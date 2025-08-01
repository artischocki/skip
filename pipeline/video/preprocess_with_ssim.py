import os
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Konfiguration
VIDEO_PATH = "/path/to/video0000030.avi"
REFERENCE_IMAGES_DIR = (
    "/path/to/KIP_Referenzbilder_Video"  # 4 Referenzbilder (eines pro Schneidkante)
)
OUTPUT_DIR = "/path/to/edge_frames"
PIXEL_SIZE_UM = 1.725

# Verzeichnisse vorbereiten
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i in range(4):
    os.makedirs(f"{OUTPUT_DIR}/edge_{i+1}", exist_ok=True)


# Funktion: Frames extrahieren, ohne das gesamte Video in den Speicher zu laden
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1
    cap.release()


# Funktion: SSIM-Vergleich zweier Bilder
def compute_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))
    score, _ = ssim(img1, img2, full=True)
    return score


# Referenzbilder laden
REFERENCE_IMAGE_PATHS = [
    "/content/drive/MyDrive/KIP_Referenzbilder_Video/edge1.png",
    "/content/drive/MyDrive/KIP_Referenzbilder_Video/edge2.png",
    "/content/drive/MyDrive/KIP_Referenzbilder_Video/edge3.png",
    "/content/drive/MyDrive/KIP_Referenzbilder_Video/edge4.png",
]

reference_images = []
for ref_path in REFERENCE_IMAGE_PATHS:
    img = cv2.imread(ref_path)
    if img is None:
        print(f"Bild konnte nicht geladen werden: {ref_path}")
    else:
        reference_images.append(img)

# Verzeichnis für die letzten Frames vorbereiten
OUTPUT_DIR_letzte_Frames = "/content/drive/MyDrive/frames_letzte_10s"
os.makedirs(OUTPUT_DIR_letzte_Frames, exist_ok=True)

# === Video laden ===
cap = cv2.VideoCapture(VIDEO_PATH)

# === Framerate (fps) und Gesamtanzahl der Frames ===
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

print(f"FPS: {fps}")
print(f"Gesamtanzahl Frames: {total_frames}")
print(f"Dauer (s): {duration_sec:.2f}")

# DEBUG
last_duration_sec = 10
start_frame = max(0, total_frames - int(fps * last_duration_sec))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_idx = int(start_frame)
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Bild speichern
    out_path = os.path.join(OUTPUT_DIR_letzte_Frames, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(out_path, frame)

    frame_idx += 1
    saved += 1

cap.release()
print(f"{saved} Frames der letzten {last_duration_sec} Sekunden gespeichert.")

# Ähnlichkeitsvergleich: Beste Matches pro Frame suchen
MATCH_THRESHOLD = 0.8  # Experimentell anpassbar
found_frames = [[] for _ in range(4)]  # Speicherpfade je Schneidkante

for frame_idx, frame in tqdm(extract_frames(VIDEO_PATH)):
    best_score = 0
    best_edge = -1

    # Ähnlichkeitsbewertung mit allen Referenzbildern
    for edge_id, ref_img in enumerate(reference_images):
        score = compute_ssim(frame, ref_img)
        if score > best_score:
            best_score = score
            best_edge = edge_id

    # Wenn der beste Score den Schwellwert überschreitet → speichern
    if best_score > MATCH_THRESHOLD:
        output_path = (
            f"{OUTPUT_DIR}/edge_{best_edge+1}/frame_{str(frame_idx).zfill(5)}.png"
        )
        cv2.imwrite(output_path, frame)
        found_frames[best_edge].append(frame_idx)

# Zusammenfassung
print("Fertig! Beste Frames je Schneidkante gespeichert unter:", OUTPUT_DIR)
