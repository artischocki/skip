import cv2
from pathlib import Path

# --- Parameters & Paths ---
# Video location
video_name = "video0000031"
base_dir = Path(__file__).parents[2] / "data" / "test" / "Video"
extracted_frames_dir = base_dir / video_name
extracted_frames_dir.mkdir(exist_ok=True)
VIDEO_PATH = str(base_dir / f"{video_name}.avi")


# --- 1. Load video frames ---
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
# first frame is 17 and then every 60th frame is a viable frame
ff = 17
idx = -1
while True:
    idx += 1
    ret, frame = cap.read()
    if not ret:
        break
    if not (idx - ff) % 60 == 0:
        continue
    print(idx)
    frames.append(frame)
    filename = str(extracted_frames_dir / f"frame_{idx:04d}.png")
    cv2.imwrite(filename, frame)
cap.release()
print(f"Extracted {len(frames)} frames from {VIDEO_PATH}")
