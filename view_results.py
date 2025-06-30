import numpy as np
import cv2
import os

video_path = 'sample_data/test/Test001.mp4'
results_path = 'results.npy'
threshold = 0.004

# Load anomaly scores
if not os.path.exists(results_path):
    print(f"[❌] {results_path} not found!")
    exit()
scores = np.load(results_path)
print(f"[✔] Loaded {len(scores)} scores from {results_path}")

# Open video
if not os.path.exists(video_path):
    print(f"[❌] Video {video_path} not found!")
    exit()
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[❌] Failed to open video: {video_path}")
    exit()
else:
    print(f"[✔] Video opened successfully.")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_id >= len(scores):
        print(f"[ℹ] Finished showing {frame_id} frames.")
        break

    score = scores[frame_id]
    is_normal = score < threshold
    label = "Normal" if is_normal else "Anomaly"
    color = (0, 255, 0) if is_normal else (0, 0, 255)

    frame_resized = cv2.resize(frame, (640, 480))
    cv2.putText(frame_resized, f"Frame {frame_id+1}, Score={score:.4f} -> {label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Anomaly Detection Frame Viewer", frame_resized)

    key = cv2.waitKey(30)
    if key == ord('q'):
        print("[✔] Quit pressed by user.")
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
