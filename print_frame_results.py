# print_frame_results.py
import numpy as np

# Load results
scores = np.load("results.npy")

# If threshold.npy exists, load it, else use a default
try:
    threshold = np.load("threshold.npy")
    print(f"[✔] Using saved threshold: {threshold:.4f}")
except FileNotFoundError:
    threshold = 0.0037  # Default fallback
    print(f"[!] Using default threshold: {threshold:.4f}")

print("\nSample Results:")

# Print first 50 results for review
for i, score in enumerate(scores[:50], start=1):
    is_normal = score < threshold
    print(f"Frame {i:04}: Score={score:.4f} -> {'True (Normal)' if is_normal else 'False (Anomaly)'}")

# Save all results to a file
with open("frame_results.txt", "w") as f:
    for i, score in enumerate(scores, start=1):
        is_normal = score < threshold
        f.write(f"Frame {i:04}: Score={score:.4f} -> {'True (Normal)' if is_normal else 'False (Anomaly)'}\n")

print("\n✅ All results saved to frame_results.txt")
