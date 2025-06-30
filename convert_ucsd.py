import os
import cv2

def convert_sequence_to_video(input_folder, output_path, fps=10):
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.tif')])
    if not frames:
        print(f"No .tif files found in {input_folder}")
        return

    sample_frame = cv2.imread(os.path.join(input_folder, frames[0]))
    height, width, _ = sample_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frames:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"[✔] Saved: {output_path}")

def process_ucsd_folders(base_folder, mode):
    output_folder = os.path.join("sample_data", mode)
    os.makedirs(output_folder, exist_ok=True)
    folder_path = os.path.join(base_folder, mode.capitalize())
    if not os.path.exists(folder_path):
        print(f"Missing folder: {folder_path}")
        return

    for sequence_folder in sorted(os.listdir(folder_path)):
        sequence_path = os.path.join(folder_path, sequence_folder)
        if os.path.isdir(sequence_path):
            output_video = os.path.join(output_folder, f"{sequence_folder}.mp4")
            convert_sequence_to_video(sequence_path, output_video)

if __name__ == "__main__":
    for category in ["train", "test"]:
        process_ucsd_folders(".", category)
