# 🔍 Surveillance Anomaly Detection (3D Autoencoder + PyTorch)

This project implements an unsupervised anomaly detection system for surveillance video footage using a 3D Convolutional Autoencoder built with PyTorch. The model is trained to reconstruct only normal behavior, and high reconstruction error indicates potential anomalies.

---

## 🧠 Problem Statement

> Detect unusual activities or behaviors in surveillance video footage using deep learning.

- **Approach:** Unsupervised Learning with Autoencoders  
- **Dataset:** UCSD Pedestrian (Ped1)  
- **Tools:** PyTorch, NumPy, OpenCV, Matplotlib  

---

## 📁 Project Structure

```text
surveillance_anomaly_detection_project/
├── anomaly_detection.py        # Model training & evaluation
├── convert_ucsd.py             # Converts UCSD tif frames to mp4
├── view_results.py             # Visualizes test video + anomaly status
├── print_frame_results.py      # Saves results to frame_results.txt
├── results.npy                 # Saved frame-wise reconstruction errors
├── anomaly_model.pth           # Trained model
├── frame_results.txt           # Output summary (frame-wise labels)
├── requirements.txt            # Dependencies
└── sample_data/
    ├── train/                  # Converted training videos
    └── test/                   # Converted testing videos


---

## ⚙️ Installation

Make sure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt

### 1️⃣ Step 1: Convert the UCSD dataset (only once)

1. Download the **UCSD Ped1 Dataset**  
   🔗 [http://www.svcl.ucsd.edu/projects/anomaly/dataset.html](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)

2. Extract the dataset and place the `Train` and `Test` folders from `UCSDped1` into your project directory.

3. Then run the following command to convert `.tif` image sequences into `.mp4` video clips:

```bash
python convert_ucsd.py
sample_data/
├── train/        # Contains Train001.mp4, Train002.mp4, ...
└── test/         # Contains Test001.mp4, Test002.mp4, ...

### 2️⃣ Step 2: Train the Model

Run the following command to train the model on normal training clips:

```bash
python anomaly_detection.py

##📝 To view frame-by-frame results in plain text:
python print_frame_results.py
##🎥 To visualize test video with anomaly labels overlaid:
python view_results.py
## ✅ Output

| File                | Purpose                                                  |
|---------------------|----------------------------------------------------------|
| `results.npy`        | Reconstruction error scores for each test frame         |
| `frame_results.txt`  | Text summary of anomaly detection (True/False)          |
| `anomaly_model.pth`  | Saved trained model                                     |
| `sample_data/test/`  | Folder containing test videos in `.mp4` format          |

## 📌 Notes

- Threshold value (default: `0.0037`) determines whether a frame is considered **Normal** or **Anomalous**.
- `convert_ucsd.py` only needs to be run **once** during setup to convert `.tif` frames to `.mp4` videos.
- This project uses an **unsupervised learning** approach — no manual anomaly labels required.
- Only **normal surveillance videos** are used during training.

## 💡 Future Improvements

- ✅ Add support for **CCTV Action Recognition** and **ShanghaiTech** datasets  
- ✅ Integrate **real-time webcam-based anomaly detection**  
- ✅ Use **Grad-CAM** or heatmaps to visualize what part of the frame is **abnormal**

## 📷 Example Output
Frame 001: Score=0.0025 → True (Normal)
Frame 056: Score=0.0072 → False (Anomaly)

## 👤 Author

**Mirza Muhammed**  
AI/ML Developer – Surveillance Anomaly Detection System

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) – feel free to use, modify, and distribute it for academic or commercial purposes.

---

✅ You can now copy this and paste it directly into your `README.md`. Let me know if you’d also like:
- `.gitignore` template
- Screenshot section
- Downloadable README file as `.md` or `.txt`

