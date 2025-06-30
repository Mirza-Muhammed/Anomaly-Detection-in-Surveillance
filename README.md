# 🔍 Surveillance Anomaly Detection (3D Autoencoder + PyTorch)

This project implements an unsupervised anomaly detection system for surveillance video footage using a 3D Convolutional Autoencoder built with PyTorch. The model is trained to reconstruct only normal behavior, and high reconstruction error indicates potential anomalies.

---

## 🧠 Problem Statement

> Detect unusual activities or behaviors in surveillance video footage using deep learning.

- **Approach:** Unsupervised Learning with Autoencoders  
- **Dataset:** UCSD Pedestrian (Ped1)  
- **Tools:** PyTorch, NumPy, OpenCV, Matplotlib  

---

# 🎯 Surveillance Anomaly Detection

An unsupervised deep learning-based system to identify unusual activities in surveillance videos.

---

## 📁 Project Structure

```
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
```

---

## ⚙️ Installation

Make sure you have **Python 3.7+** installed.

```bash
pip install -r requirements.txt
```

---

## 🔄 Step 1: Convert UCSD Dataset

1. 📥 Download the **UCSD Ped1 Dataset**  
   🔗 [UCSD Ped1 Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)

2. Extract it and place the `Train` and `Test` folders from `UCSDped1` in your project directory.

3. Run the following command to convert `.tif` image sequences into `.mp4` videos:

```bash
python convert_ucsd.py
```

**Structure after conversion:**

```
sample_data/
├── train/        # Train001.mp4, Train002.mp4, ...
└── test/         # Test001.mp4, Test002.mp4, ...
```

---

## 🧠 Step 2: Train the Model

Run the training script on normal clips:

```bash
python anomaly_detection.py
```

---

## 📄 Optional: View Results

### 📝 Text Output (frame-by-frame):

```bash
python print_frame_results.py
```

### 🎥 Video Output with Overlays:

```bash
python view_results.py
```

---

## ✅ Output Files

| File                 | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `results.npy`        | Reconstruction error scores for each test frame          |
| `frame_results.txt`  | Text summary of anomaly detection (True/False)           |
| `anomaly_model.pth`  | Trained PyTorch model                                    |
| `sample_data/test/`  | Folder containing `.mp4` test videos                     |

---

## 📌 Notes

- 📉 **Threshold**: Default is `0.0037` to classify Normal/Anomalous.
- 🛠 `convert_ucsd.py` is a one-time preprocessing step.
- 🤖 Fully **unsupervised** — no ground-truth anomaly labels needed.
- 🎓 Only **normal video clips** used during training phase.

---

## 🚀 Future Improvements

- 🔄 Add support for **CCTV Action Recognition** & **ShanghaiTech**
- 📡 Integrate **real-time webcam-based anomaly detection**
- 🔥 Use **Grad-CAM** or heatmaps to highlight anomaly regions

---

## 📷 Example Output

```
Frame 001: Score=0.0025 → ✅ Normal  
Frame 056: Score=0.0072 → ❌ Anomaly
```

---

## 👤 Author

**Mirza Muhammed**  
AI/ML Developer – Surveillance Anomaly Detection System

---

## 📜 License

Licensed under the [MIT License](https://opensource.org/licenses/MIT)  
Feel free to use, modify, or distribute for academic/commercial purposes.
