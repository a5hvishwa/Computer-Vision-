# 🖼️ Computer Vision Lab Experiments

A collection of **10 Computer Vision experiments** implemented in Python using OpenCV, TensorFlow/Keras, and Ultralytics YOLOv8 — built for a college practical examination.

---

## 📋 Experiments Overview

| # | Experiment | Key Topics |
|---|-----------|-----------|
| 1 | **Basic Image Processing** | Grayscale, HSV, LAB, resize, rotate, flip, negative, ROI |
| 2 | **Image Enhancement – Histogram Techniques** | Linear contrast stretching, global equalization, CLAHE |
| 3 | **Image Smoothing & Edge Detection** | Gaussian, median, average blur; Laplacian, Sobel |
| 4 | **Discrete Fourier Transform** | DFT, frequency shift, low-pass & high-pass filters |
| 5 | **Feature Extraction & Matching** | SIFT keypoints, HOG features, BF k-NN matching, Lowe's ratio test |
| 6 | **Image Segmentation** | Global/Otsu/Adaptive thresholding, Watershed, K-Means |
| 7 | **Motion Tracking – Optical Flow** | Shi–Tomasi, Lucas–Kanade sparse flow, Farnebäck dense flow |
| 8 | **Object Tracking & Motion Analysis** | LK tracking, trajectories, direction/magnitude, performance comparison |
| 9 | **Object Detection – YOLOv8** | YOLOv8 Nano pretrained, bounding boxes, class names, confidence scores |
| 10 | **Handwritten Digit Classification – CNN** | MNIST, CNN architecture, training curves, confusion matrix |

---

## 🗂️ Project Structure

```
computer-vision/
├── experiment1/
│   └── experiment1.py
├── experiment2/
│   └── experiment2.py
├── experiment3/
│   └── experiment3.py
├── experiment4/
│   └── experiment4.py
├── experiment5/
│   └── experiment5.py
├── experiment6/
│   └── experiment6.py
├── experiment7/
│   └── experiment7.py
├── experiment8/
│   └── experiment8.py
├── experiment9/
│   └── experiment9.py
├── experiment10/
│   └── experiment10.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Requirements

### Python version
Python 3.10+

### Install all dependencies at once
```bash
pip install -r requirements.txt
```

### Or install individually
```bash
pip install opencv-python numpy matplotlib scikit-learn scikit-image
pip install ultralytics          # Experiment 9 (YOLOv8)
pip install tensorflow seaborn   # Experiment 10 (CNN)
```

> **Kali / Debian users** — if you see `externally-managed-environment`, add `--break-system-packages`:
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```

---

## ▶️ How to Run Each Experiment

Every experiment is a self-contained Python script. Run from its own folder:

```bash
# Example – Experiment 1
cd experiment1
python3 experiment1.py

# Example – Experiment 6
cd experiment6
python3 experiment6.py

# Example – Experiment 10 (CNN)
cd experiment10
python3 experiment10.py
```

### Experiments that need an image (1–6, 9)
A **file picker** opens automatically. Press **F** to browse or **P** to type the path.

### Experiments that need a video (7, 8)
A **file picker** opens automatically. Press **F** to browse, **P** to type the path, or **W** for webcam.

### Experiment 9 (YOLOv8)
Downloads `yolov8n.pt` (~6 MB) automatically on first run — requires internet.

### Experiment 10 (CNN / MNIST)
Downloads the MNIST dataset (~11 MB) automatically on first run — requires internet.

---

## 📦 Output Files

Each experiment saves results to an **`output/`** folder next to the input image/video, or to a dedicated results directory (Experiment 10 → `experiment_10_results/`).

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| `opencv-python` | Image/video I/O, all CV operations |
| `numpy` | Numerical array operations |
| `matplotlib` | Visualization and figure saving |
| `scikit-learn` | K-Means, classification report, confusion matrix |
| `scikit-image` | HOG feature extraction |
| `ultralytics` | YOLOv8 object detection |
| `tensorflow` / `keras` | CNN model building and training |
| `seaborn` | Confusion matrix heatmap |

---

## 👤 Author

**Vishwas** — Computer Vision Lab, Antigravity IDE
# Computer-Vision-
