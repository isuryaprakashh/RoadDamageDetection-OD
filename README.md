# 🚧 AI-Based Intelligent Road Damage Detection & Risk Assessment System

> Real-time road damage detection with severity classification, impact scoring, and road health assessment powered by **YOLOv8n + PyTorch**.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 Static Image Detection | Upload images for road damage detection |
| 🎥 Real-Time Video Tracking | Process dashcam videos with BoT-SORT tracking |
| 🚗 Vehicle Speed Estimation | CV-based speed estimation across frames |
| 💥 Impact Score Calculation | Risk scoring (AreaRatio × Speed × Confidence) |
| 🏥 Road Health Index | Aggregate health score per image (0–1 scale) |
| 📊 Analytics Dashboard | Plotly charts for damage distribution & severity |
| 🗺️ GPS Damage Mapping | Pin damage locations on interactive Folium maps |
| ⚡ GPU Acceleration | Auto CUDA detection for fast inference |

## 📊 Model Performance

Trained on **5,843 images** from Indian roads, validated on **1,288 images**.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **Overall** | **0.814** | **0.761** | **0.816** | **0.462** |
| HMV | 0.910 | 0.966 | 0.978 | 0.720 |
| LMV | 0.891 | 0.932 | 0.957 | 0.616 |
| Pedestrian | 0.823 | 0.802 | 0.856 | 0.446 |
| RoadDamages | 0.683 | 0.462 | 0.542 | 0.218 |
| SpeedBump | 0.726 | 0.831 | 0.860 | 0.426 |
| UnsurfacedRoad | 0.852 | 0.574 | 0.701 | 0.347 |

**Inference Speed:** 3.0ms/image on NVIDIA RTX 3050 Laptop GPU

## 🗂️ Project Structure

```
Road-Damage-Detection/
├── app.py                 # Streamlit dashboard (main entry)
├── config.py              # Centralized configuration & GPU detection
├── train.py               # GPU-accelerated YOLOv8n training
├── evaluate.py            # Model evaluation & benchmarking
├── predict_image.py       # Static image detection (CLI)
├── predict_video.py       # Video detection with tracking (CLI)
├── risk_assessment.py     # Impact scoring & Road Health Index
├── road_damage.yaml       # Dataset configuration
├── requirements.txt       # Python dependencies
├── dataset-1/
│   └── images/
│       ├── data.yaml      # Original dataset config
│       ├── train/         # 5,843 training images + labels
│       ├── valid/         # 1,288 validation images + labels
│       └── test/          # Test images + labels
└── runs/
    └── detect/
        └── road_damage_yolov8n/
            └── weights/
                └── best.pt   # Trained model (6.2MB)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run app.py
```

### 3. Train Model (GPU)
```bash
python train.py --epochs 50 --batch 16
python train.py --resume   # Resume from checkpoint
```

### 4. Evaluate Model
```bash
python evaluate.py
```

### 5. CLI Prediction
```bash
python predict_image.py path/to/image.jpg
python predict_video.py path/to/video.mp4
```

## 📐 Mathematical Models

| Formula | Equation |
|---------|----------|
| Area Ratio | `DamageArea / ImageArea` |
| Severity | `AreaRatio × Confidence × ClassWeight` |
| Impact Score | `AreaRatio × (Speed/100) × Confidence × 20` |
| Road Health Index | `1.0 - Σ(severity)`, clamped to [0, 1] |

### Risk Classification
| Score | Risk Level |
|-------|-----------|
| 0 – 0.3 | 🟢 Low |
| 0.3 – 0.6 | 🟡 Moderate |
| 0.6+ | 🔴 Severe |

## 📊 Detection Classes

| ID | Class | Severity Weight | Description |
|----|-------|----------------|-------------|
| 0 | HMV | 0.3 | Heavy Motor Vehicle |
| 1 | LMV | 0.2 | Light Motor Vehicle |
| 2 | Pedestrian | 0.1 | Pedestrian on road |
| 3 | RoadDamages | 1.0 | Potholes, cracks, road damage |
| 4 | SpeedBump | 0.5 | Speed breakers |
| 5 | UnsurfacedRoad | 0.8 | Unpaved/damaged road surface |

## 🛠️ Tech Stack

- **Detection:** Ultralytics YOLOv8n (3M params, 6.2MB)
- **Framework:** PyTorch 2.5.1 (CUDA 12.1)
- **Python:** 3.12.4
- **Ultralytics:** 8.4.22
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB)
- **Dashboard:** Streamlit + Plotly
- **Tracking:** BoT-SORT
- **Mapping:** Folium
- **CV:** OpenCV

## ⚙️ Configuration

All settings are centralized in `config.py`:
- GPU/CPU auto-detection
- Model paths & class names
- Risk thresholds & calibration constants
- Dashboard theme colors
