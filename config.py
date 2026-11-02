"""
Centralized Configuration for Road Damage Detection System
===========================================================
Auto-detects GPU availability and provides all project-wide constants.
"""

import os
import torch

# ==============================================================================
# Device Configuration
# ==============================================================================
def get_device(verbose=False):
    """Auto-detect the best available device for inference/training."""
    if torch.cuda.is_available():
        if verbose:
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[GPU]  Using CUDA device: {device_name} ({vram:.1f} GB VRAM)")
        return 0  # CUDA device index
    else:
        if verbose:
            print("[CPU]  No CUDA GPU detected — falling back to CPU.")
            print("       Training will be significantly slower on CPU.")
        return "cpu"

DEVICE = get_device()

# ==============================================================================
# Project Paths
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATASET_YAML = os.path.join(PROJECT_ROOT, "road_damage.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Model weights — search multiple locations for trained model
MODEL_VARIANT = "yolov8n"  # Light model for new training
USER_RUNS_DIR = os.path.join(os.path.expanduser("~"), "runs")

# All possible trained model locations (checked in order)
TRAINED_MODEL_PATHS = [
    os.path.join(RUNS_DIR, "detect", "road_damage_yolov8n", "weights", "best.pt"),   # New trained model (project dir)
    os.path.join(USER_RUNS_DIR, "detect", "road_damage_yolov8n", "weights", "best.pt"),  # New trained model (user dir)
    os.path.join(RUNS_DIR, "detect", "road_damage_yolov8x", "weights", "best.pt"),   # Old trained model (fallback)
    os.path.join(USER_RUNS_DIR, "detect", "road_damage_yolov8x", "weights", "best.pt"),  # Old (user dir) 
]

PRETRAINED_MODEL = os.path.join(PROJECT_ROOT, f"{MODEL_VARIANT}.pt")

def get_model_path():
    """Return the best available model path (trained > pretrained > download)."""
    for path in TRAINED_MODEL_PATHS:
        if os.path.exists(path):
            return path
    if os.path.exists(PRETRAINED_MODEL):
        print(f"[WARN] No trained model found. Using pretrained {MODEL_VARIANT}.pt")
        return PRETRAINED_MODEL
    else:
        print(f"[WARN] No local model found. Ultralytics will download {MODEL_VARIANT}.pt")
        return f"{MODEL_VARIANT}.pt"

# Classes — matches the Indian roads dataset (dataset-1)
CLASS_NAMES = ["HMV", "LMV", "Pedestrian", "RoadDamages", "SpeedBump", "UnsurfacedRoad"]
NUM_CLASSES = len(CLASS_NAMES)

# Class-specific severity weights
CLASS_SEVERITY_WEIGHTS = {
    0: 0.3,    # HMV (Heavy Motor Vehicle) — low direct road risk
    1: 0.2,    # LMV (Light Motor Vehicle) — low direct road risk
    2: 0.1,    # Pedestrian — minimal road risk
    3: 1.0,    # RoadDamages — highest risk
    4: 0.5,    # SpeedBump — moderate risk
    5: 0.8,    # UnsurfacedRoad — high risk
}

# ==============================================================================
# Training Defaults
# ==============================================================================
TRAIN_DEFAULTS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16 if isinstance(DEVICE, int) else 4,  # Larger batch on GPU
    "optimizer": "auto",
    "lr0": 0.01,
    "patience": 20,       # Early stopping patience
    "save_period": 10,    # Save checkpoint every N epochs
    "name": "road_damage_yolov8n",
}

# ==============================================================================
# Inference Defaults
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 100

# ==============================================================================
# Risk Assessment Parameters
# ==============================================================================
# Impact score boundaries
RISK_LOW_THRESHOLD = 0.3
RISK_MODERATE_THRESHOLD = 0.6

# Speed estimation calibration
PIXEL_TO_METER = 0.05       # 1 pixel ≈ 0.05 meters (requires camera calibration)
DEFAULT_SPEED_KMH = 40.0    # Default assumed vehicle speed
MIN_SPEED_KMH = 10.0
MAX_SPEED_KMH = 120.0

# Impact score scaling factor (compensates for small bbox area ratios)
IMPACT_SCALE_FACTOR = 20.0

# ==============================================================================
# Road Health Index
# ==============================================================================
ROAD_HEALTH_CATEGORIES = {
    "Excellent": (0.8, 1.0),
    "Good": (0.6, 0.8),
    "Fair": (0.4, 0.6),
    "Poor": (0.2, 0.4),
    "Critical": (0.0, 0.2),
}

# ==============================================================================
# Dashboard Theme Colors
# ==============================================================================
THEME = {
    "bg_primary": "#0a0a1a",
    "bg_secondary": "#12122a",
    "bg_card": "rgba(25, 25, 60, 0.8)",
    "accent": "#6c5ce7",
    "accent_light": "#a29bfe",
    "danger": "#ff4757",
    "warning": "#ffa502",
    "success": "#2ed573",
    "text_primary": "#ffffff",
    "text_secondary": "#b0b0d0",
    "gradient_start": "#6c5ce7",
    "gradient_end": "#00cec9",
}
