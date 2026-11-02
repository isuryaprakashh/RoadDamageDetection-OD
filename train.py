"""
YOLOv8x Training Script — GPU-Accelerated
==========================================
Train the road damage detection model with automatic GPU detection.

Usage:
    python train.py
    python train.py --epochs 50 --batch 32 --resume
"""

import argparse
import sys
import torch
from ultralytics import YOLO
from config import (
    DEVICE, DATASET_YAML, PRETRAINED_MODEL,
    TRAIN_DEFAULTS, MODEL_VARIANT, get_model_path
)
import os


def print_system_info():
    """Print system and GPU information."""
    print("=" * 60)
    print("  🚧 Road Damage Detection — Training Pipeline")
    print("=" * 60)
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device      : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Memory      : {vram:.1f} GB")
        print(f"  CUDA Version    : {torch.version.cuda}")
    print(f"  Device Selected : {'GPU (CUDA:0)' if isinstance(DEVICE, int) else 'CPU'}")
    print("=" * 60)


def train(args):
    """Run YOLOv8n training with the specified parameters."""
    print_system_info()

    # Determine base model
    existing_model = get_model_path()
    if args.resume and "best.pt" in existing_model:
        print(f"\n[RESUME] Resuming training from: {existing_model}")
        model = YOLO(existing_model)
    else:
        base_model = PRETRAINED_MODEL if os.path.exists(PRETRAINED_MODEL) else f"{MODEL_VARIANT}.pt"
        print(f"\n[NEW] Starting fresh training with: {base_model}")
        model = YOLO(base_model)

    # Configure training parameters
    train_params = {
        "data": args.data or DATASET_YAML,
        "epochs": args.epochs or TRAIN_DEFAULTS["epochs"],
        "imgsz": args.imgsz or TRAIN_DEFAULTS["imgsz"],
        "batch": args.batch or TRAIN_DEFAULTS["batch"],
        "device": DEVICE,
        "project": os.path.join(os.path.dirname(DATASET_YAML), "runs", "detect"),
        "name": TRAIN_DEFAULTS["name"],
        "patience": TRAIN_DEFAULTS["patience"],
        "save_period": TRAIN_DEFAULTS["save_period"],
        "exist_ok": True,  # Overwrite existing run
        "verbose": True,
        "plots": True,     # Generate training plots
    }

    print("\n[CONFIG] Training Parameters:")
    for k, v in train_params.items():
        print(f"  {k:15s} : {v}")
    print()

    # Start training
    results = model.train(**train_params)

    # Evaluate after training
    print("\n" + "=" * 60)
    print("  📊 Post-Training Evaluation")
    print("=" * 60)
    metrics = model.val()
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("=" * 60)

    # Export model to ONNX for deployment (optional)
    if args.export:
        print("\n[EXPORT] Exporting model to ONNX format...")
        model.export(format="onnx")
        print("[EXPORT] Done.")

    saved_path = os.path.join(train_params["project"], train_params["name"], "weights", "best.pt")
    print(f"\n✅ Training complete! Best weights saved to:")
    print(f"   {saved_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚧 Train YOLOv8x for Road Damage Detection"
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of training epochs (default: {TRAIN_DEFAULTS['epochs']})")
    parser.add_argument("--batch", type=int, default=None,
                        help=f"Batch size (default: {TRAIN_DEFAULTS['batch']})")
    parser.add_argument("--imgsz", type=int, default=None,
                        help=f"Image size (default: {TRAIN_DEFAULTS['imgsz']})")
    parser.add_argument("--data", type=str, default=None,
                        help=f"Dataset YAML path (default: {DATASET_YAML})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--export", action="store_true",
                        help="Export trained model to ONNX after training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
