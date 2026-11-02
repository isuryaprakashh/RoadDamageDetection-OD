"""
Model Evaluation & Benchmarking
================================
Evaluate the trained YOLOv8x model and produce a metrics report.

Usage:
    python evaluate.py
    python evaluate.py --model runs/detect/road_damage_yolov8x/weights/best.pt
"""

import argparse
import os
from ultralytics import YOLO
from config import get_model_path, DATASET_YAML, DEVICE, CLASS_NAMES


def evaluate(model_path=None, data_yaml=None):
    """Evaluate the model and print a formatted metrics report."""
    model_path = model_path or get_model_path()
    data_yaml = data_yaml or DATASET_YAML

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("        Train the model first with: python train.py")
        return None

    print("=" * 60)
    print("  📊 Road Damage Detection — Model Evaluation")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Dataset : {data_yaml}")
    print(f"  Device  : {'GPU (CUDA:0)' if isinstance(DEVICE, int) else 'CPU'}")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=DEVICE, verbose=False)

    # Overall metrics
    print("\n┌─────────────────────────────────────────────┐")
    print("│           Overall Performance               │")
    print("├──────────────────┬──────────────────────────┤")
    print(f"│  mAP50           │  {metrics.box.map50:.4f}                   │")
    print(f"│  mAP50-95        │  {metrics.box.map:.4f}                   │")
    print(f"│  Precision       │  {metrics.box.mp:.4f}                   │")
    print(f"│  Recall          │  {metrics.box.mr:.4f}                   │")
    print("└──────────────────┴──────────────────────────┘")

    # Per-class metrics
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│              Per-Class Performance                  │")
    print("├──────────────┬─────────┬─────────┬─────────────────┤")
    print("│ Class        │ mAP50   │ mAP50-95│ Images          │")
    print("├──────────────┼─────────┼─────────┼─────────────────┤")

    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        for i, cls_name in enumerate(CLASS_NAMES):
            if i < len(metrics.box.maps):
                map50_cls = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') else 0
                map_cls = metrics.box.maps[i]
                print(f"│ {cls_name:<12s} │ {map50_cls:.4f}  │ {map_cls:.4f}  │                 │")

    print("└──────────────┴─────────┴─────────┴─────────────────┘")

    # Benchmark comparison table
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│          Performance Benchmark Comparison               │")
    print("├────────────┬─────────┬────────┬────────────────────────┤")
    print("│ Model      │ mAP50   │ FPS*   │ Parameters             │")
    print("├────────────┼─────────┼────────┼────────────────────────┤")
    print("│ YOLOv8n    │ -       │ ~170   │ 3.2M                   │")
    print("│ YOLOv8s    │ -       │ ~130   │ 11.2M                  │")
    print("│ YOLOv8m    │ -       │ ~100   │ 25.9M                  │")
    print("│ YOLOv8l    │ -       │ ~80    │ 43.7M                  │")
    print(f"│ YOLOv8x ★  │ {metrics.box.map50:.4f} │ ~50    │ 68.2M                  │")
    print("└────────────┴─────────┴────────┴────────────────────────┘")
    print("  ★ = Your trained model    * FPS on NVIDIA T4 GPU (batch=1)")

    print(f"\n✅ Evaluation complete.")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="📊 Evaluate Road Damage Detection Model")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(model_path=args.model, data_yaml=args.data)
