"""
Static Image Prediction Module
================================
Run road damage detection on a single image with full risk assessment.

Usage:
    python predict_image.py <image_path>
    python predict_image.py <image_path> --output result.jpg --conf 0.5
"""

import argparse
import json
import os
import sys
import cv2
from ultralytics import YOLO
from config import get_model_path, CONFIDENCE_THRESHOLD, DEVICE
from risk_assessment import (
    calculate_area_ratio, calculate_severity, calculate_impact_score,
    get_risk_level, generate_road_health_report
)


# Color map for risk levels (BGR)
RISK_COLORS = {
    "Severe": (0, 0, 255),      # Red
    "Moderate": (0, 165, 255),   # Orange
    "Low": (0, 255, 0),          # Green
}


def predict_on_image(image_path, model_path=None, output_path=None,
                     confidence=None, assumed_speed=40.0, save_json=True):
    """
    Run detection on a single image and produce annotated output + report.
    
    Args:
        image_path: Path to input image
        model_path: Path to model weights (auto-detected if None)
        output_path: Path for annotated output image
        confidence: Detection confidence threshold
        assumed_speed: Assumed vehicle speed in km/h
        save_json: Whether to save a JSON report alongside the image
    
    Returns:
        RoadHealthReport object
    """
    model_path = model_path or get_model_path()
    confidence = confidence or CONFIDENCE_THRESHOLD

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("        Train the model first: python train.py")
        return None

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detected{ext}"

    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None

    img_h, img_w = img.shape[:2]

    print(f"\n🔍 Analyzing: {image_path}")
    print(f"   Image size: {img_w}x{img_h}")
    print(f"   Confidence: {confidence}")
    print(f"   Model: {os.path.basename(model_path)}")

    # Run prediction
    results = model.predict(image_path, conf=confidence, device=DEVICE, verbose=False)
    result = results[0]

    annotated_img = result.plot()

    # Extract detections
    boxes_list = []
    confs_list = []
    cls_ids_list = []

    for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes_list.append((x1, y1, x2, y2))
        confs_list.append(conf)
        cls_ids_list.append(cls_id)

    # Generate health report
    report = generate_road_health_report(
        boxes=boxes_list,
        confs=confs_list,
        cls_ids=cls_ids_list,
        class_names=model.names,
        img_w=img_w,
        img_h=img_h,
        assumed_speed=assumed_speed,
    )

    # Draw custom risk labels on image
    for damage in report.damages:
        x1, y1, x2, y2 = damage.bbox
        color = RISK_COLORS.get(damage.risk_level, (255, 255, 255))

        label = f"Sev:{damage.severity:.2f} | Impact:{damage.impact_score:.2f} ({damage.risk_level})"
        cv2.putText(annotated_img, label, (x1, max(y1 - 25, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated image
    cv2.imwrite(output_path, annotated_img)

    # Print report
    print(f"\n{'='*50}")
    print(f"  📋 Road Health Report")
    print(f"{'='*50}")
    print(f"  Total Damages  : {report.total_damages}")
    print(f"  Severe         : {report.severe_count}")
    print(f"  Moderate       : {report.moderate_count}")
    print(f"  Low            : {report.low_count}")
    print(f"  Health Index   : {report.health_index:.4f}")
    print(f"  Health Status  : {report.health_category}")
    print(f"  Dominant Type  : {report.dominant_damage}")
    print(f"{'='*50}")

    for i, d in enumerate(report.damages):
        print(f"  [{d.class_name}] Conf:{d.confidence:.2f} "
              f"Area:{d.area_ratio:.4f} Sev:{d.severity:.4f} "
              f"Impact:{d.impact_score:.2f} Risk:{d.risk_level}")

    print(f"\n✅ Annotated image saved: {output_path}")

    # Save JSON report
    if save_json:
        json_path = os.path.splitext(output_path)[0] + "_report.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"📄 JSON report saved: {json_path}")

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="🔍 Road Damage Image Detection")
    parser.add_argument("image", type=str, nargs="?",
                        default=None, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Model weights path")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--speed", type=float, default=40.0, help="Assumed vehicle speed (km/h)")
    parser.add_argument("--no-json", action="store_true", help="Skip JSON report output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_path = args.image
    if image_path is None:
        # Default to first image in train set
        train_dir = os.path.join("dataset", "train")
        if os.path.exists(train_dir):
            images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                image_path = os.path.join(train_dir, images[0])
                print(f"[INFO] No image specified. Using: {image_path}")

    if image_path is None:
        print("[ERROR] Please provide an image path: python predict_image.py <image.jpg>")
        sys.exit(1)

    predict_on_image(
        image_path=image_path,
        model_path=args.model,
        output_path=args.output,
        confidence=args.conf,
        assumed_speed=args.speed,
        save_json=not args.no_json,
    )
