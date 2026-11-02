"""
Real-Time Video Detection Module
=================================
Process video with YOLO tracking, speed estimation, and risk assessment.

Usage:
    python predict_video.py <video_path>
    python predict_video.py <video_path> --output result.mp4 --conf 0.4
"""

import argparse
import math
import os
import sys
import cv2
from ultralytics import YOLO
from config import (
    get_model_path, CONFIDENCE_THRESHOLD, DEVICE,
    PIXEL_TO_METER, DEFAULT_SPEED_KMH, MIN_SPEED_KMH, MAX_SPEED_KMH
)
from risk_assessment import (
    calculate_area_ratio, calculate_severity, calculate_impact_score,
    get_risk_level
)


# Color map for risk levels (BGR)
RISK_COLORS = {
    "Severe": (0, 0, 255),
    "Moderate": (0, 165, 255),
    "Low": (0, 255, 0),
}


def predict_on_video(video_path, model_path=None, output_path=None,
                     confidence=None):
    """
    Process video with YOLO tracking and real-time risk assessment.
    """
    model_path = model_path or get_model_path()
    confidence = confidence or CONFIDENCE_THRESHOLD

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_detected{ext}"

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n🎥 Processing Video: {video_path}")
    print(f"   Resolution: {width}x{height} @ {fps} FPS")
    print(f"   Total Frames: {total_frames}")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Device: {'GPU' if isinstance(DEVICE, int) else 'CPU'}")
    print()

    track_history = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0

        # Print progress every 30 frames
        if frame_count % 30 == 0 or frame_count == 1:
            print(f"   Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

        # Run YOLO tracking
        results = model.track(
            frame, persist=True, conf=confidence,
            tracker="botsort.yaml", verbose=False, device=DEVICE
        )
        result = results[0]
        annotated_frame = result.plot()

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            cls_ids = result.boxes.cls.int().cpu().tolist()

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                box_w = x2 - x1
                box_h = y2 - y1

                # Update track history
                track = track_history.get(track_id, [])
                track.append((cx, cy))
                if len(track) > 30:
                    track.pop(0)
                track_history[track_id] = track

                # Speed estimation (CV-based)
                estimated_speed = DEFAULT_SPEED_KMH
                if len(track) >= 5:
                    dx = track[-1][0] - track[0][0]
                    dy = track[-1][1] - track[0][1]
                    pixel_dist = math.hypot(dx, dy)
                    time_elapsed = len(track) / fps if fps > 0 else 1
                    meters_per_sec = (pixel_dist * PIXEL_TO_METER) / time_elapsed
                    estimated_speed = meters_per_sec * 3.6

                estimated_speed = max(MIN_SPEED_KMH, min(estimated_speed, MAX_SPEED_KMH))

                # Risk calculations
                area_ratio = calculate_area_ratio(box_w, box_h, width, height)
                severity = calculate_severity(area_ratio, conf, cls_id)
                impact = calculate_impact_score(area_ratio, estimated_speed, conf, cls_id)
                risk = get_risk_level(impact)

                # Draw labels
                color = RISK_COLORS.get(risk, (255, 255, 255))
                label = f"Spd:{estimated_speed:.0f}km/h | Impact:{impact:.2f} ({risk})"
                cv2.putText(annotated_frame, label,
                            (x1, max(y1 - 25, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw tracking trail
                if len(track) > 2:
                    pts = [(int(p[0]), int(p[1])) for p in track]
                    for i in range(1, len(pts)):
                        cv2.line(annotated_frame, pts[i-1], pts[i], color, 2)

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"\n✅ Processed {frame_count} frames.")
    print(f"   Output saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="🎥 Road Damage Video Detection")
    parser.add_argument("video", type=str, nargs="?",
                        default=None, help="Path to input video")
    parser.add_argument("--model", type=str, default=None, help="Model weights path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    video_path = args.video
    if video_path is None:
        print("[ERROR] Please provide a video path: python predict_video.py <video.mp4>")
        sys.exit(1)

    predict_on_video(
        video_path=video_path,
        model_path=args.model,
        output_path=args.output,
        confidence=args.conf,
    )
