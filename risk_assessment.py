"""
Risk Assessment Engine
======================
Mathematical models for damage severity, impact scoring, and road health index.

Formulas:
    AreaRatio  = DamageArea / ImageArea
    Severity   = AreaRatio × Confidence × ClassWeight
    ImpactScore = AreaRatio × NormalizedSpeed × Confidence × ScaleFactor
    RoadHealthIndex = 1.0 - (sum of severity scores, clamped to [0,1])
"""

from dataclasses import dataclass, field
from typing import List, Optional
from config import (
    RISK_LOW_THRESHOLD, RISK_MODERATE_THRESHOLD,
    IMPACT_SCALE_FACTOR, CLASS_SEVERITY_WEIGHTS,
    ROAD_HEALTH_CATEGORIES
)


@dataclass
class DamageReport:
    """Structured report for a single detected damage instance."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    area_ratio: float
    severity: float
    impact_score: float
    risk_level: str
    estimated_speed: float = 0.0

    def to_dict(self):
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "area_ratio": round(self.area_ratio, 6),
            "severity": round(self.severity, 4),
            "impact_score": round(self.impact_score, 4),
            "risk_level": self.risk_level,
            "estimated_speed": round(self.estimated_speed, 1),
        }


@dataclass
class RoadHealthReport:
    """Aggregate road health assessment for an entire image/frame."""
    total_damages: int = 0
    severe_count: int = 0
    moderate_count: int = 0
    low_count: int = 0
    health_index: float = 1.0
    health_category: str = "Excellent"
    damages: List[DamageReport] = field(default_factory=list)
    dominant_damage: str = "None"

    def to_dict(self):
        return {
            "total_damages": self.total_damages,
            "severe_count": self.severe_count,
            "moderate_count": self.moderate_count,
            "low_count": self.low_count,
            "health_index": round(self.health_index, 4),
            "health_category": self.health_category,
            "dominant_damage": self.dominant_damage,
            "damages": [d.to_dict() for d in self.damages],
        }


# ==============================================================================
# Core Calculation Functions (backward compatible)
# ==============================================================================

def calculate_area_ratio(box_w, box_h, img_w, img_h):
    """
    Calculate the relative damage area ratio.
    Returns: damage_area / image_area
    """
    damage_area = box_w * box_h
    img_area = img_w * img_h
    if img_area == 0:
        return 0
    return damage_area / img_area


def calculate_severity(area_ratio, conf, class_id=0):
    """
    Severity Estimation with optional class-based weighting.
    Severity = AreaRatio × Confidence × ClassWeight
    """
    weight = CLASS_SEVERITY_WEIGHTS.get(class_id, 1.0)
    return area_ratio * conf * weight


def calculate_impact_score(area_ratio, vehicle_speed, conf, class_id=0):
    """
    Impact Score Calculation.
    ImpactScore = AreaRatio × NormalizedSpeed × Confidence × ScaleFactor
    Normalized to [0, 1] range.
    """
    normalized_speed = min(vehicle_speed / 100.0, 1.2)
    weight = CLASS_SEVERITY_WEIGHTS.get(class_id, 1.0)
    score = area_ratio * normalized_speed * conf * weight
    scaled_score = min(score * IMPACT_SCALE_FACTOR, 1.0)
    return scaled_score


def get_risk_level(impact_score):
    """
    Classify risk based on impact score thresholds.
    0 - 0.3  : Low
    0.3 - 0.6: Moderate
    0.6+     : Severe
    """
    if impact_score <= RISK_LOW_THRESHOLD:
        return "Low"
    elif impact_score <= RISK_MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Severe"


# ==============================================================================
# Road Health Index
# ==============================================================================

def calculate_road_health_index(damage_reports: List[DamageReport]) -> float:
    """
    Calculate the Road Health Index (RHI) for an image.
    RHI = 1.0 - sum(severity_scores), clamped to [0, 1].
    1.0 = perfect road, 0.0 = extremely damaged.
    """
    if not damage_reports:
        return 1.0
    total_severity = sum(d.severity for d in damage_reports)
    return max(0.0, min(1.0, 1.0 - total_severity))


def get_health_category(health_index: float) -> str:
    """Map a health index value to a human-readable category."""
    for category, (low, high) in ROAD_HEALTH_CATEGORIES.items():
        if low <= health_index <= high:
            return category
    return "Critical"


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_road_health_report(
    boxes, confs, cls_ids, class_names,
    img_w, img_h, assumed_speed=40.0
) -> RoadHealthReport:
    """
    Generate a complete road health report from detection results.

    Args:
        boxes: List of (x1, y1, x2, y2) bounding boxes
        confs: List of confidence scores
        cls_ids: List of class IDs
        class_names: Dict of class_id -> class_name
        img_w, img_h: Image dimensions
        assumed_speed: Vehicle speed in km/h

    Returns:
        RoadHealthReport with all damage details and health index
    """
    report = RoadHealthReport()
    damage_type_counts = {}

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, cls_ids):
        box_w = x2 - x1
        box_h = y2 - y1
        cls_name = class_names.get(cls_id, f"class_{cls_id}")

        area_ratio = calculate_area_ratio(box_w, box_h, img_w, img_h)
        severity = calculate_severity(area_ratio, conf, cls_id)
        impact = calculate_impact_score(area_ratio, assumed_speed, conf, cls_id)
        risk = get_risk_level(impact)

        damage = DamageReport(
            class_id=cls_id,
            class_name=cls_name,
            confidence=conf,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            area_ratio=area_ratio,
            severity=severity,
            impact_score=impact,
            risk_level=risk,
            estimated_speed=assumed_speed,
        )
        report.damages.append(damage)

        # Count risk levels
        if risk == "Severe":
            report.severe_count += 1
        elif risk == "Moderate":
            report.moderate_count += 1
        else:
            report.low_count += 1

        # Track damage types
        damage_type_counts[cls_name] = damage_type_counts.get(cls_name, 0) + 1

    report.total_damages = len(report.damages)
    report.health_index = calculate_road_health_index(report.damages)
    report.health_category = get_health_category(report.health_index)

    if damage_type_counts:
        report.dominant_damage = max(damage_type_counts, key=damage_type_counts.get)

    return report
