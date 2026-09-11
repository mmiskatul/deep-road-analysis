"""
Road Analyzer Module - Deep Road Scene Perception and Analytics
Powered by custom YOLO object detection.
"""

import os
import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO


# Category groupings
CATEGORY_MAP = {
    "person": "Vulnerable User",
    "rider": "Vulnerable User",
    "bike": "Two-Wheeler",
    "motor": "Two-Wheeler",
    "car": "Vehicle",
    "bus": "Heavy Vehicle",
    "truck": "Heavy Vehicle",
    "traffic light": "Infrastructure",
    "traffic sign": "Infrastructure",
    "train": "Rail Transit",
}

# Aesthetic modern color palette (BGR for OpenCV)
CLASS_COLORS = {
    "person": (90, 245, 120),       # Mint Green
    "rider": (40, 220, 180),        # Turquoise
    "bike": (230, 80, 240),         # Pink/Purple
    "motor": (210, 40, 220),        # Vivid Magenta
    "car": (245, 195, 30),          # Electric Cyan
    "bus": (30, 150, 255),          # Vivid Orange
    "truck": (0, 120, 255),         # Deep Orange-Red
    "traffic light": (20, 220, 245),# Amber/Yellow
    "traffic sign": (255, 140, 20), # Sky Azure
    "train": (180, 120, 255),       # Lavender
}

DEFAULT_COLOR = (200, 200, 200)


class RoadAnalyzer:
    """
    Analyzes road traffic scenes, detects road participants, evaluates congestion,
    computes collision/proximity risk, and generates high-fidelity visual HUD overlays.
    """

    def __init__(self, model_path: str = "best.pt", device: str = "cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def analyze(
        self,
        image_input: Any,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        track: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs YOLO detection (or tracking) and performs deep scene analytics.

        :param image_input: Filepath string or numpy array (BGR).
        :param conf: Confidence threshold (0.0 - 1.0).
        :param iou: NMS IoU threshold.
        :param imgsz: Image inference size.
        :param track: Enable ByteTrack object tracking across video frames.
        :return: Comprehensive analytics dictionary.
        """
        if track:
            results = self.model.track(
                source=image_input,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                source=image_input,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

        res = results[0]
        orig_img = res.orig_img
        height, width = orig_img.shape[:2]

        detections = []
        counts_by_class = {name: 0 for name in self.class_names.values()}
        counts_by_category = {
            "Vehicle": 0,
            "Heavy Vehicle": 0,
            "Two-Wheeler": 0,
            "Vulnerable User": 0,
            "Infrastructure": 0,
            "Rail Transit": 0,
        }

        # Define ego-vehicle danger zone
        proximity_alerts = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                class_name = self.class_names.get(cls_id, str(cls_id))

                track_id = int(box.id[0].item()) if (box.id is not None and len(box.id) > 0) else None

                category = CATEGORY_MAP.get(class_name, "Other")
                counts_by_class[class_name] = counts_by_class.get(class_name, 0) + 1
                counts_by_category[category] = counts_by_category.get(category, 0) + 1

                # Calculate spatial metrics
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                box_width = x2 - x1
                box_height = y2 - y1
                box_area_ratio = (box_width * box_height) / (width * height)

                # Distance estimation proxy
                proximity_score = (y2 / height) * 0.7 + min(1.0, box_area_ratio * 10) * 0.3

                # Hazard assessment
                is_centered = (0.28 * width) <= cx <= (0.72 * width)
                is_close = y2 >= (0.62 * height)

                is_hazard = False
                hazard_level = "NONE"

                if is_close and is_centered and category in ["Vehicle", "Heavy Vehicle", "Two-Wheeler", "Vulnerable User"]:
                    if y2 >= (0.78 * height) or box_area_ratio > 0.08:
                        hazard_level = "CRITICAL"
                        is_hazard = True
                    elif y2 >= (0.62 * height):
                        hazard_level = "CAUTION"
                        is_hazard = True

                detection_item = {
                    "id": len(detections) + 1,
                    "track_id": track_id,
                    "class": class_name,
                    "category": category,
                    "confidence": round(confidence, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                    "proximity_score": round(proximity_score, 3),
                    "is_hazard": is_hazard,
                    "hazard_level": hazard_level,
                }
                detections.append(detection_item)

                if is_hazard:
                    proximity_alerts.append({
                        "object_id": detection_item["id"],
                        "track_id": track_id,
                        "class": class_name,
                        "hazard_level": hazard_level,
                        "confidence": detection_item["confidence"],
                        "bbox": detection_item["bbox"],
                    })

        # Calculate high-level metrics
        total_vehicles = counts_by_category["Vehicle"] + counts_by_category["Heavy Vehicle"] + counts_by_category["Two-Wheeler"]
        total_vulnerable = counts_by_category["Vulnerable User"]
        total_infrastructure = counts_by_category["Infrastructure"]
        total_objects = len(detections)

        # Traffic Density Classification
        if total_vehicles <= 4:
            traffic_density = "FREE FLOW"
            density_score = 25
        elif total_vehicles <= 10:
            traffic_density = "MODERATE"
            density_score = 55
        elif total_vehicles <= 18:
            traffic_density = "HEAVY"
            density_score = 80
        else:
            traffic_density = "CONGESTED"
            density_score = 98

        # Overall Road Safety Score
        critical_count = sum(1 for a in proximity_alerts if a["hazard_level"] == "CRITICAL")
        caution_count = sum(1 for a in proximity_alerts if a["hazard_level"] == "CAUTION")

        safety_score = 100
        safety_score -= critical_count * 30
        safety_score -= caution_count * 15
        if traffic_density in ["HEAVY", "CONGESTED"]:
            safety_score -= 15
        safety_score = max(5, min(100, safety_score))

        if critical_count > 0:
            safety_status = "CRITICAL ALERT"
        elif caution_count > 0:
            safety_status = "PROXIMITY WARNING"
        elif safety_score >= 80:
            safety_status = "NORMAL / SAFE"
        else:
            safety_status = "MODERATE ATTENTION"

        analytics = {
            "scene_dimensions": {"width": width, "height": height},
            "summary": {
                "total_objects": total_objects,
                "total_vehicles": total_vehicles,
                "vulnerable_road_users": total_vulnerable,
                "infrastructure_elements": total_infrastructure,
                "traffic_density": traffic_density,
                "density_index": density_score,
                "road_safety_score": safety_score,
                "safety_status": safety_status,
                "critical_hazards": critical_count,
                "caution_hazards": caution_count,
            },
            "counts_by_category": counts_by_category,
            "counts_by_class": {k: v for k, v in counts_by_class.items() if v > 0},
            "proximity_alerts": proximity_alerts,
            "detections": detections,
        }

        return analytics

    def render_hud(
        self,
        image: np.ndarray,
        analytics: Dict[str, Any],
        show_danger_zone: bool = True,
        show_hud_banner: bool = True,
    ) -> np.ndarray:
        """
        Renders a futuristic, high-aesthetic Heads-Up Display (HUD) on the image.
        """
        annotated = image.copy()
        height, width = annotated.shape[:2]

        # 1. Draw Ego-Vehicle Danger Corridor
        if show_danger_zone:
            overlay = annotated.copy()
            pts = np.array([
                [int(width * 0.40), int(height * 0.58)],
                [int(width * 0.60), int(height * 0.58)],
                [int(width * 0.85), int(height * 0.98)],
                [int(width * 0.15), int(height * 0.98)],
            ], np.int32)
            cv2.fillPoly(overlay, [pts], (20, 80, 20))
            
            if analytics["summary"]["critical_hazards"] > 0:
                cv2.polylines(annotated, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.polylines(annotated, [pts], True, (0, 255, 120), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # 2. Draw Sleek Cyber Bounding Boxes with Corner Accents
        # Draw subtle box tints first
        box_overlay = annotated.copy()
        for det in analytics["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            is_hazard = det["is_hazard"]
            hazard_level = det["hazard_level"]

            if is_hazard and hazard_level == "CRITICAL":
                box_color = (0, 0, 255)
            elif is_hazard and hazard_level == "CAUTION":
                box_color = (0, 140, 255)
            else:
                box_color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            # Soft tinted fill
            cv2.rectangle(box_overlay, (x1, y1), (x2, y2), box_color, -1)

        cv2.addWeighted(box_overlay, 0.08, annotated, 0.92, 0, annotated)

        # Draw box outlines, corner brackets and badges
        for det in analytics["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            conf = det["confidence"]
            track_id = det.get("track_id")
            is_hazard = det["is_hazard"]
            hazard_level = det["hazard_level"]

            if is_hazard and hazard_level == "CRITICAL":
                box_color = (0, 0, 255)
                c_thick = 3
            elif is_hazard and hazard_level == "CAUTION":
                box_color = (0, 140, 255)
                c_thick = 2
            else:
                box_color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                c_thick = 2

            # Thin perimeter border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)

            # High-tech Cyber Corner Brackets
            c_len = max(5, min(16, (x2 - x1) // 4, (y2 - y1) // 4))
            # Top-Left
            cv2.line(annotated, (x1, y1), (x1 + c_len, y1), box_color, c_thick, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + c_len), box_color, c_thick, cv2.LINE_AA)
            # Top-Right
            cv2.line(annotated, (x2, y1), (x2 - c_len, y1), box_color, c_thick, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2, y1 + c_len), box_color, c_thick, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(annotated, (x1, y2), (x1 + c_len, y2), box_color, c_thick, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1, y2 - c_len), box_color, c_thick, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(annotated, (x2, y2), (x2 - c_len, y2), box_color, c_thick, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - c_len), box_color, c_thick, cv2.LINE_AA)

            # Label badge
            id_prefix = f"#{track_id} " if track_id is not None else ""
            label = f"{id_prefix}{cls_name.upper()} {int(conf * 100)}%"
            if is_hazard:
                label += f" [{hazard_level}]"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1
            )
            badge_y1 = max(0, y1 - text_h - 7)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(width, x1 + text_w + 10)

            # Badge background with slight rounding feel
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                box_color,
                -1,
            )
            # Badge text
            text_color = (0, 0, 0) if (box_color[0]*0.299 + box_color[1]*0.587 + box_color[2]*0.114) > 130 else (255, 255, 255)
            cv2.putText(
                annotated,
                label,
                (badge_x1 + 5, badge_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                text_color,
                1,
                cv2.LINE_AA,
            )

        # 3. Top Telemetry Glassmorphism HUD Banner
        if show_hud_banner:
            banner_h = 68
            banner_overlay = annotated.copy()
            cv2.rectangle(banner_overlay, (0, 0), (width, banner_h), (18, 20, 24), -1)
            cv2.addWeighted(banner_overlay, 0.82, annotated, 0.18, 0, annotated)

            # Glowing bottom accent line for banner
            summary = analytics["summary"]
            status_color = (0, 255, 120)
            if summary["safety_status"] == "CRITICAL ALERT":
                status_color = (0, 0, 255)
            elif summary["safety_status"] == "PROXIMITY WARNING":
                status_color = (0, 160, 255)

            cv2.line(annotated, (0, banner_h), (width, banner_h), status_color, 2, cv2.LINE_AA)

            # Title
            cv2.putText(
                annotated,
                "DEEP ROAD ANALYSIS // AI VISION",
                (20, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Metric telemetry tags
            metrics_line = (
                f"VEHICLES: {summary['total_vehicles']}  |  "
                f"PEDESTRIANS/VRU: {summary['vulnerable_road_users']}  |  "
                f"SIGNS/LIGHTS: {summary['infrastructure_elements']}  |  "
                f"DENSITY: {summary['traffic_density']}  |  "
                f"SAFETY SCORE: {summary['road_safety_score']}/100"
            )
            cv2.putText(
                annotated,
                metrics_line,
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (210, 215, 220),
                1,
                cv2.LINE_AA,
            )

            # Status Badge on Top Right
            status_text = f"STATUS: {summary['safety_status']}"
            (st_w, st_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            st_x = width - st_w - 25
            cv2.rectangle(annotated, (st_x - 10, 14), (width - 15, 48), (28, 32, 40), -1)
            cv2.rectangle(annotated, (st_x - 10, 14), (width - 15, 48), status_color, 1)
            cv2.putText(
                annotated,
                status_text,
                (st_x, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def process_image_file(
        self,
        image_path: str,
        output_image_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        conf: float = 0.35,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Convenience function to analyze a file on disk and optionally save outputs.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        analytics = self.analyze(img, conf=conf)
        annotated = self.render_hud(img, analytics)

        if output_image_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
            cv2.imwrite(output_image_path, annotated)

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(analytics, f, indent=2)

        return analytics, annotated
