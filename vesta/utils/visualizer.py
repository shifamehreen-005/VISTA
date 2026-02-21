"""
Video Visualizer — Annotated output with entity overlays and radar minimap.

Renders:
1. Bounding markers on detected entities in the current frame
2. A radar minimap showing all entities in the scene graph (360° view)
3. Heading indicator and category-colored entity dots
4. Status bar with frame count, heading, and entity count
"""

import math
from pathlib import Path

import cv2
import numpy as np

from vesta.registry.scene_graph import SceneGraph, Entity

# ── Color scheme ────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "person":    (255, 100, 50),    # Blue
    "equipment": (0, 200, 255),     # Yellow
    "structure": (150, 150, 150),   # Gray
    "material":  (0, 200, 0),       # Green
    "vehicle":   (0, 140, 255),     # Orange
    "signage":   (200, 200, 0),     # Cyan
    "unknown":   (180, 180, 180),   # Light gray
}
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
DARK = (20, 20, 20)
RADAR_BG = (30, 30, 30)
RADAR_RING = (60, 60, 60)
HEADING_COLOR = (255, 200, 0)  # Cyan-ish


class VideoVisualizer:
    """Renders annotated video frames with entity overlays and radar."""

    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        radar_size: int = 200,
        radar_margin: int = 20,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.radar_size = radar_size
        self.radar_margin = radar_margin
        self.radar_center = (
            frame_width - radar_margin - radar_size // 2,
            radar_margin + radar_size // 2,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        graph: SceneGraph,
        current_detections: list | None = None,
        frame_idx: int = 0,
        fps: float = 30.0,
        is_keyframe: bool = False,
    ) -> np.ndarray:
        """
        Draw all overlays on a frame and return the annotated copy.

        Args:
            frame: Original BGR frame
            graph: Current scene graph state
            current_detections: Entities detected in THIS frame (if keyframe)
            frame_idx: Current frame number
            fps: Video FPS for timestamp display
        """
        out = frame.copy()
        h, w = out.shape[:2]
        self.frame_width = w
        self.frame_height = h

        # Update radar center based on actual frame size
        self.radar_center = (
            w - self.radar_margin - self.radar_size // 2,
            self.radar_margin + self.radar_size // 2,
        )

        # 1. Draw detection markers on current-frame entities
        if current_detections:
            self._draw_detections(out, current_detections, w, h)

        # 2. Draw the radar minimap
        self._draw_radar(out, graph)

        # 3. Draw status bar
        self._draw_status_bar(out, graph, frame_idx, fps, is_keyframe)

        return out

    def _draw_detections(self, frame, detections, w, h):
        """Draw bounding boxes on entities detected in the current frame."""
        for det in detections:
            category = getattr(det, "category", "unknown")
            color = CATEGORY_COLORS.get(category, WHITE)

            # Use bounding box if available, fall back to center point
            has_bbox = hasattr(det, "x1") and det.x1 != 0 and det.x2 != 0
            if has_bbox:
                bx1 = int(det.x1 * w)
                by1 = int(det.y1 * h)
                bx2 = int(det.x2 * w)
                by2 = int(det.y2 * h)
            else:
                # Fallback: synthesize a box from center point
                px = int(det.x * w)
                py = int(det.y * h)
                size = 40
                bx1, by1 = px - size, py - size
                bx2, by2 = px + size, py + size

            # Draw bounding box with semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Solid border
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            # Corner accents (thicker L-shaped corners)
            corner_len = min(20, (bx2 - bx1) // 4, (by2 - by1) // 4)
            t = 3
            # Top-left
            cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), color, t)
            cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), color, t)
            # Top-right
            cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), color, t)
            cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), color, t)
            # Bottom-left
            cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), color, t)
            cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), color, t)
            # Bottom-right
            cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), color, t)
            cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), color, t)

            # Label above the box
            label = det.label
            conf_text = f"{det.confidence:.0%}"
            full_label = f"{label} {conf_text}"
            label_size = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Background pill for label
            lx = bx1
            ly = by1 - label_size[1] - 10
            cv2.rectangle(frame, (lx - 2, ly - 4), (lx + label_size[0] + 6, by1 - 2), color, -1)
            cv2.putText(
                frame, full_label,
                (lx + 2, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

    def _draw_radar(self, frame, graph: SceneGraph):
        """Draw a top-down radar minimap showing all entities in the scene graph."""
        cx, cy = self.radar_center
        r = self.radar_size // 2

        # Semi-transparent background
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r + 5, DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Radar rings
        for ring_r in [r // 3, 2 * r // 3, r]:
            cv2.circle(frame, (cx, cy), ring_r, RADAR_RING, 1)

        # Cross lines (N/S/E/W)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), RADAR_RING, 1)
        cv2.line(frame, (cx - r, cy), (cx + r, cy), RADAR_RING, 1)

        # Camera heading indicator (FOV cone)
        heading = graph.current_heading
        fov_half = 45  # degrees

        # Draw FOV cone
        angle_left = math.radians(-(heading - fov_half) + 90)
        angle_right = math.radians(-(heading + fov_half) + 90)
        cone_len = r - 5

        left_x = int(cx + cone_len * math.cos(angle_left))
        left_y = int(cy - cone_len * math.sin(angle_left))
        right_x = int(cx + cone_len * math.cos(angle_right))
        right_y = int(cy - cone_len * math.sin(angle_right))

        # Fill the FOV cone
        fov_overlay = frame.copy()
        pts = np.array([[cx, cy], [left_x, left_y], [right_x, right_y]], np.int32)
        cv2.fillPoly(fov_overlay, [pts], (80, 60, 0))
        cv2.addWeighted(fov_overlay, 0.3, frame, 0.7, 0, frame)

        # FOV edges
        cv2.line(frame, (cx, cy), (left_x, left_y), HEADING_COLOR, 1)
        cv2.line(frame, (cx, cy), (right_x, right_y), HEADING_COLOR, 1)

        # Heading direction arrow
        head_angle = math.radians(-heading + 90)
        arrow_len = r // 3
        arrow_x = int(cx + arrow_len * math.cos(head_angle))
        arrow_y = int(cy - arrow_len * math.sin(head_angle))
        cv2.arrowedLine(frame, (cx, cy), (arrow_x, arrow_y), HEADING_COLOR, 2, tipLength=0.4)

        # Plot entities as colored dots
        all_entities = graph.get_all(min_confidence=0.3)
        for entity in all_entities:
            color = CATEGORY_COLORS.get(entity.category, WHITE)

            # Convert allocentric angle to radar position
            angle_rad = math.radians(-entity.allo_angle + 90)
            dist_ratio = min(1.0, (1.0 - entity.distance) * 0.8 + 0.2)
            dot_r = int(dist_ratio * (r - 10))

            dot_x = int(cx + dot_r * math.cos(angle_rad))
            dot_y = int(cy - dot_r * math.sin(angle_rad))

            # Dot size based on confidence
            dot_size = max(3, int(entity.confidence * 6))
            cv2.circle(frame, (dot_x, dot_y), dot_size, color, -1)
            cv2.circle(frame, (dot_x, dot_y), dot_size, WHITE, 1)

        # Labels
        cv2.putText(frame, "N", (cx - 5, cy - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        cv2.putText(frame, f"{heading:.0f}", (cx - 15, cy + r + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, HEADING_COLOR, 1)

    def _draw_status_bar(self, frame, graph: SceneGraph, frame_idx: int,
                         fps: float, is_keyframe: bool):
        """Draw status bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        bar_h = 40

        # Semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        timestamp = frame_idx / fps
        summary = graph.get_summary()
        y = h - 12

        # Left: timestamp and frame
        cv2.putText(frame, f"T={timestamp:.1f}s  Frame {frame_idx}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        # Center: entity counts by category
        by_cat = summary.get("by_category", {})
        x_pos = w // 3
        for cat in ["person", "equipment", "structure", "material", "vehicle"]:
            count = by_cat.get(cat, 0)
            if count > 0:
                color = CATEGORY_COLORS.get(cat, WHITE)
                text = f"{cat[:3].upper()}:{count}"
                cv2.putText(frame, text, (x_pos, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                x_pos += 80

        # Right: heading + keyframe indicator
        heading_text = f"HDG {graph.current_heading:.1f}"
        cv2.putText(frame, heading_text, (w - 150, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HEADING_COLOR, 1, cv2.LINE_AA)

        if is_keyframe:
            cv2.circle(frame, (w - 20, y - 5), 8, (0, 0, 255), -1)
            cv2.putText(frame, "AI", (w - 27, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)
