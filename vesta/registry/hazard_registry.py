"""
Module 3: Hazard Registry — The Stateful Map

This is the core of VESTA. It maintains a persistent dictionary of every hazard
detected, stored in polar coordinates (angle, distance) relative to an allocentric
reference frame. When the camera rotates, all hazard angles are updated so that
spatial queries ("what's behind me?") always resolve correctly.

The key insight: we store hazards in ALLOCENTRIC coordinates (fixed world frame),
not egocentric (camera-relative). When the camera turns 90° right, we DON'T move
the hazards — we just update "current_heading" so the ego→allo conversion works.
"""

import uuid
from dataclasses import dataclass
from typing import Optional

from vesta.flow.optical_flow import CameraMotion


@dataclass
class HazardEntry:
    """A single hazard in the registry."""
    id: str
    label: str
    category: str
    severity: str
    description: str

    # Position in allocentric polar coordinates
    # angle: degrees from allocentric north (0° = initial camera facing direction)
    # distance: relative distance estimate (from pixel position, rough)
    allo_angle: float
    distance: float

    # Metadata
    confidence: float
    first_seen: float          # timestamp (seconds into video)
    last_seen: float
    times_observed: int = 1
    osha_narrative: str = ""   # Injected from OSHA CSV lookup

    @property
    def is_stale(self) -> bool:
        """Hazard hasn't been re-observed in a while."""
        return self.confidence < 0.2

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "angle": round(self.allo_angle, 1),
            "distance": round(self.distance, 2),
            "confidence": round(self.confidence, 2),
            "first_seen": round(self.first_seen, 1),
            "last_seen": round(self.last_seen, 1),
            "times_observed": self.times_observed,
            "osha_narrative": self.osha_narrative,
        }


class CoordinateTransformer:
    """
    Handles Egocentric ↔ Allocentric coordinate conversion.

    Egocentric: relative to current camera view (0° = center of frame)
    Allocentric: relative to fixed world frame (0° = initial heading)
    """

    def __init__(self, frame_width: int = 1920, fov_degrees: float = 90.0):
        self.frame_width = frame_width
        self.fov = fov_degrees

    def pixel_to_ego_angle(self, x_normalized: float) -> float:
        """
        Convert normalized x-coordinate (0-1) to egocentric angle.
        0.0 → -fov/2 (left edge), 0.5 → 0° (center), 1.0 → +fov/2 (right edge)
        """
        return (x_normalized - 0.5) * self.fov

    def pixel_to_distance(self, y_normalized: float) -> float:
        """
        Rough distance estimate from y-coordinate.
        Objects near bottom of frame (y→1) are closer, top (y→0) are farther.
        Returns 0-1 relative distance (not metric).
        """
        return 1.0 - y_normalized

    def ego_to_allo(self, ego_angle: float, heading: float) -> float:
        """Convert egocentric angle to allocentric given current heading."""
        return _normalize_angle(heading + ego_angle)

    def allo_to_ego(self, allo_angle: float, heading: float) -> float:
        """Convert allocentric angle to egocentric given current heading."""
        return _normalize_angle(allo_angle - heading)


class HazardRegistry:
    """
    The persistent hazard map.

    Stores all detected hazards in allocentric coordinates and updates them
    as the camera moves. Supports spatial queries for the agent.
    """

    def __init__(
        self,
        merge_angle_threshold: float = 15.0,
        merge_distance_threshold: float = 0.2,
        decay_rate: float = 0.02,
        fov_degrees: float = 90.0,
    ):
        self.hazards: dict[str, HazardEntry] = {}
        self.current_heading: float = 0.0   # cumulative camera rotation
        self.transformer = CoordinateTransformer(fov_degrees=fov_degrees)

        # Merge threshold: if a new detection is within this angle+distance
        # of an existing hazard with the same label, update instead of adding
        self.merge_angle_threshold = merge_angle_threshold
        self.merge_distance_threshold = merge_distance_threshold
        self.decay_rate = decay_rate

    def update_with_motion(self, motion: CameraMotion) -> None:
        """
        Update the heading based on camera rotation.
        This is called every frame with the optical flow result.
        """
        self.current_heading = _normalize_angle(self.current_heading + motion.d_theta)

    def add_detection(
        self,
        label: str,
        category: str,
        severity: str,
        description: str,
        x_normalized: float,
        y_normalized: float,
        confidence: float,
        timestamp: float,
    ) -> HazardEntry:
        """
        Add a new hazard detection from Gemini.

        Converts pixel coordinates to allocentric polar coordinates
        and either merges with an existing entry or creates a new one.
        """
        # Convert pixel → ego angle → allo angle
        ego_angle = self.transformer.pixel_to_ego_angle(x_normalized)
        allo_angle = self.transformer.ego_to_allo(ego_angle, self.current_heading)
        distance = self.transformer.pixel_to_distance(y_normalized)

        # Try to merge with existing hazard
        existing = self._find_match(label, allo_angle, distance)

        if existing:
            # Update existing entry — boost confidence, update last_seen
            existing.confidence = min(1.0, existing.confidence + 0.2)
            existing.last_seen = timestamp
            existing.times_observed += 1
            # Update position with weighted average (new observation pulls it)
            alpha = 0.3  # weight for new observation
            existing.allo_angle = _normalize_angle(
                existing.allo_angle * (1 - alpha) + allo_angle * alpha
            )
            existing.distance = existing.distance * (1 - alpha) + distance * alpha
            return existing

        # Create new entry
        entry = HazardEntry(
            id=f"HAZ_{uuid.uuid4().hex[:8].upper()}",
            label=label,
            category=category,
            severity=severity,
            description=description,
            allo_angle=allo_angle,
            distance=distance,
            confidence=confidence,
            first_seen=timestamp,
            last_seen=timestamp,
        )
        self.hazards[entry.id] = entry
        return entry

    def decay_confidence(self, dt: float) -> None:
        """Reduce confidence of all hazards that haven't been re-observed."""
        for entry in self.hazards.values():
            entry.confidence = max(0.0, entry.confidence - self.decay_rate * dt)

    def prune_stale(self) -> list[str]:
        """Remove hazards with confidence below threshold. Returns removed IDs."""
        stale_ids = [hid for hid, h in self.hazards.items() if h.is_stale]
        for hid in stale_ids:
            del self.hazards[hid]
        return stale_ids

    # ── Spatial Queries (used by the Agent as Tools) ────────────────────────

    def query_angle(
        self,
        query_angle_deg: float,
        fov: float = 90.0,
        min_confidence: float = 0.3,
    ) -> list[HazardEntry]:
        """
        Get all hazards within a field-of-view arc around a query angle.

        Args:
            query_angle_deg: Allocentric angle to query
            fov: Width of the arc in degrees
            min_confidence: Minimum confidence to include
        """
        half_fov = fov / 2
        results = []
        for entry in self.hazards.values():
            if entry.confidence < min_confidence:
                continue
            angular_diff = abs(_angle_diff(entry.allo_angle, query_angle_deg))
            if angular_diff <= half_fov:
                results.append(entry)
        return sorted(results, key=lambda h: h.confidence, reverse=True)

    def query_direction(
        self,
        direction: str,
        min_confidence: float = 0.3,
    ) -> list[HazardEntry]:
        """
        Query hazards by natural language direction relative to current heading.

        Directions: "front", "behind"/"back", "left", "right",
                    "front-left", "front-right", "behind-left", "behind-right"
        """
        direction_map = {
            "front": 0,
            "ahead": 0,
            "forward": 0,
            "right": 90,
            "behind": 180,
            "back": 180,
            "rear": 180,
            "left": -90,
            "front-right": 45,
            "front-left": -45,
            "behind-right": 135,
            "rear-right": 135,
            "behind-left": -135,
            "rear-left": -135,
        }

        direction = direction.lower().strip()
        if direction not in direction_map:
            return []

        # Convert relative direction to allocentric angle
        relative_angle = direction_map[direction]
        allo_query = _normalize_angle(self.current_heading + relative_angle)

        return self.query_angle(allo_query, fov=90, min_confidence=min_confidence)

    def query_time_range(
        self,
        start_time: float,
        end_time: float,
        min_confidence: float = 0.1,
    ) -> list[HazardEntry]:
        """
        Get all hazards that were visible during a time window.

        A hazard is included if its [first_seen, last_seen] interval overlaps
        with [start_time, end_time].
        """
        results = []
        for entry in self.hazards.values():
            if entry.confidence < min_confidence:
                continue
            # Overlap check: hazard was seen during this window
            if entry.first_seen <= end_time and entry.last_seen >= start_time:
                results.append(entry)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(results, key=lambda h: severity_order.get(h.severity, 4))

    def get_all(self, min_confidence: float = 0.1) -> list[HazardEntry]:
        """Get all hazards above minimum confidence, sorted by severity."""
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        results = [h for h in self.hazards.values() if h.confidence >= min_confidence]
        return sorted(results, key=lambda h: severity_order.get(h.severity, 4))

    def get_summary(self) -> dict:
        """Get a summary of the registry state for the agent."""
        all_hazards = self.get_all()
        return {
            "total_hazards": len(all_hazards),
            "current_heading": round(self.current_heading, 1),
            "by_severity": {
                sev: len([h for h in all_hazards if h.severity == sev])
                for sev in ["critical", "high", "medium", "low"]
            },
            "hazards": [h.to_dict() for h in all_hazards],
        }

    def describe_relative_to_camera(self, entry: HazardEntry) -> str:
        """Describe a hazard's position relative to the current camera heading."""
        ego_angle = self.transformer.allo_to_ego(entry.allo_angle, self.current_heading)

        if abs(ego_angle) < 30:
            direction = "ahead of you"
        elif abs(ego_angle) > 150:
            direction = "behind you"
        elif ego_angle > 0:
            if ego_angle < 70:
                direction = "to your front-right"
            elif ego_angle < 110:
                direction = "to your right"
            else:
                direction = "to your rear-right"
        else:
            if ego_angle > -70:
                direction = "to your front-left"
            elif ego_angle > -110:
                direction = "to your left"
            else:
                direction = "to your rear-left"

        return (
            f"[{entry.severity.upper()}] {entry.label} — {direction} "
            f"(~{abs(ego_angle):.0f}° {'right' if ego_angle > 0 else 'left'}). "
            f"{entry.description}"
        )

    # ── Internal ────────────────────────────────────────────────────────────

    def _find_match(
        self,
        label: str,
        allo_angle: float,
        distance: float,
    ) -> Optional[HazardEntry]:
        """Find an existing hazard that matches a new detection (for merging)."""
        for entry in self.hazards.values():
            if entry.label.lower() != label.lower():
                continue
            angle_diff = abs(_angle_diff(entry.allo_angle, allo_angle))
            dist_diff = abs(entry.distance - distance)
            if (angle_diff <= self.merge_angle_threshold and
                    dist_diff <= self.merge_distance_threshold):
                return entry
        return None


# ── Utility functions ───────────────────────────────────────────────────────

def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] range."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def _angle_diff(a: float, b: float) -> float:
    """Shortest angular distance between two angles."""
    diff = (a - b + 180) % 360 - 180
    return diff
