"""
Trajectory Predictor — Predictive Path Warning System

Accumulates worker world position from optical flow, extrapolates forward
3-5 seconds, and checks for collisions with known hazards.

This is the "world model" capability: given current state + actions,
predict future state. Frontier VLMs can't do this — they process each
frame independently with no spatial memory or motion accumulation.
"""

import math
from collections import deque
from dataclasses import dataclass

from vesta.flow.optical_flow import CameraMotion
from vesta.registry.hazard_registry import HazardRegistry


@dataclass
class CollisionWarning:
    """A predicted collision with a hazard."""
    hazard_id: str
    label: str
    eta_seconds: float      # estimated time to reach hazard
    distance: float          # current distance to hazard
    hazard_x: float
    hazard_y: float


class TrajectoryPredictor:
    """
    Tracks worker world position and predicts future trajectory.

    Uses the same accumulation math as spatial_map.py:compute_camera_path
    but runs incrementally per-frame instead of batch.
    """

    def __init__(
        self,
        scale: float = 0.02,
        history_size: int = 30,
        prediction_seconds: float = 5.0,
        prediction_steps: int = 10,
        collision_radius: float = 0.5,
    ):
        self.scale = scale
        self.prediction_seconds = prediction_seconds
        self.prediction_steps = prediction_steps
        self.collision_radius = collision_radius

        # Current world position
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

        # Rolling velocity history (dx_world, dy_world per frame)
        self._velocity_history: deque[tuple[float, float]] = deque(maxlen=history_size)

        # Full path (for visualization)
        self._path: deque[tuple[float, float]] = deque(maxlen=500)
        self._path.append((0.0, 0.0))

    def update(self, motion: CameraMotion) -> None:
        """Update world position with new frame's optical flow."""
        self.heading += motion.d_theta
        heading_rad = math.radians(self.heading)

        # Project pixel displacement into world coordinates
        dx_world = (motion.dx * self.scale * math.cos(heading_rad)
                     - motion.dy * self.scale * math.sin(heading_rad))
        dy_world = (motion.dx * self.scale * math.sin(heading_rad)
                     + motion.dy * self.scale * math.cos(heading_rad))

        self.x += dx_world
        self.y += dy_world

        self._velocity_history.append((dx_world, dy_world))
        self._path.append((self.x, self.y))

    def predict_path(self, fps: float = 30.0) -> list[tuple[float, float]]:
        """
        Extrapolate the worker's future position for the next N seconds.

        Uses average velocity over recent frames to project forward.
        Returns list of (x, y) predicted positions.
        """
        if len(self._velocity_history) < 3:
            return []

        # Average velocity over recent frames
        recent = list(self._velocity_history)[-15:]  # last ~0.5s
        avg_vx = sum(v[0] for v in recent) / len(recent)
        avg_vy = sum(v[1] for v in recent) / len(recent)

        # Velocity in world units per second
        vx_per_sec = avg_vx * fps
        vy_per_sec = avg_vy * fps

        # Speed check — if barely moving, no meaningful prediction
        speed = math.sqrt(vx_per_sec**2 + vy_per_sec**2)
        if speed < 0.01:
            return []

        # Generate prediction points
        dt = self.prediction_seconds / self.prediction_steps
        points = []
        px, py = self.x, self.y
        for i in range(1, self.prediction_steps + 1):
            px += vx_per_sec * dt
            py += vy_per_sec * dt
            points.append((px, py))

        return points

    def check_collisions(
        self,
        registry: HazardRegistry,
        predicted_path: list[tuple[float, float]],
        fps: float = 30.0,
    ) -> list[CollisionWarning]:
        """
        Check if the predicted path intersects any known hazards.

        Projects hazards from allocentric polar into world XY, then checks
        distance from each prediction point to each hazard.
        """
        if not predicted_path:
            return []

        # Project hazards to world coordinates
        hazard_positions = []
        for hazard in registry.get_all(min_confidence=0.3):
            # Project from allo_angle + distance relative to current worker position
            # Hazard's world position: from origin + allo_angle direction + distance
            dist = (1.0 - hazard.distance) * 3.0 + 0.5
            angle_rad = math.radians(hazard.allo_angle)
            hx = dist * math.sin(angle_rad)
            hy = dist * math.cos(angle_rad)
            hazard_positions.append((hazard, hx, hy))

        warnings = []
        dt = self.prediction_seconds / self.prediction_steps

        for step_idx, (px, py) in enumerate(predicted_path):
            eta = (step_idx + 1) * dt
            for hazard, hx, hy in hazard_positions:
                dist = math.sqrt((px - hx)**2 + (py - hy)**2)
                if dist < self.collision_radius:
                    # Check we haven't already warned about this hazard
                    if not any(w.hazard_id == hazard.id for w in warnings):
                        warnings.append(CollisionWarning(
                            hazard_id=hazard.id,
                            label=hazard.label,
                            eta_seconds=eta,
                            distance=dist,
                            hazard_x=hx,
                            hazard_y=hy,
                        ))

        warnings.sort(key=lambda w: w.eta_seconds)
        return warnings

    def get_path(self) -> list[tuple[float, float]]:
        """Get the full path history for visualization."""
        return list(self._path)

    def get_hazard_world_positions(
        self, registry: HazardRegistry
    ) -> list[dict]:
        """Get all hazards projected into world XY for visualization."""
        positions = []
        for hazard in registry.get_all(min_confidence=0.3):
            dist = (1.0 - hazard.distance) * 3.0 + 0.5
            angle_rad = math.radians(hazard.allo_angle)
            positions.append({
                "id": hazard.id,
                "label": hazard.label,
                "severity": hazard.severity,
                "confidence": hazard.confidence,
                "x": dist * math.sin(angle_rad),
                "y": dist * math.cos(angle_rad),
            })
        return positions
