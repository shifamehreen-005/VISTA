"""
3D Spatial Map — Bird's-eye and 3D visualization of entity positions.

Uses monocular depth estimation (MiDaS via torch.hub) to estimate real-world
distances to entities, then projects them into a 3D coordinate system using
accumulated camera motion from optical flow.

Output: Interactive Plotly HTML that judges can rotate, zoom, and explore.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from vesta.flow.optical_flow import CameraMotion
from vesta.registry.scene_graph import SceneGraph, Entity


# ── Depth Estimation ────────────────────────────────────────────────────────

_depth_model = None
_depth_transform = None
_device = None


def _load_depth_model():
    """Load MiDaS depth estimation model (cached, one-time)."""
    global _depth_model, _depth_transform, _device

    if _depth_model is not None:
        return

    _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[DEPTH] Loading MiDaS on {_device}...")

    # MiDaS small — fast, good enough for relative depth
    _depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _depth_model.to(_device)
    _depth_model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    _depth_transform = midas_transforms.small_transform

    print("[DEPTH] MiDaS loaded.")


def estimate_depth_at_points(
    frame: np.ndarray,
    points: list[tuple[float, float]],
) -> list[float]:
    """
    Estimate relative depth at specific normalized (x, y) coordinates.

    Args:
        frame: BGR image
        points: List of (x_norm, y_norm) where 0-1

    Returns:
        List of relative depth values (higher = farther)
    """
    _load_depth_model()

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform and run model
    input_batch = _depth_transform(rgb).to(_device)

    with torch.no_grad():
        prediction = _depth_model(input_batch)

    # Resize to original frame size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # MiDaS outputs inverse depth (higher value = closer)
    # Invert so higher = farther
    depth_map = depth_map.max() - depth_map + 1e-6

    h, w = frame.shape[:2]
    depths = []
    for x_norm, y_norm in points:
        px = int(min(max(x_norm, 0), 1) * (w - 1))
        py = int(min(max(y_norm, 0), 1) * (h - 1))
        depths.append(float(depth_map[py, px]))

    # Normalize to 0-1 range
    if depths:
        d_min, d_max = min(depths), max(depths)
        if d_max > d_min:
            depths = [(d - d_min) / (d_max - d_min) for d in depths]
        else:
            depths = [0.5] * len(depths)

    return depths


# ── Camera Path Tracking ────────────────────────────────────────────────────

@dataclass
class CameraPathPoint:
    frame_idx: int
    timestamp: float
    x: float
    y: float
    heading: float  # degrees


@dataclass
class EntityWorldPosition:
    """An entity projected into world coordinates."""
    entity_id: str
    label: str
    category: str
    confidence: float
    world_x: float
    world_y: float
    world_z: float  # depth-based height estimate
    allo_angle: float
    distance: float
    first_seen: float


def compute_camera_path(
    motions: list[CameraMotion],
    fps: float = 30.0,
    scale: float = 0.02,
) -> list[CameraPathPoint]:
    """
    Accumulate optical flow deltas into a world-space camera path.

    Args:
        motions: Per-frame CameraMotion from optical flow
        fps: Video FPS
        scale: Conversion factor from pixel displacement to world units

    Returns:
        List of camera positions in world coordinates
    """
    path = []
    x, y, heading = 0.0, 0.0, 0.0

    for i, motion in enumerate(motions):
        if i > 0:
            heading += motion.d_theta
            # Move in the direction of current heading
            heading_rad = math.radians(heading)
            x += motion.dx * scale * math.cos(heading_rad) - motion.dy * scale * math.sin(heading_rad)
            y += motion.dx * scale * math.sin(heading_rad) + motion.dy * scale * math.cos(heading_rad)

        path.append(CameraPathPoint(
            frame_idx=i,
            timestamp=i / fps,
            x=x,
            y=y,
            heading=heading,
        ))

    return path


def project_entities_to_world(
    graph: SceneGraph,
    camera_path: list[CameraPathPoint],
    depth_scale: float = 3.0,
) -> list[EntityWorldPosition]:
    """
    Project all entities from the scene graph into world XY coordinates.

    Uses each entity's allocentric angle and distance estimate to place it
    relative to the camera position at the time it was first detected.
    """
    positions = []

    for entity in graph.get_all(min_confidence=0.2):
        # Find camera position at first detection
        best_idx = 0
        best_diff = float("inf")
        for i, pt in enumerate(camera_path):
            diff = abs(pt.timestamp - entity.first_seen)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        cam = camera_path[best_idx]

        # Project entity position from camera location + allocentric angle
        dist = (1.0 - entity.distance) * depth_scale + 0.5
        angle_rad = math.radians(entity.allo_angle)

        world_x = cam.x + dist * math.sin(angle_rad)
        world_y = cam.y + dist * math.cos(angle_rad)

        # Z is based on the y-position in the frame (higher in frame = higher up)
        world_z = entity.distance * 2.0

        positions.append(EntityWorldPosition(
            entity_id=entity.id,
            label=entity.label,
            category=entity.category,
            confidence=entity.confidence,
            world_x=world_x,
            world_y=world_y,
            world_z=world_z,
            allo_angle=entity.allo_angle,
            distance=dist,
            first_seen=entity.first_seen,
        ))

    return positions


# ── Visualization ───────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "person":    "#3388FF",
    "equipment": "#FFCC00",
    "structure": "#999999",
    "material":  "#00CC00",
    "vehicle":   "#FF8800",
    "signage":   "#00CCCC",
    "unknown":   "#BBBBBB",
}

CATEGORY_SIZES = {
    "person":    14,
    "equipment": 12,
    "structure": 16,
    "material":  10,
    "vehicle":   16,
    "signage":   10,
    "unknown":   8,
}


def build_3d_map(
    camera_path: list[CameraPathPoint],
    entity_positions: list[EntityWorldPosition],
    output_path: str = "results/site_map_3d.html",
) -> str:
    """
    Build an interactive 3D Plotly visualization.

    Returns the output file path.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # ── Camera path (blue line) ─────────────────────────────────────────
    sampled = camera_path[::5]
    fig.add_trace(go.Scatter3d(
        x=[p.x for p in sampled],
        y=[p.y for p in sampled],
        z=[0] * len(sampled),
        mode="lines",
        line=dict(color="#4488FF", width=3),
        name="Camera Path",
        hovertemplate="T=%{customdata:.1f}s<extra>Camera Path</extra>",
        customdata=[p.timestamp for p in sampled],
    ))

    # Camera start marker
    fig.add_trace(go.Scatter3d(
        x=[camera_path[0].x],
        y=[camera_path[0].y],
        z=[0],
        mode="markers+text",
        marker=dict(size=10, color="#00FF00", symbol="diamond"),
        text=["START"],
        textposition="top center",
        textfont=dict(color="white", size=12),
        name="Start",
        showlegend=False,
    ))

    # Camera end marker
    fig.add_trace(go.Scatter3d(
        x=[camera_path[-1].x],
        y=[camera_path[-1].y],
        z=[0],
        mode="markers+text",
        marker=dict(size=10, color="#FF4444", symbol="diamond"),
        text=["END"],
        textposition="top center",
        textfont=dict(color="white", size=12),
        name="End",
        showlegend=False,
    ))

    # ── Entities by category ─────────────────────────────────────────────
    for category in ["person", "equipment", "structure", "material", "vehicle", "signage", "unknown"]:
        entities = [e for e in entity_positions if e.category == category]
        if not entities:
            continue

        color = CATEGORY_COLORS.get(category, "#BBBBBB")
        size = CATEGORY_SIZES.get(category, 10)

        fig.add_trace(go.Scatter3d(
            x=[e.world_x for e in entities],
            y=[e.world_y for e in entities],
            z=[e.world_z for e in entities],
            mode="markers+text",
            marker=dict(
                size=size,
                color=color,
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            text=[e.label for e in entities],
            textposition="top center",
            textfont=dict(color="white", size=9),
            name=f"{category.upper()} ({len(entities)})",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Category: " + category.upper() + "<br>"
                "Confidence: %{customdata[0]:.0%}<br>"
                "First seen: T=%{customdata[1]:.1f}s<br>"
                "Angle: %{customdata[2]:.0f}°"
                "<extra></extra>"
            ),
            customdata=[[e.confidence, e.first_seen, e.allo_angle] for e in entities],
        ))

        # Draw vertical lines from ground to entity
        for e in entities:
            fig.add_trace(go.Scatter3d(
                x=[e.world_x, e.world_x],
                y=[e.world_y, e.world_y],
                z=[0, e.world_z],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── Layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="VESTA — 3D Scene Entity Map",
            font=dict(size=20, color="white"),
        ),
        scene=dict(
            xaxis=dict(title="X (meters)", color="gray", gridcolor="#333"),
            yaxis=dict(title="Y (meters)", color="gray", gridcolor="#333"),
            zaxis=dict(title="Height", color="gray", gridcolor="#333"),
            bgcolor="#111111",
            aspectmode="data",
        ),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(30,30,30,0.8)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"[MAP] 3D map saved: {output_path}")
    return output_path


def build_2d_radar_map(
    camera_path: list[CameraPathPoint],
    entity_positions: list[EntityWorldPosition],
    output_path: str = "results/site_map_2d.html",
) -> str:
    """Build a 2D bird's-eye-view map (top-down)."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Camera path
    sampled = camera_path[::5]
    fig.add_trace(go.Scatter(
        x=[p.x for p in sampled],
        y=[p.y for p in sampled],
        mode="lines",
        line=dict(color="#4488FF", width=2),
        name="Camera Path",
    ))

    # Start/End
    fig.add_trace(go.Scatter(
        x=[camera_path[0].x], y=[camera_path[0].y],
        mode="markers+text", marker=dict(size=12, color="#00FF00", symbol="diamond"),
        text=["START"], textposition="top center", textfont=dict(color="white"),
        name="Start", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[camera_path[-1].x], y=[camera_path[-1].y],
        mode="markers+text", marker=dict(size=12, color="#FF4444", symbol="diamond"),
        text=["END"], textposition="top center", textfont=dict(color="white"),
        name="End", showlegend=False,
    ))

    # Heading arrows at intervals
    for pt in camera_path[::30]:
        angle_rad = math.radians(pt.heading)
        dx = 0.3 * math.sin(angle_rad)
        dy = 0.3 * math.cos(angle_rad)
        fig.add_annotation(
            x=pt.x + dx, y=pt.y + dy, ax=pt.x, ay=pt.y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor="#4488FF",
        )

    # Entities by category
    for category in ["person", "equipment", "structure", "material", "vehicle", "signage", "unknown"]:
        entities = [e for e in entity_positions if e.category == category]
        if not entities:
            continue

        color = CATEGORY_COLORS.get(category, "#BBBBBB")
        size = CATEGORY_SIZES.get(category, 10)

        fig.add_trace(go.Scatter(
            x=[e.world_x for e in entities],
            y=[e.world_y for e in entities],
            mode="markers+text",
            marker=dict(
                size=[size] * len(entities),
                color=color,
                line=dict(width=1, color="white"),
            ),
            text=[e.label for e in entities],
            textposition="top center",
            textfont=dict(color="white", size=8),
            name=f"{category.upper()} ({len(entities)})",
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Category: {category.upper()}<br>"
                "Confidence: %{customdata[0]:.0%}<br>"
                "T=%{customdata[1]:.1f}s"
                "<extra></extra>"
            ),
            customdata=[[e.confidence, e.first_seen] for e in entities],
        ))

    fig.update_layout(
        title=dict(text="VESTA — Bird's Eye Scene Map", font=dict(size=18, color="white")),
        xaxis=dict(title="X", color="gray", gridcolor="#222", scaleanchor="y"),
        yaxis=dict(title="Y", color="gray", gridcolor="#222"),
        paper_bgcolor="#0D1117",
        plot_bgcolor="#161B22",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", bordercolor="#444", borderwidth=1),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"[MAP] 2D map saved: {output_path}")
    return output_path
