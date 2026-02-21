#!/usr/bin/env python3
"""
Generate 3D and 2D spatial maps from a saved VESTA registry.

Usage:
    python scripts/generate_map.py --registry results/test_registry.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vesta.flow.optical_flow import CameraMotion


def main():
    parser = argparse.ArgumentParser(description="Generate VESTA spatial maps")
    parser.add_argument("--registry", required=True, help="Path to saved registry .pkl")
    args = parser.parse_args()

    # Load saved state
    with open(args.registry, "rb") as f:
        state = pickle.load(f)

    registry = state["registry"]
    fps = state["fps"]
    frame_count = state["frame_count"]
    motions = state.get("motions", None)

    print(f"[MAP] Registry: {len(registry.hazards)} hazards, {frame_count} frames @ {fps} FPS")

    from vesta.utils.spatial_map import (
        compute_camera_path,
        project_hazards_to_world,
        build_3d_map,
        build_2d_radar_map,
    )

    # If no motions saved, generate synthetic straight-line path
    if motions is None:
        print("[MAP] No motions in registry — generating synthetic camera path")
        motions = [CameraMotion(dx=2.0, dy=0.0, d_theta=0.0, confidence=1.0)
                   for _ in range(frame_count)]
        motions[0] = CameraMotion(dx=0.0, dy=0.0, d_theta=0.0, confidence=1.0)

    camera_path = compute_camera_path(motions, fps=fps)
    print(f"[MAP] Camera path: {len(camera_path)} points")

    hazard_positions = project_hazards_to_world(registry, camera_path)
    print(f"[MAP] Projected {len(hazard_positions)} hazards to world coordinates")

    output_dir = Path(args.registry).parent
    stem = Path(args.registry).stem.replace("_registry", "")

    map_3d = build_3d_map(camera_path, hazard_positions,
                          output_path=str(output_dir / f"{stem}_map_3d.html"))
    map_2d = build_2d_radar_map(camera_path, hazard_positions,
                                output_path=str(output_dir / f"{stem}_map_2d.html"))

    print(f"\n[MAP] Done!")
    print(f"  3D Map: {map_3d}")
    print(f"  2D Map: {map_2d}")


if __name__ == "__main__":
    main()
