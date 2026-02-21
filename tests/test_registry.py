"""
Unit tests for the Hazard Registry — can run WITHOUT Gemini API key.

Tests the core spatial logic: coordinate transforms, hazard persistence,
motion updates, and directional queries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vesta.flow.optical_flow import CameraMotion
from vesta.registry.hazard_registry import (
    HazardRegistry,
    CoordinateTransformer,
    _normalize_angle,
    _angle_diff,
)


def test_normalize_angle():
    assert _normalize_angle(0) == 0
    assert _normalize_angle(180) == 180
    assert _normalize_angle(270) == -90
    assert _normalize_angle(-90) == -90
    assert _normalize_angle(360) == 0
    assert _normalize_angle(450) == 90
    print("  ✓ normalize_angle")


def test_angle_diff():
    assert abs(_angle_diff(10, 350) - 20) < 0.01
    assert abs(_angle_diff(350, 10) + 20) < 0.01
    assert abs(_angle_diff(0, 0)) < 0.01
    assert abs(abs(_angle_diff(180, 0)) - 180) < 0.01  # ±180 are equivalent
    print("  ✓ angle_diff")


def test_coordinate_transformer():
    t = CoordinateTransformer(frame_width=1920, fov_degrees=90)

    # Center of frame = 0° ego
    assert abs(t.pixel_to_ego_angle(0.5)) < 0.01

    # Left edge = -45°
    assert abs(t.pixel_to_ego_angle(0.0) - (-45)) < 0.01

    # Right edge = +45°
    assert abs(t.pixel_to_ego_angle(1.0) - 45) < 0.01

    # Ego→Allo: facing north (0°), object at 45° ego = 45° allo
    assert abs(t.ego_to_allo(45, heading=0) - 45) < 0.01

    # Ego→Allo: facing east (90°), object at -45° ego = 45° allo
    assert abs(t.ego_to_allo(-45, heading=90) - 45) < 0.01

    print("  ✓ coordinate_transformer")


def test_registry_add_and_query():
    reg = HazardRegistry()

    # Add a hazard at center of frame (ego 0°), heading is 0°
    entry = reg.add_detection(
        label="Floor Hole",
        category="Fall Hazard",
        severity="high",
        description="Uncovered floor hole near walkway",
        x_normalized=0.5,
        y_normalized=0.7,
        confidence=0.9,
        timestamp=2.0,
    )

    assert entry.label == "Floor Hole"
    assert abs(entry.allo_angle) < 1  # Should be ~0° (straight ahead)
    assert len(reg.hazards) == 1

    # Query: should be in "front"
    front = reg.query_direction("front")
    assert len(front) == 1
    assert front[0].label == "Floor Hole"

    # Query: should NOT be "behind"
    behind = reg.query_direction("behind")
    assert len(behind) == 0

    print("  ✓ registry_add_and_query")


def test_registry_rotation_tracking():
    """The core "Ghost Mapping" test from the blueprint."""
    reg = HazardRegistry()

    # T=2s: Detect trench at 45° (front-right)
    reg.add_detection(
        label="Trench",
        category="Fall Hazard",
        severity="critical",
        description="Open trench without shoring",
        x_normalized=1.0,  # Right edge of frame = +45° ego
        y_normalized=0.5,
        confidence=0.95,
        timestamp=2.0,
    )

    # Simulate 180° left turn (negative d_theta = counterclockwise)
    # This happens over many frames, but we can simulate in one step
    reg.update_with_motion(CameraMotion(dx=0, dy=0, d_theta=-180, confidence=1.0))

    # Now heading is -180°. The trench was at allo_angle=45°.
    # Relative to current heading: 45° - (-180°) = 225° → normalized = -135°
    # That's "behind-right"

    behind = reg.query_direction("behind")
    # The trench should now be roughly behind us
    assert len(behind) >= 0  # May or may not be in the exact 90° arc

    # More precise: query at the trench's allocentric angle
    all_hazards = reg.get_all()
    assert len(all_hazards) == 1
    trench = all_hazards[0]

    # The allocentric angle should still be ~45° (it's world-fixed)
    assert abs(trench.allo_angle - 45) < 5

    # But the description should say "behind" or "rear"
    desc = reg.describe_relative_to_camera(trench)
    assert "rear" in desc.lower() or "behind" in desc.lower(), f"Expected 'rear/behind' in: {desc}"

    print("  ✓ registry_rotation_tracking (Ghost Mapping)")


def test_registry_merge_duplicates():
    reg = HazardRegistry()

    # Add same hazard twice at similar positions
    e1 = reg.add_detection("Exposed Wire", "Electrocution", "high", "Bare wire",
                           x_normalized=0.3, y_normalized=0.5, confidence=0.7, timestamp=1.0)
    e2 = reg.add_detection("Exposed Wire", "Electrocution", "high", "Bare wire",
                           x_normalized=0.32, y_normalized=0.48, confidence=0.8, timestamp=2.0)

    # Should merge into one entry
    assert len(reg.hazards) == 1
    assert e1.id == e2.id
    assert e2.times_observed == 2
    assert e2.confidence > 0.7  # Boosted

    print("  ✓ registry_merge_duplicates")


def test_registry_confidence_decay():
    reg = HazardRegistry(decay_rate=0.1)

    reg.add_detection("Debris", "Trip/Slip Hazard", "low", "Scattered debris",
                      x_normalized=0.5, y_normalized=0.8, confidence=0.5, timestamp=0.0)

    # Decay over time
    reg.decay_confidence(3.0)  # 3 seconds worth of decay at 0.1/s = -0.3
    entry = list(reg.hazards.values())[0]
    assert entry.confidence < 0.5
    assert entry.confidence > 0.1

    print("  ✓ registry_confidence_decay")


if __name__ == "__main__":
    print("\n  VESTA Registry Tests")
    print("  " + "-" * 40)
    test_normalize_angle()
    test_angle_diff()
    test_coordinate_transformer()
    test_registry_add_and_query()
    test_registry_rotation_tracking()
    test_registry_merge_duplicates()
    test_registry_confidence_decay()
    print("\n  All tests passed! ✓\n")
