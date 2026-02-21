"""
Unit tests for the Scene Graph — can run WITHOUT Gemini API key.

Tests the core spatio-temporal logic: entity creation, merging,
observation timelines, relationship tracking, direction/time queries,
and spatial relation computation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vesta.flow.optical_flow import CameraMotion
from vesta.registry.scene_graph import (
    SceneGraph, Entity, Observation, Relationship,
    StateChange, _description_similarity,
)


def test_entity_creation():
    g = SceneGraph()

    entity = g.add_entity(
        label="worker_1",
        category="person",
        description="Worker in orange vest",
        current_state="laying_blocks",
        x_normalized=0.5,
        y_normalized=0.7,
        confidence=0.9,
        timestamp=2.0,
        frame_idx=60,
        bbox=(0.3, 0.2, 0.7, 0.8),
    )

    assert entity.label == "worker_1"
    assert entity.category == "person"
    assert abs(entity.allo_angle) < 1  # center of frame
    assert entity.times_observed == 1
    assert len(entity.observations) == 1
    assert entity.first_seen == 2.0
    assert entity.last_seen == 2.0
    print("  ✓ entity_creation")


def test_entity_merging():
    g = SceneGraph()

    e1 = g.add_entity("crane", "equipment", "Yellow crane", "idle",
                       0.3, 0.5, 0.7, 1.0)
    e2 = g.add_entity("crane", "equipment", "Yellow crane moving", "moving",
                       0.32, 0.48, 0.8, 2.0)

    assert len(g.entities) == 1
    assert e1.id == e2.id
    assert e2.times_observed == 2
    assert e2.confidence > 0.7
    assert e2.current_state == "moving"
    assert len(e2.observations) == 2
    print("  ✓ entity_merging")


def test_no_merge_different_category():
    g = SceneGraph()

    g.add_entity("crane", "equipment", "Yellow crane", "idle", 0.5, 0.5, 0.8, 1.0)
    g.add_entity("crane", "structure", "Crane structure", "static", 0.5, 0.5, 0.8, 2.0)

    assert len(g.entities) == 2
    print("  ✓ no_merge_different_category")


def test_direction_query():
    g = SceneGraph()

    # Add entity at center = front
    g.add_entity("worker", "person", "Worker", "idle", 0.5, 0.5, 0.9, 1.0)

    front = g.query_direction("front")
    assert len(front) == 1
    assert front[0].label == "worker"

    behind = g.query_direction("behind")
    assert len(behind) == 0
    print("  ✓ direction_query")


def test_rotation_tracking():
    """Object permanence: entity persists after camera rotation."""
    g = SceneGraph()

    g.add_entity("scaffold", "structure", "Metal scaffolding", "static",
                 1.0, 0.5, 0.95, 2.0)  # right edge = +45° ego

    # Rotate 180° left
    g.update_with_motion(CameraMotion(dx=0, dy=0, d_theta=-180, confidence=1.0))

    all_entities = g.get_all()
    assert len(all_entities) == 1
    scaffold = all_entities[0]

    # Allo angle should still be ~45°
    assert abs(scaffold.allo_angle - 45) < 5

    # Relative to camera should say "rear" or "behind"
    desc = g.describe_relative_to_camera(scaffold)
    assert "rear" in desc.lower() or "behind" in desc.lower(), f"Expected rear/behind in: {desc}"
    print("  ✓ rotation_tracking (object permanence)")


def test_time_range_query():
    g = SceneGraph()

    g.add_entity("worker_1", "person", "Worker A", "walking", 0.3, 0.5, 0.9, 2.0)
    g.add_entity("crane", "equipment", "Yellow crane", "idle", 0.7, 0.3, 0.8, 8.0)

    # Query at T=2: should find worker but not crane
    early = g.query_time_range(1.0, 3.0)
    labels = [e.label for e in early]
    assert "worker_1" in labels
    assert "crane" not in labels

    # Query at T=8: should find both
    later = g.query_time_range(1.0, 9.0)
    assert len(later) == 2
    print("  ✓ time_range_query")


def test_label_query():
    g = SceneGraph()

    g.add_entity("worker_1", "person", "Worker", "idle", 0.3, 0.5, 0.9, 1.0)
    g.add_entity("worker_2", "person", "Worker", "idle", 0.7, 0.5, 0.8, 1.0)
    g.add_entity("crane", "equipment", "Crane", "idle", 0.5, 0.3, 0.7, 1.0)

    workers = g.query_by_label("worker")
    assert len(workers) == 2

    cranes = g.query_by_label("crane")
    assert len(cranes) == 1
    print("  ✓ label_query")


def test_category_query():
    g = SceneGraph()

    g.add_entity("worker_1", "person", "Worker", "idle", 0.3, 0.5, 0.9, 1.0)
    g.add_entity("crane", "equipment", "Crane", "idle", 0.5, 0.3, 0.7, 1.0)
    g.add_entity("scaffold", "structure", "Scaffold", "static", 0.7, 0.5, 0.8, 1.0)

    people = g.query_by_category("person")
    assert len(people) == 1
    assert people[0].label == "worker_1"

    equip = g.query_by_category("equipment")
    assert len(equip) == 1
    print("  ✓ category_query")


def test_relationships():
    g = SceneGraph()

    g.add_entity("worker_1", "person", "Worker", "laying_blocks", 0.3, 0.5, 0.9, 1.0)
    g.add_entity("scaffold", "structure", "Scaffold", "static", 0.7, 0.5, 0.8, 1.0)

    rel = g.add_relationship("worker_1", "scaffold", "standing_on", 1.0)
    assert rel is not None
    assert rel.relation == "standing_on"

    rels = g.query_relationships(entity_label="worker_1")
    assert len(rels) == 1
    assert rels[0]["relation"] == "standing_on"

    rels_typed = g.query_relationships(relation_type="standing_on")
    assert len(rels_typed) == 1
    print("  ✓ relationships")


def test_spatial_relation():
    g = SceneGraph()

    # Entity A at center (0° allo), Entity B at right edge (+45° allo)
    g.add_entity("worker", "person", "Worker", "idle", 0.5, 0.5, 0.9, 1.0)
    g.add_entity("crane", "equipment", "Crane", "idle", 1.0, 0.5, 0.8, 1.0)

    desc = g.describe_spatial_relation("worker", "crane")
    assert "right" in desc.lower() or "front" in desc.lower(), f"Expected right/front in: {desc}"
    print("  ✓ spatial_relation")


def test_entity_timeline():
    g = SceneGraph()

    g.add_entity("wall", "structure", "Brick wall partially built", "building",
                 0.5, 0.5, 0.8, 2.0, frame_idx=60)
    g.add_entity("wall", "structure", "Brick wall half complete", "building",
                 0.52, 0.48, 0.85, 5.0, frame_idx=150)
    g.add_entity("wall", "structure", "Brick wall nearly done", "finishing",
                 0.51, 0.49, 0.9, 10.0, frame_idx=300)

    assert len(g.entities) == 1
    wall = list(g.entities.values())[0]
    timeline = g.get_entity_timeline(wall.id)
    assert len(timeline) == 3
    assert timeline[0]["timestamp"] == 2.0
    assert timeline[2]["timestamp"] == 10.0
    print("  ✓ entity_timeline")


def test_scene_descriptions():
    g = SceneGraph()

    g.add_scene_description("Workers laying blocks on upper floor", 2.0, 60)
    g.add_scene_description("Crane operating in background", 5.0, 150)

    assert len(g.scene_descriptions) == 2
    assert g.scene_descriptions[0]["description"] == "Workers laying blocks on upper floor"
    print("  ✓ scene_descriptions")


def test_confidence_decay_and_prune():
    g = SceneGraph(decay_rate=0.1)

    g.add_entity("debris", "material", "Scattered debris", "static",
                 0.5, 0.8, 0.5, 0.0)

    g.decay_confidence(3.0)  # -0.3
    entity = list(g.entities.values())[0]
    assert entity.confidence < 0.5
    assert entity.confidence > 0.1

    g.decay_confidence(5.0)  # another -0.5 → should be near 0
    pruned = g.prune_stale()
    assert len(pruned) == 1
    assert len(g.entities) == 0
    print("  ✓ confidence_decay_and_prune")


def test_summary():
    g = SceneGraph()

    g.add_entity("worker_1", "person", "Worker", "idle", 0.3, 0.5, 0.9, 1.0)
    g.add_entity("crane", "equipment", "Crane", "idle", 0.7, 0.3, 0.8, 1.0)
    g.add_relationship("worker_1", "crane", "near", 1.0)
    g.add_scene_description("Construction site", 1.0)

    summary = g.get_summary()
    assert summary["total_entities"] == 2
    assert summary["total_relationships"] == 1
    assert summary["by_category"]["person"] == 1
    assert summary["by_category"]["equipment"] == 1
    assert len(summary["scene_descriptions"]) == 1
    print("  ✓ summary")


def test_description_similarity():
    # Same descriptions
    assert _description_similarity("Worker in orange vest", "Worker in orange vest") > 0.8
    # Similar descriptions
    assert _description_similarity("Worker in orange vest laying bricks",
                                   "Worker in orange vest building wall") > 0.3
    # Very different descriptions
    assert _description_similarity("Worker in red hard hat operating crane",
                                   "Worker in blue vest carrying rebar") < 0.3
    # Empty
    assert _description_similarity("", "Something") == 0.0
    print("  ✓ description_similarity")


def test_change_detection():
    g = SceneGraph()

    # Add wall with changing descriptions across observations
    g.add_entity("wall", "structure", "Bare concrete wall, no rebar", "bare",
                 0.5, 0.5, 0.8, 2.0, frame_idx=60)
    g.add_entity("wall", "structure", "Concrete wall with rebar installed and formwork", "rebar_installed",
                 0.52, 0.48, 0.85, 8.0, frame_idx=240)
    g.add_entity("wall", "structure", "Completed block wall with mortar finish", "complete",
                 0.51, 0.49, 0.9, 15.0, frame_idx=450)

    changes = g.detect_changes()
    assert len(changes) >= 1  # At least one change detected
    # The change from bare concrete to rebar should be detected
    assert any("rebar" in c.description_after.lower() or "block" in c.description_after.lower()
               for c in changes)
    print("  ✓ change_detection")


def test_progress_summary():
    g = SceneGraph()

    g.add_entity("wall", "structure", "Empty foundation", "empty",
                 0.5, 0.5, 0.8, 1.0)
    g.add_entity("wall", "structure", "First row of blocks laid", "building",
                 0.52, 0.48, 0.85, 5.0)
    g.add_entity("wall", "structure", "Wall half complete with mortar", "half_done",
                 0.51, 0.49, 0.9, 10.0)

    summary = g.get_progress_summary()
    assert summary["total_changes"] >= 1
    assert summary["entities_with_changes"] >= 1
    print("  ✓ progress_summary")


def test_no_merge_different_descriptions():
    """Two workers with same generic label but different descriptions should NOT merge."""
    g = SceneGraph()

    g.add_entity("worker", "person",
                 "Worker in red hard hat operating crane controls",
                 "operating_crane", 0.3, 0.5, 0.9, 1.0)
    g.add_entity("worker", "person",
                 "Worker in blue safety vest carrying steel rebar bundle",
                 "carrying_rebar", 0.35, 0.48, 0.85, 2.0)

    # These should NOT merge because descriptions are very different
    # even though labels match and positions are close
    assert len(g.entities) == 2, f"Expected 2 entities but got {len(g.entities)}"
    print("  ✓ no_merge_different_descriptions")


def test_query_classifier():
    from vesta.agent.scene_agent import SceneAgent

    assert SceneAgent._classify_query("What's behind me?") == "direction"
    assert SceneAgent._classify_query("What is to my left?") == "direction"
    assert SceneAgent._classify_query("Where is the crane relative to the worker?") == "spatial_relation"
    assert SceneAgent._classify_query("What was visible at 10 seconds?") == "temporal"
    assert SceneAgent._classify_query("What changed over time?") == "changes"
    assert SceneAgent._classify_query("How did the wall progress?") == "changes"
    assert SceneAgent._classify_query("Describe the scene") is None  # ambiguous
    print("  ✓ query_classifier")


if __name__ == "__main__":
    print("\n  VESTA Scene Graph Tests")
    print("  " + "-" * 40)
    test_entity_creation()
    test_entity_merging()
    test_no_merge_different_category()
    test_direction_query()
    test_rotation_tracking()
    test_time_range_query()
    test_label_query()
    test_category_query()
    test_relationships()
    test_spatial_relation()
    test_entity_timeline()
    test_scene_descriptions()
    test_confidence_decay_and_prune()
    test_summary()
    test_description_similarity()
    test_change_detection()
    test_progress_summary()
    test_no_merge_different_descriptions()
    test_query_classifier()
    print("\n  All tests passed! ✓\n")
