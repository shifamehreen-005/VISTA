"""
Spatio-Temporal Scene Graph — The core data structure for video understanding.

Generalizes HazardRegistry from hazard-only to full scene understanding.
Entities persist with object permanence, relationships track spatial context,
and temporal observation chains enable tracking changes over time.

Every entity has world-fixed (allocentric) coordinates computed from optical
flow heading — this is the key capability that RAVU and other video understanding
approaches lack. It enables spatial retrieval: "what's behind me?", "what's near
the crane?", and geometric precision: "the scaffolding is 45 degrees to your left."
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from vesta.flow.optical_flow import CameraMotion
from vesta.registry.hazard_registry import (
    CoordinateTransformer,
    _normalize_angle,
    _angle_diff,
)


# ── Lightweight text similarity ─────────────────────────────────────────────

def _extract_label_words(label: str) -> set[str]:
    """Extract significant words from a compound label like 'crane_boom_1'."""
    parts = label.lower().replace("-", "_").split("_")
    # Filter out numbers and very short words
    stopwords = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "is"}
    return {p for p in parts if len(p) > 2 and not p.isdigit() and p not in stopwords}


def _labels_share_root(label_a: str, label_b: str) -> bool:
    """
    Check if two labels share a significant root word.

    Examples that should match:
      worker_1 ↔ worker_arm (share "worker")
      crane_boom_1 ↔ crane_tower_1 (share "crane")
      scaffolding_1 ↔ scaffolding_structure_1 (share "scaffolding")

    Examples that should NOT match:
      worker_1 ↔ crane_1 (no shared roots)
      hand_1 ↔ crane_hook_1 (no shared roots)
    """
    words_a = _extract_label_words(label_a)
    words_b = _extract_label_words(label_b)
    if not words_a or not words_b:
        return False
    shared = words_a & words_b
    return len(shared) > 0


def _description_similarity(desc_a: str, desc_b: str) -> float:
    """
    Compute similarity between two descriptions using word overlap (Jaccard).
    Fast, no model loading. Returns 0.0-1.0.
    """
    if not desc_a or not desc_b:
        return 0.0
    # Normalize and tokenize
    words_a = set(desc_a.lower().replace(",", " ").replace(".", " ").split())
    words_b = set(desc_b.lower().replace(",", " ").replace(".", " ").split())
    # Remove very common words
    stopwords = {"a", "an", "the", "in", "on", "at", "is", "are", "was", "with", "and", "or", "of", "to"}
    words_a -= stopwords
    words_b -= stopwords
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


@dataclass
class Observation:
    """A single sighting of an entity in one keyframe."""
    timestamp: float
    frame_idx: int
    ego_angle: float
    allo_angle: float
    distance: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    confidence: float
    description: str
    heading_at_observation: float


@dataclass
class Entity:
    """A persistent entity in the scene graph with full observation history."""
    id: str
    label: str
    category: str  # "person", "equipment", "structure", "material", "vehicle", "signage"

    # Current best-estimate position (allocentric)
    allo_angle: float
    distance: float

    # Rich description (updated with latest observation)
    description: str
    current_state: str = ""

    # Observation history (temporal chain)
    observations: list[Observation] = field(default_factory=list)

    # Metadata
    confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    times_observed: int = 0

    @property
    def is_stale(self) -> bool:
        return self.confidence < 0.2

    @property
    def lifespan(self) -> float:
        return self.last_seen - self.first_seen

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "current_state": self.current_state,
            "angle": round(self.allo_angle, 1),
            "distance": round(self.distance, 2),
            "confidence": round(self.confidence, 2),
            "first_seen": round(self.first_seen, 1),
            "last_seen": round(self.last_seen, 1),
            "times_observed": self.times_observed,
            "observation_count": len(self.observations),
        }


@dataclass
class Relationship:
    """A spatial or action relationship between two entities at a point in time."""
    subject_id: str
    object_id: str
    relation: str  # "near", "left_of", "operating", "standing_on", etc.
    timestamp: float
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation": self.relation,
            "timestamp": round(self.timestamp, 1),
        }


@dataclass
class StateChange:
    """A detected state change for an entity between two observations."""
    entity_id: str
    entity_label: str
    category: str
    timestamp_before: float
    timestamp_after: float
    state_before: str
    state_after: str
    description_before: str
    description_after: str
    similarity: float  # 0-1, lower = bigger change

    def to_dict(self) -> dict:
        return {
            "entity_label": self.entity_label,
            "category": self.category,
            "time_before": round(self.timestamp_before, 1),
            "time_after": round(self.timestamp_after, 1),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "description_before": self.description_before,
            "description_after": self.description_after,
            "change_magnitude": round(1.0 - self.similarity, 2),
        }


class SceneGraph:
    """
    The Spatio-Temporal Scene Graph.

    Generalizes HazardRegistry: same allocentric coordinate system,
    same optical-flow heading tracking, same entity merging logic,
    but stores arbitrary entities + relationships.
    """

    def __init__(
        self,
        merge_angle_threshold: float = 20.0,
        merge_distance_threshold: float = 0.25,
        decay_rate: float = 0.015,
        fov_degrees: float = 90.0,
    ):
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.current_heading: float = 0.0
        self.transformer = CoordinateTransformer(fov_degrees=fov_degrees)

        self.merge_angle_threshold = merge_angle_threshold
        self.merge_distance_threshold = merge_distance_threshold
        self.decay_rate = decay_rate

        # Scene-level metadata per keyframe
        self.scene_descriptions: list[dict] = []

        # Heading history: list of (timestamp, heading) for temporal spatial queries
        self.heading_history: list[tuple[float, float]] = []

    # ── Motion tracking ──────────────────────────────────────────────────

    def update_with_motion(self, motion: CameraMotion, timestamp: float | None = None) -> None:
        self.current_heading = _normalize_angle(self.current_heading + motion.d_theta)
        if timestamp is not None:
            self.heading_history.append((timestamp, self.current_heading))

    def record_heading(self, timestamp: float) -> None:
        """Explicitly record current heading at a timestamp."""
        self.heading_history.append((timestamp, self.current_heading))

    def heading_at_time(self, timestamp: float) -> float:
        """Look up the camera heading at a given timestamp via interpolation."""
        if not self.heading_history:
            return self.current_heading
        # Find closest entry
        best = min(self.heading_history, key=lambda h: abs(h[0] - timestamp))
        return best[1]

    # ── Entity management ────────────────────────────────────────────────

    def add_entity(
        self,
        label: str,
        category: str,
        description: str,
        current_state: str,
        x_normalized: float,
        y_normalized: float,
        confidence: float,
        timestamp: float,
        frame_idx: int = 0,
        bbox: tuple[float, float, float, float] = (0, 0, 0, 0),
    ) -> Entity:
        """
        Add or merge an entity observation.

        Converts pixel coordinates to allocentric polar coordinates
        and either merges with an existing entity or creates a new one.
        """
        ego_angle = self.transformer.pixel_to_ego_angle(x_normalized)
        allo_angle = self.transformer.ego_to_allo(ego_angle, self.current_heading)
        distance = self.transformer.pixel_to_distance(y_normalized)

        observation = Observation(
            timestamp=timestamp,
            frame_idx=frame_idx,
            ego_angle=ego_angle,
            allo_angle=allo_angle,
            distance=distance,
            bbox=bbox,
            confidence=confidence,
            description=description,
            heading_at_observation=self.current_heading,
        )

        existing = self._find_match(label, category, allo_angle, distance, description)

        if existing:
            existing.confidence = min(1.0, existing.confidence + 0.15)
            existing.last_seen = timestamp
            existing.times_observed += 1
            existing.description = description
            existing.current_state = current_state
            existing.observations.append(observation)
            alpha = 0.3
            existing.allo_angle = _normalize_angle(
                existing.allo_angle * (1 - alpha) + allo_angle * alpha
            )
            existing.distance = existing.distance * (1 - alpha) + distance * alpha
            return existing

        entity = Entity(
            id=f"ENT_{uuid.uuid4().hex[:8].upper()}",
            label=label,
            category=category,
            allo_angle=allo_angle,
            distance=distance,
            description=description,
            current_state=current_state,
            observations=[observation],
            confidence=confidence,
            first_seen=timestamp,
            last_seen=timestamp,
            times_observed=1,
        )
        self.entities[entity.id] = entity
        return entity

    def add_relationship(
        self,
        subject_label: str,
        object_label: str,
        relation: str,
        timestamp: float,
        confidence: float = 1.0,
    ) -> Optional[Relationship]:
        """Add a relationship between two entities (looked up by label)."""
        subject = self._find_entity_by_label(subject_label)
        obj = self._find_entity_by_label(object_label)
        if not subject or not obj:
            return None

        rel = Relationship(
            subject_id=subject.id,
            object_id=obj.id,
            relation=relation,
            timestamp=timestamp,
            confidence=confidence,
        )
        self.relationships.append(rel)
        return rel

    def add_scene_description(self, description: str, timestamp: float, frame_idx: int = 0):
        """Store a scene-level description for a keyframe."""
        self.scene_descriptions.append({
            "timestamp": round(timestamp, 2),
            "frame_idx": frame_idx,
            "description": description,
        })

    # ── Spatial Queries ──────────────────────────────────────────────────

    def query_angle(
        self,
        query_angle_deg: float,
        fov: float = 90.0,
        min_confidence: float = 0.3,
    ) -> list[Entity]:
        """Get all entities within an angular arc."""
        half_fov = fov / 2
        results = []
        for entity in self.entities.values():
            if entity.confidence < min_confidence:
                continue
            angular_diff = abs(_angle_diff(entity.allo_angle, query_angle_deg))
            if angular_diff <= half_fov:
                results.append(entity)
        return sorted(results, key=lambda e: e.confidence, reverse=True)

    def query_direction(
        self,
        direction: str,
        min_confidence: float = 0.3,
        category: str | None = None,
    ) -> list[Entity]:
        """Query entities by direction relative to current heading."""
        direction_map = {
            "front": 0, "ahead": 0, "forward": 0,
            "right": 90,
            "behind": 180, "back": 180, "rear": 180,
            "left": -90,
            "front-right": 45, "front-left": -45,
            "behind-right": 135, "rear-right": 135,
            "behind-left": -135, "rear-left": -135,
        }

        direction = direction.lower().strip()
        if direction not in direction_map:
            return []

        relative_angle = direction_map[direction]
        allo_query = _normalize_angle(self.current_heading + relative_angle)
        results = self.query_angle(allo_query, fov=90, min_confidence=min_confidence)

        if category:
            results = [e for e in results if e.category.lower() == category.lower()]
        return results

    def query_direction_at_time(
        self,
        direction: str,
        timestamp: float,
        min_confidence: float = 0.1,
        category: str | None = None,
    ) -> list[Entity]:
        """Query entities by direction relative to heading at a specific past time."""
        direction_map = {
            "front": 0, "ahead": 0, "forward": 0,
            "right": 90,
            "behind": 180, "back": 180, "rear": 180,
            "left": -90,
            "front-right": 45, "front-left": -45,
            "behind-right": 135, "rear-right": 135,
            "behind-left": -135, "rear-left": -135,
        }

        direction = direction.lower().strip()
        if direction not in direction_map:
            return []

        heading = self.heading_at_time(timestamp)
        relative_angle = direction_map[direction]
        allo_query = _normalize_angle(heading + relative_angle)

        # Only include entities that were visible around that time
        window = 3.0
        results = []
        half_fov = 45.0
        for entity in self.entities.values():
            if entity.confidence < min_confidence:
                continue
            # Was it visible near this time?
            if not (entity.first_seen <= timestamp + window and entity.last_seen >= timestamp - window):
                continue
            angular_diff = abs(_angle_diff(entity.allo_angle, allo_query))
            if angular_diff <= half_fov:
                results.append(entity)

        if category:
            results = [e for e in results if e.category.lower() == category.lower()]
        return sorted(results, key=lambda e: e.confidence, reverse=True)

    def query_time_range(
        self,
        start_time: float,
        end_time: float,
        min_confidence: float = 0.1,
    ) -> list[Entity]:
        """Get entities visible during a time window."""
        results = []
        for entity in self.entities.values():
            if entity.confidence < min_confidence:
                continue
            if entity.first_seen <= end_time and entity.last_seen >= start_time:
                results.append(entity)
        return sorted(results, key=lambda e: e.confidence, reverse=True)

    def query_by_label(self, label: str, fuzzy: bool = True) -> list[Entity]:
        """Find entities matching a label."""
        results = []
        label_lower = label.lower()
        for entity in self.entities.values():
            if fuzzy:
                if label_lower in entity.label.lower() or entity.label.lower() in label_lower:
                    results.append(entity)
            else:
                if entity.label.lower() == label_lower:
                    results.append(entity)
        return results

    def query_by_category(self, category: str, min_confidence: float = 0.3) -> list[Entity]:
        """Find all entities of a given category."""
        return [
            e for e in self.entities.values()
            if e.category.lower() == category.lower() and e.confidence >= min_confidence
        ]

    def query_relationships(
        self,
        entity_label: str | None = None,
        relation_type: str | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> list[dict]:
        """Query relationships with optional filters. Returns enriched dicts."""
        results = []
        for rel in self.relationships:
            if relation_type and rel.relation != relation_type:
                continue
            if time_range and not (time_range[0] <= rel.timestamp <= time_range[1]):
                continue
            subject = self.entities.get(rel.subject_id)
            obj = self.entities.get(rel.object_id)
            if not subject or not obj:
                continue
            if entity_label:
                el = entity_label.lower()
                if el not in subject.label.lower() and el not in obj.label.lower():
                    continue
            results.append({
                "subject": subject.label,
                "relation": rel.relation,
                "object": obj.label,
                "timestamp": round(rel.timestamp, 1),
            })
        return results

    def get_entity_timeline(self, entity_id: str) -> list[dict]:
        """Get the full observation timeline for an entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            return []
        return [
            {
                "timestamp": round(obs.timestamp, 1),
                "frame_idx": obs.frame_idx,
                "angle": round(obs.allo_angle, 1),
                "distance": round(obs.distance, 2),
                "description": obs.description,
            }
            for obs in entity.observations
        ]

    def detect_changes(
        self,
        similarity_threshold: float = 0.5,
        min_observations: int = 2,
    ) -> list[StateChange]:
        """
        Detect state changes across entity timelines.

        Compares consecutive observations for each entity. If the description
        similarity drops below the threshold OR the state field changes,
        it's flagged as a state change.

        Returns list of StateChange objects sorted by time.
        """
        changes = []
        for entity in self.entities.values():
            if len(entity.observations) < min_observations:
                continue
            for i in range(1, len(entity.observations)):
                prev = entity.observations[i - 1]
                curr = entity.observations[i]

                # Check state field change
                state_changed = (
                    prev.description != curr.description
                    and prev.description and curr.description
                )
                if not state_changed:
                    continue

                sim = _description_similarity(prev.description, curr.description)
                if sim < similarity_threshold:
                    changes.append(StateChange(
                        entity_id=entity.id,
                        entity_label=entity.label,
                        category=entity.category,
                        timestamp_before=prev.timestamp,
                        timestamp_after=curr.timestamp,
                        state_before=prev.description,
                        state_after=curr.description,
                        description_before=prev.description,
                        description_after=curr.description,
                        similarity=sim,
                    ))

        return sorted(changes, key=lambda c: c.timestamp_after)

    def get_changes_for_entity(self, label: str) -> list[StateChange]:
        """Get state changes for a specific entity."""
        all_changes = self.detect_changes()
        label_lower = label.lower()
        return [
            c for c in all_changes
            if label_lower in c.entity_label.lower()
            or c.entity_label.lower() in label_lower
        ]

    def get_progress_summary(self) -> dict:
        """Get a summary of all detected changes — the progress tracking view."""
        changes = self.detect_changes()
        if not changes:
            return {
                "total_changes": 0,
                "message": "No state changes detected.",
                "changes": [],
            }

        by_entity = {}
        for c in changes:
            if c.entity_label not in by_entity:
                by_entity[c.entity_label] = []
            by_entity[c.entity_label].append(c.to_dict())

        return {
            "total_changes": len(changes),
            "entities_with_changes": len(by_entity),
            "time_range": {
                "first_change": round(changes[0].timestamp_after, 1),
                "last_change": round(changes[-1].timestamp_after, 1),
            },
            "by_entity": by_entity,
            "changes": [c.to_dict() for c in changes],
        }

    def describe_spatial_relation(self, entity_a_label: str, entity_b_label: str) -> str:
        """Compute geometric spatial relationship between two entities."""
        a_list = self.query_by_label(entity_a_label)
        b_list = self.query_by_label(entity_b_label)
        if not a_list:
            return f"Entity '{entity_a_label}' not found."
        if not b_list:
            return f"Entity '{entity_b_label}' not found."
        a, b = a_list[0], b_list[0]
        angle_diff = _angle_diff(b.allo_angle, a.allo_angle)

        if abs(angle_diff) < 30:
            direction = "directly ahead of"
        elif abs(angle_diff) > 150:
            direction = "behind"
        elif 30 <= angle_diff < 70:
            direction = "to the front-right of"
        elif 70 <= angle_diff < 110:
            direction = "to the right of"
        elif angle_diff >= 110:
            direction = "to the rear-right of"
        elif -70 < angle_diff <= -30:
            direction = "to the front-left of"
        elif -110 < angle_diff <= -70:
            direction = "to the left of"
        else:
            direction = "to the rear-left of"

        return (
            f"{b.label} is {direction} {a.label} "
            f"(angular separation: {abs(angle_diff):.0f} degrees)"
        )

    def describe_relative_to_camera(self, entity: Entity) -> str:
        """Describe entity position relative to current camera heading."""
        ego_angle = self.transformer.allo_to_ego(entity.allo_angle, self.current_heading)

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
            f"{entity.label} ({entity.category}) — {direction} "
            f"(~{abs(ego_angle):.0f}°). {entity.description}"
        )

    # ── Summary ──────────────────────────────────────────────────────────

    def get_all(self, min_confidence: float = 0.1) -> list[Entity]:
        """Get all entities above confidence threshold."""
        return sorted(
            [e for e in self.entities.values() if e.confidence >= min_confidence],
            key=lambda e: e.confidence,
            reverse=True,
        )

    def get_summary(self) -> dict:
        """Get a summary of the scene graph state."""
        all_entities = self.get_all()
        categories = {}
        for e in all_entities:
            categories[e.category] = categories.get(e.category, 0) + 1
        changes = self.detect_changes()
        return {
            "total_entities": len(all_entities),
            "total_relationships": len(self.relationships),
            "total_state_changes": len(changes),
            "current_heading": round(self.current_heading, 1),
            "by_category": categories,
            "entities": [e.to_dict() for e in all_entities],
            "scene_descriptions": self.scene_descriptions[-5:],
        }

    # ── Decay + pruning ──────────────────────────────────────────────────

    def decay_confidence(self, dt: float) -> None:
        for entity in self.entities.values():
            entity.confidence = max(0.0, entity.confidence - self.decay_rate * dt)

    def prune_stale(self) -> list[str]:
        stale_ids = [eid for eid, e in self.entities.items() if e.is_stale]
        for eid in stale_ids:
            del self.entities[eid]
        return stale_ids

    # ── Internal ─────────────────────────────────────────────────────────

    def _find_match(
        self,
        label: str,
        category: str,
        allo_angle: float,
        distance: float,
        description: str = "",
    ) -> Optional[Entity]:
        """
        Find existing entity for merging.

        Matching criteria:
        1. Label match (exact or substring)
        2. Category match (exact)
        3. Spatial proximity (angle ≤ 20°, distance ≤ 0.25)
        4. Description similarity check for "person" category — prevents
           merging two different workers with generic labels like "worker"
        """
        label_lower = label.lower()
        cat_lower = category.lower()
        best_match = None
        best_score = -1.0

        for entity in self.entities.values():
            # Label match: exact, substring, or shared root words
            ent_label = entity.label.lower()
            label_match = (
                ent_label == label_lower
                or (len(label_lower) > 3 and label_lower in ent_label)
                or (len(ent_label) > 3 and ent_label in label_lower)
                or _labels_share_root(label, entity.label)
            )
            if not label_match:
                continue
            if entity.category.lower() != cat_lower:
                continue
            angle_diff = abs(_angle_diff(entity.allo_angle, allo_angle))
            dist_diff = abs(entity.distance - distance)
            # Wider merge threshold for non-person entities (structures/background
            # are stationary, person positions vary more meaningfully)
            angle_thresh = self.merge_angle_threshold
            dist_thresh = self.merge_distance_threshold
            if cat_lower != "person":
                angle_thresh *= 2.0  # 40° for structures/equipment
                dist_thresh *= 2.0   # 0.5 for structures/equipment
            if angle_diff > angle_thresh or dist_diff > dist_thresh:
                continue

            # Check description similarity to prevent merging distinct entities
            # that happen to share a generic label (e.g., two different "worker"s)
            if description and entity.description and cat_lower == "person":
                desc_sim = _description_similarity(description, entity.description)
                # For persons: if descriptions are very different, don't merge
                # even if label is exact match. This prevents merging two
                # workers both labeled "worker" who look completely different.
                # Numbered labels (worker_1 == worker_1) still merge because
                # Gemini assigns specific numbers intentionally.
                has_specific_id = any(c.isdigit() for c in label_lower)
                if desc_sim < 0.2 and not has_specific_id:
                    continue

            # Score: prefer closer spatial matches
            spatial_score = 1.0 - (angle_diff / self.merge_angle_threshold * 0.5
                                   + dist_diff / self.merge_distance_threshold * 0.5)
            if spatial_score > best_score:
                best_score = spatial_score
                best_match = entity

        return best_match

    def _find_entity_by_label(self, label: str) -> Optional[Entity]:
        """Find best-matching entity by label."""
        label_lower = label.lower()
        # Exact match first
        for entity in self.entities.values():
            if entity.label.lower() == label_lower:
                return entity
        # Fuzzy fallback
        for entity in self.entities.values():
            if label_lower in entity.label.lower() or entity.label.lower() in label_lower:
                return entity
        return None
