"""
Scene Agent — Spatio-Temporal Video Understanding via Scene Graph

Processes video into a spatio-temporal scene graph and answers
"what, where, when" questions using tool-calling with Gemini.

Replaces VestaAgent for the scene graph pipeline.
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from vesta.flow.optical_flow import CameraMotion, estimate_camera_motion
from vesta.detection.gemini_detector import KeyframeSampler
from vesta.detection.scene_descriptor import describe_scene_async, SceneAnalysis
from vesta.registry.scene_graph import SceneGraph


class SceneAgent:
    """
    Processes video into a scene graph and answers spatio-temporal queries.

    Pipeline:
      Pass 1+2 (overlapped): Optical flow + submit keyframes to Gemini
      Pass 3 (fast): Build scene graph from results + write annotated video
    """

    def __init__(
        self,
        video_path: str | None = None,
        keyframe_interval: int = 30,
        model: str = "gemini-2.5-flash",
        fov_degrees: float = 90.0,
        max_workers: int = 10,
        verbose: bool = True,
    ):
        self.video_path = video_path
        self.model = model
        self.verbose = verbose
        self.max_workers = max_workers

        self.graph = SceneGraph(fov_degrees=fov_degrees)
        self.sampler = KeyframeSampler(interval=keyframe_interval)

        self.frame_count = 0
        self.fps = 30.0
        self.processed = False
        self.motions: list[CameraMotion] = []

    def process(
        self,
        max_frames: int | None = None,
        output_video: str | None = None,
    ) -> dict:
        """
        Run the pipeline with overlapped optical flow and Gemini detection.
        """
        if not self.video_path:
            raise ValueError("No video_path set")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.verbose:
            print(f"[VESTA] Processing: {self.video_path}")
            print(f"[VESTA] {total_frames} frames @ {self.fps:.1f} FPS = {total_frames/self.fps:.1f}s")

        # ── Pass 1: Optical flow (read all frames, compute motion) ──
        t_start = time.time()
        if self.verbose:
            print(f"[VESTA] Pass 1: Optical flow...")

        all_frames = []
        motions = []
        keyframe_indices = []
        keyframe_headings = []
        keyframe_frames = {}  # frame_idx → frame copy

        prev_frame = None
        frame_idx = 0
        cumulative_heading = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            motion = CameraMotion(0, 0, 0, 1.0)
            if prev_frame is not None:
                motion = estimate_camera_motion(prev_frame, frame)
                cumulative_heading += motion.d_theta

            motions.append(motion)
            if output_video:
                all_frames.append(frame)

            if self.sampler.should_sample(frame_idx, motion):
                keyframe_indices.append(frame_idx)
                keyframe_headings.append(cumulative_heading)
                keyframe_frames[frame_idx] = frame.copy()
                if self.verbose:
                    print(f"[VESTA] Keyframe #{len(keyframe_indices)} @ frame {frame_idx}")

            prev_frame = frame
            frame_idx += 1

        cap.release()
        self.frame_count = frame_idx

        t_flow_done = time.time() - t_start
        if self.verbose:
            print(f"[VESTA] Pass 1 done: {frame_idx} frames in {t_flow_done:.1f}s, "
                  f"{len(keyframe_indices)} keyframes found")

        # ── Pass 2: Async Gemini calls (all keyframes at once) ──
        if self.verbose:
            print(f"[VESTA] Pass 2: Sending {len(keyframe_indices)} keyframes to Gemini "
                  f"(max {self.max_workers} concurrent)...")

        scene_results = self._run_gemini_async(keyframe_frames)

        t_pass12 = time.time() - t_start
        if self.verbose:
            print(f"[VESTA] Pass 1+2 done: {len(scene_results)} responses in {t_pass12:.1f}s")

        # ── Pass 3: Build scene graph + write video ──
        t_pass3 = time.time()
        if self.verbose:
            print(f"[VESTA] Pass 3: Building scene graph + writing video...")

        writer = None
        visualizer = None
        if output_video and all_frames:
            from vesta.utils.visualizer import VideoVisualizer
            visualizer = VideoVisualizer(frame_w, frame_h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, self.fps, (frame_w, frame_h))

        keyframe_data = {}
        for ki, kh in zip(keyframe_indices, keyframe_headings):
            if ki in scene_results:
                keyframe_data[ki] = (scene_results[ki], kh)

        entity_count = 0
        rel_count = 0
        self.graph.current_heading = 0.0

        for i in range(self.frame_count):
            timestamp = i / self.fps

            if i > 0:
                self.graph.update_with_motion(motions[i], timestamp=timestamp)
            else:
                self.graph.record_heading(0.0)

            is_keyframe = i in keyframe_data
            current_detections = None

            if is_keyframe:
                analysis, _ = keyframe_data[i]

                if analysis.scene_description:
                    self.graph.add_scene_description(analysis.scene_description, timestamp, i)

                if self.verbose:
                    kf_num = keyframe_indices.index(i) + 1
                    print(
                        f"[VESTA] Keyframe #{kf_num} @ T={timestamp:.1f}s "
                        f"(frame {i}), heading={self.graph.current_heading:.1f}°"
                    )
                    if analysis.scene_description:
                        print(f"  Scene: {analysis.scene_description[:100]}")

                # Add entities
                current_detections = analysis.entities
                for ent in analysis.entities:
                    entry = self.graph.add_entity(
                        label=ent.label,
                        category=ent.category,
                        description=ent.description,
                        current_state=ent.state,
                        x_normalized=ent.x,
                        y_normalized=ent.y,
                        confidence=ent.confidence,
                        timestamp=timestamp,
                        frame_idx=i,
                        bbox=(ent.x1, ent.y1, ent.x2, ent.y2),
                    )
                    entity_count += 1
                    if self.verbose:
                        print(f"  + {entry.label} ({entry.category}) at {entry.allo_angle:.0f}°")

                # Add relationships
                for rel in analysis.relationships:
                    r = self.graph.add_relationship(
                        rel.subject, rel.object, rel.relation, timestamp
                    )
                    if r:
                        rel_count += 1

            # Write annotated frame
            if writer and visualizer and all_frames:
                annotated = visualizer.annotate_frame(
                    all_frames[i],
                    self.graph,
                    current_detections=current_detections,
                    frame_idx=i,
                    fps=self.fps,
                    is_keyframe=is_keyframe,
                )
                writer.write(annotated)

            if i % 30 == 0:
                self.graph.decay_confidence(1.0 / self.fps * 30)

        if writer:
            writer.release()

        self.motions = motions
        self.processed = True
        t_total = time.time() - t_start

        summary = self.graph.get_summary()
        summary["frames_processed"] = self.frame_count
        summary["keyframes_analyzed"] = len(keyframe_indices)
        summary["total_entity_observations"] = entity_count
        summary["total_relationship_observations"] = rel_count

        if self.verbose:
            print(f"\n[VESTA] Done in {t_total:.1f}s total.")
            print(f"[VESTA]   Flow + Gemini (overlapped): {t_pass12:.1f}s")
            print(f"[VESTA]   Scene graph + video:        {time.time() - t_pass3:.1f}s")
            print(f"[VESTA] {self.frame_count} frames, {len(keyframe_indices)} keyframes, "
                  f"{summary['total_entities']} entities, "
                  f"{summary['total_relationships']} relationships.")
            if output_video:
                print(f"[VESTA] Video saved: {output_video}")

        return summary

    def _run_gemini_async(self, keyframe_frames: dict[int, any]) -> dict[int, SceneAnalysis]:
        """
        Run all Gemini keyframe calls concurrently using async I/O.

        Uses asyncio + semaphore for controlled concurrency — much faster than
        ThreadPoolExecutor because we're I/O bound (waiting for HTTP responses).
        """
        async def _process_all():
            sem = asyncio.Semaphore(self.max_workers)
            results = {}

            async def _process_one(fidx, frame):
                async with sem:
                    try:
                        result = await describe_scene_async(frame, self.model)
                        if self.verbose and result.entities:
                            print(f"[VESTA] ✓ Frame {fidx}: {len(result.entities)} entities")
                        elif self.verbose:
                            print(f"[VESTA] ✗ Frame {fidx}: parse failed, 0 entities")
                        return fidx, result
                    except Exception as e:
                        if self.verbose:
                            print(f"[VESTA] ✗ Frame {fidx}: {e}")
                        return fidx, SceneAnalysis()

            tasks = [_process_one(fidx, frame) for fidx, frame in keyframe_frames.items()]
            completed = await asyncio.gather(*tasks)

            for fidx, result in completed:
                results[fidx] = result
            return results

        # Run the async event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context (e.g., Jupyter) — use thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    return pool.submit(lambda: asyncio.run(_process_all())).result()
            else:
                return loop.run_until_complete(_process_all())
        except RuntimeError:
            return asyncio.run(_process_all())

    # ── Tool Functions ───────────────────────────────────────────────────

    def tool_get_entities_in_direction(self, direction: str, category: str = "") -> str:
        """Get all entities in a named direction."""
        entities = self.graph.query_direction(
            direction, category=category if category else None
        )
        if not entities:
            return json.dumps({"entities": [], "message": f"No entities detected {direction}."})
        return json.dumps({
            "direction": direction,
            "entities": [
                {
                    **e.to_dict(),
                    "relative_description": self.graph.describe_relative_to_camera(e),
                }
                for e in entities
            ],
        })

    def tool_get_entity_info(self, label: str) -> str:
        """Get detailed info about a specific entity."""
        entities = self.graph.query_by_label(label)
        if not entities:
            return json.dumps({"error": f"Entity '{label}' not found."})
        entity = entities[0]
        timeline = self.graph.get_entity_timeline(entity.id)
        rels = self.graph.query_relationships(entity_label=label)
        return json.dumps({
            "entity": entity.to_dict(),
            "relative_position": self.graph.describe_relative_to_camera(entity),
            "timeline": timeline,
            "relationships": rels,
        })

    def tool_get_all_entities(self) -> str:
        """Get the full scene graph summary."""
        return json.dumps(self.graph.get_summary())

    def tool_get_entities_at_time(self, time_seconds: float, window: float = 2.0) -> str:
        """Get entities visible at a specific timestamp."""
        start = max(0, time_seconds - window / 2)
        end = time_seconds + window / 2
        entities = self.graph.query_time_range(start, end)
        if not entities:
            return json.dumps({
                "entities": [],
                "message": f"No entities detected between T={start:.1f}s and T={end:.1f}s",
            })
        return json.dumps({
            "time_query": f"T={time_seconds:.1f}s (window: {start:.1f}s-{end:.1f}s)",
            "entities": [e.to_dict() for e in entities],
        })

    def tool_get_spatial_relation(self, entity_a: str, entity_b: str) -> str:
        """Get the spatial relationship between two entities."""
        result = self.graph.describe_spatial_relation(entity_a, entity_b)
        return json.dumps({"spatial_relation": result})

    def tool_get_entity_timeline(self, label: str) -> str:
        """Get the observation history of an entity over time."""
        entities = self.graph.query_by_label(label)
        if not entities:
            return json.dumps({"error": f"Entity '{label}' not found."})
        entity = entities[0]
        timeline = self.graph.get_entity_timeline(entity.id)
        return json.dumps({
            "entity": entity.label,
            "category": entity.category,
            "first_seen": entity.first_seen,
            "last_seen": entity.last_seen,
            "times_observed": entity.times_observed,
            "observations": timeline,
        })

    def tool_get_relationships(self, entity_label: str = "", relation_type: str = "") -> str:
        """Get relationships filtered by entity and/or relation type."""
        rels = self.graph.query_relationships(
            entity_label=entity_label if entity_label else None,
            relation_type=relation_type if relation_type else None,
        )
        if not rels:
            return json.dumps({"relationships": [], "message": "No matching relationships found."})
        return json.dumps({"relationships": rels})

    def tool_get_direction_at_time(self, direction: str, time_seconds: float, category: str = "") -> str:
        """Get entities in a direction relative to where the camera was facing at a specific time."""
        entities = self.graph.query_direction_at_time(
            direction, time_seconds,
            category=category if category else None,
        )
        heading = self.graph.heading_at_time(time_seconds)
        if not entities:
            return json.dumps({
                "entities": [],
                "message": f"No entities detected {direction} at T={time_seconds:.1f}s (heading was {heading:.1f}°).",
            })
        return json.dumps({
            "direction": direction,
            "time": time_seconds,
            "heading_at_time": round(heading, 1),
            "entities": [
                {
                    **e.to_dict(),
                    "description": e.description,
                }
                for e in entities
            ],
        })

    def tool_get_changes(self, entity_label: str = "") -> str:
        """Get detected state changes across entity timelines."""
        if entity_label:
            changes = self.graph.get_changes_for_entity(entity_label)
            if not changes:
                return json.dumps({
                    "changes": [],
                    "message": f"No state changes detected for '{entity_label}'.",
                })
            return json.dumps({
                "entity": entity_label,
                "total_changes": len(changes),
                "changes": [c.to_dict() for c in changes],
            })
        else:
            return json.dumps(self.graph.get_progress_summary())

    # ── Tool Declarations for Gemini Function Calling ───────────────────

    def _get_tool_declarations(self):
        """Return Gemini function declarations for all scene graph tools."""
        from google.genai import types

        return [
            types.FunctionDeclaration(
                name="get_entities_in_direction",
                description="Get all tracked entities in a compass direction relative to the camera's current heading. Use for 'what's behind/left/right/in front of me?' questions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "direction": types.Schema(type="STRING", description="Direction: front, behind, left, right, front-left, front-right, behind-left, behind-right"),
                        "category": types.Schema(type="STRING", description="Optional filter: person, equipment, structure, material, vehicle"),
                    },
                    required=["direction"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_entity_info",
                description="Get detailed info about a specific entity including its position, timeline, and relationships. Use when asking about a specific object/person.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "label": types.Schema(type="STRING", description="Entity label (e.g. 'worker_1', 'crane_1', 'wall_1')"),
                    },
                    required=["label"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_entities_at_time",
                description="Get all entities visible at a specific timestamp. Use for 'what was visible at Xs?' questions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "time_seconds": types.Schema(type="NUMBER", description="Timestamp in seconds"),
                        "window": types.Schema(type="NUMBER", description="Time window in seconds (default 2.0)"),
                    },
                    required=["time_seconds"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_spatial_relation",
                description="Get the spatial relationship between two entities (angle, direction, distance). Use for 'is X left or right of Y?' questions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "entity_a": types.Schema(type="STRING", description="First entity label"),
                        "entity_b": types.Schema(type="STRING", description="Second entity label"),
                    },
                    required=["entity_a", "entity_b"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_entity_timeline",
                description="Get the full observation history of an entity over time (timestamps, positions, state changes). Use for 'when did X appear?', 'how long was X visible?', temporal questions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "label": types.Schema(type="STRING", description="Entity label"),
                    },
                    required=["label"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_relationships",
                description="Get spatial relationships between entities (near, left_of, on_top_of, etc). Use for 'what's near X?', 'what's on top of Y?' questions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "entity_label": types.Schema(type="STRING", description="Filter by entity label (optional)"),
                        "relation_type": types.Schema(type="STRING", description="Filter by relation type (optional)"),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="get_direction_at_time",
                description="Get entities in a direction relative to where the camera was facing at a SPECIFIC PAST TIME. Use for 'what was behind me at 30s?' questions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "direction": types.Schema(type="STRING", description="Direction: front, behind, left, right"),
                        "time_seconds": types.Schema(type="NUMBER", description="Timestamp in seconds"),
                        "category": types.Schema(type="STRING", description="Optional category filter"),
                    },
                    required=["direction", "time_seconds"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_changes",
                description="Get detected state changes for an entity or the whole scene over time. Use for 'what changed?', 'how did X change?', 'idle time' questions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "entity_label": types.Schema(type="STRING", description="Entity label (optional, empty for all changes)"),
                    },
                ),
            ),
        ]

    def _execute_tool_call(self, function_call) -> str:
        """Execute a tool call and return the result as a string."""
        name = function_call.name
        args = dict(function_call.args) if function_call.args else {}

        tool_map = {
            "get_entities_in_direction": lambda: self.tool_get_entities_in_direction(
                args.get("direction", "front"), args.get("category", "")),
            "get_entity_info": lambda: self.tool_get_entity_info(args.get("label", "")),
            "get_all_entities": lambda: self.tool_get_all_entities(),
            "get_entities_at_time": lambda: self.tool_get_entities_at_time(
                args.get("time_seconds", 0), args.get("window", 2.0)),
            "get_spatial_relation": lambda: self.tool_get_spatial_relation(
                args.get("entity_a", ""), args.get("entity_b", "")),
            "get_entity_timeline": lambda: self.tool_get_entity_timeline(args.get("label", "")),
            "get_relationships": lambda: self.tool_get_relationships(
                args.get("entity_label", ""), args.get("relation_type", "")),
            "get_direction_at_time": lambda: self.tool_get_direction_at_time(
                args.get("direction", "front"), args.get("time_seconds", 0), args.get("category", "")),
            "get_changes": lambda: self.tool_get_changes(args.get("entity_label", "")),
        }

        if name in tool_map:
            return tool_map[name]()
        return json.dumps({"error": f"Unknown tool: {name}"})
        return "[VESTA] Unable to generate response."

    # ── Natural Language Q&A (RAG approach) ──────────────────────────────

    def _build_context(self, question: str) -> str:
        """
        Build a rich context document from the scene graph, tailored to the question.

        This is the RAG retrieval step: we pull ALL relevant data from the scene
        graph and inject it as context, so Gemini can reason freely over it.
        """
        import re
        q = question.lower()
        sections = []
        video_duration = self.frame_count / self.fps

        # ── Always include: scene overview ──
        summary = self.graph.get_summary()
        cats = ", ".join(f"{v} {k}" for k, v in summary["by_category"].items())
        sections.append(
            f"## Video Overview\n"
            f"Duration: {video_duration:.1f}s ({self.frame_count} frames @ {self.fps:.0f} FPS)\n"
            f"Entities tracked: {summary['total_entities']} ({cats})\n"
            f"Relationships: {summary['total_relationships']}\n"
            f"Camera final heading: {summary['current_heading']}°"
        )

        # ── Scene descriptions (sampled) ──
        descs = self.graph.scene_descriptions
        if descs:
            sampled = descs[::max(1, len(descs) // 6)]  # ~6 evenly spaced
            lines = [f"## Scene Descriptions (sampled across video)"]
            for d in sampled:
                lines.append(f"- T={d['timestamp']:.1f}s: {d['description']}")
            sections.append("\n".join(lines))

        # ── Entity list with positions ──
        all_entities = self.graph.get_all(min_confidence=0.15)
        if all_entities:
            lines = ["## All Tracked Entities"]
            for e in all_entities:
                rel = self.graph.describe_relative_to_camera(e)
                lines.append(
                    f"- **{e.label}** [{e.category}]: {e.description} | "
                    f"angle={e.allo_angle:.1f}° dist={e.distance:.2f} | "
                    f"seen {e.first_seen:.1f}s–{e.last_seen:.1f}s ({e.times_observed}x) | "
                    f"state: {e.current_state or 'unknown'} | "
                    f"confidence: {e.confidence:.2f} | "
                    f"Position: {rel}"
                )
            sections.append("\n".join(lines))

        # ── Detect if question is about a specific time ──
        time_match = re.search(r"(\d+)\s*s(?:econds?)?", q)
        if not time_match:
            time_match = re.search(r"(\d+)\s*min(?:ute)?s?", q)
            if time_match:
                query_time = float(time_match.group(1)) * 60
            else:
                query_time = None
        else:
            query_time = float(time_match.group(1))

        if query_time is not None:
            # Add entities visible at that time
            entities_at_t = self.graph.query_time_range(
                max(0, query_time - 2), query_time + 2
            )
            if entities_at_t:
                lines = [f"## Entities visible around T={query_time:.0f}s"]
                for e in entities_at_t:
                    lines.append(f"- {e.label} [{e.category}]: {e.description} (state: {e.current_state})")
                sections.append("\n".join(lines))

            # Add heading at that time
            heading = self.graph.heading_at_time(query_time)
            sections.append(f"## Camera heading at T={query_time:.0f}s: {heading:.1f}°")

            # Add spatial directions at that time
            for direction in ["front", "behind", "left", "right"]:
                ents = self.graph.query_direction_at_time(direction, query_time)
                if ents:
                    names = ", ".join(f"{e.label} ({e.description[:40]})" for e in ents[:5])
                    sections.append(f"At T={query_time:.0f}s, {direction}: {names}")

        # ── If question mentions a specific entity, include its full timeline ──
        mentioned_entities = []
        for e in all_entities:
            if e.label.lower() in q or any(
                word in q for word in e.label.lower().split("_") if len(word) > 3
            ):
                mentioned_entities.append(e)

        # Also check for self-referential questions
        self_words = ["i ", "me ", "my ", "self", "idle", "working", "doing", "person"]
        if any(w in q for w in self_words):
            for label in ["camera_self", "worker_1"]:
                ents = self.graph.query_by_label(label)
                if ents and ents[0] not in mentioned_entities:
                    mentioned_entities.append(ents[0])

        for e in mentioned_entities[:3]:  # Limit to 3 entity timelines
            timeline = self.graph.get_entity_timeline(e.id)
            if timeline:
                lines = [f"## Timeline for {e.label} ({e.category})"]
                lines.append(f"First seen: {e.first_seen:.1f}s, Last seen: {e.last_seen:.1f}s")
                for obs in timeline:
                    lines.append(
                        f"- T={obs['timestamp']:.1f}s: {obs['description']} "
                        f"(angle={obs['angle']:.1f}°)"
                    )
                sections.append("\n".join(lines))

            # Also get relationships for mentioned entities
            rels = self.graph.query_relationships(entity_label=e.label)
            if rels:
                lines = [f"## Relationships for {e.label}"]
                for r in rels[:10]:
                    lines.append(f"- {r['subject']} {r['relation']} {r['object']} (T={r['timestamp']:.1f}s)")
                sections.append("\n".join(lines))

        # ── State changes (if question is about changes/progress) ──
        change_words = ["change", "different", "progress", "evolve", "before", "after", "idle", "working"]
        if any(w in q for w in change_words):
            changes = self.graph.detect_changes()
            if changes:
                lines = ["## Detected State Changes"]
                for c in changes[:15]:
                    lines.append(
                        f"- {c.entity_label}: T={c.timestamp_before:.1f}s→{c.timestamp_after:.1f}s | "
                        f"'{c.state_before[:60]}' → '{c.state_after[:60]}'"
                    )
                sections.append("\n".join(lines))

        # ── Spatial relationships (if question is spatial) ──
        spatial_words = ["where", "left", "right", "behind", "front", "near", "next to", "relative", "direction"]
        if any(w in q for w in spatial_words):
            # Include key spatial relations
            rels = self.graph.query_relationships()
            if rels:
                # Deduplicate and take most recent
                seen = set()
                unique_rels = []
                for r in reversed(rels):
                    key = (r["subject"], r["relation"], r["object"])
                    if key not in seen:
                        seen.add(key)
                        unique_rels.append(r)
                if unique_rels:
                    lines = ["## Key Spatial Relationships"]
                    for r in unique_rels[:15]:
                        lines.append(f"- {r['subject']} {r['relation']} {r['object']}")
                    sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def ask(self, question: str, max_rounds: int = 5) -> str:
        """
        Agentic RAG: context injection + tool calling in a multi-turn loop.

        1. Build scene graph context (RAG retrieval)
        2. Attach video if available (visual fallback)
        3. Provide tools for precise spatial/temporal queries
        4. Gemini reads context, optionally calls tools, then answers
        5. Loop up to max_rounds if Gemini needs more tool calls
        """
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        context = self._build_context(question)

        system_prompt = (
            f"You are VESTA, a spatio-temporal video understanding agent.\n\n"
            f"You have THREE sources of information:\n"
            f"1. SCENE GRAPH — structured data with precise entity positions (angles, directions), "
            f"temporal timestamps (first_seen, last_seen), object permanence, and state changes. "
            f"This is your PRIMARY advantage over raw video models.\n"
            f"2. VIDEO — the original first-person (egocentric) construction site video is attached. "
            f"Use it when the scene graph lacks a specific entity or event.\n"
            f"3. TOOLS — you can call scene graph tools for precise spatial/temporal queries "
            f"(directions, entity timelines, spatial relations, changes). Use them when you need "
            f"exact data that isn't already in the context.\n\n"
            f"HOW TO ANSWER:\n"
            f"- Read the scene graph context FIRST. If it has what you need, answer directly.\n"
            f"- If you need more precision (exact angle, timeline, spatial relation), call a tool.\n"
            f"- If the scene graph doesn't have the entity/event at all, use the video.\n"
            f"- ALWAYS answer. NEVER refuse, say 'can't determine', return -1, or say 'unknown'.\n"
            f"- For spatial questions, give precise directions (left/right/front/back + angle).\n"
            f"- For temporal questions, give specific timestamps.\n"
            f"- Be concise. Match the format the question asks for.\n"
        )

        # Build initial message: video (optional) + context + question
        user_parts = []

        # Attach video if available
        if self.video_path and os.path.exists(self.video_path):
            video_bytes = open(self.video_path, "rb").read()
            video_size = len(video_bytes)
            if video_size <= 20 * 1024 * 1024:  # 20MB inline limit
                user_parts.append(
                    types.Part(inline_data=types.Blob(
                        data=video_bytes, mime_type="video/mp4"
                    ))
                )
            else:
                try:
                    uploaded = client.files.upload(
                        file=self.video_path,
                        config=types.UploadFileConfig(mime_type="video/mp4"),
                    )
                    import time as _time
                    while uploaded.state == "PROCESSING":
                        _time.sleep(2)
                        uploaded = client.files.get(name=uploaded.name)
                    if uploaded.state != "FAILED":
                        user_parts.append(
                            types.Part.from_uri(file_uri=uploaded.uri, mime_type="video/mp4")
                        )
                except Exception:
                    pass  # Skip video if upload fails

        user_parts.append(types.Part(text=(
            f"--- SCENE GRAPH CONTEXT ---\n\n"
            f"{context}\n\n"
            f"--- END CONTEXT ---\n\n"
            f"Question: {question}"
        )))

        # Set up conversation history
        history = [
            types.Content(role="user", parts=user_parts),
        ]

        # Tool declarations
        tool_declarations = self._get_tool_declarations()

        # Agentic loop
        for round_num in range(max_rounds):
            response = client.models.generate_content(
                model=self.model,
                contents=history,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    tools=[types.Tool(function_declarations=tool_declarations)],
                    system_instruction=system_prompt,
                ),
            )

            if not response.candidates:
                return "[VESTA] No response generated."

            candidate = response.candidates[0]

            # Check if the model wants to call tools
            has_function_calls = False
            function_call_parts = []
            text_parts = []

            for part in candidate.content.parts:
                if part.function_call:
                    has_function_calls = True
                    function_call_parts.append(part)
                elif part.text:
                    text_parts.append(part.text)

            if not has_function_calls:
                # Model gave a final text answer — we're done
                return "\n".join(text_parts) if text_parts else "[VESTA] No response."

            # Execute tool calls and build response
            history.append(candidate.content)  # Add model's tool call to history

            tool_response_parts = []
            for part in function_call_parts:
                fc = part.function_call
                if self.verbose:
                    print(f"  [VESTA] Tool call: {fc.name}({dict(fc.args) if fc.args else {}})")

                result_str = self._execute_tool_call(fc)

                tool_response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=json.loads(result_str),
                    )
                ))

            # Add tool results to history
            history.append(types.Content(role="user", parts=tool_response_parts))

        # If we exhausted rounds, return whatever text we have
        return "\n".join(text_parts) if text_parts else "[VESTA] Max tool-calling rounds reached."
