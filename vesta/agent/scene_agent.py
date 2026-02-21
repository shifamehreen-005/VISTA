"""
Scene Agent — Spatio-Temporal Video Understanding via Scene Graph

Processes video into a spatio-temporal scene graph and answers
"what, where, when" questions using tool-calling with Gemini.

Replaces VestaAgent for the scene graph pipeline.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

from vesta.flow.optical_flow import CameraMotion, estimate_camera_motion
from vesta.detection.gemini_detector import KeyframeSampler
from vesta.detection.scene_descriptor import describe_scene, SceneAnalysis
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
        verbose: bool = True,
    ):
        self.video_path = video_path
        self.model = model
        self.verbose = verbose

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

        # ── Pass 1+2: Optical flow + Gemini (overlapped) ──
        t_start = time.time()
        if self.verbose:
            print(f"[VESTA] Pass 1+2: Optical flow + Gemini (overlapped)...")

        all_frames = []
        motions = []
        keyframe_indices = []
        keyframe_headings = []

        executor = ThreadPoolExecutor(max_workers=4)
        gemini_futures = {}

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
                future = executor.submit(describe_scene, frame.copy(), self.model)
                gemini_futures[future] = frame_idx
                if self.verbose:
                    print(f"[VESTA] Keyframe #{len(keyframe_indices)} @ frame {frame_idx} → submitted")

            prev_frame = frame
            frame_idx += 1

        cap.release()
        self.frame_count = frame_idx

        t_flow_done = time.time() - t_start
        if self.verbose:
            print(f"[VESTA] Flow done: {frame_idx} frames in {t_flow_done:.1f}s, "
                  f"{len(keyframe_indices)} keyframes submitted")

        # Collect Gemini results
        scene_results = {}
        for future in as_completed(gemini_futures):
            fidx = gemini_futures[future]
            try:
                scene_results[fidx] = future.result()
            except Exception as e:
                if self.verbose:
                    print(f"[VESTA] Gemini error for frame {fidx}: {e}")
                scene_results[fidx] = SceneAnalysis()

        executor.shutdown(wait=False)
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

    # ── Query Classifier ──────────────────────────────────────────────────

    @staticmethod
    def _classify_query(question: str) -> str | None:
        """
        Rule-based query classifier. Returns a tool hint or None for ambiguous.

        Forces the right tool for obvious query patterns, preventing Gemini
        from picking the wrong tool for spatial/temporal/change questions.
        """
        q = question.lower().strip()

        # Spatial + temporal combined: "what's behind me at 30 seconds?"
        import re
        spatial_keywords = [
            "behind", "left", "right", "front", "ahead", "rear",
            "front-left", "front-right", "behind-left", "behind-right",
        ]
        has_time = re.search(r"at\s+\d+\s*s(?:econds?)?|at\s+\d+\s*min(?:ute)?", q)
        if has_time:
            for kw in spatial_keywords:
                if kw in q:
                    return "direction_at_time"

        # Spatial direction queries (no time component)
        for kw in spatial_keywords:
            if f"what's {kw}" in q or f"whats {kw}" in q or f"what is {kw}" in q or f"to my {kw}" in q:
                return "direction"

        # Spatial relation queries (two entities)
        if "relative to" in q or "compared to" in q:
            return "spatial_relation"
        if ("where is" in q or "position of" in q) and ("relative" in q or "from" in q):
            return "spatial_relation"

        # Temporal queries
        import re
        if re.search(r"at \d+ ?s(econds?)?", q) or "at the start" in q or "at the end" in q:
            return "temporal"
        if "when was" in q or "when did" in q or "what time" in q or "at what time" in q:
            return "temporal"
        if "how long" in q or "how much time" in q or "duration" in q:
            return "temporal"

        # Change/progress queries
        change_keywords = [
            "changed", "change", "different", "progress", "evolved",
            "before and after", "state change", "what happened to",
        ]
        for kw in change_keywords:
            if kw in q:
                return "changes"

        return None

    def _force_tool_call(self, question: str, hint: str) -> str:
        """Execute the forced tool call based on query classification."""
        import re
        q = question.lower()

        if hint == "direction_at_time":
            # Extract direction and time
            direction = None
            for d in ["behind", "rear", "back", "front", "ahead", "forward",
                       "left", "right", "front-left", "front-right",
                       "behind-left", "behind-right"]:
                if d in q:
                    direction = d
                    break
            time_match = re.search(r"at\s+(\d+)\s*s(?:econds?)?", q)
            if not time_match:
                time_match = re.search(r"at\s+(\d+)\s*min", q)
                if time_match:
                    t = float(time_match.group(1)) * 60
                else:
                    return None
            else:
                t = float(time_match.group(1))
            if direction:
                result_json = self.tool_get_direction_at_time(direction, t)
                result = json.loads(result_json)
                return self._synthesize_answer(question, "get_direction_at_time", result)
            return None

        if hint == "direction":
            # Extract the direction keyword
            for d in ["behind", "rear", "back", "front", "ahead", "forward",
                       "left", "right", "front-left", "front-right",
                       "behind-left", "behind-right"]:
                if d in q:
                    result_json = self.tool_get_entities_in_direction(d)
                    result = json.loads(result_json)
                    break
            else:
                return None
            # Still use Gemini to synthesize the answer from the tool result
            return self._synthesize_answer(question, "get_entities_in_direction", result)

        elif hint == "changes":
            # Check if a specific entity is mentioned
            entity_label = ""
            for entity in self.graph.get_all():
                if entity.label.lower() in q:
                    entity_label = entity.label
                    break
            result_json = self.tool_get_changes(entity_label)
            result = json.loads(result_json)
            return self._synthesize_answer(question, "get_changes", result)

        elif hint == "temporal":
            import re
            # If asking about a specific time, use get_entities_at_time
            match = re.search(r"(\d+)\s*s(?:econds?)?", q)
            if match:
                t = float(match.group(1))
                result_json = self.tool_get_entities_at_time(t)
                result = json.loads(result_json)
                return self._synthesize_answer(question, "get_entities_at_time", result)

            # If asking "at what time..." or "when..." or "how long...",
            # find the entity mentioned and get its timeline
            for entity in self.graph.get_all():
                if entity.label.lower() in q or entity.category.lower() in q:
                    result_json = self.tool_get_entity_timeline(entity.label)
                    result = json.loads(result_json)
                    return self._synthesize_answer(question, "get_entity_timeline", result)

            # Fallback: try worker_1 or camera_self for self-referential questions
            self_words = ["i ", "me ", "my ", "self", "idle", "working", "doing"]
            if any(w in q for w in self_words):
                for label in ["camera_self", "worker_1"]:
                    entities = self.graph.query_by_label(label)
                    if entities:
                        result_json = self.tool_get_entity_timeline(entities[0].label)
                        result = json.loads(result_json)
                        return self._synthesize_answer(question, "get_entity_timeline", result)

        return None

    def _synthesize_answer(self, question: str, tool_name: str, tool_result: dict) -> str:
        """Use Gemini to synthesize a natural language answer from tool output."""
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        prompt = (
            f"You are VESTA, a video understanding agent. ALWAYS answer — never refuse.\n\n"
            f"Question: {question}\n\n"
            f"Tool used: {tool_name}\n"
            f"Tool result: {json.dumps(tool_result, indent=2)}\n\n"
            f"Give a concise, precise answer based on the tool result. "
            f"For spatial queries, state directions and angles. "
            f"If the result is empty or incomplete, give your best answer and note the limitation. "
            f"NEVER say 'I cannot answer this'."
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )

        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        return "[VESTA] Unable to generate response."

    # ── Natural Language Q&A ─────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        Ask a natural language question about the scene.

        First checks if a rule-based classifier can route the query directly
        to the right tool. Falls back to Gemini tool-calling for ambiguous queries.
        """
        # Try rule-based routing first for reliable spatial/temporal/change queries
        hint = self._classify_query(question)
        if hint:
            forced = self._force_tool_call(question, hint)
            if forced:
                return forced

        # Fall back to full Gemini tool-calling for ambiguous queries
        return self._ask_with_tools(question)

    def _ask_with_tools(self, question: str) -> str:
        """Full Gemini tool-calling loop for complex/ambiguous queries."""
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="get_entities_in_direction",
                    description="Get all entities in a named direction relative to the camera: front, behind, left, right, front-left, front-right, behind-left, behind-right. Optionally filter by category.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "direction": types.Schema(type="STRING", description="Direction name"),
                            "category": types.Schema(type="STRING", description="Optional category filter: person, equipment, structure, material, vehicle"),
                        },
                        required=["direction"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_entity_info",
                    description="Get detailed info about a specific entity including its position, timeline of observations, and relationships. Use for questions like 'tell me about the crane' or 'describe the worker'.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "label": types.Schema(type="STRING", description="Entity label to look up (e.g., 'worker_1', 'crane', 'scaffolding')"),
                        },
                        required=["label"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_all_entities",
                    description="Get a complete summary of the scene graph: all entities, their categories, and scene descriptions.",
                    parameters=types.Schema(type="OBJECT", properties={}),
                ),
                types.FunctionDeclaration(
                    name="get_entities_at_time",
                    description="Get entities visible at a specific timestamp. Use for questions like 'what was visible at 10 seconds' or 'what was happening at the start'.",
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
                    description="Get the precise spatial relationship between two entities. Returns direction and angular separation. Use for questions like 'where is the crane relative to the worker?' or 'is the scaffolding to the left of the wall?'.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "entity_a": types.Schema(type="STRING", description="Reference entity label"),
                            "entity_b": types.Schema(type="STRING", description="Target entity label"),
                        },
                        required=["entity_a", "entity_b"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_entity_timeline",
                    description="Get the full observation history of an entity over time. Shows how its state/position changed. Use for questions like 'how did the wall change?' or 'what did the worker do over time?'.",
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
                    description="Get observed spatial and action relationships between entities. Filter by entity name or relation type. Use for questions like 'what is near the crane?' or 'who is operating equipment?'.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "entity_label": types.Schema(type="STRING", description="Filter by entity label"),
                            "relation_type": types.Schema(type="STRING", description="Filter by relation type: near, left_of, right_of, above, below, operating, standing_on, etc."),
                        },
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_changes",
                    description="Get detected state changes for entities over time. Shows what changed, when, and by how much. Use for progress tracking, change detection, or questions like 'what changed?' or 'how did X evolve?'.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "entity_label": types.Schema(type="STRING", description="Optional entity label to filter changes for"),
                        },
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_direction_at_time",
                    description="Get entities in a direction (front, behind, left, right, etc.) relative to where the camera was facing at a SPECIFIC PAST TIME. Use for questions like 'what was behind me at 30 seconds?' or 'what was to my left at the start?'. This uses the historical heading, not the current heading.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "direction": types.Schema(type="STRING", description="Direction: front, behind, left, right, front-left, front-right, behind-left, behind-right"),
                            "time_seconds": types.Schema(type="NUMBER", description="Timestamp in seconds"),
                            "category": types.Schema(type="STRING", description="Optional category filter"),
                        },
                        required=["direction", "time_seconds"],
                    ),
                ),
            ])
        ]

        summary = self.graph.get_summary()
        cats = ", ".join(f"{v} {k}" for k, v in summary["by_category"].items())

        system_prompt = (
            f"You are VESTA, a spatio-temporal video understanding agent. You have processed "
            f"a video ({self.frame_count} frames, {self.frame_count / self.fps:.0f} seconds) and "
            f"built a scene graph with {summary['total_entities']} tracked entities ({cats}), "
            f"{summary['total_relationships']} observed relationships, and "
            f"{summary['total_state_changes']} detected state changes.\n\n"
            f"The camera is currently facing heading {summary['current_heading']}°.\n"
            f"Video duration: {self.frame_count / self.fps:.1f} seconds.\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER refuse to answer. NEVER say 'I cannot answer this'. ALWAYS try your best.\n"
            "2. Use tools to look up data, then give your best answer with what you find.\n"
            "3. If the data is incomplete, still answer and note the uncertainty.\n"
            "4. The user is wearing the camera (egocentric/first-person video). 'camera_self' = the user.\n"
            "5. For activity questions ('when was I idle?', 'what was I doing?'), check the camera_self "
            "or worker timelines — their states describe what was happening.\n\n"
            "Tool routing:\n"
            "- SPATIAL (where is X? what's behind me?) → get_entities_in_direction or get_spatial_relation\n"
            "- TEMPORAL (what at 10s? when was X?) → get_entities_at_time or get_entity_timeline\n"
            "- PAST SPATIAL (what was behind me at 30s?) → get_direction_at_time\n"
            "- CHANGES (what changed? how did X evolve?) → get_changes\n"
            "- ACTIVITY (what was I doing? when was I idle?) → get_entity_timeline for relevant person\n\n"
            "Be precise and concise. State directions + angles for spatial queries."
        )

        response = client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools,
                temperature=0.3,
            ),
        )

        # Tool-calling loop
        dispatch = {
            "get_entities_in_direction": self.tool_get_entities_in_direction,
            "get_entity_info": self.tool_get_entity_info,
            "get_all_entities": self.tool_get_all_entities,
            "get_entities_at_time": self.tool_get_entities_at_time,
            "get_spatial_relation": self.tool_get_spatial_relation,
            "get_entity_timeline": self.tool_get_entity_timeline,
            "get_relationships": self.tool_get_relationships,
            "get_changes": self.tool_get_changes,
            "get_direction_at_time": self.tool_get_direction_at_time,
        }

        max_rounds = 5
        for _ in range(max_rounds):
            if not response.candidates or not response.candidates[0].content.parts:
                break

            tool_calls = [
                p for p in response.candidates[0].content.parts
                if p.function_call is not None
            ]
            if not tool_calls:
                break

            tool_responses = []
            for tc in tool_calls:
                fn_name = tc.function_call.name
                fn_args = dict(tc.function_call.args) if tc.function_call.args else {}

                handler = dispatch.get(fn_name)
                if handler:
                    result = handler(**fn_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                try:
                    parsed = json.loads(result)
                except json.JSONDecodeError:
                    parsed = {"text": result}

                tool_responses.append(
                    types.Part.from_function_response(name=fn_name, response=parsed)
                )

            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=question)]),
                    response.candidates[0].content,
                    types.Content(role="user", parts=tool_responses),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=tools,
                    temperature=0.3,
                ),
            )

        if response.candidates and response.candidates[0].content.parts:
            text_parts = [
                p.text for p in response.candidates[0].content.parts if p.text
            ]
            return "\n".join(text_parts)

        return "[VESTA] Unable to generate response."
