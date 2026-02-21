"""
Module 4: VESTA Agent — The Orchestrator

Runs the 3-stage detection loop over video and exposes tool functions
that Gemini can call to query the hazard registry.

Usage:
    agent = VestaAgent(video_path="factory_walkthrough.mp4")
    agent.process()          # Run the full pipeline
    agent.ask("What's behind me?")  # Query the registry via Gemini
"""

import json
import os

import cv2

from vesta.flow.optical_flow import CameraMotion, estimate_camera_motion
from vesta.detection.gemini_detector import (
    KeyframeSampler,
    detect_hazards,
)
from vesta.registry.hazard_registry import HazardRegistry
from vesta.utils.osha_lookup import get_risk_context


class VestaAgent:
    """
    The VESTA agent: processes video through the 3-stage loop and answers
    spatial safety questions using the hazard registry.
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

        # Core components
        self.registry = HazardRegistry(fov_degrees=fov_degrees)
        self.sampler = KeyframeSampler(interval=keyframe_interval)

        # State
        self.frame_count = 0
        self.fps = 30.0
        self.processed = False
        self.motions: list[CameraMotion] = []  # stored for spatial map

    def process(
        self,
        max_frames: int | None = None,
        output_video: str | None = None,
    ) -> dict:
        """
        Run the pipeline with overlapped optical flow and Gemini detection.

        Pass 1+2 (overlapped): Read frames, compute optical flow, and submit
        keyframes to Gemini IMMEDIATELY as they're identified. By the time the
        last frame is read, most Gemini results are already back.

        Pass 3 (fast): Replay motion sequence, inject detections, write video.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # ══════════════════════════════════════════════════════════════
        # PASS 1+2: Optical flow + keyframe selection + Gemini (overlapped)
        # Keyframes are submitted to Gemini immediately as they're found,
        # so Gemini inference runs in parallel with the remaining flow.
        # ══════════════════════════════════════════════════════════════
        t_start = time.time()
        if self.verbose:
            print(f"[VESTA] Pass 1+2: Optical flow + Gemini (overlapped)...")

        all_frames = []        # Store frames for video output
        motions = []           # Per-frame camera motion
        keyframe_indices = []  # Which frames are keyframes
        keyframe_headings = [] # Heading at each keyframe

        # Thread pool for Gemini calls — submit immediately, collect later
        executor = ThreadPoolExecutor(max_workers=4)
        gemini_futures = {}    # future → frame_idx

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

            # Decide if this is a keyframe — submit to Gemini immediately
            if self.sampler.should_sample(frame_idx, motion):
                keyframe_indices.append(frame_idx)
                keyframe_headings.append(cumulative_heading)
                future = executor.submit(detect_hazards, frame.copy(), self.model)
                gemini_futures[future] = frame_idx
                if self.verbose:
                    print(f"[VESTA] Keyframe #{len(keyframe_indices)} @ frame {frame_idx} → submitted to Gemini")

            prev_frame = frame
            frame_idx += 1

        cap.release()
        self.frame_count = frame_idx

        t_flow_done = time.time() - t_start
        if self.verbose:
            print(f"[VESTA] Flow done: {frame_idx} frames in {t_flow_done:.1f}s, "
                  f"{len(keyframe_indices)} keyframes submitted")

        # Wait for any remaining Gemini results
        detection_results = {}
        for future in as_completed(gemini_futures):
            fidx = gemini_futures[future]
            try:
                detection_results[fidx] = future.result()
            except Exception as e:
                if self.verbose:
                    print(f"[VESTA] Gemini error for frame {fidx}: {e}")
                from vesta.detection.gemini_detector import FrameAnalysis
                detection_results[fidx] = FrameAnalysis()

        executor.shutdown(wait=False)
        t_pass12 = time.time() - t_start
        if self.verbose:
            already_done = sum(1 for f in gemini_futures if f.done())
            print(f"[VESTA] Pass 1+2 done: {len(detection_results)} Gemini responses in {t_pass12:.1f}s total")

        # ══════════════════════════════════════════════════════════════
        # PASS 3: Apply detections to registry + write video
        # ══════════════════════════════════════════════════════════════
        t_pass3 = time.time()
        if self.verbose:
            print(f"[VESTA] Pass 3: Building registry + writing video...")

        # Set up video writer
        writer = None
        visualizer = None
        if output_video and all_frames:
            from vesta.utils.visualizer import VideoVisualizer
            visualizer = VideoVisualizer(frame_w, frame_h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, self.fps, (frame_w, frame_h))

        # Build a lookup: frame_idx → (analysis, heading_at_keyframe)
        keyframe_data = {}
        for ki, kh in zip(keyframe_indices, keyframe_headings):
            if ki in detection_results:
                keyframe_data[ki] = (detection_results[ki], kh)

        # Replay the motion sequence and inject detections at keyframes
        detections_count = 0
        self.registry.current_heading = 0.0  # Reset heading

        for i in range(self.frame_count):
            timestamp = i / self.fps

            # Apply optical flow
            if i > 0:
                self.registry.update_with_motion(motions[i])

            # Inject detections at keyframes
            is_keyframe = i in keyframe_data
            current_detections = None

            if is_keyframe:
                analysis, _ = keyframe_data[i]
                current_detections = analysis.hazards
                detections_count += len(analysis.hazards)

                if self.verbose:
                    kf_num = keyframe_indices.index(i) + 1
                    print(
                        f"[VESTA] Keyframe #{kf_num} @ T={timestamp:.1f}s "
                        f"(frame {i}), heading={self.registry.current_heading:.1f}°"
                    )

                for h in analysis.hazards:
                    entry = self.registry.add_detection(
                        label=h.label,
                        category=h.category,
                        severity=h.severity,
                        description=h.description,
                        x_normalized=h.x,
                        y_normalized=h.y,
                        confidence=h.confidence,
                        timestamp=timestamp,
                    )
                    if self.verbose:
                        print(f"  → {entry.label} [{entry.severity}] at {entry.allo_angle:.0f}°")

            # Write annotated frame
            if writer and visualizer and output_video and all_frames:
                annotated = visualizer.annotate_frame(
                    all_frames[i],
                    self.registry,
                    current_detections=current_detections,
                    frame_idx=i,
                    fps=self.fps,
                    is_keyframe=is_keyframe,
                )
                writer.write(annotated)

            # Decay confidence
            if i % 30 == 0:
                self.registry.decay_confidence(1.0 / self.fps * 30)

        if writer:
            writer.release()

        self.motions = motions  # save for spatial map generation
        self.processed = True
        t_total = time.time() - t_start

        summary = self.registry.get_summary()
        summary["frames_processed"] = self.frame_count
        summary["keyframes_analyzed"] = len(keyframe_indices)
        summary["total_detections"] = detections_count

        if self.verbose:
            print(f"\n[VESTA] Done in {t_total:.1f}s total.")
            print(f"[VESTA]   Flow + Gemini (overlapped): {t_pass12:.1f}s")
            print(f"[VESTA]   Registry + video:           {time.time() - t_pass3:.1f}s")
            print(f"[VESTA] {self.frame_count} frames, {len(keyframe_indices)} keyframes, "
                  f"{summary['total_hazards']} unique hazards.")
            if output_video:
                print(f"[VESTA] Video saved: {output_video}")

        return summary

    # ── Agent Tool Functions (exposed to Gemini) ────────────────────────────

    def tool_get_hazards_at_angle(self, angle: float, fov: float = 90.0) -> str:
        """Tool: Get hazards at a specific angle relative to current heading."""
        allo_angle = self.registry.transformer.ego_to_allo(
            angle, self.registry.current_heading
        )
        hazards = self.registry.query_angle(allo_angle, fov=fov)
        if not hazards:
            return json.dumps({"hazards": [], "message": f"No hazards detected at {angle}°"})
        return json.dumps({
            "hazards": [
                {
                    **h.to_dict(),
                    "relative_description": self.registry.describe_relative_to_camera(h),
                }
                for h in hazards
            ]
        })

    def tool_get_all_hazards(self) -> str:
        """Tool: Get all hazards in the registry."""
        return json.dumps(self.registry.get_summary())

    def tool_get_direction(self, direction: str) -> str:
        """Tool: Get hazards in a named direction (front, behind, left, right, etc.)"""
        hazards = self.registry.query_direction(direction)
        if not hazards:
            return json.dumps({
                "hazards": [],
                "message": f"No hazards detected {direction}.",
            })
        return json.dumps({
            "direction": direction,
            "hazards": [
                {
                    **h.to_dict(),
                    "relative_description": self.registry.describe_relative_to_camera(h),
                }
                for h in hazards
            ],
        })

    def tool_get_hazards_at_time(self, time_seconds: float, window: float = 2.0) -> str:
        """Tool: Get hazards visible at a specific timestamp in the video."""
        start = max(0, time_seconds - window / 2)
        end = time_seconds + window / 2
        hazards = self.registry.query_time_range(start, end)
        if not hazards:
            return json.dumps({
                "hazards": [],
                "message": f"No hazards detected between T={start:.1f}s and T={end:.1f}s",
            })
        return json.dumps({
            "time_query": f"T={time_seconds:.1f}s (window: {start:.1f}s-{end:.1f}s)",
            "hazards": [
                {
                    **h.to_dict(),
                    "relative_description": self.registry.describe_relative_to_camera(h),
                }
                for h in hazards
            ],
        })

    def tool_get_osha_context(self, hazard_label: str, category: str = "") -> str:
        """Tool: Get OSHA incident context for a hazard type."""
        return get_risk_context(category, hazard_label)

    # ── Natural Language Query Interface ────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        Ask VESTA a natural language question about the scene.
        Uses Gemini with tool-calling to query the registry.
        """
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Define the tools Gemini can call
        tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="get_hazards_at_angle",
                    description="Get hazards at a specific angle in degrees relative to current camera direction. 0=ahead, 90=right, 180=behind, -90=left.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "angle": types.Schema(type="NUMBER", description="Angle in degrees"),
                            "fov": types.Schema(type="NUMBER", description="Field of view arc width (default 90)"),
                        },
                        required=["angle"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_all_hazards",
                    description="Get a complete list of all detected hazards and their positions.",
                    parameters=types.Schema(type="OBJECT", properties={}),
                ),
                types.FunctionDeclaration(
                    name="get_direction",
                    description="Get hazards in a named direction: front, behind, left, right, front-left, front-right, behind-left, behind-right.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "direction": types.Schema(type="STRING", description="Direction name"),
                        },
                        required=["direction"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_hazards_at_time",
                    description="Get hazards that were visible at a specific timestamp in the video. Use this when the user asks about what was seen at a particular time (e.g., 'at 5 seconds', 'around the 10 second mark', 'in the first minute').",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "time_seconds": types.Schema(type="NUMBER", description="Timestamp in seconds into the video"),
                            "window": types.Schema(type="NUMBER", description="Time window in seconds to search around the timestamp (default 2.0)"),
                        },
                        required=["time_seconds"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_osha_context",
                    description="Get OSHA incident data and risk context for a specific hazard type.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "hazard_label": types.Schema(type="STRING", description="Hazard name"),
                            "category": types.Schema(type="STRING", description="OSHA hazard category"),
                        },
                        required=["hazard_label"],
                    ),
                ),
            ])
        ]

        # System context with registry state
        registry_summary = self.registry.get_summary()
        system_prompt = (
            "You are VESTA, a construction site safety AI agent. You have processed "
            f"a video of a construction site walkthrough ({self.frame_count} frames, "
            f"{self.frame_count / self.fps:.0f} seconds). "
            f"You have detected {registry_summary['total_hazards']} hazards. "
            f"The camera is currently facing heading {registry_summary['current_heading']}°. "
            "Use your tools to look up specific hazard information. "
            "Always cite OSHA data when warning about hazards. "
            "Be direct, specific, and safety-focused in your responses."
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

        # Handle tool calls in a loop
        max_rounds = 5
        for _ in range(max_rounds):
            # Check if there are function calls in the response
            if not response.candidates or not response.candidates[0].content.parts:
                break

            tool_calls = [
                p for p in response.candidates[0].content.parts
                if p.function_call is not None
            ]

            if not tool_calls:
                break

            # Execute each tool call
            tool_responses = []
            for tc in tool_calls:
                fn_name = tc.function_call.name
                fn_args = dict(tc.function_call.args) if tc.function_call.args else {}

                if fn_name == "get_hazards_at_angle":
                    result = self.tool_get_hazards_at_angle(**fn_args)
                elif fn_name == "get_all_hazards":
                    result = self.tool_get_all_hazards()
                elif fn_name == "get_direction":
                    result = self.tool_get_direction(**fn_args)
                elif fn_name == "get_hazards_at_time":
                    result = self.tool_get_hazards_at_time(**fn_args)
                elif fn_name == "get_osha_context":
                    result = self.tool_get_osha_context(**fn_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                # Parse result — could be JSON string or plain text
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except json.JSONDecodeError:
                        parsed = {"text": result}
                else:
                    parsed = result

                tool_responses.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response=parsed,
                    )
                )

            # Send tool results back to Gemini
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

        # Extract final text response
        if response.candidates and response.candidates[0].content.parts:
            text_parts = [
                p.text for p in response.candidates[0].content.parts
                if p.text
            ]
            return "\n".join(text_parts)

        return "[VESTA] Unable to generate response."
