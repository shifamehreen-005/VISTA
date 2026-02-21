"""
Real-Time VESTA Pipeline

Processes a webcam or video feed frame-by-frame, running optical flow on every
frame in the main thread and Gemini hazard detection in background threads,
with live annotated display via cv2.imshow().
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from vesta.flow.optical_flow import CameraMotion, estimate_camera_motion
from vesta.detection.gemini_detector import KeyframeSampler, detect_hazards, FrameAnalysis
from vesta.registry.hazard_registry import HazardRegistry
from vesta.utils.visualizer import VideoVisualizer
from realtime.audio_alerts import ProximityTracker, AlertSpeaker, AlertLevel
from realtime.trajectory import TrajectoryPredictor


class RealtimeVesta:
    """
    Real-time VESTA pipeline.

    Main loop (~30fps):
      read frame -> optical flow -> update heading -> maybe submit to Gemini
      -> check results -> annotate -> display

    Background thread(s):
      detect_hazards(frame_snapshot) -> queue results back
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        keyframe_interval: int = 60,
        motion_threshold: float = 150.0,
        min_cooldown: int = 20,
        fov_degrees: float = 90.0,
        max_workers: int = 2,
        decay_every_n: int = 30,
        audio_alerts: bool = True,
        enable_viz: bool = False,
        verbose: bool = True,
    ):
        self.model = model
        self.verbose = verbose
        self.max_workers = max_workers
        self.decay_every_n = decay_every_n

        # Core components (reused from vesta)
        self.registry = HazardRegistry(fov_degrees=fov_degrees)
        self.sampler = KeyframeSampler(
            interval=keyframe_interval,
            motion_threshold=motion_threshold,
            min_cooldown=min_cooldown,
        )
        self.visualizer = VideoVisualizer()

        # Trajectory prediction
        self._trajectory = TrajectoryPredictor()

        # 3D web visualization
        self._viz_server = None
        self._viz_push_interval = 5  # push to browser every N frames
        if enable_viz:
            from realtime.web_viz import LiveVizServer
            self._viz_server = LiveVizServer()

        # Audio alert system
        self._proximity_tracker = ProximityTracker()
        self._speaker = AlertSpeaker(enabled=audio_alerts)
        self._alert_check_interval = 5  # check proximity every N frames

        # Thread safety
        self._lock = threading.Lock()

        # Background detection state
        self._executor: ThreadPoolExecutor | None = None
        self._pending_futures: list[Future] = []

        # Latest detections for overlay (written by result collector, read by main)
        self._latest_detections: list | None = None

        # Stats
        self.frame_idx = 0
        self.fps = 30.0
        self._paused = False
        self._screenshot_requested = False

        # Performance tracking (rolling averages)
        self._perf_flow_ms = 0.0
        self._perf_annotate_ms = 0.0
        self._perf_total_ms = 0.0
        self._perf_actual_fps = 0.0
        self._perf_gemini_ms = 0.0  # last Gemini round-trip
        self._perf_alpha = 0.1  # EMA smoothing factor

    def run(self, source=0):
        """
        Main entry point. Runs the real-time pipeline.

        Args:
            source: Webcam index (int, e.g. 0) or path to video file (str).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        is_file = isinstance(source, str)

        if self.verbose:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
            src_label = source if is_file else f"webcam {source}"
            print(f"[VESTA RT] Source: {src_label} ({w}x{h} @ {self.fps:.0f}fps)")
            if total > 0:
                print(f"[VESTA RT] Total frames: {total} ({total / self.fps:.1f}s)")

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Start 3D viz server if enabled
        if self._viz_server is not None:
            self._viz_server.start()

        prev_frame = None
        self.frame_idx = 0

        window_name = "VESTA Real-Time"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                # Handle pause
                if self._paused:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord(" "):
                        self._paused = False
                    elif key == ord("q"):
                        break
                    continue

                ret, frame = cap.read()
                if not ret:
                    if is_file:
                        if self.verbose:
                            print("[VESTA RT] End of video.")
                        break
                    continue

                t0 = time.perf_counter()

                # 1. Optical flow (timed)
                t_flow = time.perf_counter()
                motion = CameraMotion(0, 0, 0, 1.0)
                if prev_frame is not None:
                    motion = estimate_camera_motion(prev_frame, frame)
                flow_ms = (time.perf_counter() - t_flow) * 1000
                a = self._perf_alpha
                self._perf_flow_ms = a * flow_ms + (1 - a) * self._perf_flow_ms

                # 2. Update registry heading + trajectory (thread-safe)
                with self._lock:
                    self.registry.update_with_motion(motion)
                self._trajectory.update(motion)

                # 3. Keyframe sampling -> submit to background
                is_keyframe = self.sampler.should_sample(self.frame_idx, motion)
                if is_keyframe:
                    self._submit_detection(frame.copy(), self.frame_idx)

                # 4. Collect completed Gemini results
                self._collect_results()

                # 5. Proximity-based audio alerts
                if self.frame_idx % self._alert_check_interval == 0:
                    timestamp = self.frame_idx / self.fps
                    with self._lock:
                        alerts = self._proximity_tracker.update(self.registry, timestamp)
                    for hazard, level, message in alerts:
                        self._speaker.speak(level, message)
                        if self.verbose:
                            print(f"[VESTA AUDIO] [{level.name}] {message}")

                # 5b. Push state to 3D web viz
                if self._viz_server and self.frame_idx % self._viz_push_interval == 0:
                    self._push_viz_state()

                # 6. Periodic confidence decay
                if self.frame_idx % self.decay_every_n == 0 and self.frame_idx > 0:
                    with self._lock:
                        self.registry.decay_confidence(self.decay_every_n / self.fps)
                        self.registry.prune_stale()

                # 7. Annotate frame (timed)
                t_ann = time.perf_counter()
                with self._lock:
                    annotated = self.visualizer.annotate_frame(
                        frame,
                        self.registry,
                        current_detections=self._latest_detections,
                        frame_idx=self.frame_idx,
                        fps=self.fps,
                        is_keyframe=is_keyframe,
                    )
                ann_ms = (time.perf_counter() - t_ann) * 1000
                self._perf_annotate_ms = a * ann_ms + (1 - a) * self._perf_annotate_ms

                # Draw "LIVE" indicator + performance metrics
                self._draw_rt_overlay(annotated)

                # 8. Display
                cv2.imshow(window_name, annotated)

                # Update total frame time + actual FPS
                total_ms = (time.perf_counter() - t0) * 1000
                self._perf_total_ms = a * total_ms + (1 - a) * self._perf_total_ms
                if self._perf_total_ms > 0:
                    self._perf_actual_fps = 1000.0 / self._perf_total_ms

                # 9. Screenshot
                if self._screenshot_requested:
                    self._save_screenshot(annotated)
                    self._screenshot_requested = False

                prev_frame = frame
                self.frame_idx += 1

                # Frame timing (try to match source fps for files)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                wait_ms = max(1, int(1000 / self.fps - elapsed_ms)) if is_file else 1
                key = cv2.waitKey(wait_ms) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord(" "):
                    self._paused = True
                elif key == ord("s"):
                    self._screenshot_requested = True

        finally:
            # ── Drain pending Gemini results before shutting down ──
            # These are valuable for Q&A — don't throw them away
            pending = [f for f in self._pending_futures if not f.done()]
            if pending and self.verbose:
                print(f"\n[VESTA RT] Waiting for {len(pending)} pending Gemini calls...")
            for future in self._pending_futures:
                if future.done():
                    # Collect any already-completed results
                    try:
                        analysis, timestamp, rtt_ms = future.result()
                        if analysis.hazards:
                            with self._lock:
                                for h in analysis.hazards:
                                    self.registry.add_detection(
                                        label=h.label, category=h.category,
                                        severity=h.severity, description=h.description,
                                        x_normalized=h.x, y_normalized=h.y,
                                        confidence=h.confidence, timestamp=timestamp,
                                    )
                    except Exception:
                        pass
                else:
                    # Wait up to 10s for each pending future
                    try:
                        analysis, timestamp, rtt_ms = future.result(timeout=10)
                        if analysis.hazards:
                            with self._lock:
                                for h in analysis.hazards:
                                    entry = self.registry.add_detection(
                                        label=h.label, category=h.category,
                                        severity=h.severity, description=h.description,
                                        x_normalized=h.x, y_normalized=h.y,
                                        confidence=h.confidence, timestamp=timestamp,
                                    )
                                    if self.verbose:
                                        print(f"[VESTA RT] Late result: {entry.label} [{entry.severity}]")
                    except Exception:
                        pass

            # ── Cleanup (each step wrapped so one failure doesn't kill Q&A) ──
            try:
                self._speaker.shutdown()
            except Exception:
                pass

            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

            try:
                cap.release()
            except Exception:
                pass

            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            if self.verbose:
                with self._lock:
                    summary = self.registry.get_summary()
                print(f"\n[VESTA RT] Session ended.")
                print(f"[VESTA RT] Processed {self.frame_idx} frames.")
                print(f"[VESTA RT] {summary['total_hazards']} hazards in registry.")

    def _push_viz_state(self):
        """Compute trajectory prediction and push full state to the 3D web viz."""
        predicted_path = self._trajectory.predict_path(fps=self.fps)

        with self._lock:
            collisions = self._trajectory.check_collisions(
                self.registry, predicted_path, fps=self.fps
            )
            hazard_positions = self._trajectory.get_hazard_world_positions(self.registry)

        # Trim path for network efficiency (every 3rd point)
        full_path = self._trajectory.get_path()
        sparse_path = full_path[::3] if len(full_path) > 30 else full_path

        data = {
            "worker": {
                "x": self._trajectory.x,
                "y": self._trajectory.y,
                "heading": self.registry.current_heading,
            },
            "path": sparse_path,
            "hazards": hazard_positions,
            "prediction": {
                "points": predicted_path,
                "collisions": [
                    {
                        "hazard_id": c.hazard_id,
                        "label": c.label,
                        "eta_seconds": c.eta_seconds,
                        "hazard_x": c.hazard_x,
                        "hazard_y": c.hazard_y,
                    }
                    for c in collisions
                ],
            },
        }
        self._viz_server.push_state(data)

    def _submit_detection(self, frame_snapshot: np.ndarray, frame_idx: int):
        """Submit a frame to background Gemini detection."""
        timestamp = frame_idx / self.fps

        if self.verbose:
            pending = len([f for f in self._pending_futures if not f.done()])
            print(f"[VESTA RT] Submitting keyframe #{frame_idx} (T={timestamp:.1f}s, {pending} pending)")

        future = self._executor.submit(
            self._detect_worker, frame_snapshot, timestamp
        )
        self._pending_futures.append(future)

    def _detect_worker(self, frame: np.ndarray, timestamp: float) -> tuple[FrameAnalysis, float, float]:
        """Background worker: runs Gemini detection and returns results + round-trip time."""
        t0 = time.perf_counter()
        analysis = detect_hazards(frame, model=self.model)
        rtt_ms = (time.perf_counter() - t0) * 1000
        return analysis, timestamp, rtt_ms

    def _collect_results(self):
        """Check for completed futures and inject detections into registry."""
        still_pending = []
        new_detections = None

        for future in self._pending_futures:
            if future.done():
                try:
                    analysis, timestamp, rtt_ms = future.result()
                    self._perf_gemini_ms = rtt_ms
                except Exception as e:
                    if self.verbose:
                        print(f"[VESTA RT] Detection error: {e}")
                    continue

                if analysis.hazards:
                    new_detections = analysis.hazards
                    with self._lock:
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
                                print(
                                    f"[VESTA RT] Detected: {entry.label} [{entry.severity}] "
                                    f"at {entry.allo_angle:.0f}"
                                )

                    if self.verbose:
                        print(
                            f"[VESTA RT] +{len(analysis.hazards)} hazards "
                            f"(scene: {analysis.scene_type}, workers: {analysis.workers_visible})"
                        )
            else:
                still_pending.append(future)

        self._pending_futures = still_pending

        # Update latest detections for overlay (show for a few frames)
        if new_detections is not None:
            self._latest_detections = new_detections
        elif len(still_pending) == 0:
            # Clear overlay when no pending work and no new results
            pass  # Keep showing last detections until next keyframe

    def _draw_rt_overlay(self, frame: np.ndarray):
        """Draw real-time status indicators and performance metrics."""
        h, w = frame.shape[:2]
        pending = len([f for f in self._pending_futures if not f.done()])

        # "LIVE" or "PAUSED" indicator (top-left)
        if self._paused:
            cv2.putText(frame, "PAUSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (20, 25), 6, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE", (32, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Pending Gemini calls indicator
        if pending > 0:
            cv2.putText(frame, f"AI: {pending} pending", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        # ── Performance metrics panel (top-left, below LIVE) ──
        y_perf = 85
        fps_color = (0, 255, 0) if self._perf_actual_fps >= 25 else (0, 200, 255) if self._perf_actual_fps >= 15 else (0, 0, 255)

        # Background for perf panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, y_perf - 15), (230, y_perf + 75), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"{self._perf_actual_fps:.0f} FPS", (10, y_perf + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"flow: {self._perf_flow_ms:.1f}ms", (10, y_perf + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"draw: {self._perf_annotate_ms:.1f}ms", (10, y_perf + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"total: {self._perf_total_ms:.1f}ms", (10, y_perf + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        if self._perf_gemini_ms > 0:
            cv2.putText(frame, f"gemini: {self._perf_gemini_ms:.0f}ms", (120, y_perf + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        # Controls hint (top-right, small)
        cv2.putText(frame, "Q:quit  SPACE:pause  S:screenshot", (w - 350, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    def _save_screenshot(self, frame: np.ndarray):
        """Save current annotated frame as a screenshot."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"vesta_screenshot_{ts}.png"
        cv2.imwrite(path, frame)
        if self.verbose:
            print(f"[VESTA RT] Screenshot saved: {path}")

    def get_registry(self) -> HazardRegistry:
        """Get the hazard registry (for post-session Q&A)."""
        return self.registry
