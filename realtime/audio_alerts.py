"""
Intelligent Audio Alert System for VESTA Real-Time Pipeline

Tracks per-hazard proximity over time and generates escalating spoken warnings:
- First detection: "Floor opening detected behind you"
- Approaching:     "Caution, you're moving toward the floor opening behind you"
- Very close:      "Warning — floor opening very close, directly behind you"
- Imminent:        "STOP! Floor opening right behind you!"

Uses ego-angle trend (is the hazard moving toward center of view?) and
distance trend (is the y-position dropping = getting closer?) to determine
if the worker is approaching a hazard.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import pyttsx3

from vesta.registry.hazard_registry import HazardRegistry, HazardEntry


class AlertLevel(IntEnum):
    """Escalating alert levels. Higher = more urgent."""
    DETECTED = 0       # First seen
    NEARBY = 1         # Within awareness zone
    APPROACHING = 2    # Distance/angle closing
    CLOSE = 3          # Very close, high urgency
    IMMINENT = 4       # About to walk into it


# How many seconds before we can re-alert the same hazard at the same level
COOLDOWNS = {
    AlertLevel.DETECTED: 30.0,
    AlertLevel.NEARBY: 15.0,
    AlertLevel.APPROACHING: 8.0,
    AlertLevel.CLOSE: 4.0,
    AlertLevel.IMMINENT: 2.0,
}

# TTS speech rate by alert level (words per minute) — faster = more urgent
SPEECH_RATES = {
    AlertLevel.DETECTED: 160,
    AlertLevel.NEARBY: 170,
    AlertLevel.APPROACHING: 180,
    AlertLevel.CLOSE: 200,
    AlertLevel.IMMINENT: 220,
}


@dataclass
class HazardTrack:
    """Tracks a single hazard's proximity history over time."""
    hazard_id: str
    label: str
    severity: str

    # Rolling history of (timestamp, ego_angle, distance)
    history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Last alert state
    last_alert_level: AlertLevel = AlertLevel.DETECTED
    last_alert_time: float = 0.0
    alert_count: int = 0


def _direction_word(ego_angle: float) -> str:
    """Convert ego angle to a natural direction phrase."""
    a = abs(ego_angle)
    if a < 20:
        return "directly ahead"
    elif a < 50:
        side = "right" if ego_angle > 0 else "left"
        return f"to your front-{side}"
    elif a < 80:
        side = "right" if ego_angle > 0 else "left"
        return f"to your {side}"
    elif a < 130:
        side = "right" if ego_angle > 0 else "left"
        return f"behind you to the {side}"
    else:
        return "directly behind you"


def _closeness_word(distance: float) -> str:
    """Describe how close something is based on registry distance (0=close, 1=far)."""
    if distance < 0.2:
        return "very close"
    elif distance < 0.4:
        return "close"
    elif distance < 0.6:
        return "nearby"
    else:
        return ""


class ProximityTracker:
    """
    Evaluates all hazards in the registry each tick and determines
    which ones need audio alerts based on proximity trends.
    """

    def __init__(self):
        self._tracks: dict[str, HazardTrack] = {}

    def update(
        self,
        registry: HazardRegistry,
        timestamp: float,
    ) -> list[tuple[HazardEntry, AlertLevel, str]]:
        """
        Evaluate all hazards and return those that need an alert now.

        Returns:
            List of (hazard_entry, alert_level, spoken_message)
        """
        alerts = []
        active_ids = set()

        for hazard in registry.get_all(min_confidence=0.3):
            active_ids.add(hazard.id)

            # Get or create track
            if hazard.id not in self._tracks:
                self._tracks[hazard.id] = HazardTrack(
                    hazard_id=hazard.id,
                    label=hazard.label,
                    severity=hazard.severity,
                )

            track = self._tracks[hazard.id]

            # Compute current ego angle
            ego_angle = registry.transformer.allo_to_ego(
                hazard.allo_angle, registry.current_heading
            )
            distance = hazard.distance

            # Record this snapshot
            track.history.append((timestamp, ego_angle, distance))

            # Determine alert level
            level = self._evaluate_level(track, ego_angle, distance, hazard)

            # Check cooldown
            cooldown = COOLDOWNS[level]
            if timestamp - track.last_alert_time < cooldown and level <= track.last_alert_level:
                continue

            # Only alert if level is >= NEARBY (don't spam on first detection of
            # far-away things, but always announce close ones)
            if level == AlertLevel.DETECTED and hazard.severity not in ("critical", "high"):
                continue

            # Generate message
            message = self._build_message(track, ego_angle, distance, level, hazard)

            track.last_alert_level = level
            track.last_alert_time = timestamp
            track.alert_count += 1

            alerts.append((hazard, level, message))

        # Clean up tracks for pruned hazards
        stale = [hid for hid in self._tracks if hid not in active_ids]
        for hid in stale:
            del self._tracks[hid]

        # Sort by urgency (highest level first)
        alerts.sort(key=lambda x: x[1], reverse=True)
        return alerts

    def _evaluate_level(
        self,
        track: HazardTrack,
        ego_angle: float,
        distance: float,
        hazard: HazardEntry,
    ) -> AlertLevel:
        """Determine the current alert level for a hazard based on proximity and trend."""

        # Base level from absolute proximity
        abs_angle = abs(ego_angle)

        # Is it in front of us (within ~90 degree cone)?
        in_front = abs_angle < 60
        # Is it close behind us (can't see it)?
        blind_spot = abs_angle > 120

        # Distance thresholds
        if distance < 0.15:
            base_level = AlertLevel.IMMINENT
        elif distance < 0.3:
            base_level = AlertLevel.CLOSE
        elif distance < 0.5:
            base_level = AlertLevel.NEARBY
        else:
            base_level = AlertLevel.DETECTED

        # Boost urgency for blind-spot hazards (behind the worker)
        if blind_spot and base_level >= AlertLevel.NEARBY:
            base_level = min(AlertLevel.IMMINENT, AlertLevel(base_level + 1))

        # Boost for critical/high severity
        if hazard.severity in ("critical", "high") and base_level < AlertLevel.CLOSE:
            base_level = max(base_level, AlertLevel.NEARBY)

        # Check approach trend: is the distance decreasing over recent history?
        approach_boost = self._detect_approach(track)
        if approach_boost:
            base_level = min(AlertLevel.IMMINENT, AlertLevel(base_level + 1))

        return base_level

    def _detect_approach(self, track: HazardTrack) -> bool:
        """
        Analyze recent history to determine if the worker is approaching this hazard.

        Checks two signals:
        1. Ego angle magnitude decreasing (hazard moving toward center of view)
        2. Distance decreasing (hazard getting closer in depth)
        """
        if len(track.history) < 4:
            return False

        recent = list(track.history)[-6:]  # Last ~6 samples

        # Check distance trend
        distances = [d for _, _, d in recent]
        if len(distances) >= 3:
            # Is distance consistently decreasing?
            decreasing = sum(
                1 for i in range(1, len(distances)) if distances[i] < distances[i - 1]
            )
            if decreasing >= len(distances) * 0.6:
                return True

        # Check ego angle trend (magnitude decreasing = moving toward center)
        angles = [abs(a) for _, a, _ in recent]
        if len(angles) >= 3:
            angle_decreasing = sum(
                1 for i in range(1, len(angles)) if angles[i] < angles[i - 1]
            )
            # Only count angle approach for hazards that started off-screen
            if angle_decreasing >= len(angles) * 0.6 and angles[0] > 60:
                return True

        return False

    def _build_message(
        self,
        track: HazardTrack,
        ego_angle: float,
        distance: float,
        level: AlertLevel,
        hazard: HazardEntry,
    ) -> str:
        """Generate a natural spoken warning appropriate to the alert level."""
        label = track.label.lower()
        direction = _direction_word(ego_angle)
        closeness = _closeness_word(distance)

        if level == AlertLevel.IMMINENT:
            return f"Stop! {label} right {direction}!"

        if level == AlertLevel.CLOSE:
            return f"Warning, {label} {closeness}, {direction}."

        if level == AlertLevel.APPROACHING:
            return f"Caution, you're moving toward the {label}, {direction}."

        if level == AlertLevel.NEARBY:
            return f"{label} detected {direction}, {closeness}." if closeness else f"{label} detected {direction}."

        # DETECTED — first announcement for critical/high
        return f"Hazard detected: {label}, {direction}."


class AlertSpeaker:
    """
    Non-blocking TTS speaker that runs speech on a background thread.

    Queues messages and speaks them in priority order.
    Drops stale messages if the queue backs up.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._queue: deque[tuple[AlertLevel, str]] = deque(maxlen=5)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._engine_lock = threading.Lock()

        if self.enabled:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 170)
            # Try to pick a clear voice
            voices = self._engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower() or "samantha" in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break
            self._start_worker()

    def speak(self, level: AlertLevel, message: str):
        """Queue a message for speaking. Higher level = higher priority."""
        if not self.enabled:
            return
        self._queue.append((level, message))

    def _start_worker(self):
        """Start the background speech worker thread."""
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background thread that pulls from queue and speaks."""
        while not self._stop_event.is_set():
            if not self._queue:
                time.sleep(0.1)
                continue

            # Grab highest priority message in queue
            messages = []
            while self._queue:
                messages.append(self._queue.popleft())
            messages.sort(key=lambda x: x[0], reverse=True)
            level, text = messages[0]

            try:
                with self._engine_lock:
                    self._engine.setProperty("rate", SPEECH_RATES.get(level, 170))
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception:
                pass  # pyttsx3 can throw on macOS — don't crash the thread

    def shutdown(self):
        """Stop the speaker thread cleanly."""
        self._stop_event.set()
        # Clear the queue so the worker exits faster
        self._queue.clear()
        if self._thread:
            self._thread.join(timeout=3)
        if self.enabled:
            try:
                with self._engine_lock:
                    self._engine.stop()
            except Exception:
                pass  # pyttsx3 shutdown can segfault on macOS — don't kill the process
