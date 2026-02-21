"""
Module 2: Gemini Flash Hazard Detector

Sends keyframes to Gemini 2.0 Flash and extracts structured hazard detections
with pixel coordinates, labels, and OSHA risk categories.
"""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

# Lazy import — only load google.genai when actually calling the API
_client = None


def _get_client():
    """Lazy-init Gemini client."""
    global _client
    if _client is None:
        import google.genai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Add it to .env or environment.")
        _client = genai.Client(api_key=api_key)
    return _client


DETECTION_PROMPT = """You are a construction safety AI. Analyze this hardhat camera image.

Your ONLY task: detect **floor openings** — any unprotected hole, gap, or opening in the floor/deck that a worker could fall through or step into. This includes:
- Holes cut for plumbing, HVAC, or stairwells
- Missing deck sections or plywood covers
- Open trenches or pits in the floor
- Gaps between floor joists or framing
- Any unguarded floor edge where the floor simply ends

For EACH floor opening detected, return a bounding box:
- "label": Short name (e.g., "Floor Opening", "Stairwell Hole", "Deck Gap")
- "x1": Left edge of bounding box (0-1 normalized, left=0, right=1)
- "y1": Top edge of bounding box (0-1 normalized, top=0, bottom=1)
- "x2": Right edge of bounding box (0-1 normalized)
- "y2": Bottom edge of bounding box (0-1 normalized)
- "confidence": Your confidence 0.0-1.0
- "description": One sentence about the specific risk

Return ONLY valid JSON:
{{
  "hazards": [
    {{
      "label": "Floor Opening",
      "x1": 0.2,
      "y1": 0.4,
      "x2": 0.6,
      "y2": 0.8,
      "confidence": 0.9,
      "description": "Large unguarded opening in subfloor near stairwell"
    }}
  ]
}}

If NO floor openings are visible, return {{"hazards": []}}. Do NOT hallucinate. Only detect what you can clearly see.
Respond ONLY with JSON, no markdown fences."""


# ── Pydantic models for structured output ───────────────────────────────────

class HazardDetection(BaseModel):
    label: str
    category: str = "Fall Hazard"
    x: float = 0.0       # center x (computed from bbox)
    y: float = 0.0       # center y (computed from bbox)
    x1: float = 0.0      # bbox left
    y1: float = 0.0      # bbox top
    x2: float = 0.0      # bbox right
    y2: float = 0.0      # bbox bottom
    confidence: float = 0.0
    severity: str = "critical"
    description: str = ""


class FrameAnalysis(BaseModel):
    scene_type: str = "unknown"
    workers_visible: int = 0
    hazards: list[HazardDetection] = []


# ── Core detection functions ────────────────────────────────────────────────

# Max dimension sent to Gemini — 720p is plenty for hazard detection,
# cuts JPEG payload ~4x vs 1080p and reduces upload time significantly
GEMINI_MAX_WIDTH = 1280
GEMINI_JPEG_QUALITY = 70


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = GEMINI_JPEG_QUALITY) -> bytes:
    """Convert an OpenCV frame to JPEG bytes for API submission.
    Automatically downscales to GEMINI_MAX_WIDTH to reduce payload size."""
    h, w = frame.shape[:2]
    if w > GEMINI_MAX_WIDTH:
        scale = GEMINI_MAX_WIDTH / w
        frame = cv2.resize(frame, (GEMINI_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return buffer.tobytes()


def detect_hazards(
    frame: np.ndarray,
    model: str = "gemini-2.5-flash",
) -> FrameAnalysis:
    """
    Send a single frame to Gemini Flash and get structured hazard detections.

    Args:
        frame: BGR image (OpenCV format)
        model: Gemini model ID

    Returns:
        FrameAnalysis with scene info and list of HazardDetection
    """
    client = _get_client()
    import google.genai as genai
    from google.genai import types

    # Encode frame as JPEG
    jpeg_bytes = frame_to_jpeg_bytes(frame)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            DETECTION_PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )

    # Parse the JSON response
    return _parse_response(response.text)


def detect_hazards_from_file(
    image_path: str,
    model: str = "gemini-2.5-flash",
) -> FrameAnalysis:
    """Detect hazards from an image file path."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return detect_hazards(frame, model=model)


def _parse_response(text: str) -> FrameAnalysis:
    """Parse Gemini's JSON response into a FrameAnalysis object."""
    # Strip any markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)

        # Compute center x/y from bounding box for each hazard
        for h in data.get("hazards", []):
            x1 = h.get("x1", 0)
            y1 = h.get("y1", 0)
            x2 = h.get("x2", 0)
            y2 = h.get("y2", 0)
            h["x"] = (x1 + x2) / 2
            h["y"] = (y1 + y2) / 2
            # Default fields the new prompt doesn't return
            h.setdefault("category", "Fall Hazard")
            h.setdefault("severity", "critical")

        return FrameAnalysis(**data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[VESTA] Warning: Failed to parse Gemini response: {e}")
        print(f"[VESTA] Raw response: {text[:200]}")
        return FrameAnalysis()


# ── Keyframe Sampler ────────────────────────────────────────────────────────

class KeyframeSampler:
    """
    Decides which frames to send to Gemini for analysis.

    Strategy: Sample every N frames, but also trigger on significant
    camera motion (the scene changed enough that we should re-check).
    A minimum cooldown prevents rapid-fire API calls during fast rotation.
    """

    def __init__(
        self,
        interval: int = 30,
        motion_threshold: float = 150.0,
        min_cooldown: int = 10,
    ):
        self.interval = interval
        self.motion_threshold = motion_threshold
        self.min_cooldown = min_cooldown  # minimum frames between keyframes
        self._last_keyframe = 0
        self._accumulated_motion = 0.0

    def should_sample(self, frame_idx: int, camera_motion=None) -> bool:
        """Return True if this frame should be sent to Gemini."""
        # Always sample first frame
        if frame_idx == 0:
            self._last_keyframe = 0
            return True

        frames_since_last = frame_idx - self._last_keyframe

        # Enforce minimum cooldown — never sample faster than this
        if frames_since_last < self.min_cooldown:
            if camera_motion is not None:
                self._accumulated_motion += camera_motion.magnitude
            return False

        # Interval-based sampling
        if frames_since_last >= self.interval:
            self._last_keyframe = frame_idx
            self._accumulated_motion = 0.0
            return True

        # Motion-triggered sampling
        if camera_motion is not None:
            self._accumulated_motion += camera_motion.magnitude
            if self._accumulated_motion >= self.motion_threshold:
                self._last_keyframe = frame_idx
                self._accumulated_motion = 0.0
                return True

        return False


# ── Batch Detection (Parallel API Calls) ────────────────────────────────────

def detect_hazards_batch(
    frames: list[tuple[int, np.ndarray]],
    model: str = "gemini-2.5-flash",
    max_workers: int = 4,
) -> dict[int, FrameAnalysis]:
    """
    Send multiple frames to Gemini in parallel using a thread pool.

    Args:
        frames: List of (frame_index, frame_array) tuples
        model: Gemini model ID
        max_workers: Number of parallel API calls

    Returns:
        Dict mapping frame_index → FrameAnalysis
    """
    results = {}

    def _detect_one(item):
        idx, frame = item
        return idx, detect_hazards(frame, model=model)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_detect_one, item): item[0] for item in frames}
        for future in as_completed(futures):
            try:
                idx, analysis = future.result()
                results[idx] = analysis
            except Exception as e:
                idx = futures[future]
                print(f"[VESTA] Batch detection error for frame {idx}: {e}")
                results[idx] = FrameAnalysis()

    return results
