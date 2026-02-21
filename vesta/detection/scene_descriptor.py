"""
Scene Descriptor — Rich Scene Graph Extraction via Gemini Flash

Sends keyframes to Gemini and extracts structured scene descriptions:
entities (with bounding boxes, categories, states), spatial relationships
between entities, and an overall scene description.

This replaces the hazard-only detection with full scene understanding.
"""

import json
import os

import cv2
import numpy as np
from pydantic import BaseModel

from vesta.detection.gemini_detector import frame_to_jpeg_bytes, _get_client


SCENE_PROMPT = """Extract a scene graph from this first-person construction camera frame. Return JSON only.

For each entity: label (consistent short name like "worker_1", "crane_1"), category (person/equipment/structure/material/vehicle), bounding box (x1,y1,x2,y2 as 0-1), confidence (0-1), description (brief: color + key feature), state (action or condition).

Rules:
- MAX 10 entities. Prioritize: people > equipment > vehicles > structures > materials.
- Skip background elements (sky, hills, trees, sun, clouds) — only include things ON the construction site.
- Use CONSISTENT labels across frames: "worker_1" not "person_arm_1" or "viewer_arm_hand".
- If you see a person's arm/hand holding the camera, label it "camera_self" not a unique name.
- Keep descriptions under 15 words.
- Include 3-5 key spatial relationships.

JSON format:
{{
  "scene_description": "One sentence.",
  "entities": [{{"label":"worker_1","category":"person","x1":0.3,"y1":0.2,"x2":0.5,"y2":0.8,"confidence":0.95,"description":"Worker in orange vest, yellow hardhat","state":"laying_blocks"}}],
  "relationships": [{{"subject":"worker_1","relation":"standing_on","object":"platform_1"}}]
}}"""


# ── Pydantic models ──────────────────────────────────────────────────────

class SceneEntity(BaseModel):
    label: str
    category: str = "unknown"
    x: float = 0.0       # center (computed from bbox)
    y: float = 0.0       # center (computed from bbox)
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    confidence: float = 0.0
    description: str = ""
    state: str = ""


class SceneRelationship(BaseModel):
    subject: str
    relation: str
    object: str


class SceneAnalysis(BaseModel):
    scene_description: str = ""
    entities: list[SceneEntity] = []
    relationships: list[SceneRelationship] = []


# ── Core function ────────────────────────────────────────────────────────

def describe_scene(
    frame: np.ndarray,
    model: str = "gemini-2.5-flash",
) -> SceneAnalysis:
    """
    Send a frame to Gemini Flash and get full scene graph extraction.

    Args:
        frame: BGR image (OpenCV format)
        model: Gemini model ID

    Returns:
        SceneAnalysis with entities, relationships, and scene description
    """
    client = _get_client()
    from google.genai import types

    jpeg_bytes = frame_to_jpeg_bytes(frame)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            SCENE_PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )

    return _parse_scene_response(response.text)


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair JSON truncated mid-output by closing open structures."""
    # Find the last complete entity or relationship object
    # Strategy: truncate to last complete "}", then close arrays/objects
    last_complete = -1
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth >= 0:
                last_complete = i

    if last_complete <= 0:
        return None

    # Truncate to last complete closing brace
    repaired = text[:last_complete + 1]

    # Count unclosed brackets/braces
    open_brackets = repaired.count('[') - repaired.count(']')
    open_braces = repaired.count('{') - repaired.count('}')

    # Close them
    repaired += ']' * max(0, open_brackets)
    repaired += '}' * max(0, open_braces)

    return repaired


def _parse_scene_response(text: str) -> SceneAnalysis:
    """Parse Gemini's JSON response into a SceneAnalysis object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    def _process_data(data: dict) -> SceneAnalysis:
        for ent in data.get("entities", []):
            x1 = ent.get("x1", 0)
            y1 = ent.get("y1", 0)
            x2 = ent.get("x2", 0)
            y2 = ent.get("y2", 0)
            ent["x"] = (x1 + x2) / 2
            ent["y"] = (y1 + y2) / 2
            valid_cats = {"person", "equipment", "structure", "material", "vehicle", "signage"}
            if ent.get("category", "").lower() not in valid_cats:
                ent["category"] = "unknown"
        return SceneAnalysis(**data)

    try:
        data = json.loads(cleaned)
        return _process_data(data)
    except (json.JSONDecodeError, ValueError) as e:
        # Try to repair truncated JSON
        repaired = _repair_truncated_json(cleaned)
        if repaired:
            try:
                data = json.loads(repaired)
                n_ents = len(data.get("entities", []))
                print(f"[VESTA] Repaired truncated JSON — recovered {n_ents} entities")
                return _process_data(data)
            except (json.JSONDecodeError, ValueError):
                pass

        print(f"[VESTA] Warning: Failed to parse scene response: {e}")
        print(f"[VESTA] Raw response: {text[:300]}")
        return SceneAnalysis()
