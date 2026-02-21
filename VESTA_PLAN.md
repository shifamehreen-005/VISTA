# VESTA — Vision-Enhanced Spatial Tracking Agent
## 24-Hour Hackathon Build Plan (UMD Ironsite)

---

## The Problem: Spatial Amnesia
VLMs process frames independently. A trench seen at T=5s is forgotten by T=60s if the camera
pans away. Workers need an agent that **remembers every hazard it has ever seen**, regardless of
where the camera is currently pointing.

## The Solution: A Stateful Hazard Registry
VESTA maintains an **allocentric hazard map** — a persistent dictionary of every detected threat,
continuously updated with camera motion so that "behind you" and "to your left" always resolve
correctly.

---

## Architecture Overview

```
Video Feed
    │
    ├──[every N frames]──► Stage A: DETECTION (Gemini 2.0 Flash)
    │                        Extract hazards → {label, [x,y], confidence}
    │                                │
    │                                ▼
    │                        ┌─────────────────┐
    ├──[every frame]────────►│  Stage C: REGISTRY  │◄── OSHA Narratives DB
    │   Stage B: FLOW        │  (Hazard Map)       │     (18,694 incidents)
    │   Optical flow →       │  {id, label, vector,│
    │   [dx, dy, dθ]        │   risk, timestamp}  │
    │                        └────────┬────────────┘
    │                                 │
    │                                 ▼
    │                        Stage D: AGENT (Gemini 2.0 Flash)
    │                        Tools: get_hazards_at_angle()
    │                                get_all_hazards()
    │                                get_osha_context()
    └────────────────────────────────────────────────────
```

---

## Key Improvements Over Original Blueprint

1. **Gemini 2.0 Flash over Pro** — 10x cheaper, ~3x faster, sufficient accuracy for
   construction hazard detection. We can afford to sample MORE keyframes (every 15-30
   frames instead of 30-60), giving better coverage.

2. **Hybrid Flow Strategy** — Use ORB feature matching for rotation estimation (more
   robust than pure dense optical flow for large camera rotations). Fall back to
   Farneback dense flow for translation.

3. **Polar Coordinate Registry** — Store hazards in (distance, angle) relative to an
   allocentric north, not pixel coords. This makes "what's behind me?" a simple angle
   query rather than a coordinate transform.

4. **Confidence Decay** — Hazards that haven't been re-observed for N seconds get a
   decaying confidence score. Prevents phantom hazards from dominating.

5. **OSHA Narrative Injection** — Don't just name the hazard. Pull the most relevant
   incident narrative from the CSV data to give the agent real-world context for its
   warnings.

---

## Folder Structure

```
Ironsite/
├── VESTA_PLAN.md              ← This file
├── requirements.txt
├── .env                       ← GEMINI_API_KEY
│
├── data/
│   ├── osha/                  ← 13 OSHA CSV files (~2.3GB)
│   ├── test_videos/           ← Sample construction footage
│   └── test_frames/           ← Extracted keyframes for testing
│
├── vesta/
│   ├── __init__.py
│   ├── flow/
│   │   ├── __init__.py
│   │   └── optical_flow.py    ← Module 1: Camera motion estimation
│   ├── detection/
│   │   ├── __init__.py
│   │   └── gemini_detector.py ← Module 2: Gemini Flash hazard detection
│   ├── registry/
│   │   ├── __init__.py
│   │   └── hazard_registry.py ← Module 3: Stateful hazard map
│   ├── agent/
│   │   ├── __init__.py
│   │   └── vesta_agent.py     ← Module 4: Orchestrator + Gemini tools
│   └── utils/
│       ├── __init__.py
│       └── osha_lookup.py     ← OSHA narrative search utility
│
├── scripts/
│   └── run_pipeline.py        ← End-to-end pipeline runner
├── tests/
│   └── test_registry.py       ← Unit tests for the registry
│
├── analyze_osha.py            ← (existing) OSHA analysis plots
├── benchmark/                 ← (existing) VLM spatial benchmarks
└── plots/                     ← (existing) Generated charts
```

---

## Module Specifications

### Module 1: Optical Flow (`vesta/flow/optical_flow.py`)
- **Input**: Two consecutive video frames
- **Output**: `CameraMotion(dx, dy, d_theta)` — pixel translation + rotation in degrees
- **Method**: ORB keypoint matching → estimate affine transform → decompose into
  translation + rotation. Farneback dense flow as fallback.
- **Key function**: `estimate_camera_motion(prev_frame, curr_frame) → CameraMotion`

### Module 2: Gemini Detector (`vesta/detection/gemini_detector.py`)
- **Input**: Single video frame (JPEG bytes)
- **Output**: `list[HazardDetection(label, x, y, confidence, osha_category)]`
- **Model**: `gemini-2.0-flash` via `google-genai` SDK
- **Prompt strategy**: System prompt with OSHA High-Risk categories, structured JSON output
- **Key function**: `detect_hazards(frame) → list[HazardDetection]`

### Module 3: Hazard Registry (`vesta/registry/hazard_registry.py`)
- **The brain of VESTA**
- **Data structure**: Dict of `HazardEntry(id, label, angle, distance, confidence,
  osha_risk, first_seen, last_seen, osha_narrative)`
- **Key operations**:
  - `add_hazard(detection, frame_timestamp)` — Ego→Allo conversion, dedup nearby
  - `update_with_motion(camera_motion)` — Rotate all hazard angles by -d_theta
  - `query_angle(angle, fov=90)` — "What's at 180°?" returns all hazards in that arc
  - `get_all()` — Full registry dump for agent reasoning
  - `decay_confidence(dt)` — Reduce confidence of stale entries

### Module 4: VESTA Agent (`vesta/agent/vesta_agent.py`)
- **Orchestrator** that runs the 3-stage loop
- **Gemini Tools exposed**:
  - `get_hazard_at_angle(degrees)` → Registry query
  - `get_all_hazards()` → Full map
  - `get_osha_context(hazard_label)` → Relevant OSHA narrative from CSV
- **Query interface**: Natural language questions about the scene

---

## Build Order

| Phase | Hours | What                          | Depends On |
|-------|-------|-------------------------------|------------|
| 1a    | 0-3   | Optical Flow engine           | Nothing    |
| 1b    | 0-3   | Gemini detector prompt+parser | API key    |
| 2     | 3-6   | Hazard Registry + transforms  | 1a, 1b     |
| 3     | 6-9   | OSHA narrative lookup         | CSVs       |
| 4     | 9-12  | Agent orchestrator + tools    | 2, 3       |
| 5     | 12-15 | End-to-end pipeline           | 4          |
| 6     | 15-18 | Demo UI / polish              | 5          |

---

## Tech Stack

- **VLM**: Gemini 2.0 Flash (`google-genai` SDK)
- **Vision**: OpenCV (optical flow, ORB features)
- **Data**: Pydantic models for type safety
- **OSHA**: pandas for CSV search, pre-indexed by hazard category
- **Python**: 3.11+
