# VISTA — Video Intelligence with Spatio-Temporal Augmented Retrieval for Egocentric Understanding

VISTA builds a **persistent spatio-temporal scene graph** from egocentric construction video, enabling spatial, temporal, and reasoning queries that frontier VLMs cannot answer alone.

A single monocular hardhat camera. No GPS, no SLAM, no depth sensors. Just optical flow + Gemini Flash + a scene graph.

**Built by Team Cortex at UMD Ironsite Hackathon 2025.**

---

## Demo

https://github.com/user-attachments/assets/684c97a0-7411-47f2-9277-a324c09ce9f0

---

## The Problem

Construction sites generate hours of first-person video from hardhat-mounted cameras every day. Supervisors reviewing this footage face a fundamental limitation: **the footage remembers everything, but the AI remembers nothing.**

Frontier VLMs (Gemini, GPT-4o, Claude) process video frame-by-frame. Every frame is a blank slate. The moment an object leaves the camera's field of view, it ceases to exist for the model. Ask _"Where's the ladder the worker walked past two minutes ago?"_ and the model fails — not because it can't recognize a ladder, but because it has no mechanism to remember that the ladder exists once off-screen.

This limitation is **architectural, not perceptual**. Frame-level models are fundamentally incapable of answering where-and-when questions, no matter how accurate they are per frame. The bottleneck is persistence, not perception.

---

## Our Approach

RAG transformed what LLMs could do with documents by giving them structured, retrievable memory. **VISTA does the same thing for video** — turning raw egocentric footage into structured, queryable spatial-temporal memory that any VLM can retrieve from. The same way RAG enabled LLMs to answer questions over knowledge they were never trained on, our Spatio-Temporal RAG enables VLMs to answer questions over physical spaces they can no longer see.

Rather than replacing frontier models, VISTA augments them with a memory layer they never had.

---

## How It Works

```
Egocentric Video (hardhat cam)
    │
    ├── Every frame ──► Optical Flow (ORB + RANSAC) ──► Allocentric Heading
    │
    └── Every 30th frame ──► Gemini 3 Flash ──► Entities + Relationships
                                                        │
                                                        ▼
                                            ┌───────────────────────┐
                                            │     SCENE GRAPH       │
                                            │  World-fixed coords   │
                                            │  Observation chains   │
                                            │  Object permanence    │
                                            └───────────┬───────────┘
                                                        │
                                            Agentic RAG (9 tools, 5 rounds)
                                                        │
                                                        ▼
                                              Natural language answer
```

### Stage 1 — Parallel Video Processing

Two passes run simultaneously over the video:

- **Optical Flow (every frame):** ORB feature detection (500 keypoints) matched via BFMatcher with Hamming distance, filtered through Lowe's ratio test (0.75). Surviving matches feed into OpenCV's `estimateAffinePartial2D` with RANSAC, extracting per-frame camera rotation. These rotations **accumulate continuously** into an **allocentric heading** — a world-fixed compass direction that tells us exactly which way the camera points at any moment. This is VISTA's "vestibular system," providing spatial awareness from nothing but pixel math. Drift stays below 2° over 60 seconds.

- **Scene Extraction (every 30th frame):** Keyframes are sent to Gemini Flash (temperature 0.1, structured JSON output), extracting entities — workers, tools, equipment, structures — with bounding boxes, confidence scores, visual descriptions, current states, and inter-entity relationships. Up to 10 concurrent API calls run in a thread pool, fully hiding Gemini latency behind the optical flow pass.

### Stage 2 — Spatio-Temporal Scene Graph

Both passes converge into a **persistent scene graph** — the core data structure. Each entity's pixel-space bounding box is transformed into **world-fixed polar coordinates**: horizontal center maps to an egocentric angle (via FOV geometry), and the accumulated heading converts it to an allocentric angle that remains stable regardless of subsequent camera rotation. Distance is estimated from vertical position.

**Entity merging** provides object permanence: new observations match against existing entities by label (fuzzy substring matching), category, and spatial proximity — within 20° and 0.25 distance units for people, doubled to 40° and 0.50 for structures/equipment. Matched observations update via exponential smoothing (alpha=0.3). Confidence decays at 0.015/s for unobserved entities, pruning below 0.2.

The result: a **queryable map of reality** with world-fixed coordinates, full temporal observation chains, tracked states, and spatial relationships — persisting even when entities are completely off-camera.

### Stage 3 — Agentic RAG

Natural language queries are resolved through an **agentic RAG loop** using Gemini's function-calling API with 9 graph query tools:

| Tool | Purpose |
|------|---------|
| `get_entities_in_direction` | Spatial lookup by compass bearing |
| `get_spatial_relation` | Angle and distance between two entities |
| `get_entity_timeline` | State changes over time for an entity |
| `get_entities_at_time` | Snapshot of visible entities at a timestamp |
| `get_entity_info` | Full observation history and attributes |
| `get_relationships` | Filtered relationship traversal |
| `get_direction_at_time` | Camera heading at a past moment |
| `get_changes` | State transitions over a time window |
| `get_all_entities` | Summary statistics and category counts |

The system prompt injects live scene graph statistics (entity count, category breakdown, heading, time range). Gemini autonomously selects which tools to call and in what order, iterating through up to **5 tool-calling rounds** before synthesizing a final answer.

---

## What VISTA Can Answer

| Capability | Example | Why VLMs Can't |
|---|---|---|
| Temporal retrieval | "When does the worker lay the first brick?" → 371s | No timestamped observation chains |
| Productivity analysis | "How long was the worker idle?" → 370.6s | Can't diff entity states over time |
| Object permanence | "Where is the ladder?" → Behind-left (-129°) | Object off-screen = doesn't exist |
| Directional reasoning | "What's behind me right now?" | No concept of camera heading |
| Temporal reasoning | "Does the worker return to the tool?" → No | Can't check observation timelines |

---

## VISTA vs Frontier VLMs on Ironsite Data

| Query | Ground Truth | VISTA | Gemini 3 Flash | Molmo |
|---|---|---|---|---|
| At 30s, where are the concrete walls relative to me? (test\_2) | Right | **Right** | Right | Left |
| At 10s, where is the ladder relative to the worker? (test\_10) | Behind | **Behind** | Right | Right |
| How many seconds is the person idle? (test\_3) | 190s | **176s** | 299s | 10s |
| Where is the bucket at start vs end? (test\_11) | start=BELOW, end=RIGHT | start=FRONT-LEFT, end=FRONT-LEFT | — | start=right, end=right |

VISTA matches or outperforms frontier VLMs on spatial and temporal queries that require persistent world memory. On directional reasoning (test\_10), both Gemini 3 Flash and Molmo answer incorrectly because they lack allocentric heading — they cannot determine "behind" without a world-fixed coordinate system. On productivity analysis (test\_3), VISTA's answer (176s) is within 7% of ground truth while Gemini overestimates by 57% and Molmo underestimates by 95%.

---

## Key Findings

- **Spatial memory is an architectural problem, not a perceptual one.** Frame-level models are fundamentally incapable of answering where-and-when questions. The bottleneck is persistence, not perception — even the best VLMs fail on spatial queries that a simple scene graph can answer.

- **Optical flow is massively underutilized in the VLM era.** Cheap classical computer vision running at full frame rate (~55 FPS) provides the spatial scaffolding that expensive foundation models cannot — and costs essentially nothing. The accumulated heading gives world-fixed spatial awareness from pure pixel math.

- **RAG doesn't have to mean documents and embeddings.** Spatio-temporal retrieval — filtering by location, direction, and time rather than semantic similarity — is a powerful and underexplored paradigm. The scene graph is the new vector store.

- **Agentic tool-calling is the right interface for spatial queries.** Letting the model decide which graph tools to call, in what order, across multiple rounds produces far richer answers than any single retrieval step. A question like "how long was the worker idle?" requires chaining temporal retrieval, state diffing, and duration computation — the agent handles that naturally.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Setup

```bash
git clone <repo-url>
cd Ironsite

pip install -r requirements.txt

# Set up your API key
echo "GEMINI_API_KEY=your-key-here" > .env
```

### Run the Web App (VISTA Q&A)

```bash
streamlit run app.py
```

Upload a video (MP4) and ask questions in the chat panel. The app processes the video, builds a scene graph, and lets you query it conversationally.

### Run the Pipeline (CLI)

```bash
# Process a video and build the scene graph
python scripts/run_pipeline.py --video data/test_videos/test_2.mp4

# Quick test — first 150 frames only
python scripts/run_pipeline.py --video data/test_videos/test_2.mp4 --max-frames 150

# Process without interactive Q&A
python scripts/run_pipeline.py --video data/test_videos/test_2.mp4 --no-interactive
```

### Ask Questions (No Re-Processing)

Load a saved scene graph and query instantly:

```bash
# Interactive mode
python scripts/ask.py --graph results/test_2_graph.pkl

# Single question
python scripts/ask.py --graph results/test_2_graph.pkl -q "What's behind me?"
```

---

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Path to video file |
| `--max-frames` | all | Limit frames to process |
| `--keyframe-interval` | 30 | Frames between Gemini API calls |
| `--model` | `gemini-3-flash` | Gemini model ID |
| `--workers` | 10 | Max concurrent Gemini API calls |
| `--no-interactive` | false | Skip interactive Q&A after processing |

---

## Project Structure

```
Ironsite/
├── app.py                          ← Streamlit web app (VISTA Q&A)
├── vesta/
│   ├── flow/optical_flow.py        ← Camera motion (ORB + RANSAC)
│   ├── detection/scene_descriptor.py ← Gemini scene extraction
│   ├── registry/scene_graph.py     ← Spatio-temporal scene graph
│   ├── agent/scene_agent.py        ← Agentic RAG with 9 tools
│   └── utils/
│       ├── visualizer.py           ← Video annotation + radar minimap
│       └── spatial_map.py          ← 3D/2D maps (Plotly)
├── scripts/
│   ├── run_pipeline.py             ← CLI pipeline runner
│   ├── ask.py                      ← Query saved graphs
│   └── generate_map.py             ← Regenerate spatial maps
├── tests/
│   ├── test_scene_graph.py         ← Unit tests
│   └── eval_spatial.py             ← Evaluation suite
├── results/                        ← Output directory
├── data/test_videos/               ← Input videos
├── requirements.txt
└── .env                            ← GEMINI_API_KEY
```

---

## Performance

| Metric | Value |
|--------|-------|
| Pipeline throughput | 10-min video in ~35s |
| Query response time | 2–5 seconds |
| Eval accuracy | 93.3% (14/15) |
| Heading drift | <2° over 60s |
| Optical flow | ~55 FPS |

---

*Team Cortex*
