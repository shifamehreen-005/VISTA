# VISTA — Video Intelligence with Spatio-Temporal Augmented Retrieval for Egocentric Understanding

VISTA (referenced as VESTA, in some places) builds a **persistent, spatially-grounded scene graph** from egocentric video. Unlike standard VLMs that process each frame independently with no memory, VESTA tracks every entity (workers, equipment, structures, materials) with world-fixed coordinates and temporal observation history — enabling precise "what, where, when" queries that no frontier model can answer alone.

Built for the UMD Ironsite Hackathon. Powered by Google Gemini 3 Flash + OpenCV optical flow.

## How It Works

```
Video (hardhat-cam / egocentric)
    │
    ├── Every frame ──► Optical Flow (ORB + Affine) ──► Camera heading
    │
    └── Keyframes ────► Gemini Flash ──► Entities + Relationships + Scene description
                                                │
                                                ▼
                                    ┌──────────────────────┐
                                    │   SCENE GRAPH         │
                                    │   Entities (nodes)    │
                                    │   + world-fixed coords│
                                    │   + observation chain  │
                                    │   + relationships      │
                                    └───────────┬──────────┘
                                                │
                                   "Where is the crane relative
                                    to the worker at 10s?"
                                                │
                                    ┌───────────┴──────────┐
                                    │ Spatial → geometric   │
                                    │   "right, ~45°"       │
                                    │ Temporal → retrieve    │
                                    │   observation history  │
                                    └──────────────────────┘
```

The pipeline runs in **2 overlapped passes**:

1. **Pass 1+2 (overlapped):** Read frames, run ORB optical flow on every frame, and **submit keyframes to Gemini in parallel** (4 workers). Gemini latency is hidden behind flow computation.
2. **Pass 3 (fast, <1s):** Replay motion, inject entities + relationships into scene graph, write annotated video with bounding boxes + radar minimap.

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd Ironsite

# Install dependencies
pip install -r requirements.txt

# Set up your API key
echo "GEMINI_API_KEY=your-key-here" > .env
```

## Usage

### 1. Process a Video

Analyze video, build scene graph, generate outputs.

```bash
# Full video processing
python scripts/run_pipeline.py --video data/test_videos/test_2.mp4

# Quick test — first 150 frames (~10 seconds)
python scripts/run_pipeline.py --video data/test_videos/test_2.mp4 --max-frames 150

# Process without interactive Q&A (just save outputs)
python scripts/run_pipeline.py --video data/test_videos/test_2.mp4 --no-interactive
```

**Output files** (saved to `results/`):

| File | What |
|------|------|
| `<video>_annotated.mp4` | Video with entity bounding boxes + radar minimap + status bar |
| `<video>_results.json` | Full scene graph as JSON (entities, relationships, positions) |
| `<video>_graph.pkl` | Saved agent state — use with `scripts/ask.py` for instant Q&A |
| `<video>_map_3d.html` | Interactive 3D Plotly map (camera path + entity positions) |
| `<video>_map_2d.html` | Bird's-eye 2D map with heading arrows |

After processing, it drops into interactive Q&A mode.

### 2. Ask Questions (No Re-Processing)

Load a saved scene graph and ask unlimited questions instantly:

```bash
# Interactive mode
python scripts/ask.py --graph results/test_2_graph.pkl

# Single question
python scripts/ask.py --graph results/test_2_graph.pkl -q "What's behind me?"

# Multiple questions
python scripts/ask.py --graph results/test_2_graph.pkl \
  -q "Where is the crane relative to the worker?" \
  -q "What was visible at 10 seconds?" \
  -q "How many workers are on site?"
```

### Example Questions

```
Ask VESTA: What's behind me?
Ask VESTA: Where is the scaffolding relative to the worker?
Ask VESTA: What entities were visible at 5 seconds?
Ask VESTA: How did the wall change over time?
Ask VESTA: What's near the crane?
Ask VESTA: How many workers are on site?
Ask VESTA: map                   ← dumps all entities with positions
Ask VESTA: quit                  ← exit
```

### 3. Run Tests (No API Key Needed)

```bash
# Scene graph tests (14 tests)
python tests/test_scene_graph.py

# Hazard registry tests (7 tests)
python tests/test_registry.py
```

Tests cover: entity creation, merging, object permanence, direction queries, rotation tracking, temporal queries, relationships, spatial relations, confidence decay.

## Real-Time Mode

Live processing on webcam or video feed with spoken audio alerts and optional 3D web visualization.

```bash
# Live from webcam
python realtime/run.py

# From a video file
python realtime/run.py --video data/test_videos/test_2.mp4

# With Q&A after video ends
python realtime/run.py --video data/test_videos/test_2.mp4 --ask

# With 3D web visualization (opens localhost:8080)
python realtime/run.py --video data/test_videos/test_2.mp4 --viz

# Disable audio alerts
python realtime/run.py --no-audio
```

**Controls:** `Q` = quit, `Space` = pause, `S` = screenshot

## Configuration

### Standalone Pipeline (`scripts/run_pipeline.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Path to video file (.mp4) |
| `--max-frames` | all | Limit frames to process |
| `--keyframe-interval` | 30 | Frames between Gemini API calls |
| `--model` | `gemini-2.5-flash` | Gemini model to use |
| `--no-interactive` | false | Skip interactive Q&A mode |
| `--output-video` | auto | Custom path for annotated video |

### Real-Time Pipeline (`realtime/run.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | webcam | Path to video file |
| `--webcam` | 0 | Webcam device index |
| `--ask` | false | Q&A mode after session |
| `--viz` | false | Launch 3D web viz (localhost:8080) |
| `--no-audio` | false | Disable spoken alerts |
| `--keyframe-interval` | 60 | Frames between Gemini samples |
| `--workers` | 2 | Background Gemini threads |
| `--model` | `gemini-2.5-flash` | Gemini model ID |

## Project Structure

```
Ironsite/
├── vesta/                              ← Core library
│   ├── flow/
│   │   └── optical_flow.py             ← Camera motion estimation (ORB + RANSAC)
│   ├── detection/
│   │   ├── gemini_detector.py          ← Gemini hazard detection (legacy)
│   │   └── scene_descriptor.py         ← Full scene extraction (entities + relationships)
│   ├── registry/
│   │   ├── hazard_registry.py          ← Allocentric hazard map (legacy)
│   │   └── scene_graph.py              ← Spatio-temporal scene graph
│   ├── agent/
│   │   ├── vesta_agent.py              ← Hazard-only agent (legacy)
│   │   └── scene_agent.py              ← Scene graph agent with 7 tools
│   └── utils/
│       ├── visualizer.py               ← Video annotation (bboxes + radar minimap)
│       └── spatial_map.py              ← 3D/2D maps (MiDaS depth + Plotly)
│
├── realtime/                           ← Real-time pipeline
│   ├── realtime_pipeline.py            ← Live processing loop (threading)
│   ├── audio_alerts.py                 ← Escalating spoken warnings (pyttsx3)
│   ├── trajectory.py                   ← Predictive collision warning
│   ├── web_viz.py                      ← Flask-SocketIO 3D server
│   ├── run.py                          ← CLI entry point
│   └── static/index.html              ← Three.js 3D visualization
│
├── scripts/
│   ├── run_pipeline.py                 ← Process video → scene graph → outputs
│   ├── ask.py                          ← Query saved graph (instant Q&A)
│   └── generate_map.py                 ← Regenerate spatial maps
│
├── tests/
│   ├── test_scene_graph.py             ← 14 scene graph tests
│   └── test_registry.py               ← 7 hazard registry tests
│
├── data/
│   └── test_videos/                    ← Drop your test videos here
│
├── results/                            ← Output directory
├── requirements.txt
└── .env                                ← GEMINI_API_KEY (not committed)
```

## Performance

| Metric | Value |
|--------|-------|
| Optical flow | ~55 FPS (ORB, 640x480) |
| Gemini Flash latency | ~5s per keyframe |
| Pipeline throughput (overlapped) | 150 frames in 33s |
| Scene graph size | ~20-30 entities per minute |
| Q&A response time | 2-5s |
| Heading drift | <2° over 60s |
