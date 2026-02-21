# VESTA — Vision-Enhanced Spatial Tracking Agent

A stateful AI agent that detects and **remembers** construction site hazards from hardhat camera video. Unlike standard VLMs that forget everything between frames, VESTA maintains a persistent 360-degree hazard map — so a hole seen 30 seconds ago is still tracked even when the camera is facing the opposite direction.

Built for the UMD Ironsite Hackathon. Powered by Google Gemini 2.5 Flash + OpenCV optical flow + 18,694 real OSHA incident records.

## How It Works

```
Video Frame ──► Optical Flow (every frame) ──► Camera rotation [dx, dy, dθ]
                                                        │
Video Frame ──► Gemini Flash (every 30 frames) ──► Hazard detections [label, x, y]
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  HAZARD REGISTRY │
                                                │  (persistent map)│
                                                └────────┬────────┘
                                                         │
                                        "What's behind me?" ──► Agent answers
                                                                 using the map
```

The pipeline runs in **2 overlapped passes** for speed:

1. **Pass 1+2 (overlapped):** Read frames, run ORB optical flow, and **submit keyframes to Gemini instantly** as they're identified (4 parallel workers). By the time the last frame is read, most Gemini results are already back. Flow + Gemini run simultaneously.
2. **Pass 3 (fast, ~1s):** Replay the motion sequence, convert pixel detections to world-fixed polar angles, build the hazard registry, write annotated video

After processing, the **agent** answers spatial questions ("what's behind me?") by querying the registry and citing real OSHA incident data.

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

## Usage — Two Commands

### Command 1: Process a video

This is the heavy step — it analyzes the video, detects hazards, and saves everything.

```bash
# Full video processing (generates annotated video + JSON + saved registry)
python scripts/run_pipeline.py --video data/test_videos/test.mp4

# Quick test — only first 150 frames (~5 seconds)
python scripts/run_pipeline.py --video data/test_videos/test.mp4 --max-frames 150

# Process without interactive mode (just save outputs and exit)
python scripts/run_pipeline.py --video data/test_videos/test.mp4 --no-interactive
```

**What it produces** (saved to `results/`):

| File | What |
|------|------|
| `results/<video>_annotated.mp4` | Video with hazard crosshairs + radar minimap + status bar |
| `results/<video>_results.json` | Full hazard registry as JSON (every hazard with angle, severity, confidence, timestamps) |
| `results/<video>_registry.pkl` | Saved agent state — use this with `scripts/ask.py` to ask questions without re-processing |
| `results/<video>_map_3d.html` | Interactive 3D Plotly map — camera path + hazard positions (rotate, zoom, explore) |
| `results/<video>_map_2d.html` | Bird's-eye 2D map — top-down view with heading arrows |

After processing, it drops you into interactive mode where you can type questions.

### Command 2: Ask questions (no re-processing)

Once you've processed a video, you can ask unlimited questions instantly using the saved registry:

```bash
# Interactive mode — ask multiple questions in a conversation
python scripts/ask.py --registry results/test_registry.pkl

# Single question (great for scripting/demos)
python scripts/ask.py --registry results/test_registry.pkl -q "What hazards are behind me?"

# Multiple questions at once
python scripts/ask.py --registry results/test_registry.pkl \
  -q "List all fall risks and how severe they are" \
  -q "Give me a full safety briefing for this site"
```

### Interactive Commands

When in interactive mode (either script), you can type:

```
Ask VESTA: What hazards are in front of me?
Ask VESTA: List all fall risks and their severity
Ask VESTA: Give me a full safety briefing for this site
Ask VESTA: map                          ← dumps the full hazard registry
Ask VESTA: quit                         ← exit
```

### Run Tests (No API Key Needed)

```bash
python tests/test_registry.py
```

Tests the core spatial logic — coordinate transforms, hazard persistence, rotation tracking, and directional queries — all without hitting the Gemini API.

## Real-Time Mode

VESTA also runs **live** on a webcam or video feed — processing optical flow on every frame in the main thread and running Gemini detection in background threads, with live annotated display and spoken audio alerts.

### Usage

```bash
# Live from webcam
python realtime/run.py

# From a video file
python realtime/run.py --video data/test_videos/test.mp4

# With interactive Q&A after the video ends
python realtime/run.py --video data/test_videos/test.mp4 --ask

# Disable audio alerts
python realtime/run.py --video data/test_videos/test.mp4 --no-audio
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `Space` | Pause / Resume |
| `S` | Save screenshot |

### How It Works

```
┌──────────── MAIN THREAD (~30+ fps) ──────────────────────────┐
│  cap.read() → optical flow (14ms) → update heading           │
│  → keyframe? submit to background → collect results           │
│  → proximity check → audio alert → annotate → display         │
├──────────── BACKGROUND THREADS ──────────────────────────────┤
│  ThreadPoolExecutor (2 workers): Gemini detection (~2-5s)     │
│  AlertSpeaker thread: pyttsx3 TTS for spoken warnings         │
└───────────────────────────────────────────────────────────────┘
```

The main thread never blocks on Gemini. Keyframes are submitted to a thread pool; results are collected on the next frame loop iteration. Hazards appear in the overlay ~2-5s after first entering the frame.

### Intelligent Audio Alerts

The proximity tracker evaluates every hazard's position relative to the worker each frame and generates **escalating spoken warnings**:

| Situation | Example Alert |
|-----------|---------------|
| First detection (critical/high) | "Hazard detected: floor opening, directly behind you" |
| Nearby | "Floor opening detected behind you, nearby" |
| Approaching (distance decreasing) | "Caution, you're moving toward the floor opening behind you" |
| Very close | "Warning, floor opening very close, directly behind you" |
| Imminent danger | "Stop! Floor opening right behind you!" |

Alerts escalate based on distance trend (are you getting closer?), blind spot detection (hazards behind you are more urgent), and severity. Each level has its own cooldown to prevent spam.

### Real-Time Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | webcam | Path to video file |
| `--webcam` | 0 | Webcam device index |
| `--ask` | false | Interactive Q&A after session ends |
| `--no-audio` | false | Disable spoken audio alerts |
| `--keyframe-interval` | 60 | Frames between Gemini samples |
| `--workers` | 2 | Background Gemini threads |
| `--model` | `gemini-2.5-flash` | Gemini model ID |

## Performance

Both pipelines benefit from automatic resolution-adaptive processing:

| Optimization | Before | After | Improvement |
|---|---|---|---|
| Optical flow (1080p input) | 65ms/frame | 14ms/frame | **4.7x faster** |
| Gemini payload size | 1,531 KB | 365 KB | **76% smaller** |
| Flow FPS ceiling | ~15 fps | ~74 fps | Headroom for all other processing |

- **Optical flow auto-downscales** to 640px width internally — ORB feature detection is O(w*h), so 480p is 4.4x faster than 1080p with no accuracy loss on rotation estimation
- **Gemini frames auto-downscale** to 1280px and encode at JPEG quality 70 — 720p is more than enough for hazard detection, and 4x smaller payloads mean faster network transfer
- **Live performance metrics** are shown on-screen in real-time mode: FPS, flow time, annotation time, and Gemini round-trip

## Project Structure

```
Ironsite/
├── vesta/                          ← Core VESTA package
│   ├── flow/
│   │   └── optical_flow.py         ← Camera motion estimation (ORB features + RANSAC)
│   ├── detection/
│   │   └── gemini_detector.py      ← Gemini 2.5 Flash hazard detection + parallel batching
│   ├── registry/
│   │   └── hazard_registry.py      ← Stateful hazard map with ego/allo coordinate transforms
│   ├── agent/
│   │   └── vesta_agent.py          ← Orchestrator: 3-pass pipeline + Gemini tool-calling for Q&A
│   └── utils/
│       ├── osha_lookup.py          ← OSHA incident narrative search
│       ├── visualizer.py           ← Video annotation: crosshairs, radar minimap, status bar
│       └── spatial_map.py          ← 3D/2D spatial maps with camera path + hazard projection
│
├── realtime/                       ← Real-time pipeline
│   ├── realtime_pipeline.py        ← Live processing: main loop + background Gemini threads
│   ├── audio_alerts.py             ← Proximity tracker + escalating TTS warnings
│   └── run.py                      ← Entry point: webcam/video with CLI args
│
├── scripts/
│   ├── run_pipeline.py             ← Command 1: Process video → detect → save results + maps
│   ├── ask.py                      ← Command 2: Ask questions from saved registry (instant)
│   └── generate_map.py             ← Regenerate spatial maps from saved registry
│
├── data/
│   ├── osha/                       ← 13 OSHA CSV files (~2.3GB, not in repo)
│   ├── test_videos/                ← Drop your test videos here
│   └── test_frames/                ← Extracted keyframes
│
├── results/                        ← Output: annotated videos, JSON, saved registries
├── tests/
│   └── test_registry.py            ← Unit tests (7 tests, no API key needed)
├── docs/
│   ├── VESTA_CORE_IDEA.md          ← One-paragraph explanation of the core concept
│   └── OSHA_ENGINE_SPEC.md         ← Build spec for the OSHA engine module
│
├── requirements.txt
├── VESTA_PLAN.md                   ← Full architecture plan
└── .env                            ← Your GEMINI_API_KEY (not committed)
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Path to video file (.mp4) |
| `--max-frames` | all | Limit frames to process |
| `--keyframe-interval` | 30 | Frames between Gemini API calls |
| `--model` | `gemini-2.5-flash` | Gemini model to use |
| `--no-interactive` | false | Skip interactive query mode |
| `--output-video` | auto | Custom path for annotated video |

## How Each File Works

| File | What It Does | How |
|------|-------------|-----|
| `optical_flow.py` | Estimates camera rotation between frames | ORB feature detection → match features → RANSAC affine fit → decompose to dx, dy, rotation angle |
| `gemini_detector.py` | Detects hazards in keyframes | Encodes frame as JPEG → sends to Gemini 2.5 Flash with OSHA hazard prompt → parses JSON response with hazard labels + pixel coordinates |
| `hazard_registry.py` | Maintains the persistent hazard map | Converts pixel coords → egocentric angle → allocentric (world-fixed) angle. Merges duplicates. Decays stale entries. Answers directional queries. |
| `vesta_agent.py` | Orchestrates everything | Runs 3-pass pipeline (flow → parallel Gemini → registry). Exposes tools for Gemini to query the registry. Handles natural language Q&A. |
| `visualizer.py` | Draws overlays on video frames | Crosshair markers on detections, 360° radar minimap with FOV cone, severity-colored dots, status bar |
| `osha_lookup.py` | Searches OSHA incident records | Keyword search over 18,694 construction incident narratives. Returns real stories + days-away-from-work data. |
| `spatial_map.py` | Generates interactive spatial maps | Accumulates camera motion into a world-space path, projects hazards using allocentric angles + distance, outputs interactive Plotly 3D and 2D bird's-eye HTML maps |
| `realtime_pipeline.py` | Live real-time processing loop | Main thread: optical flow + heading update + annotate at ~30fps. Background: ThreadPoolExecutor submits keyframes to Gemini. Lock-based thread safety around registry. |
| `audio_alerts.py` | Intelligent spoken warnings | Tracks per-hazard proximity history (ego angle + distance trend). Detects approach via rolling window analysis. Escalates through 5 alert levels with per-level cooldowns. pyttsx3 TTS on daemon thread. |

## OSHA Data

The `data/osha/` directory contains 13 OSHA ITA (Injury Tracking Application) CSV files spanning 2016-2025. The two Case Detail files contain **18,694 construction-specific incident records** with narratives describing what happened, what injuries occurred, and how many days workers were away.

This data is not included in the repo due to size (~2.3GB). Download from [OSHA ITA Data](https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data).

## Team

| Role | What They Build |
|------|----------------|
| Core Pipeline | VESTA agent, optical flow, registry, detection |
| OSHA Engine | SQLite indexer, RAG retriever, risk scorer, stats |
| Demo UI | Streamlit/Gradio app with radar map + chat |
| Voice I/O | Whisper STT + TTS for hands-free interaction |
| Spatial Map | Bird's-eye-view visualization of camera path + hazards |
