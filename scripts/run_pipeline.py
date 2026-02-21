#!/usr/bin/env python3
"""
VESTA Pipeline Runner

Step 1: Process a video → detect hazards → build registry → save results.
Step 2: Ask questions interactively OR use scripts/ask.py for single questions.

Usage:
    # Process a video (generates annotated video + JSON results)
    python scripts/run_pipeline.py --video data/test_videos/test.mp4

    # Process only first 150 frames (quick test)
    python scripts/run_pipeline.py --video data/test_videos/test.mp4 --max-frames 150

    # Process without interactive mode (just save outputs)
    python scripts/run_pipeline.py --video data/test_videos/test.mp4 --no-interactive

Output files (saved to results/):
    results/<video>_annotated.mp4   ← Video with hazard overlays + radar minimap
    results/<video>_results.json    ← Full hazard registry as JSON
    results/<video>_registry.pkl    ← Saved agent state (for scripts/ask.py)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from vesta.agent.vesta_agent import VestaAgent


def main():
    parser = argparse.ArgumentParser(
        description="VESTA — Process a construction site video and detect hazards"
    )
    parser.add_argument("--video", required=True, help="Path to video file (.mp4)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames to process (default: all)")
    parser.add_argument("--keyframe-interval", type=int, default=30,
                        help="Frames between Gemini API calls (default: 30)")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model ID (default: gemini-2.5-flash)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive query mode after processing")
    parser.add_argument("--output-video", type=str, default=None,
                        help="Custom path for annotated video output")
    args = parser.parse_args()

    # Set up output paths
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    video_name = Path(args.video).stem

    output_video = args.output_video or str(output_dir / f"{video_name}_annotated.mp4")
    results_json = output_dir / f"{video_name}_results.json"
    registry_pkl = output_dir / f"{video_name}_registry.pkl"

    # ── STEP 1: Process the video ───────────────────────────────────────
    agent = VestaAgent(
        video_path=args.video,
        keyframe_interval=args.keyframe_interval,
        model=args.model,
    )

    print("=" * 60)
    print("  VESTA — Vision-Enhanced Spatial Tracking Agent")
    print("=" * 60)

    summary = agent.process(max_frames=args.max_frames, output_video=output_video)

    # Print summary
    print("\n" + "=" * 60)
    print("  HAZARD REGISTRY SUMMARY")
    print("=" * 60)
    print(f"  Frames processed:    {summary['frames_processed']}")
    print(f"  Keyframes analyzed:  {summary['keyframes_analyzed']}")
    print(f"  Total detections:    {summary['total_detections']}")
    print(f"  Unique hazards:      {summary['total_hazards']}")
    print(f"  Current heading:     {summary['current_heading']}°")
    print(f"  By severity:")
    for sev, count in summary['by_severity'].items():
        if count > 0:
            print(f"    {sev}: {count}")

    if summary['hazards']:
        print(f"\n  Hazards:")
        for h in summary['hazards']:
            print(f"    [{h['severity'].upper():8s}] {h['label']} @ {h['angle']}° (conf: {h['confidence']})")

    # ── Save outputs ────────────────────────────────────────────────────
    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(registry_pkl, "wb") as f:
        pickle.dump({
            "registry": agent.registry,
            "frame_count": agent.frame_count,
            "fps": agent.fps,
            "model": agent.model,
            "motions": agent.motions,
        }, f)

    # ── Generate spatial maps ─────────────────────────────────────────
    map_3d_path = str(output_dir / f"{video_name}_map_3d.html")
    map_2d_path = str(output_dir / f"{video_name}_map_2d.html")

    try:
        from vesta.utils.spatial_map import (
            compute_camera_path,
            project_hazards_to_world,
            build_3d_map,
            build_2d_radar_map,
        )

        print("\n[VESTA] Generating spatial maps...")
        camera_path = compute_camera_path(agent.motions, fps=agent.fps)
        hazard_positions = project_hazards_to_world(agent.registry, camera_path)

        build_3d_map(camera_path, hazard_positions, output_path=map_3d_path)
        build_2d_radar_map(camera_path, hazard_positions, output_path=map_2d_path)
    except Exception as e:
        print(f"[VESTA] Warning: Could not generate spatial maps: {e}")
        map_3d_path = "(skipped)"
        map_2d_path = "(skipped)"

    print(f"\n  Output files:")
    print(f"    Video:    {output_video}")
    print(f"    JSON:     {results_json}")
    print(f"    Registry: {registry_pkl}  (use with scripts/ask.py)")
    print(f"    3D Map:   {map_3d_path}")
    print(f"    2D Map:   {map_2d_path}")

    # ── STEP 2: Interactive mode (optional) ─────────────────────────────
    if args.no_interactive:
        return

    _interactive_loop(agent)


def _interactive_loop(agent: VestaAgent):
    """Run the interactive question-answer loop."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE — Ask VESTA about hazards")
    print("  Commands: 'map' (full registry), 'quit' (exit)")
    print("=" * 60)

    while True:
        try:
            question = input("\nAsk VESTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "map":
            print()
            for h in agent.registry.get_all():
                print(f"  {agent.registry.describe_relative_to_camera(h)}")
            continue

        print()
        response = agent.ask(question)
        print(response)

    print("\n[VESTA] Session ended.")


if __name__ == "__main__":
    main()
