#!/usr/bin/env python3
"""
VESTA Pipeline Runner — Scene Graph Edition

Step 1: Process a video → extract scene graph → save results.
Step 2: Ask questions interactively OR use scripts/ask.py for single questions.

Usage:
    # Process a video (generates annotated video + JSON results)
    python scripts/run_pipeline.py --video data/test_videos/test.mp4

    # Process only first 150 frames (quick test)
    python scripts/run_pipeline.py --video data/test_videos/test.mp4 --max-frames 150

    # Process without interactive mode (just save outputs)
    python scripts/run_pipeline.py --video data/test_videos/test.mp4 --no-interactive

Output files (saved to results/):
    results/<video>_annotated.mp4   ← Video with entity overlays + radar minimap
    results/<video>_results.json    ← Full scene graph as JSON
    results/<video>_graph.pkl       ← Saved agent state (for scripts/ask.py)
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

from vesta.agent.scene_agent import SceneAgent


def main():
    parser = argparse.ArgumentParser(
        description="VESTA — Process video and build spatio-temporal scene graph"
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
    graph_pkl = output_dir / f"{video_name}_graph.pkl"

    # ── STEP 1: Process the video ───────────────────────────────────────
    agent = SceneAgent(
        video_path=args.video,
        keyframe_interval=args.keyframe_interval,
        model=args.model,
    )

    print("=" * 60)
    print("  VESTA — Spatio-Temporal Scene Graph")
    print("=" * 60)

    summary = agent.process(max_frames=args.max_frames, output_video=output_video)

    # Print summary
    print("\n" + "=" * 60)
    print("  SCENE GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Frames processed:    {summary['frames_processed']}")
    print(f"  Keyframes analyzed:  {summary['keyframes_analyzed']}")
    print(f"  Total entities:      {summary['total_entities']}")
    print(f"  Total relationships: {summary['total_relationships']}")
    print(f"  Current heading:     {summary['current_heading']}°")
    print(f"  By category:")
    for cat, count in summary.get('by_category', {}).items():
        if count > 0:
            print(f"    {cat}: {count}")

    if summary.get('entities'):
        print(f"\n  Entities:")
        for e in summary['entities']:
            print(f"    [{e['category']:10s}] {e['label']} @ {e['angle']}° (conf: {e['confidence']})")

    # ── Save outputs ────────────────────────────────────────────────────
    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(graph_pkl, "wb") as f:
        pickle.dump({
            "graph": agent.graph,
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
            project_entities_to_world,
            build_3d_map,
            build_2d_radar_map,
        )

        print("\n[VESTA] Generating spatial maps...")
        camera_path = compute_camera_path(agent.motions, fps=agent.fps)
        entity_positions = project_entities_to_world(agent.graph, camera_path)

        build_3d_map(camera_path, entity_positions, output_path=map_3d_path)
        build_2d_radar_map(camera_path, entity_positions, output_path=map_2d_path)
    except Exception as e:
        print(f"[VESTA] Warning: Could not generate spatial maps: {e}")
        map_3d_path = "(skipped)"
        map_2d_path = "(skipped)"

    print(f"\n  Output files:")
    print(f"    Video:    {output_video}")
    print(f"    JSON:     {results_json}")
    print(f"    Graph:    {graph_pkl}  (use with scripts/ask.py)")
    print(f"    3D Map:   {map_3d_path}")
    print(f"    2D Map:   {map_2d_path}")

    # ── STEP 2: Interactive mode (optional) ─────────────────────────────
    if args.no_interactive:
        return

    _interactive_loop(agent)


def _interactive_loop(agent: SceneAgent):
    """Run the interactive question-answer loop."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE — Ask VESTA about the scene")
    print("  Commands: 'map' (all entities), 'quit' (exit)")
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
            for e in agent.graph.get_all():
                print(f"  {agent.graph.describe_relative_to_camera(e)}")
            continue

        print()
        response = agent.ask(question)
        print(response)

    print("\n[VESTA] Session ended.")


if __name__ == "__main__":
    main()
