#!/usr/bin/env python3
"""
VESTA Real-Time Pipeline — Entry Point

Usage:
    python realtime/run.py                          # webcam (device 0)
    python realtime/run.py --video path/to/file.mp4 # video file
    python realtime/run.py --video file.mp4 --ask   # video + interactive Q&A after
    python realtime/run.py --webcam 1               # specific webcam index
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so vesta imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realtime.realtime_pipeline import RealtimeVesta


def main():
    parser = argparse.ArgumentParser(description="VESTA Real-Time Hazard Detection")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (default: webcam)")
    parser.add_argument("--webcam", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--ask", action="store_true",
                        help="Drop into interactive Q&A mode after video ends")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model ID")
    parser.add_argument("--keyframe-interval", type=int, default=60,
                        help="Frames between keyframe samples (default: 60)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Max background Gemini workers (default: 2)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable spoken audio alerts")
    parser.add_argument("--viz", action="store_true",
                        help="Launch live 3D web visualization (localhost:8080)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    source = args.video if args.video else args.webcam

    # Validate video file exists
    if args.video and not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    pipeline = RealtimeVesta(
        model=args.model,
        keyframe_interval=args.keyframe_interval,
        max_workers=args.workers,
        audio_alerts=not args.no_audio,
        enable_viz=args.viz,
        verbose=not args.quiet,
    )

    print("[VESTA RT] Starting real-time pipeline...")
    print("[VESTA RT] Controls: Q=quit, SPACE=pause, S=screenshot")
    if args.viz:
        print("[VESTA RT] 3D visualization: http://localhost:8080")
    print()

    pipeline.run(source=source)

    # Interactive Q&A mode
    if args.ask:
        _interactive_qa(pipeline)


def _interactive_qa(pipeline: RealtimeVesta):
    """Post-session interactive Q&A using VestaAgent.ask()."""
    from vesta.agent.vesta_agent import VestaAgent

    registry = pipeline.get_registry()
    summary = registry.get_summary()

    if summary["total_hazards"] == 0:
        print("\n[VESTA] No hazards were detected during the session.")
        return

    # Create an agent and inject our registry into it
    agent = VestaAgent(model=pipeline.model, verbose=False)
    agent.registry = registry
    agent.frame_count = pipeline.frame_idx
    agent.fps = pipeline.fps
    agent.processed = True

    print(f"\n[VESTA Q&A] Session complete. {summary['total_hazards']} hazards detected.")
    print("[VESTA Q&A] Ask questions about what was observed. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        try:
            answer = agent.ask(question)
            print(f"\nVESTA: {answer}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")

    print("[VESTA Q&A] Goodbye.")


if __name__ == "__main__":
    main()
