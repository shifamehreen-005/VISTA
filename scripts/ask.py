#!/usr/bin/env python3
"""
Ask VESTA — Query a previously processed video without re-processing.

This loads the saved scene graph from a prior run of run_pipeline.py and lets you
ask questions immediately (no video processing, no waiting).

Usage:
    # Interactive mode (loads saved graph)
    python scripts/ask.py --graph results/test_graph.pkl

    # Single question (no interactive prompt)
    python scripts/ask.py --graph results/test_graph.pkl --question "What's behind me?"

    # Multiple questions from command line
    python scripts/ask.py --graph results/test_graph.pkl \
        --question "Where is the crane relative to the worker?" \
        --question "What was visible at 10 seconds?"
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from vesta.agent.scene_agent import SceneAgent


def main():
    parser = argparse.ArgumentParser(
        description="Ask VESTA questions about a previously processed video"
    )
    parser.add_argument("--graph", required=True,
                        help="Path to saved graph (.pkl from run_pipeline.py)")
    parser.add_argument("--question", "-q", action="append",
                        help="Question to ask (can specify multiple). If omitted, enters interactive mode.")
    parser.add_argument("--model", default=None,
                        help="Override Gemini model (default: uses model from processing)")
    args = parser.parse_args()

    # Load saved state
    pkl_path = Path(args.graph)
    if not pkl_path.exists():
        print(f"Error: Graph file not found: {pkl_path}")
        print(f"Run 'python scripts/run_pipeline.py --video <your_video>' first.")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    # Reconstruct the agent with the saved graph
    agent = SceneAgent(verbose=False)
    agent.graph = state["graph"]
    agent.frame_count = state["frame_count"]
    agent.fps = state["fps"]
    agent.model = args.model or state["model"]
    agent.processed = True

    summary = agent.graph.get_summary()
    print(f"[VESTA] Loaded scene graph: {summary['total_entities']} entities, "
          f"{summary['total_relationships']} relationships, "
          f"heading {summary['current_heading']}°, "
          f"{agent.frame_count} frames @ {agent.fps:.0f} FPS")

    # Single question mode
    if args.question:
        for q in args.question:
            print(f"\nQ: {q}")
            print("-" * 50)
            response = agent.ask(q)
            print(response)
        return

    # Interactive mode
    print("\nCommands: 'map' (all entities), 'quit' (exit)")
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
