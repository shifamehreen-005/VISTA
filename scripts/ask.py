#!/usr/bin/env python3
"""
Ask VESTA — Query a previously processed video without re-processing.

This loads the saved registry from a prior run of run_pipeline.py and lets you
ask questions immediately (no video processing, no waiting).

Usage:
    # Interactive mode (loads saved registry)
    python scripts/ask.py --registry results/test_registry.pkl

    # Single question (no interactive prompt)
    python scripts/ask.py --registry results/test_registry.pkl --question "What's behind me?"

    # Multiple questions from command line
    python scripts/ask.py --registry results/test_registry.pkl \
        --question "What's the most dangerous hazard?" \
        --question "Any fall risks to my left?"
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from vesta.agent.vesta_agent import VestaAgent


def main():
    parser = argparse.ArgumentParser(
        description="Ask VESTA questions about a previously processed video"
    )
    parser.add_argument("--registry", required=True,
                        help="Path to saved registry (.pkl from run_pipeline.py)")
    parser.add_argument("--question", "-q", action="append",
                        help="Question to ask (can specify multiple). If omitted, enters interactive mode.")
    parser.add_argument("--model", default=None,
                        help="Override Gemini model (default: uses model from processing)")
    args = parser.parse_args()

    # Load saved state
    pkl_path = Path(args.registry)
    if not pkl_path.exists():
        print(f"Error: Registry file not found: {pkl_path}")
        print(f"Run 'python scripts/run_pipeline.py --video <your_video>' first.")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    # Reconstruct the agent with the saved registry
    agent = VestaAgent(verbose=False)
    agent.registry = state["registry"]
    agent.frame_count = state["frame_count"]
    agent.fps = state["fps"]
    agent.model = args.model or state["model"]
    agent.processed = True

    summary = agent.registry.get_summary()
    print(f"[VESTA] Loaded registry: {summary['total_hazards']} hazards, "
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
    print("\nCommands: 'map' (full registry), 'quit' (exit)")
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
