#!/usr/bin/env python3
"""
Direct VLM Baseline — Ask Gemini questions about a video with NO scene graph.

Sends sampled frames directly to Gemini and asks questions.
This is the baseline to compare against VESTA's scene graph approach.

Usage:
    # Single question
    python benchmark/ask_vlm_direct.py --video data/test_videos/test_2.mp4 \
        -q "What's behind me?"

    # Interactive mode
    python benchmark/ask_vlm_direct.py --video data/test_videos/test_2.mp4

    # Custom number of frames to sample
    python benchmark/ask_vlm_direct.py --video data/test_videos/test_2.mp4 \
        --frames 10 -q "Where is the scaffolding relative to the worker?"

    # Run the eval set for comparison
    python benchmark/ask_vlm_direct.py --video data/test_videos/test_2.mp4 --eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import cv2
import google.genai as genai
from google.genai import types


def sample_frames(video_path: str, num_frames: int = 8) -> list[bytes]:
    """Sample evenly-spaced frames from a video, return as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Downscale to 720p max
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frames.append(buf.tobytes())

    cap.release()
    print(f"[BASELINE] Sampled {len(frames)} frames from {total} total ({total/fps:.1f}s)")
    return frames


def ask_direct(frames: list[bytes], question: str, model: str = "gemini-2.5-flash") -> str:
    """Send frames + question directly to Gemini. No scene graph, no tools."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    parts = []
    for i, jpg in enumerate(frames):
        parts.append(types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))

    parts.append(
        f"These are {len(frames)} evenly-sampled frames from a first-person (egocentric) "
        f"video. Frame 1 is the start, frame {len(frames)} is the end.\n\n"
        f"Question: {question}\n\n"
        f"Answer precisely and concisely."
    )

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(temperature=0.3),
    )

    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text
    return "[No response]"


def run_eval(frames: list[bytes], model: str):
    """Run the same eval set used for VESTA, score the baseline."""
    from tests.eval_spatial import EVAL_SET

    print("=" * 60)
    print("  BASELINE (Direct VLM) Evaluation")
    print("=" * 60)
    print()

    correct = 0
    total = len(EVAL_SET)

    for i, q in enumerate(EVAL_SET):
        question = q["question"]
        expected_any = q["expected_any"]
        expected_none = q["expected_none"]

        print(f"  [{i+1}/{total}] {question}")

        try:
            answer = ask_direct(frames, question, model)
        except Exception as e:
            answer = f"[ERROR] {e}"

        answer_lower = answer.lower()
        has_expected = any(kw.lower() in answer_lower for kw in expected_any)
        has_forbidden = any(kw.lower() in answer_lower for kw in expected_none) if expected_none else False
        passed = has_expected and not has_forbidden

        if passed:
            correct += 1

        print(f"         → {'PASS' if passed else 'FAIL'}")
        print(f"         {answer[:120]}...")
        print()

    print("=" * 60)
    print(f"  BASELINE RESULTS: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print("=" * 60)

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Direct VLM Baseline — no scene graph")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--question", "-q", action="append", help="Question(s) to ask")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to sample (default: 8)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model ID")
    parser.add_argument("--eval", action="store_true", help="Run the evaluation set")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    frames = sample_frames(args.video, args.frames)

    if args.eval:
        run_eval(frames, args.model)
        return

    if args.question:
        for q in args.question:
            print(f"\nQ: {q}")
            print("-" * 50)
            answer = ask_direct(frames, q, args.model)
            print(answer)
        return

    # Interactive mode
    print("\nType questions. 'quit' to exit.\n")
    while True:
        try:
            question = input("Ask Gemini: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        print()
        answer = ask_direct(frames, question, args.model)
        print(answer)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
