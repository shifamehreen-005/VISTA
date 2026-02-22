#!/usr/bin/env python3
"""
Direct VLM Baseline — Ask Gemini questions about a video with NO scene graph.

Sends the video directly to Gemini (inline for <20MB, File API for larger).
This is the baseline to compare against VESTA's scene graph approach.

Usage:
    # Single question
    python benchmark/ask_vlm_direct.py --video data/test_videos/test_2.mp4 \
        -q "What's behind me?"

    # Interactive mode
    python benchmark/ask_vlm_direct.py --video data/test_videos/test_2.mp4

    # Run the eval set for comparison
    python benchmark/ask_vlm_direct.py --video data/test_videos/test_2.mp4 --eval
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import google.genai as genai
from google.genai import types


MAX_INLINE_SIZE = 20 * 1024 * 1024  # 20MB


def load_video(video_path: str) -> types.Part:
    """Load video as a Gemini Part. Inline for <20MB, File API for larger."""
    file_size = os.path.getsize(video_path)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    if file_size <= MAX_INLINE_SIZE:
        print(f"[BASELINE] Loading video inline ({file_size / 1024 / 1024:.1f}MB)")
        video_bytes = open(video_path, "rb").read()
        return types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
        )
    else:
        print(f"[BASELINE] Uploading video via File API ({file_size / 1024 / 1024:.1f}MB)...")
        uploaded = client.files.upload(
            file=video_path,
            config=types.UploadFileConfig(mime_type="video/mp4"),
        )
        # Wait for processing
        while uploaded.state == "PROCESSING":
            print(f"[BASELINE] Processing... ({uploaded.name})")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state == "FAILED":
            raise RuntimeError(f"File upload failed: {uploaded.name}")
        print(f"[BASELINE] Upload complete: {uploaded.name}")
        return types.Part.from_uri(file_uri=uploaded.uri, mime_type="video/mp4")


def ask_direct(video_part: types.Part, question: str, model: str = "gemini-2.5-flash") -> str:
    """Send video + question directly to Gemini. No scene graph, no tools."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    contents = types.Content(
        parts=[
            video_part,
            types.Part(text=
                f"This is a first-person (egocentric) video from a construction site camera.\n\n"
                f"Question: {question}\n\n"
                f"Answer precisely and concisely."
            ),
        ]
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.3),
    )

    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text
    return "[No response]"


def run_eval(video_part: types.Part, model: str):
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
        print(f"         Expected: {q.get('short_answer', '?')}")

        try:
            answer = ask_direct(video_part, question, model)
        except Exception as e:
            answer = f"[ERROR] {e}"

        answer_lower = answer.lower()
        has_expected = any(kw.lower() in answer_lower for kw in expected_any)
        has_forbidden = any(kw.lower() in answer_lower for kw in expected_none) if expected_none else False
        passed = has_expected and not has_forbidden

        if passed:
            correct += 1

        print(f"         → {'PASS' if passed else 'FAIL'}")
        print(f"         VLM: {answer[:120]}...")
        if not passed:
            print(f"         Why: {q.get('why_vlm_fails', '')}")
        print()

    print("=" * 60)
    print(f"  BASELINE RESULTS: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print("=" * 60)

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Direct VLM Baseline — no scene graph")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--question", "-q", action="append", help="Question(s) to ask")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model ID")
    parser.add_argument("--eval", action="store_true", help="Run the evaluation set")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    video_part = load_video(args.video)

    if args.eval:
        run_eval(video_part, args.model)
        return

    if args.question:
        for q in args.question:
            print(f"\nQ: {q}")
            print("-" * 50)
            answer = ask_direct(video_part, q, args.model)
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
        answer = ask_direct(video_part, question, args.model)
        print(answer)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
