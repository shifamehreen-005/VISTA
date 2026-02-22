#!/usr/bin/env python3
"""
Molmo 2 Baseline — Ask Molmo2-8B questions about a video with NO scene graph.

Requires: GPU server (H100/A100 on Vast.ai)

Setup on Vast.ai:
    conda create --name molmo python=3.11 -y
    conda activate molmo
    pip install transformers==4.57.1 torch pillow einops torchvision accelerate decord2 molmo_utils

Usage:
    # Single question
    python benchmark/ask_molmo.py --video data/test_videos/test_2.mp4 \
        -q "What's behind me?"

    # Interactive mode
    python benchmark/ask_molmo.py --video data/test_videos/test_2.mp4

    # Run the eval set
    python benchmark/ask_molmo.py --video data/test_videos/test_2.mp4 --eval
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_id: str = "allenai/Molmo2-8B"):
    """Load Molmo2 model and processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[MOLMO] Loading {model_id}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", device_map="auto"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", device_map="auto"
    )

    print(f"[MOLMO] Model loaded in {time.time() - t0:.1f}s")
    return model, processor


def ask_molmo(model, processor, video_path: str, question: str) -> str:
    """Send video + question to Molmo2. Returns text answer."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {
                    "type": "text",
                    "text": (
                        f"This is a first-person (egocentric) video from a "
                        f"construction site camera.\n\n"
                        f"Question: {question}\n\n"
                        f"Answer precisely and concisely."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    new_tokens = output_ids[0, inputs["input_ids"].size(1):]
    answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.strip()


def run_eval(model, processor, video_path: str):
    """Run the same eval set used for VESTA, score Molmo2."""
    from tests.eval_spatial import EVAL_SET

    print("=" * 60)
    print("  MOLMO 2 (8B) Baseline Evaluation")
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
            answer = ask_molmo(model, processor, video_path, question)
        except Exception as e:
            answer = f"[ERROR] {e}"

        answer_lower = answer.lower()
        has_expected = any(kw.lower() in answer_lower for kw in expected_any)
        has_forbidden = any(kw.lower() in answer_lower for kw in expected_none) if expected_none else False
        passed = has_expected and not has_forbidden

        if passed:
            correct += 1

        print(f"         -> {'PASS' if passed else 'FAIL'}")
        print(f"         Molmo: {answer[:150]}...")
        if not passed:
            print(f"         Why: {q.get('why_vlm_fails', '')}")
        print()

    print("=" * 60)
    print(f"  MOLMO RESULTS: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print("=" * 60)

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Molmo2 Baseline — no scene graph")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--question", "-q", action="append", help="Question(s) to ask")
    parser.add_argument("--model", default="allenai/Molmo2-8B", help="Model ID")
    parser.add_argument("--eval", action="store_true", help="Run the evaluation set")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    model, processor = load_model(args.model)

    if args.eval:
        run_eval(model, processor, args.video)
        return

    if args.question:
        for q in args.question:
            print(f"\nQ: {q}")
            print("-" * 50)
            answer = ask_molmo(model, processor, args.video, q)
            print(answer)
        return

    # Interactive mode
    print("\nType questions. 'quit' to exit.\n")
    while True:
        try:
            question = input("Ask Molmo: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        print()
        answer = ask_molmo(model, processor, args.video, question)
        print(answer)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
