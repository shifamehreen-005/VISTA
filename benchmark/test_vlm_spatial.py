#!/usr/bin/env python3
"""
VLM Spatial Reasoning Benchmark — Floor Opening Detection
Tests whether VLMs can distinguish floor openings/holes from visual lookalikes
(shadows, dark surfaces, tarps, puddles) in construction egocentric images.

Usage:
  1. Put test images in benchmark/images/positive/ and benchmark/images/negative/
  2. Set API keys as environment variables
  3. Run: python3 test_vlm_spatial.py

Supports: Claude (Anthropic), GPT-4o/GPT-5 (OpenAI), Gemini (Google)
"""

import os
import sys
import json
import base64
import time
import csv
from pathlib import Path
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

BENCHMARK_DIR = Path(__file__).parent
IMAGES_DIR = BENCHMARK_DIR / "images"
POSITIVE_DIR = IMAGES_DIR / "positive"   # actual floor openings/holes
NEGATIVE_DIR = IMAGES_DIR / "negative"   # shadows, dark surfaces, tarps (no hole)
RESULTS_DIR = BENCHMARK_DIR / "results"

# Create directories
for d in [POSITIVE_DIR, NEGATIVE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── The Prompts ────────────────────────────────────────────────────────────────
# We test multiple prompt strategies to show baseline vs. augmented

PROMPTS = {
    # Baseline: simple direct question
    "baseline_simple": """Look at this image taken from a worker's helmet-mounted camera on a construction site.
Is there an unguarded floor opening, uncovered hole, or missing decking in the worker's walking path?
Answer with: YES or NO, then explain your reasoning in 2-3 sentences.""",

    # Baseline: more detailed
    "baseline_detailed": """You are analyzing a first-person view from a construction worker's helmet camera.
Your task is to determine if there is a fall hazard in the worker's path.

Specifically, look for:
- Unguarded floor openings (holes in the floor/deck with no cover or guardrail)
- Uncovered holes (removed manhole covers, pipe penetrations, skylight openings)
- Missing decking or flooring sections (gaps in the walking surface)
- Unprotected edges leading to a lower level

Answer with: YES or NO
Confidence: HIGH, MEDIUM, or LOW
Reasoning: 2-3 sentences explaining what you see and why you made this judgment.""",

    # Spatial reasoning prompt (chain-of-thought)
    "spatial_cot": """You are analyzing a first-person view from a construction worker's helmet camera for fall hazards.

Before answering, reason through these spatial analysis steps:
1. SURFACE CONTINUITY: Trace the walking surface from the bottom of the image forward. Is the floor/deck continuous, or is there a break/gap?
2. DEPTH CUES: Look for depth discontinuities — edges where the surface drops away, visible lower levels, or areas where you can see through the floor.
3. SHADOW vs HOLE: If you see a dark area, determine: does it have visible depth (you can see walls/sides of an opening)? Or is it flat darkness (a shadow, stain, or dark material)?
4. CONTEXT: Are there guardrails, hole covers, warning tape, or barriers around any openings? If an opening exists but is guarded, it's not an immediate hazard.

Based on this analysis:
Answer: YES or NO (is there an unguarded floor opening/hole in the walking path?)
Confidence: HIGH, MEDIUM, or LOW
Step-by-step reasoning: Walk through steps 1-4 above.""",

    # Adversarial: designed to test if the model falls for visual tricks
    "adversarial": """This image is from a construction site helmet camera.
I need you to be extremely precise about spatial geometry here.

Tell me: is the dark area visible in this image (a) a shadow/stain on a solid surface, or (b) an actual opening/hole leading to a lower level or void?

How can you tell the difference from this single image? What visual evidence supports your conclusion?

Final answer: HOLE or NOT_A_HOLE""",
}


# ── Image encoding ─────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> tuple[str, str]:
    """Read and base64-encode an image. Returns (base64_data, media_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


# ── API Callers ────────────────────────────────────────────────────────────────

def call_claude(image_path: str, prompt: str, model: str = "claude-sonnet-4-5-20250929") -> dict:
    """Call Anthropic Claude API with an image."""
    try:
        import anthropic
    except ImportError:
        return {"error": "pip install anthropic"}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "Set ANTHROPIC_API_KEY environment variable"}

    client = anthropic.Anthropic(api_key=api_key)
    img_data, media_type = encode_image(image_path)

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    elapsed = time.time() - start
    text = response.content[0].text
    return {"response": text, "latency_s": round(elapsed, 2), "model": model}


def call_openai(image_path: str, prompt: str, model: str = "gpt-4o") -> dict:
    """Call OpenAI GPT-4o/GPT-5 API with an image."""
    try:
        import openai
    except ImportError:
        return {"error": "pip install openai"}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "Set OPENAI_API_KEY environment variable"}

    client = openai.OpenAI(api_key=api_key)
    img_data, _ = encode_image(image_path)

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content
    return {"response": text, "latency_s": round(elapsed, 2), "model": model}


def call_gemini(image_path: str, prompt: str, model: str = "gemini-2.5-pro-preview-05-06") -> dict:
    """Call Google Gemini API with an image."""
    try:
        import google.generativeai as genai
    except ImportError:
        return {"error": "pip install google-generativeai"}

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "Set GOOGLE_API_KEY environment variable"}

    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)

    import PIL.Image
    img = PIL.Image.open(image_path)

    start = time.time()
    response = model_obj.generate_content([prompt, img])
    elapsed = time.time() - start
    text = response.text
    return {"response": text, "latency_s": round(elapsed, 2), "model": model}


# ── Answer Parsing ─────────────────────────────────────────────────────────────

def parse_answer(response_text: str, prompt_key: str) -> str:
    """Extract YES/NO/HOLE/NOT_A_HOLE from model response."""
    text = response_text.upper()
    if prompt_key == "adversarial":
        if "NOT_A_HOLE" in text or "NOT A HOLE" in text:
            return "NO_HOLE"
        elif "HOLE" in text:
            return "HOLE"
        return "UNCLEAR"
    else:
        # Find the first YES or NO
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("YES") or line.startswith("ANSWER: YES") or line.startswith("**YES"):
                return "YES"
            if line.startswith("NO") or line.startswith("ANSWER: NO") or line.startswith("**NO"):
                return "NO"
        if "YES" in text[:100]:
            return "YES"
        if "NO" in text[:100]:
            return "NO"
        return "UNCLEAR"


def is_correct(parsed_answer: str, ground_truth: str, prompt_key: str) -> bool:
    """Check if the model's answer matches ground truth."""
    if prompt_key == "adversarial":
        if ground_truth == "positive":
            return parsed_answer == "HOLE"
        else:
            return parsed_answer == "NO_HOLE"
    else:
        if ground_truth == "positive":
            return parsed_answer == "YES"
        else:
            return parsed_answer == "NO"


# ── Main Benchmark Runner ─────────────────────────────────────────────────────

def collect_images():
    """Collect all test images with ground truth labels."""
    images = []
    for img_path in sorted(POSITIVE_DIR.glob("*")):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
            images.append({"path": str(img_path), "label": "positive", "name": img_path.name})
    for img_path in sorted(NEGATIVE_DIR.glob("*")):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
            images.append({"path": str(img_path), "label": "negative", "name": img_path.name})
    return images


def run_benchmark(models=None, prompts_to_test=None):
    """Run the full benchmark across models and prompts."""

    if models is None:
        models = []
        if os.environ.get("ANTHROPIC_API_KEY"):
            models.append(("claude", call_claude, "claude-sonnet-4-5-20250929"))
        if os.environ.get("OPENAI_API_KEY"):
            models.append(("openai", call_openai, "gpt-4o"))
        if os.environ.get("GOOGLE_API_KEY"):
            models.append(("gemini", call_gemini, "gemini-2.5-pro-preview-05-06"))

        if not models:
            print("ERROR: No API keys found. Set at least one of:")
            print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
            sys.exit(1)

    if prompts_to_test is None:
        prompts_to_test = list(PROMPTS.keys())

    images = collect_images()
    if not images:
        print(f"ERROR: No images found!")
        print(f"  Put floor opening images in:  {POSITIVE_DIR}/")
        print(f"  Put shadow/dark surface images in: {NEGATIVE_DIR}/")
        print()
        print("See README or download test images first.")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"VLM SPATIAL REASONING BENCHMARK — FLOOR OPENING DETECTION")
    print(f"{'='*70}")
    print(f"Images:  {len(images)} ({sum(1 for i in images if i['label']=='positive')} positive, "
          f"{sum(1 for i in images if i['label']=='negative')} negative)")
    print(f"Models:  {', '.join(m[0]+'/'+m[2] for m in models)}")
    print(f"Prompts: {', '.join(prompts_to_test)}")
    print(f"{'='*70}\n")

    all_results = []
    summary = {}  # (model, prompt) -> {correct, total, tp, fp, tn, fn}

    for model_name, model_fn, model_id in models:
        for prompt_key in prompts_to_test:
            prompt_text = PROMPTS[prompt_key]
            key = (model_name, model_id, prompt_key)
            stats = {"correct": 0, "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}

            print(f"\n── {model_name}/{model_id} × {prompt_key} ──")

            for img in images:
                print(f"  Testing: {img['name']} (GT: {img['label']})...", end=" ", flush=True)

                try:
                    result = model_fn(img["path"], prompt_text, model_id)
                except Exception as e:
                    result = {"error": str(e)}

                if "error" in result:
                    print(f"ERROR: {result['error']}")
                    parsed = "ERROR"
                    correct = False
                else:
                    parsed = parse_answer(result["response"], prompt_key)
                    correct = is_correct(parsed, img["label"], prompt_key)

                    # Confusion matrix
                    if img["label"] == "positive" and correct:
                        stats["tp"] += 1
                    elif img["label"] == "positive" and not correct:
                        stats["fn"] += 1
                    elif img["label"] == "negative" and correct:
                        stats["tn"] += 1
                    elif img["label"] == "negative" and not correct:
                        stats["fp"] += 1

                    stats["total"] += 1
                    if correct:
                        stats["correct"] += 1

                    status = "✓" if correct else "✗"
                    print(f"{status} (answered: {parsed}, {result.get('latency_s', '?')}s)")

                all_results.append({
                    "model": model_id,
                    "model_provider": model_name,
                    "prompt_strategy": prompt_key,
                    "image": img["name"],
                    "ground_truth": img["label"],
                    "parsed_answer": parsed,
                    "correct": correct,
                    "raw_response": result.get("response", result.get("error", "")),
                    "latency_s": result.get("latency_s", None),
                })

                time.sleep(0.5)  # rate limiting

            summary[key] = stats

    # ── Print Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Prompt':<22} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'F1':>6}")
    print(f"{'-'*30} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for (model_name, model_id, prompt_key), stats in sorted(summary.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{model_id:<30} {prompt_key:<22} {acc:>5.1f}% {precision:>5.1f}% {recall:>5.1f}% {f1:>5.1f}%")

    # ── Save detailed results ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in summary.items()},
            "results": all_results,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # ── Save CSV for easy plotting ─────────────────────────────────────────
    csv_file = RESULTS_DIR / f"benchmark_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "prompt_strategy", "image", "ground_truth",
            "parsed_answer", "correct", "latency_s",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"CSV results saved to: {csv_file}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM Spatial Reasoning Benchmark")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS.keys()),
                        default=list(PROMPTS.keys()), help="Which prompts to test")
    parser.add_argument("--models", nargs="+", help="Model IDs to test (e.g. gpt-4o claude-sonnet-4-5-20250929)")
    args = parser.parse_args()

    run_benchmark(prompts_to_test=args.prompts)
