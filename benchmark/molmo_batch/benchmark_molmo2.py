#!/usr/bin/env python3
"""
Batch benchmark runner for Molmo2 on video + prompt pairs.

Usage:
  python benchmark/molmo_batch/benchmark_molmo2.py \
    --config benchmark/molmo_batch/benchmark_config.json \
    --video-dir /path/to/videos
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(model_id: str, load_in_4bit: bool = False):
    print(f"Loading model: {model_id} (4bit={load_in_4bit})")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    print(
        f"Model loaded in {time.time() - t0:.1f}s | device={model.device} | dtype={model.dtype}"
    )
    return model, processor


def prepare_inputs_for_model(inputs, model):
    prepared = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            prepared[key] = value
            continue

        is_visual = any(k in key.lower() for k in ("pixel", "image", "video", "vision"))
        if is_visual and value.dtype == torch.uint8:
            # Fixes LayerNorm Byte-type failures in some processor paths.
            value = value.to(torch.float32) / 255.0

        if value.is_floating_point():
            prepared[key] = value.to(device=model.device, dtype=model.dtype)
        else:
            prepared[key] = value.to(model.device)

    return prepared


def run_inference(
    model,
    processor,
    video_path: str,
    prompt: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": video_path},
            ],
        }
    ]

    preprocess_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = prepare_inputs_for_model(inputs, model)
    preprocess_time = time.time() - preprocess_start

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample and temperature is not None:
        generate_kwargs["temperature"] = temperature
    if do_sample and top_p is not None:
        generate_kwargs["top_p"] = top_p

    gen_start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    gen_time = time.time() - gen_start

    new_tokens = output_ids[0, inputs["input_ids"].size(1) :]
    answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    token_count = len(new_tokens)
    return {
        "response": answer,
        "preprocess_time_s": round(preprocess_time, 3),
        "generate_time_s": round(gen_time, 3),
        "total_time_s": round(preprocess_time + gen_time, 3),
        "output_tokens": token_count,
        "tokens_per_second": round(token_count / gen_time, 2) if gen_time > 0 else 0,
    }


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Batch benchmark Molmo2 on videos")
    parser.add_argument(
        "--config",
        default="benchmark/molmo_batch/benchmark_config.json",
        help="Path to benchmark JSON config",
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--video-dir", default="./data/test_videos", help="Directory with videos")
    parser.add_argument("--model", default="allenai/Molmo2-8B", help="HF model id")
    parser.add_argument("--max-tokens", type=int, default=512, help="Default max new tokens")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top_p")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization (for smaller GPUs)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    video_dir = Path(args.video_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("results") / f"molmo2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    model, processor = load_model(args.model, load_in_4bit=args.load_in_4bit)

    tests = config.get("test_cases", [])
    print(f"\nRunning {len(tests)} test cases...\n")

    all_results = []
    for i, test in enumerate(tests, 1):
        test_id = test.get("id", f"test_{i}")
        prompt = test["prompt"]
        video_file = test["video"]
        max_tokens = int(test.get("max_new_tokens", args.max_tokens))

        video_path = str(video_dir / video_file)
        print(f"[{i}/{len(tests)}] {test_id}")
        print(f"  Video: {video_file}")
        print(f"  Max new tokens: {max_tokens}")

        if not os.path.exists(video_path):
            print(f"  SKIP - video not found: {video_path}\n")
            all_results.append(
                {
                    "id": test_id,
                    "video": video_file,
                    "prompt": prompt,
                    "error": f"Video not found: {video_path}",
                }
            )
            continue

        try:
            result = run_inference(
                model,
                processor,
                video_path=video_path,
                prompt=prompt,
                max_new_tokens=max_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            result.update({"id": test_id, "video": video_file, "prompt": prompt})
            all_results.append(result)
            print(f"  OK  ({result['total_time_s']}s)\n")
        except Exception as exc:
            all_results.append(
                {"id": test_id, "video": video_file, "prompt": prompt, "error": str(exc)}
            )
            print(f"  ERROR: {exc}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": str(model.device),
        "torch_dtype": str(model.dtype),
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens_default": args.max_tokens,
        "test_results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in all_results if "error" not in r)
    print(f"Results saved to: {output_path}")
    print(f"Summary: {success} succeeded, {len(all_results) - success} failed")


if __name__ == "__main__":
    main()
