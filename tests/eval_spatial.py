#!/usr/bin/env python3
"""
VESTA Spatial Accuracy Evaluation

Runs a set of ground-truth queries against a saved scene graph and scores
the system's spatial/temporal accuracy.

Ground truth was manually labeled from test_2.mp4 (150 frames, 10s,
construction worker laying cinder blocks on an upper floor).

Usage:
    python tests/eval_spatial.py --graph results/test_2_graph.pkl
    python tests/eval_spatial.py --graph results/test_2_graph.pkl --verbose
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ── Ground Truth Questions ──────────────────────────────────────────────────
# Each entry: (question, expected_keywords, query_type)
# expected_keywords: list of strings that SHOULD appear in the answer
# At least one keyword from the list must appear for the answer to be "correct"

EVAL_SET = [
    # Spatial: direction queries
    {
        "question": "What's ahead of me?",
        "expected_any": ["worker", "wall", "cinder", "block", "rebar"],
        "expected_none": ["behind", "nothing"],
        "type": "spatial",
    },
    {
        "question": "What's to my front-left?",
        "expected_any": ["scaffold", "yellow", "rebar", "tower", "lattice"],
        "expected_none": [],
        "type": "spatial",
    },
    {
        "question": "What's to my front-right?",
        "expected_any": ["wall", "netting", "steel", "tree", "safety", "forest", "lattice"],
        "expected_none": [],
        "type": "spatial",
    },
    # Spatial: entity location
    {
        "question": "Where is the worker?",
        "expected_any": ["ahead", "front", "center", "0°", "straight"],
        "expected_none": ["behind", "left", "right"],
        "type": "spatial",
    },
    {
        "question": "Where is the scaffolding?",
        "expected_any": ["left", "front-left", "ahead"],
        "expected_none": ["right", "behind"],
        "type": "spatial",
    },
    # Spatial: relative position
    {
        "question": "Where is the scaffolding relative to the worker?",
        "expected_any": ["left", "front-left", "ahead"],
        "expected_none": ["right", "behind"],
        "type": "spatial_relation",
    },
    {
        "question": "Where is the cinder block wall relative to the worker?",
        "expected_any": ["right", "front-right", "ahead"],
        "expected_none": ["behind", "left"],
        "type": "spatial_relation",
    },
    # Entity queries
    {
        "question": "How many workers are on the site?",
        "expected_any": ["1", "one", "single"],
        "expected_none": ["two", "three", "2", "3"],
        "type": "entity",
    },
    {
        "question": "What is the worker doing?",
        "expected_any": ["laying", "block", "building", "wall", "construct", "working"],
        "expected_none": [],
        "type": "entity",
    },
    {
        "question": "Describe the worker",
        "expected_any": ["orange", "vest", "safety", "hardhat", "hat"],
        "expected_none": [],
        "type": "entity",
    },
    # Temporal queries
    {
        "question": "What was visible at 0 seconds?",
        "expected_any": ["worker", "scaffold", "wall", "rebar"],
        "expected_none": [],
        "type": "temporal",
    },
    {
        "question": "When was the worker first and last seen?",
        "expected_any": ["0", "first", "9", "last"],
        "expected_none": [],
        "type": "temporal",
    },
    # Category queries
    {
        "question": "What equipment is on the site?",
        "expected_any": ["scaffold", "tower", "support", "steel"],
        "expected_none": [],
        "type": "category",
    },
    {
        "question": "What structures are visible?",
        "expected_any": ["wall", "cinder", "block", "platform", "truss"],
        "expected_none": [],
        "type": "category",
    },
    # Change detection
    {
        "question": "What changed during the video?",
        "expected_any": ["wall", "worker", "block", "change", "state", "description", "observation"],
        "expected_none": [],
        "type": "change",
    },
]


def evaluate(graph_path: str, verbose: bool = False):
    """Run all eval questions and score results."""
    from vesta.agent.scene_agent import SceneAgent

    # Load saved graph
    with open(graph_path, "rb") as f:
        state = pickle.load(f)

    agent = SceneAgent(verbose=False)
    agent.graph = state["graph"]
    agent.frame_count = state["frame_count"]
    agent.fps = state["fps"]
    agent.model = state["model"]
    agent.processed = True

    print("=" * 60)
    print("  VESTA Spatial Accuracy Evaluation")
    print("=" * 60)

    summary = agent.graph.get_summary()
    print(f"  Graph: {summary['total_entities']} entities, "
          f"{summary['total_relationships']} relationships")
    print(f"  Video: {agent.frame_count} frames @ {agent.fps:.0f} FPS")
    print()

    results = []
    correct = 0
    total = len(EVAL_SET)

    for i, q in enumerate(EVAL_SET):
        question = q["question"]
        expected_any = q["expected_any"]
        expected_none = q["expected_none"]

        print(f"  [{i+1}/{total}] {question}")

        try:
            answer = agent.ask(question)
        except Exception as e:
            answer = f"[ERROR] {e}"

        answer_lower = answer.lower()

        # Check: at least one expected keyword present
        has_expected = any(kw.lower() in answer_lower for kw in expected_any)
        # Check: none of the forbidden keywords present
        has_forbidden = any(kw.lower() in answer_lower for kw in expected_none) if expected_none else False

        passed = has_expected and not has_forbidden

        if passed:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        results.append({
            "question": question,
            "type": q["type"],
            "answer": answer[:200],
            "passed": passed,
            "has_expected": has_expected,
            "has_forbidden": has_forbidden,
        })

        if verbose:
            print(f"         → {status}")
            print(f"         Answer: {answer[:150]}...")
            if not passed:
                if not has_expected:
                    print(f"         Missing: {expected_any}")
                if has_forbidden:
                    print(f"         Forbidden found: {[k for k in expected_none if k.lower() in answer_lower]}")
        else:
            print(f"         → {status}")

        print()

    # Summary
    print("=" * 60)
    print(f"  RESULTS: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print("=" * 60)

    by_type = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        if r["passed"]:
            by_type[t]["correct"] += 1

    print("  By query type:")
    for t, counts in by_type.items():
        pct = counts["correct"] / counts["total"] * 100
        print(f"    {t:20s}  {counts['correct']}/{counts['total']}  ({pct:.0f}%)")

    if any(not r["passed"] for r in results):
        print("\n  Failed questions:")
        for r in results:
            if not r["passed"]:
                print(f"    - {r['question']}")

    # Save results
    output_path = Path(graph_path).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": f"{correct}/{total}",
            "accuracy_pct": round(correct / total * 100, 1),
            "by_type": by_type,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VESTA Spatial Accuracy Evaluation")
    parser.add_argument("--graph", required=True, help="Path to saved graph (.pkl)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers")
    args = parser.parse_args()

    if not Path(args.graph).exists():
        print(f"Error: Graph file not found: {args.graph}")
        print("Run 'python scripts/run_pipeline.py --video <video> --max-frames 150' first.")
        sys.exit(1)

    evaluate(args.graph, verbose=args.verbose)
