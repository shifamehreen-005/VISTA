#!/usr/bin/env python3
"""
VESTA vs VLM Comparison Evaluation

12 questions across multiple videos where VESTA clearly beats frontier VLMs.
Each question targets a specific video and expects a SHORT answer (1-5 words)
so the difference is immediately visible in a side-by-side benchmark.

Usage:
    # Step 1: Process videos and save PKLs
    python scripts/run_pipeline.py --video data/test_videos/test_7.mp4 --save results/test_7_graph.pkl
    python scripts/run_pipeline.py --video data/test_videos/test_2.mp4 --save results/test_2_graph.pkl
    # ... repeat for each video

    # Step 2: Run VESTA eval (loads each PKL per question)
    python tests/eval_spatial.py --results-dir results/ --verbose

    # Step 3: Run VLM baseline comparison
    python benchmark/ask_vlm_direct.py --eval-multi
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ── Evaluation Questions ─────────────────────────────────────────────────────
#
# Each question specifies its video. Only categories where VESTA wins.
#
# FORMAT FOR TEAM — to add more questions, copy this template:
#
#   {
#       "video": "test_7",                            # video name (no extension)
#       "question": "Your question here?",
#       "expected_any": ["keyword1", "keyword2"],      # at least one must appear
#       "expected_none": ["bad1", "bad2"],              # none should appear
#       "type": "direction|spatial_relation|temporal|permanence|change",
#       "short_answer": "1-5 word ideal answer",
#       "why_vlm_fails": "One sentence explaining the VLM gap",
#   },
#
# CATEGORIES:
#   direction          — "What direction is X?" Requires heading + allocentric coords
#   spatial_relation   — "What's happening relative to Y?" Requires world-fixed coords
#   temporal           — "When does X happen?" Requires timestamped observation chain
#   permanence         — "Where is X now (off-screen)?" Requires persistent spatial memory
#   change             — "How did X change?" Requires diffing entity state over time
#

# ── DEMO QUESTIONS ────────────────────────────────────────────────────────
#
# Scenario: A construction MANAGER is reviewing hardhat cam footage
# from a LABORER (the one wearing the camera). The manager wants to
# monitor productivity, track equipment, review safety, and understand
# what happened on site — all by querying the laborer's POV video.
#
# "the worker" = the laborer wearing the camera (camera_self / worker_1)
# The manager is the one asking questions, NOT the one in the video.
#
# For the demo: after VESTA gives a temporal answer (e.g. "at 12 seconds"),
# scrub the video to that timestamp to verify live.
#

EVAL_SET = [
    # ── PRODUCTIVITY MONITORING ────────────────────────────────────────────
    # "Was this worker actually working or standing around?"
    {
        "video": "test_3",
        "question": "How long was the worker idle in this video?",
        "expected_any": [],
        "expected_none": [],
        "type": "productivity",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM can't track worker state over time or compute idle duration",
        "demo_note": "Manager checks if the laborer was productive — VESTA computes idle time from state changes",
    },
    {
        "video": "test_3",
        "question": "What was the worker doing at the start vs the end of the video?",
        "expected_any": [],
        "expected_none": [],
        "type": "change",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM needs to compare first and last frame, no entity state tracking",
        "demo_note": "Manager reviews worker activity progression across the shift",
    },

    # ── EQUIPMENT & MATERIAL TRACKING ──────────────────────────────────────
    # "Where did the worker leave the tools/materials?"
    {
        "video": "test_10",
        "question": "Where is the ladder relative to the worker right now?",
        "expected_any": [],
        "expected_none": [],
        "type": "permanence",
        "short_answer": "TBD",
        "why_vlm_fails": "If ladder is off-screen, VLM says 'I don't see a ladder'",
        "demo_note": "Manager locating equipment that the worker walked away from",
    },
    {
        "video": "test_11",
        "question": "Where did the bucket move during this video?",
        "expected_any": [],
        "expected_none": [],
        "type": "change",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM can't compare entity positions at different timestamps",
        "demo_note": "Manager tracking if materials were moved to the right location",
    },
    {
        "video": "test_7",
        "question": "Where are the bricks relative to the worker?",
        "expected_any": [],
        "expected_none": [],
        "type": "spatial",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM doesn't know brick's position relative to camera heading",
        "demo_note": "Manager checking if materials are within reach of the worker",
    },

    # ── SITE AWARENESS / SURROUNDINGS ──────────────────────────────────────
    # "What's around the worker that I should know about?"
    {
        "video": "test_2",
        "question": "What's behind the worker right now?",
        "expected_any": [],
        "expected_none": [],
        "type": "spatial",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM can only see what's in frame, not behind the camera",
        "demo_note": "Manager checking surroundings for safety — VESTA sees full 360°",
    },
    {
        "video": "test_6",
        "question": "Where is the staircase relative to the worker at the end of the video?",
        "expected_any": [],
        "expected_none": [],
        "type": "permanence",
        "short_answer": "TBD",
        "why_vlm_fails": "Staircase may be off-screen — VLM has no memory of past positions",
        "demo_note": "Manager checking egress routes — VESTA remembers even off-camera",
    },

    # ── TEMPORAL / FINDING MOMENTS ─────────────────────────────────────────
    # "When exactly did the worker do X? Let me jump to that moment."
    {
        "video": "test_5",
        "question": "When does the worker first pick up the diagonal cutters?",
        "expected_any": [],
        "expected_none": [],
        "type": "temporal",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM has no temporal index — can't pinpoint when an action started",
        "demo_note": "Manager searching footage for a specific action → VESTA gives timestamp → scrub to verify",
    },
    {
        "video": "test_9",
        "question": "When does the bathtub first appear in the worker's view?",
        "expected_any": [],
        "expected_none": [],
        "type": "temporal",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM processes frames independently, no first_seen tracking",
        "demo_note": "Manager reviewing when the worker reached a specific area",
    },

    # ── PROGRESS TRACKING ──────────────────────────────────────────────────
    # "Is the work actually getting done?"
    {
        "video": "test_13",
        "question": "Is the brick wall getting taller over the course of this video?",
        "expected_any": ["rising", "increasing", "higher", "yes", "taller", "growing"],
        "expected_none": ["same", "no change"],
        "type": "change",
        "short_answer": "Yes, rising",
        "why_vlm_fails": "VLM can't track entity state changes across multiple observations",
        "demo_note": "Manager verifying construction progress from the laborer's POV",
    },
    {
        "video": "test_2",
        "question": "What changed on this site between the start and end of the video?",
        "expected_any": [],
        "expected_none": [],
        "type": "change",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM needs to diff frames manually, no structured state tracking",
        "demo_note": "Manager getting a quick progress summary from a shift",
    },

    # ── SPATIAL AT SPECIFIC TIME ───────────────────────────────────────────
    # "What was near the worker at a specific moment?"
    {
        "video": "test_10",
        "question": "At the 10-second mark, where is the ladder relative to the worker?",
        "expected_any": [],
        "expected_none": [],
        "type": "spatial_at_time",
        "short_answer": "TBD",
        "why_vlm_fails": "VLM can't compute spatial relations between entities at a past time",
        "demo_note": "Manager reviewing a specific moment — e.g. incident investigation",
    },
]


def _load_agent(graph_path: str, video_name: str | None = None):
    """Load a SceneAgent from a saved graph PKL."""
    from vesta.agent.scene_agent import SceneAgent

    with open(graph_path, "rb") as f:
        state = pickle.load(f)

    # Resolve video path: from PKL, or infer from video_name
    video_path = state.get("video_path")
    if not video_path and video_name:
        candidate = Path(__file__).parent.parent / "data" / "test_videos" / f"{video_name}.mp4"
        if candidate.exists():
            video_path = str(candidate)

    agent = SceneAgent(video_path=video_path, verbose=False)
    agent.graph = state["graph"]
    agent.frame_count = state["frame_count"]
    agent.fps = state["fps"]
    agent.model = state["model"]
    agent.processed = True
    return agent


def evaluate(results_dir: str, verbose: bool = False):
    """Run all eval questions across multiple videos and score results."""
    results_dir = Path(results_dir)

    # Pre-load all needed agents (one per video)
    needed_videos = set(q["video"] for q in EVAL_SET)
    agents = {}

    print("=" * 70)
    print(f"  VESTA Evaluation — {len(EVAL_SET)} Questions across {len(needed_videos)} videos")
    print("=" * 70)
    print()

    for video_name in sorted(needed_videos):
        pkl_path = results_dir / f"{video_name}_graph.pkl"
        if pkl_path.exists():
            agents[video_name] = _load_agent(str(pkl_path), video_name=video_name)
            summary = agents[video_name].graph.get_summary()
            print(f"  Loaded {video_name}: {summary['total_entities']} entities, "
                  f"{summary['total_relationships']} rels")
        else:
            print(f"  MISSING: {pkl_path} — questions for {video_name} will be skipped")
    print()

    results = []
    correct = 0
    skipped = 0
    total = len(EVAL_SET)

    for i, q in enumerate(EVAL_SET):
        video_name = q["video"]
        question = q["question"]
        expected_any = q["expected_any"]
        expected_none = q["expected_none"]

        print(f"  [{i+1}/{total}] [{video_name}] {question}")
        print(f"         Expected: {q['short_answer']}")

        if video_name not in agents:
            print(f"         -> SKIP (no graph for {video_name})")
            skipped += 1
            results.append({
                "video": video_name,
                "question": question,
                "type": q["type"],
                "short_answer": q["short_answer"],
                "answer": "[SKIPPED]",
                "passed": False,
                "skipped": True,
            })
            print()
            continue

        try:
            answer = agents[video_name].ask(question)
        except Exception as e:
            answer = f"[ERROR] {e}"

        answer_lower = answer.lower()

        # If expected_any is empty (TBD questions), just print the answer — no scoring
        if not expected_any:
            passed = None  # unscored
            print(f"         -> UNSCORED (fill in expected_any after reviewing)")
        else:
            has_expected = any(kw.lower() in answer_lower for kw in expected_any)
            has_forbidden = any(kw.lower() in answer_lower for kw in expected_none) if expected_none else False
            passed = has_expected and not has_forbidden
            if passed:
                correct += 1
            print(f"         -> {'PASS' if passed else 'FAIL'}")

        if verbose or passed is None:
            print(f"         VESTA: {answer[:150]}")
            if passed is False and expected_any:
                print(f"         Missing: {expected_any}")

        results.append({
            "video": video_name,
            "question": question,
            "type": q["type"],
            "short_answer": q["short_answer"],
            "answer": answer[:300],
            "passed": passed,
            "skipped": False,
        })
        print()

    # Summary
    scored = [r for r in results if r["passed"] is not None and not r.get("skipped")]
    unscored = [r for r in results if r["passed"] is None]

    print("=" * 70)
    if scored:
        score_total = len(scored)
        score_correct = sum(1 for r in scored if r["passed"])
        print(f"  SCORED: {score_correct}/{score_total} correct ({score_correct/score_total*100:.0f}%)")
    if unscored:
        print(f"  UNSCORED: {len(unscored)} questions (fill in expected_any)")
    if skipped:
        print(f"  SKIPPED: {skipped} questions (missing graph PKLs)")
    print("=" * 70)

    by_type = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0, "skipped": 0, "unscored": 0}
        if r.get("skipped"):
            by_type[t]["skipped"] += 1
        elif r["passed"] is None:
            by_type[t]["unscored"] += 1
        else:
            by_type[t]["total"] += 1
            if r["passed"]:
                by_type[t]["correct"] += 1

    print("  By category:")
    for t, counts in by_type.items():
        if counts["total"] > 0:
            pct = counts["correct"] / counts["total"] * 100
            bar = "=" * counts["correct"] + "." * (counts["total"] - counts["correct"])
            extra = ""
            if counts["unscored"]:
                extra += f" +{counts['unscored']} unscored"
            if counts["skipped"]:
                extra += f" +{counts['skipped']} skipped"
            print(f"    {t:20s}  [{bar}]  {counts['correct']}/{counts['total']}  ({pct:.0f}%){extra}")
        else:
            print(f"    {t:20s}  [no scored questions]")

    failed = [r for r in results if r["passed"] is False]
    if failed:
        print("\n  Failed:")
        for r in failed:
            print(f"    X [{r['video']}] {r['question']}")

    output_path = results_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "scored": f"{sum(1 for r in scored if r['passed'])}/{len(scored)}" if scored else "0/0",
            "unscored": len(unscored),
            "skipped": skipped,
            "by_type": by_type,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved: {output_path}")


def evaluate_single(graph_path: str, question: str, verbose: bool = True):
    """Ask a single question against a graph. For manual testing."""
    # Infer video_name from graph filename (e.g. test_9_graph.pkl → test_9)
    video_name = Path(graph_path).stem.replace("_graph", "")
    agent = _load_agent(graph_path, video_name=video_name)
    summary = agent.graph.get_summary()
    print(f"  Graph: {summary['total_entities']} entities, "
          f"{summary['total_relationships']} rels")
    print(f"  Q: {question}")
    print()
    answer = agent.ask(question)
    print(f"  VESTA: {answer}")
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VESTA Evaluation")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory with *_graph.pkl files")
    parser.add_argument("--graph", help="Single graph PKL (for --question mode)")
    parser.add_argument("--question", "-q", help="Ask a single question against --graph")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full answers")
    args = parser.parse_args()

    if args.question:
        if not args.graph:
            print("Error: --question requires --graph")
            sys.exit(1)
        if not Path(args.graph).exists():
            print(f"Error: Graph file not found: {args.graph}")
            sys.exit(1)
        evaluate_single(args.graph, args.question, verbose=args.verbose)
    else:
        if not Path(args.results_dir).exists():
            print(f"Error: Results dir not found: {args.results_dir}")
            sys.exit(1)
        evaluate(args.results_dir, verbose=args.verbose)
