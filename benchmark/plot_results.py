#!/usr/bin/env python3
"""
Generate presentation-ready plots from benchmark results.
Run after test_vlm_spatial.py produces results.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG = "#0D1117"
CARD_BG = "#161B22"
ACCENT = "#58A6FF"
ACCENT2 = "#F78166"
ACCENT3 = "#7EE787"
ACCENT4 = "#D2A8FF"
ACCENT5 = "#FFA657"
ACCENT6 = "#FF7B72"
TEXT = "#E6EDF3"
TEXT_DIM = "#8B949E"
GRID = "#21262D"

MODEL_COLORS = {
    "claude": ACCENT4,
    "openai": ACCENT3,
    "gemini": ACCENT,
}

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=11)
    ax.tick_params(colors=TEXT_DIM, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.5)

def save(fig, name, out_dir):
    fig.patch.set_facecolor(DARK_BG)
    fig.savefig(out_dir / name, dpi=200, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {name}")


def plot_benchmark(results_file: str):
    with open(results_file) as f:
        data = json.load(f)

    results = data["results"]
    summary = data["summary"]
    out_dir = Path(results_file).parent

    # ── Plot 1: Accuracy by Model × Prompt Strategy ───────────────────────
    # Group by (provider, prompt)
    grouped = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for r in results:
        key = (r["model_provider"], r["model"], r["prompt_strategy"])
        grouped[key]["total"] += 1
        if r["correct"]:
            grouped[key]["correct"] += 1
        if r["ground_truth"] == "positive" and r["correct"]:
            grouped[key]["tp"] += 1
        elif r["ground_truth"] == "positive" and not r["correct"]:
            grouped[key]["fn"] += 1
        elif r["ground_truth"] == "negative" and r["correct"]:
            grouped[key]["tn"] += 1
        elif r["ground_truth"] == "negative" and not r["correct"]:
            grouped[key]["fp"] += 1

    providers = sorted(set(k[0] for k in grouped))
    prompts = sorted(set(k[2] for k in grouped))

    # Accuracy grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(prompts))
    width = 0.8 / len(providers)

    for i, prov in enumerate(providers):
        accs = []
        for prompt in prompts:
            matching = [(k, v) for k, v in grouped.items() if k[0] == prov and k[2] == prompt]
            if matching:
                _, v = matching[0]
                accs.append(v["correct"] / v["total"] * 100 if v["total"] > 0 else 0)
            else:
                accs.append(0)

        color = MODEL_COLORS.get(prov, ACCENT)
        bars = ax.bar(x + i * width - (len(providers)-1)*width/2, accs, width,
                      label=prov.title(), color=color, alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.0f}%', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in prompts], fontsize=10)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color=ACCENT6, linestyle='--', alpha=0.5, label='Random baseline (50%)')
    style_ax(ax, "VLM Accuracy on Floor Opening Detection by Prompt Strategy", "Prompt Strategy", "Accuracy (%)")
    ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    save(fig, "benchmark_accuracy_by_prompt.png", out_dir)

    # ── Plot 2: Confusion matrix style — False positives vs False negatives ──
    fig, axes = plt.subplots(1, len(providers), figsize=(6 * len(providers), 5))
    if len(providers) == 1:
        axes = [axes]

    for idx, prov in enumerate(providers):
        ax = axes[idx]
        # Aggregate across all prompts for this provider
        tp = sum(v["tp"] for k, v in grouped.items() if k[0] == prov)
        fp = sum(v["fp"] for k, v in grouped.items() if k[0] == prov)
        fn = sum(v["fn"] for k, v in grouped.items() if k[0] == prov)
        tn = sum(v["tn"] for k, v in grouped.items() if k[0] == prov)

        matrix = np.array([[tp, fn], [fp, tn]])
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]}', ha='center', va='center',
                        color='black', fontsize=18, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nHole', 'Predicted\nNo Hole'], color=TEXT_DIM)
        ax.set_yticklabels(['Actual\nHole', 'Actual\nNo Hole'], color=TEXT_DIM)
        ax.set_title(f'{prov.title()}', color=TEXT, fontsize=14, fontweight='bold')
        ax.tick_params(colors=TEXT_DIM)

    fig.suptitle("Confusion Matrices — Where Do VLMs Fail?", color=TEXT, fontsize=16, fontweight='bold', y=1.05)
    fig.tight_layout()
    save(fig, "benchmark_confusion_matrices.png", out_dir)

    # ── Plot 3: The key failure chart — False Negative Rate ──────────────
    # (Model says "no hole" when there IS a hole — the dangerous failure)
    fig, ax = plt.subplots(figsize=(12, 6))

    miss_rates = []
    labels = []
    colors = []
    for prov in providers:
        for prompt in prompts:
            matching = [(k, v) for k, v in grouped.items() if k[0] == prov and k[2] == prompt]
            if matching:
                _, v = matching[0]
                fn_rate = v["fn"] / (v["tp"] + v["fn"]) * 100 if (v["tp"] + v["fn"]) > 0 else 0
                miss_rates.append(fn_rate)
                labels.append(f"{prov.title()}\n{prompt.replace('_', ' ')}")
                colors.append(MODEL_COLORS.get(prov, ACCENT))

    bars = ax.bar(range(len(miss_rates)), miss_rates, color=colors, alpha=0.85, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    style_ax(ax, "Miss Rate: % of Real Holes the Model Failed to Detect", "", "False Negative Rate (%)")

    for bar, rate in zip(bars, miss_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', color=ACCENT6, fontsize=10, fontweight='bold')

    ax.axhline(y=30, color=ACCENT6, linestyle='--', alpha=0.5)
    ax.text(len(miss_rates) - 0.5, 32, "30% miss rate = unacceptable for safety",
            color=ACCENT6, fontsize=9, ha='right')
    save(fig, "benchmark_miss_rate.png", out_dir)

    # ── Plot 4: Per-image breakdown — which images fool the models? ───────
    image_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        image_difficulty[r["image"]]["total"] += 1
        if r["correct"]:
            image_difficulty[r["image"]]["correct"] += 1
        image_difficulty[r["image"]]["label"] = r["ground_truth"]

    sorted_images = sorted(image_difficulty.items(), key=lambda x: x[1]["correct"]/x[1]["total"])

    fig, ax = plt.subplots(figsize=(14, max(6, len(sorted_images) * 0.4)))
    names = [f"{img} ({'hole' if d['label']=='positive' else 'no hole'})" for img, d in sorted_images]
    accs = [d["correct"] / d["total"] * 100 for _, d in sorted_images]
    bar_colors = [ACCENT6 if d["label"] == "positive" else ACCENT for _, d in sorted_images]

    ax.barh(range(len(names)), accs, color=bar_colors, height=0.7, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 105)
    ax.axvline(x=50, color=TEXT_DIM, linestyle='--', alpha=0.5)
    style_ax(ax, "Per-Image Accuracy (hardest images at top)", "% of Models × Prompts Correct", "")
    ax.spines['left'].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ACCENT6, label='Actual hole (positive)'),
                       Patch(facecolor=ACCENT, label='No hole (negative)')]
    ax.legend(handles=legend_elements, facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc='lower right')
    save(fig, "benchmark_per_image_difficulty.png", out_dir)

    print(f"\nAll benchmark plots saved to: {out_dir}/")


if __name__ == "__main__":
    # Find the most recent results file
    results_dir = Path(__file__).parent / "results"
    json_files = sorted(results_dir.glob("benchmark_*.json"))
    if not json_files:
        print("No results found. Run test_vlm_spatial.py first.")
        sys.exit(1)
    latest = json_files[-1]
    print(f"Plotting results from: {latest.name}")
    plot_benchmark(str(latest))
