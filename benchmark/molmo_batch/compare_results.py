#!/usr/bin/env python3
"""
Compare Molmo and VISTA benchmark JSON outputs and write markdown report.
"""

import argparse
import json
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_avg(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark result files")
    parser.add_argument("--molmo", required=True, help="Molmo benchmark JSON")
    parser.add_argument("--ours", required=True, help="VISTA benchmark JSON")
    parser.add_argument(
        "--output",
        default="results/comparison_report.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    molmo = load_json(args.molmo)
    ours = load_json(args.ours)

    molmo_ok = [r for r in molmo["test_results"] if "error" not in r]
    ours_ok = [r for r in ours["test_results"] if "error" not in r]

    molmo_by_id = {r["id"]: r for r in molmo_ok}
    ours_by_id = {r["id"]: r for r in ours_ok}
    all_ids = sorted(set(molmo_by_id.keys()) | set(ours_by_id.keys()))

    lines = [
        "# Benchmark Comparison Report",
        "",
        "| Metric | Molmo2 | VISTA |",
        "|---|---|---|",
        f"| Model | {molmo.get('model', 'N/A')} | {ours.get('model', 'N/A')} |",
        f"| Successful runs | {len(molmo_ok)} | {len(ours_ok)} |",
        f"| Avg inference time (s) | {safe_avg([x['total_time_s'] for x in molmo_ok]):.2f} | {safe_avg([x['total_time_s'] for x in ours_ok]):.2f} |",
        f"| Avg generation speed (tok/s) | {safe_avg([x['tokens_per_second'] for x in molmo_ok]):.1f} | {safe_avg([x['tokens_per_second'] for x in ours_ok]):.1f} |",
        "",
    ]

    for test_id in all_ids:
        m = molmo_by_id.get(test_id)
        o = ours_by_id.get(test_id)
        ref = m or o

        lines.extend(
            [
                f"## {test_id}",
                "",
                f"**Video:** `{ref['video']}`",
                "",
                f"**Prompt:** {ref['prompt']}",
                "",
                "### Molmo2",
                "",
            ]
        )

        if m:
            lines.extend(["```", m["response"], "```"])
        else:
            lines.append("*(No successful result)*")

        lines.extend(["", "### VISTA", ""])
        if o:
            lines.extend(["```", o["response"], "```"])
        else:
            lines.append("*(No successful result)*")

        lines.extend(["", "---", ""])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()
