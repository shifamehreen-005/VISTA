#!/usr/bin/env python3
"""
Helper to collect and organize test images for the floor opening benchmark.

Strategies for gathering images BEFORE the hackathon:

1. OSHA inspection photos (public domain)
2. YouTube POV construction footage (screenshot frames)
3. Construction safety training materials
4. Your own photos

This script helps organize and annotate them.
"""

import os
import json
import shutil
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
POSITIVE_DIR = BENCHMARK_DIR / "images" / "positive"
NEGATIVE_DIR = BENCHMARK_DIR / "images" / "negative"
METADATA_FILE = BENCHMARK_DIR / "images" / "metadata.json"

for d in [POSITIVE_DIR, NEGATIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Image sources and search terms ────────────────────────────────────────────

SEARCH_GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║              FLOOR OPENING BENCHMARK — IMAGE COLLECTION             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  POSITIVE IMAGES (actual floor openings/holes):                      ║
║  Save to: benchmark/images/positive/                                 ║
║                                                                      ║
║  Search terms:                                                       ║
║  • "construction floor opening hazard"                               ║
║  • "unguarded floor hole construction site"                          ║
║  • "missing decking construction"                                    ║
║  • "skylight fall through construction"                              ║
║  • "OSHA floor opening violation"                                    ║
║  • "construction floor penetration unguarded"                        ║
║  • "hole in floor construction site"                                 ║
║  • "open manhole construction"                                       ║
║                                                                      ║
║  NEGATIVE / HARD NEGATIVE IMAGES (no actual hole):                   ║
║  Save to: benchmark/images/negative/                                 ║
║                                                                      ║
║  These should LOOK like they might be holes but aren't:              ║
║  • Dark shadows on concrete floors                                   ║
║  • Black tarps/covers on construction floors                         ║
║  • Puddles/water on dark surfaces                                    ║
║  • Dark-colored floor mats or rubber sheets                          ║
║  • Covered/guarded floor openings (properly secured)                 ║
║  • Dark stains on concrete                                           ║
║  • Expansion joints or control joints in concrete                    ║
║  • Painted dark sections of floor                                    ║
║                                                                      ║
║  IDEAL: First-person / egocentric viewpoint (as if from a helmet)    ║
║                                                                      ║
║  MINIMUM: 10 positive + 10 negative = 20 images                     ║
║  BETTER:  20 positive + 20 negative = 40 images                     ║
║  BEST:    30 positive + 30 negative = 60 images                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def add_image(source_path: str, label: str, description: str = "", source: str = ""):
    """Add an image to the benchmark with metadata."""
    src = Path(source_path)
    if not src.exists():
        print(f"Error: {source_path} does not exist")
        return

    dest_dir = POSITIVE_DIR if label == "positive" else NEGATIVE_DIR
    dest = dest_dir / src.name

    # Avoid overwriting
    if dest.exists():
        stem = src.stem
        suffix = src.suffix
        i = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{suffix}"
            i += 1

    shutil.copy2(src, dest)

    # Update metadata
    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

    metadata[dest.name] = {
        "label": label,
        "description": description,
        "source": source,
        "original_name": src.name,
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Added: {dest.name} ({label}) -> {dest_dir.name}/")


def status():
    """Show current benchmark image collection status."""
    pos = list(POSITIVE_DIR.glob("*"))
    pos = [p for p in pos if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.gif')]
    neg = list(NEGATIVE_DIR.glob("*"))
    neg = [p for p in neg if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.gif')]

    print(f"\n  Positive images (floor openings): {len(pos)}")
    for p in sorted(pos):
        print(f"    {p.name}")
    print(f"\n  Negative images (hard negatives):  {len(neg)}")
    for n in sorted(neg):
        print(f"    {n.name}")
    print(f"\n  Total: {len(pos) + len(neg)} images")

    if len(pos) < 10 or len(neg) < 10:
        print(f"\n  ⚠ Need at least 10 of each. Currently short by "
              f"{max(0, 10-len(pos))} positive and {max(0, 10-len(neg))} negative.")
    else:
        print(f"\n  ✓ Ready to run benchmark!")
        print(f"    Run: python3 test_vlm_spatial.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect test images for floor opening benchmark")
    parser.add_argument("command", choices=["guide", "add", "status"], help="Command to run")
    parser.add_argument("--path", help="Image path (for 'add' command)")
    parser.add_argument("--label", choices=["positive", "negative"], help="Ground truth label")
    parser.add_argument("--description", default="", help="Description of what's in the image")
    parser.add_argument("--source", default="", help="Where the image came from")
    args = parser.parse_args()

    if args.command == "guide":
        print(SEARCH_GUIDE)
    elif args.command == "status":
        status()
    elif args.command == "add":
        if not args.path or not args.label:
            print("Error: --path and --label required for 'add' command")
        else:
            add_image(args.path, args.label, args.description, args.source)
