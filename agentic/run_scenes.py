#!/usr/bin/env python3
"""Run Stage 1 perception over the six worked-example scenes.

Expects the scene images in experiments/agentic_scenes/ with these names
(any of .png/.jpg/.jpeg/.webp):

    A_fire            night house fire (core)
    B_pool            drowning child at the pool (core)
    C_tanker_fire     leaking tanker beside brush fire (core)
    D_aerial_spill    overturned tanker, aerial/drone view (check set)
    E_collapse        partial building collapse (check set)
    F_park_control    ordinary park, no hazard (negative control)

Usage (from repo root, Ollama running for live VLM naming):
    python -m agentic.run_scenes                 # all six, live
    python -m agentic.run_scenes --only B_pool   # one scene
    python -m agentic.run_scenes --no-masks      # skip SAM (faster first pass)

Output per scene, in experiments/agentic_scenes/perception/:
    <name>__perception.json   the frozen Stage 1 record (after review)
    <name>__overlay.png       boxes + state badges on the image
    <name>__<id>_mask.png     SAM mask per localized entity
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.perception import run_perception  # noqa: E402

SCENES_DIR = REPO_ROOT / "experiments" / "agentic_scenes"
SCENE_NAMES = [
    "A_fire", "B_pool", "C_tanker_fire",
    "D_aerial_spill", "E_collapse", "F_park_control",
]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def find_scene(name: str) -> Path | None:
    for ext in IMG_EXTS:
        p = SCENES_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run perception on the worked-example scenes.")
    p.add_argument("--only", help="Run a single scene by name (e.g. B_pool).")
    p.add_argument("--no-masks", action="store_true")
    args = p.parse_args(argv)

    names = [args.only] if args.only else SCENE_NAMES
    out_dir = SCENES_DIR / "perception"
    missing = [n for n in names if find_scene(n) is None]
    if missing:
        print(f"Missing images in {SCENES_DIR}: {', '.join(missing)}", file=sys.stderr)
        print("Save the scene images there first (see module docstring).", file=sys.stderr)
        if len(missing) == len(names):
            return 2

    rc = 0
    for name in names:
        path = find_scene(name)
        if path is None:
            continue
        print(f"\n=== {name} ({path.name}) ===", flush=True)
        t0 = time.perf_counter()
        try:
            result = run_perception(path, with_masks=not args.no_masks, out_dir=out_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            rc = 1
            continue
        dt = time.perf_counter() - t0
        ext = sum(1 for o in result.detected_objects if o.vocab_extension)
        print(f"  {len(result.detected_objects)} entities, "
              f"{len(result.unlocalized)} unlocalized, "
              f"{ext} vocab extensions, {dt:.1f}s")
        for o in result.detected_objects:
            box = "boxed" if o.bbox else "NO BOX"
            print(f"    {o.object_id:<22} {o.state:<12} {box}  conf={o.box_confidence:.2f}")
    print(f"\nOverlays and JSON in {out_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
