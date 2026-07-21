#!/usr/bin/env python3
"""Headless REPLAY of a saved intervention run — recompute the groundedness synthesis from the
stored tries. No browser, no VLM: deterministic and instant. Sweep the basis/rule freely without
re-running the model.

Usage (from repo root, in the clip_dash env):
    python tools/headless_synth.py <run_dir> [--basis edge|consequence] [--rule above_mean|half_max|top_k]

Example:
    python tools/headless_synth.py exports/runs/run_20260715T143550 --basis consequence --rule half_max
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402


def load(run_dir: str):
    norm = json.load(open(f"{run_dir}/structured_response.json"))["structured_response"]
    iv = json.load(open(f"{run_dir}/intervention.json"))
    return norm, {"run_key": iv.get("run_id"), "tries": iv["tries"]}


def report(run_dir: str, basis: str, rule: str) -> None:
    norm, hist = load(run_dir)
    syn = main._synthesis_cell(norm, hist, core_basis=basis, core_rule=rule)
    dist = main._distributional_groundedness(syn)
    pit = norm.get("pre_intervention_trust", {})

    print(f"\nRUN: {os.path.basename(run_dir.rstrip('/'))}   basis={basis}  rule={rule}")
    print(f"Baseline trust: {pit.get('level')} ({pit.get('score')})")
    print(f"\n{'hazard':14}{'GT':>7}{'core?':>7}{'op':>8}  {'OP dep':>7}  status / direction")
    print("-" * 72)
    for h in sorted(syn["hazard_status"], key=lambda x: -dist["gt_dist"].get(x["object_id"], 0)):
        o, op = h["object_id"], h.get("operative")
        d = ("untested" if op is None else
             "down de-escalated" if op > 0 else
             "up escalated" if op < 0 else "unchanged")
        opv = f"{op:8.2f}" if op is not None else f"{'--':>8}"
        print(f"{o:14}{dist['gt_dist'].get(o, 0):7.2f}{str(h['is_core']):>7}{opv}  "
              f"{dist['op_dist'].get(o, 0):7.2f}  {h['status']:10} {d}")
    print("-" * 72)
    print(f"VERDICT: {dist['verdict']}   alignment={dist['overlap']}   "
          f"core_op_mass={dist['core_op_mass']}   non-GT_leakage={dist['nongt_op_mass']}")
    print(f"coverage: {dist['coverage'][0]}/{dist['coverage'][1]} suppressed   "
          f"gt_core={sorted(dist['core'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Headless replay of a saved intervention run.")
    ap.add_argument("run_dir")
    ap.add_argument("--basis", default="edge", choices=["edge", "consequence"])
    ap.add_argument("--rule", default="above_mean", choices=["above_mean", "half_max", "top_k"])
    a = ap.parse_args()
    report(a.run_dir, a.basis, a.rule)
