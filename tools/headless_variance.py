#!/usr/bin/env python3
"""VARIANCE harness: run a saved intervention run's counterfactuals through the VLM N times and
report how STABLE the outcome is.

Why: a single live re-run can reproduce a verdict by luck. Repeating it separates a genuine,
deterministic-ish failure mode (e.g. "suppressing one of several identical burning houses
reliably ESCALATES") from sampling noise. Verdict stability is the headline; the content_shift
spread shows how much the model wobbles even when the verdict holds.

Requires a reachable VLM (Ollama). Each repeat costs (#tries) vision calls (~20-60s each), so
N=3 on a 4-try run is ~12 calls — RUN THIS IN THE BACKGROUND.

Usage (from repo root, in the clip_dash env):
    python tools/headless_variance.py <run_dir> --repeats 3 [--basis edge] [--rule above_mean]
"""
import argparse
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from headless_rerun import rerun_once  # noqa: E402


def variance(run_dir: str, repeats: int, basis: str, rule: str) -> None:
    per_hazard_cells: dict = defaultdict(list)
    per_hazard_shift: dict = defaultdict(list)
    per_hazard_dir: dict = defaultdict(list)
    saved_cell: dict = {}
    syn_verdicts: list = []
    syn_align: list = []

    for i in range(1, repeats + 1):
        print(f"[pass {i}/{repeats}] running {run_dir} ...", flush=True)
        out = rerun_once(run_dir, basis, rule)
        for oid, r in out["per_hazard"].items():
            saved_cell[oid] = r["saved"]
            per_hazard_cells[oid].append(r["cell"])
            per_hazard_dir[oid].append(r["direction"])
            if isinstance(r["content_shift"], (int, float)):
                per_hazard_shift[oid].append(float(r["content_shift"]))
        syn_verdicts.append(out["synthesis"]["verdict"])
        syn_align.append(out["synthesis"]["alignment"])
        print(f"  -> synthesis: {out['synthesis']['verdict']} "
              f"(alignment {out['synthesis']['alignment']})", flush=True)

    def _spread(vals):
        if not vals:
            return "--"
        if len(vals) == 1:
            return f"{vals[0]:.2f}"
        return (f"{statistics.mean(vals):.2f} +/-{statistics.pstdev(vals):.2f} "
                f"[{min(vals):.2f}-{max(vals):.2f}]")

    print(f"\n{'='*78}\nVARIANCE over {repeats} live passes — {os.path.basename(run_dir.rstrip('/'))}"
          f"  (basis={basis}, rule={rule})\n{'='*78}")
    print(f"{'hazard':12}{'saved':>12}  {'verdicts across passes':38} {'content_shift'}")
    print("-" * 78)
    for oid, cells in per_hazard_cells.items():
        c = Counter(cells)
        stable = "STABLE" if len(c) == 1 else "VARIES"
        summary = ", ".join(f"{k}x{v}" for k, v in c.most_common())
        print(f"{oid:12}{str(saved_cell.get(oid)):>12}  {stable:7}{summary:31}"
              f"{_spread(per_hazard_shift[oid])}")
    print("-" * 78)
    sc = Counter(syn_verdicts)
    print(f"synthesis verdict: {'STABLE' if len(sc)==1 else 'VARIES'}  "
          f"{', '.join(f'{k}x{v}' for k, v in sc.most_common())}   |   "
          f"alignment {_spread(syn_align)}")
    n_stable = sum(1 for c in per_hazard_cells.values() if len(set(c)) == 1)
    print(f"per-hazard stability: {n_stable}/{len(per_hazard_cells)} hazards gave the SAME "
          f"verdict on every pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure verdict variance across repeated live re-runs.")
    ap.add_argument("run_dir")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--basis", default="edge", choices=["edge", "consequence"])
    ap.add_argument("--rule", default="above_mean", choices=["above_mean", "half_max", "top_k"])
    a = ap.parse_args()
    variance(a.run_dir, a.repeats, a.basis, a.rule)
