#!/usr/bin/env python3
"""Headless LIVE re-run of a saved intervention run — feed each SAVED counterfactual image+caption
back to the VLM and recompute. No browser, no manual editing: reuses the run's own do() inputs
(counterfactual_try{n}.jpg + counterfactual_caption). Stochastic: the model will not return
identical JSON each time, so this is for stability / do-not-applied checks, not reproduction.

Requires a reachable VLM (Ollama). Defaults are set if unset:
    QWEN_API_URL=http://localhost:11434/v1/chat/completions
    QWEN_MODEL_NAME=qwen2.5vl:7b

Usage (from repo root, in the clip_dash env):
    python tools/headless_rerun.py <run_dir> [--basis edge|consequence] [--rule above_mean|half_max|top_k]

`rerun_once()` is importable — tools/headless_variance.py calls it N times to measure variance.
Note: N vision calls are slow (~20-60s each); run in the background for runs with several tries.
"""
import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_env() -> None:
    os.environ.setdefault("QWEN_API_URL", "http://localhost:11434/v1/chat/completions")
    os.environ.setdefault("QWEN_MODEL_NAME", "qwen2.5vl:7b")


def _data_url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()


def rerun_once(run_dir: str, basis: str = "edge", rule: str = "above_mean") -> dict:
    """One full live pass over every saved counterfactual in `run_dir`.

    Returns {"per_hazard": {oid: {saved, cell, content_shift, direction, leaked}},
             "synthesis": {verdict, alignment, core_op_mass, nongt_leakage}}.
    """
    _ensure_env()
    import main
    import intervention

    norm = json.load(open(f"{run_dir}/structured_response.json"))["structured_response"]
    iv = json.load(open(f"{run_dir}/intervention.json"))
    baseline = intervention.intervention_baseline(norm, None, gt_dir=main.GT_VERIFIED_DIR)

    history = {"run_key": iv.get("run_id"), "tries": []}
    per_hazard: dict = {}
    for t in iv["tries"]:
        sel = t["selection"]
        oid = sel["target_object_id"]
        img_file = t.get("counterfactual_image_file")
        if not img_file:
            continue
        cap = t.get("counterfactual_caption") or ""
        img = _data_url(f"{run_dir}/{img_file}")

        def vlm(_a, _b, _c, _cap=cap, _img=img):
            return main.query_qwen(main.DEFAULT_PROMPT, _cap, _img, allow_inferred=False)

        edit = {"modality": sel.get("modality", "both"), "caption_changed": True,
                "image_changed": True, "applied": True}
        res = intervention.run_intervention(
            baseline, {"target_object_id": oid, "modality": sel.get("modality", "both")},
            vlm, run_control=False, edit=edit, core_basis=basis, core_rule=rule)
        sig = res.get("signals") or {}
        history["tries"].append({
            "selection": sel, "verdict": res.get("verdict"), "u_check": res.get("u_check"),
            "signals": sig, "spec": res.get("spec"),
            "before": {"graph_a": baseline.get("graph_a"),
                       "recommendations": baseline.get("recommendations"),
                       "hazard_level": baseline.get("hazard_level"),
                       "detected_objects": baseline.get("detected_objects")}})
        per_hazard[oid] = {
            "saved": (t.get("verdict") or {}).get("cell"),
            "cell": (res.get("verdict") or {}).get("cell"),
            "content_shift": sig.get("content_shift"),
            "direction": sig.get("rec_direction") or sig.get("graph_direction"),
            "leaked": (res.get("u_check") or {}).get("leaked"),
        }

    syn = main._synthesis_cell(norm, history, core_basis=basis, core_rule=rule)
    dist = main._distributional_groundedness(syn)
    return {"per_hazard": per_hazard,
            "synthesis": {"verdict": dist["verdict"], "alignment": dist["overlap"],
                          "core_op_mass": dist["core_op_mass"],
                          "nongt_leakage": dist["nongt_op_mass"]}}


def main_cli(run_dir: str, basis: str, rule: str) -> None:
    out = rerun_once(run_dir, basis, rule)
    print(f"\nLIVE re-run of {os.path.basename(run_dir.rstrip('/'))} "
          f"vs {os.getenv('QWEN_MODEL_NAME')} (saved do() inputs)\n")
    print(f"{'target':12}{'saved':>16}{'FRESH':>16}   fresh direction / u-leak")
    print("-" * 74)
    for oid, r in out["per_hazard"].items():
        print(f"{oid:12}{str(r['saved']):>16}{str(r['cell']):>16}   "
              f"{r['direction']}  (u-leak={r['leaked']})")
    s = out["synthesis"]
    print("-" * 74)
    print(f"FRESH synthesis verdict: {s['verdict']}   alignment={s['alignment']}   "
          f"core_op_mass={s['core_op_mass']}   non-GT_leakage={s['nongt_leakage']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Headless live re-run of a saved intervention run.")
    ap.add_argument("run_dir")
    ap.add_argument("--basis", default="edge", choices=["edge", "consequence"])
    ap.add_argument("--rule", default="above_mean", choices=["above_mean", "half_max", "top_k"])
    a = ap.parse_args()
    main_cli(a.run_dir, a.basis, a.rule)
