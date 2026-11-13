"""The intervention, step 1: the measuring, imported from the frozen arm.

Remove one hazard from the scene, keep everything else the same, ask the
model again, and measure how far the answer moved. This module is the
MEASURING only: it takes a finished "before" run and a finished "after"
run and returns the frozen arm's result. It makes no edit and calls no
model.

    before run ─┐
                ├─►  measure()  ─►  five shifts · moved or not · verdict
    after run  ─┘                   (frozen: intervention.run_intervention)

Arm B is scored by IMPORTING Arm A, so the two arms stay comparable: every
number here comes from `intervention.py` at the repo root, which is frozen
like main.py (import only, never edit). This file only translates a run of
ours into the shape that module reads.

Ground truth: the frozen arm looks for `<image name>.gt.json` in its own
verified folder. Without that file the verdict cell is "not_adjudicable"
and the shifts are still measured. GT is read here for the dev-side
verdict only; it never reaches a prompt (iron rule 7).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MODALITIES = ("caption", "image", "both")       # the frozen arm's three channels


def _frozen():
    """The frozen intervention module, imported lazily (it lives at the
    repo root and pulls main.py in on first use)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import intervention
    return intervention


def _plain(x: Any) -> Any:
    """A pydantic model, a dict or anything else -> plain JSON-like data."""
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return x


def _dict(x: Any) -> dict:
    x = _plain(x)
    return x if isinstance(x, dict) else {}


def _list(x: Any) -> list:
    return x if isinstance(x, list) else []


def frozen_result(record: Any, assessment: Any, stage4: Any,
                  image_name: str = "", run_id: str = "") -> dict:
    """One finished run of ours, in the shape the frozen arm reads.

    `assessment` may be the bare assessment or the saved assessment.json
    wrapper ({"assessment": {...}, ...}). Anything missing or malformed
    becomes an empty field; this never raises."""
    rec, s4 = _dict(record), _dict(stage4)
    a = _dict(assessment)
    if isinstance(a.get("assessment"), dict):
        a = a["assessment"]
    trust = _dict(s4.get("trust"))
    name = image_name or Path(str(rec.get("image_path") or "")).name
    return {
        "run_id": str(run_id or ""),
        "image_filename": str(name or ""),
        "caption": str(rec.get("caption") or ""),
        "disaster_level": a.get("disaster_level"),
        "detected_objects": [o for o in _list(rec.get("detected_objects"))
                             if isinstance(o, dict)],
        "threats": [t for t in _list(a.get("threats")) if isinstance(t, dict)],
        "recommendations": [r for r in _list(s4.get("recommendations"))
                            if isinstance(r, dict)],
        "causal_graph": _dict(s4.get("graph_a")) or None,
        "graph_b": _dict(s4.get("graph_b")) or None,
        "pre_intervention_trust": {"score": trust.get("score", 0.0),
                                   "level": trust.get("band", "unknown")},
    }


def result_from_run_dir(run_dir: Path) -> dict:
    """A saved UI run folder -> the frozen shape. Reads the three files a
    finished run leaves: *__perception.json, assessment.json, stage4.json.
    A missing or unreadable file is an empty part, not an error."""
    run_dir = Path(run_dir)

    def load(p: Path | None) -> dict:
        try:
            data = json.loads(p.read_text()) if p else {}
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}
    perc = next(iter(sorted(run_dir.glob("*__perception.json"))), None)
    return frozen_result(load(perc), load(run_dir / "assessment.json"),
                         load(run_dir / "stage4.json"), run_id=run_dir.name)


def clean_result(result: Any) -> dict:
    """A frozen-shape result with every malformed part emptied, so the
    frozen code (which trusts its input) never sees a None where it
    expects a record. Lists keep only their dict entries; a graph keeps
    only dict nodes and edges."""
    r = dict(_dict(result))
    for key in ("detected_objects", "threats", "recommendations"):
        r[key] = [x for x in _list(r.get(key)) if isinstance(x, dict)]
    for key in ("causal_graph", "graph_b"):
        g = _dict(r.get(key))
        if g:
            g = dict(g)
            for part in ("nodes", "edges", "intervention_candidates"):
                if part in g or part != "intervention_candidates":
                    g[part] = [x for x in _list(g.get(part)) if isinstance(x, dict)]
        r[key] = g or None
    if not isinstance(r.get("pre_intervention_trust"), dict):
        r["pre_intervention_trust"] = {}
    for key in ("image_filename", "caption", "run_id"):
        r[key] = str(r.get(key) or "")
    return r


def edit_bundle(modality: str, caption_changed: bool, image_changed: bool) -> dict:
    """What was changed on the INPUT. The fair-test reads only this, never
    the model's answer. "both" needs both channels changed."""
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    cap, img = bool(caption_changed), bool(image_changed)
    applied = cap if modality == "caption" else img if modality == "image" else (cap and img)
    return {"modality": modality, "caption_changed": cap,
            "image_changed": img, "applied": applied}


def candidates(before: dict, *, gt_dir: Path | None = None,
               core_basis: str = "edge", core_rule: str = "half_max") -> dict:
    """The hazards that can be removed, ranked by the frozen arm."""
    iv = _frozen()
    baseline = iv.intervention_baseline(clean_result(before), None, gt_dir=gt_dir)
    return iv.enumerate_candidates(baseline, core_basis=core_basis, core_rule=core_rule)


def measure(before: dict, after: dict, target_oid: str, *,
            modality: str = "both", caption_changed: bool = True,
            image_changed: bool = True, gt_dir: Path | None = None,
            core_basis: str = "edge", core_rule: str = "half_max") -> dict:
    """Compare a before run with an after run for ONE removed hazard.

    `before` and `after` are frozen-shape results (frozen_result /
    result_from_run_dir). Returns the frozen arm's result unchanged
    (signals, verdict, u_check, spec, ...) plus:
      "after_missing": True when the after run has no recommendations and
                       no objects, i.e. it is not a real answer to compare.
    When the input was not changed for the chosen channel there is no
    intervention to measure: the frozen arm marks the fair-test as failed
    and the verdict as void, exactly as it does on its own screen."""
    iv = _frozen()
    before, after = clean_result(before), clean_result(after)
    edit = edit_bundle(modality, caption_changed, image_changed)
    baseline = iv.intervention_baseline(before, None, gt_dir=gt_dir)

    def _after(_image, _prompt, _spec):
        # the do() is the edited INPUT; the answer to it is the after run
        return after
    result = iv.run_intervention(
        baseline, {"target_object_id": str(target_oid or ""), "modality": modality},
        _after, run_control=False, edit=edit,
        core_basis=core_basis, core_rule=core_rule)
    result["edit"] = edit
    result["after_missing"] = not (_list(after.get("recommendations"))
                                   or _list(after.get("detected_objects")))
    return result


def summary(result: dict) -> dict:
    """The few facts a screen or a table needs from a measure() result."""
    result = _dict(result)
    sig, ver = _dict(result.get("signals")), _dict(result.get("verdict"))
    u = _dict(result.get("u_check"))
    target = _dict(_dict(result.get("spec")).get("target"))
    return {
        "target": target.get("object_id"),
        "fair": (not u.get("leaked")) if u else None,
        "moved": ver.get("moved"),
        "cell": ver.get("cell"),
        "total_shift": sig.get("total_shift"),
        "content_shift": sig.get("content_shift"),
        "shifts": {k: sig.get(k) for k in ("hazard_shift", "graph_shift",
                                           "recommendation_shift",
                                           "structural_shift", "semantic_shift")},
        "level_delta": sig.get("hazard_level_delta"),
        "direction": sig.get("rec_direction"),
        "explanation": ver.get("explanation"),
    }
