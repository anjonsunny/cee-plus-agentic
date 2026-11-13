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

# Ground truth in the frozen arm's format. "verified" = accepted by Sunny in
# the frozen arm's Ground Truth tab. "candidate" = proposed, not verified.
GT_VERIFIED = REPO_ROOT / "exports" / "ground_truth" / "verified"
GT_CANDIDATES = REPO_ROOT / "exports" / "ground_truth" / "candidates"
# Where copies of the frozen arm's scene images live, to recognise one of
# our scenes as the SAME picture under another file name.
GT_IMAGE_POOLS = (REPO_ROOT / "tests" / "fixtures" / "golden_scenes", GT_CANDIDATES)
_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
_POOL_CACHE: dict[tuple, dict[str, str]] = {}


def _md5(path: Path) -> str:
    import hashlib
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _pool_index(pools: tuple) -> dict[str, str]:
    """{content hash: file name} over the image pools (cached per pool set)."""
    key = tuple(str(p) for p in pools)
    if key not in _POOL_CACHE:
        idx: dict[str, str] = {}
        for pool in pools:
            try:
                files = sorted(Path(pool).iterdir())
            except OSError:
                continue
            for f in files:
                if f.suffix.lower() in _IMG_EXT:
                    h = _md5(f)
                    if h:
                        idx.setdefault(h, f.name)
        _POOL_CACHE[key] = idx
    return _POOL_CACHE[key]


def gt_lookup(image_path: Any, *, verified: Path | None = None,
              candidates_dir: Path | None = None, pools: tuple | None = None) -> dict:
    """Find the frozen-format ground truth for a scene image.

    The frozen arm looks ground truth up by FILE NAME. Three of our scenes
    are byte-for-byte the frozen arm's pictures under other names (B_pool is
    push_06_drowning_pool, C_tanker_fire is push_09_tanker_near_fire,
    F_park_control is push_61_park_saturday), so the name lookup found
    nothing and every verdict read "not adjudicable". Here the picture is
    recognised by its content too.

    Returns {"status": "verified" | "candidate" | "none", "image_filename":
    the name the ground-truth file is stored under, "dir": its folder,
    "matched_by": "name" | "content" | None}. A verified file always wins
    over a candidate. Dev-side only: this never reaches a prompt."""
    verified = Path(verified) if verified else GT_VERIFIED
    candidates_dir = Path(candidates_dir) if candidates_dir else GT_CANDIDATES
    none = {"status": "none", "image_filename": "", "dir": None, "matched_by": None}
    if not image_path:
        return none
    path = Path(str(image_path))
    names = [(path.name, "name")]
    same = _pool_index(tuple(pools) if pools is not None else GT_IMAGE_POOLS).get(_md5(path))
    if same and same != path.name:
        names.append((same, "content"))
    for folder, status in ((verified, "verified"), (candidates_dir, "candidate")):
        for name, how in names:
            if name and (folder / f"{name}.gt.json").exists():
                return {"status": status, "image_filename": name,
                        "dir": str(folder), "matched_by": how}
    return none


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


def result_from_run_dir(run_dir: Path, *, gt_verified: Path | None = None,
                        gt_candidates: Path | None = None,
                        gt_pools: tuple | None = None) -> dict:
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
    result = frozen_result(load(perc), load(run_dir / "assessment.json"),
                           load(run_dir / "stage4.json"), run_id=run_dir.name)
    # the run's own input picture decides which ground truth applies
    image = next((f for ext in _IMG_EXT for f in sorted(run_dir.glob(f"*{ext}"))
                  if "__" not in f.name), None)
    gt = gt_lookup(image, verified=gt_verified, candidates_dir=gt_candidates, pools=gt_pools)
    result["scene_image"] = result["image_filename"]
    result["gt"] = gt
    if gt["status"] != "none":
        result["image_filename"] = gt["image_filename"]   # the name the frozen lookup uses
    return result


def _gt_dir(before: dict, gt_dir: Path | None) -> Path | None:
    """An explicit folder wins; else the folder gt_lookup found for the
    before run; else the frozen arm's default."""
    if gt_dir is not None:
        return gt_dir
    found = _dict(_dict(before).get("gt")).get("dir")
    return Path(found) if found else None


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
    baseline = iv.intervention_baseline(clean_result(before), None,
                                        gt_dir=_gt_dir(before, gt_dir))
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
    gt_dir = _gt_dir(before, gt_dir)
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


def synthesis(before: dict, tries: list[tuple[dict, dict]], *,
              core_basis: str = "edge", core_rule: str = "half_max") -> dict:
    """The frozen arm's labels that need ground truth, over all tries of one
    before run: a status per hazard, the scene cell (declared x operative),
    and the distribution verdict (core and spurious against ground truth).

    `tries`: (saved try, after run in the frozen shape). Each try is measured
    again here, so a try saved before its ground truth was found still gets
    its real cell. Imported from main.py (`_synthesis_cell`,
    `_distributional_groundedness`); nothing is rewritten.

    main.py reads the VERIFIED folder only, so with a candidate (proposed)
    file or none the answer is {"available": False, "why": ...}: without
    ground truth the frozen code would call every moved hazard "spurious"."""
    gt = _dict(_dict(before).get("gt"))
    out = {"gt": gt, "available": False, "tries": []}
    if gt.get("status") != "verified":
        out["why"] = ("the ground truth for this scene is proposed, not verified"
                      if gt.get("status") == "candidate"
                      else "no ground truth for this scene")
        return out
    if Path(str(gt.get("dir"))) != GT_VERIFIED:
        out["why"] = "the frozen labels read the frozen arm's own verified folder only"
        return out
    clean_before = clean_result(before)
    history = []
    for one, after in tries or []:
        one = _dict(one)
        try:
            res = measure(before, after, str(one.get("target") or ""),
                          modality=str(one.get("modality") or "both"),
                          caption_changed=bool(one.get("caption_changed")),
                          image_changed=bool(one.get("image_changed")),
                          core_basis=core_basis, core_rule=core_rule)
        except Exception:
            continue
        if not res.get("signals"):
            continue
        history.append({
            "selection": {"target_object_id": one.get("target")},
            "spec": res.get("spec"), "u_check": res.get("u_check"),
            "verdict": res.get("verdict"), "signals": res.get("signals"),
            "before": {"recommendations": clean_before["recommendations"],
                       "graph_a": clean_before["causal_graph"] or {}}})
        out["tries"].append({"index": one.get("index"), "target": one.get("target"),
                             "cell": _dict(res.get("verdict")).get("cell")})
    try:
        _frozen()
        import main
        syn = main._synthesis_cell(clean_before, {"tries": history},
                                   core_basis=core_basis, core_rule=core_rule)
        dist = main._distributional_groundedness(syn)
    except Exception as exc:
        out["why"] = f"the frozen synthesis could not run: {exc}"
        return out
    out.update({
        "available": True, "cell": syn.get("cell"),
        "declared_match": syn.get("declared_match"),
        "operative_match": syn.get("operative_match"),
        "next_to_test": syn.get("next_to_test"),
        "gt_core": sorted(syn.get("gt_core_ids") or []),
        "hazards": [{k: h.get(k) for k in ("object_id", "state", "status", "operative",
                                           "is_core", "has_gt")}
                    for h in (syn.get("hazard_status") or []) if isinstance(h, dict)],
        "distribution": {"verdict": dist.get("verdict"), "overlap": dist.get("overlap"),
                         "coverage": list(dist.get("coverage") or ()),
                         "provisional": dist.get("provisional"),
                         "core_op_mass": dist.get("core_op_mass"),
                         "nongt_op_mass": dist.get("nongt_op_mass"),
                         "gt_dist": dist.get("gt_dist"), "op_dist": dist.get("op_dist")},
    })
    return out
