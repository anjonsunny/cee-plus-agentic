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

# How much a ground-truth hazard SHOULD matter. The frozen arm offers two
# ways and defaults to "edge" (count the arrows leaving the hazard). Ours
# uses its other way, "consequence": each arrow counts by its effect, and
# double when it lands on a life (Sunny, 2026-11-13: with the arrow count
# tanker, spill and fire on C all weighed 0.33, "all scores seem the same";
# it counts arrows only). Both numbers are the frozen arm's own.
CORE_BASIS = "consequence"

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
        # which hazards the model called one and the same (agentic/same4.py)
        "same_hazards": [p for p in _list(_dict(s4.get("frame")).get("same_hazards"))
                         if isinstance(p, list)],
        "pre_intervention_trust": {"score": trust.get("score", 0.0),
                                   "level": trust.get("band", "unknown")},
        # the picture's size, so boxes of two runs can be compared as
        # fractions of the picture (agentic/align4.py)
        "image_size": rec.get("image_size") if isinstance(rec.get("image_size"), list) else None,
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


def after_result(before: Any, after_dir: Any) -> dict:
    """The edited run of a try, in the BEFORE run's names: two runs can
    hand out the names differently (the floating child is child_2 in one
    and child_1 in the other), so the same object is recognised by its
    place in the picture and renamed in a copy (agentic/align4.py). The
    copy carries `renamed` when anything was. Never raises on a bad run."""
    from agentic import align4                   # lazy: pure code
    after = result_from_run_dir(Path(after_dir)) if after_dir else {}
    return align4.aligned(before, after)


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
               core_basis: str = CORE_BASIS, core_rule: str = "half_max") -> dict:
    """The hazards that can be removed, ranked by the frozen arm."""
    iv = _frozen()
    baseline = iv.intervention_baseline(clean_result(before), None,
                                        gt_dir=_gt_dir(before, gt_dir))
    return iv.enumerate_candidates(baseline, core_basis=core_basis, core_rule=core_rule)


def _graph_ends(result: dict) -> tuple[set[str], set[str]]:
    """(sources, targets) over both graphs of a run; ids stripped of ' · state'."""
    src: set[str] = set()
    tgt: set[str] = set()
    for key in ("causal_graph", "graph_b"):
        for e in _list(_dict(result.get(key)).get("edges")):
            if isinstance(e, dict):
                a = str(e.get("source") or "").split("·")[0].strip()
                b = str(e.get("target") or "").split("·")[0].strip()
                if a:
                    src.add(a)
                if b:
                    tgt.add(b)
    return src, tgt


def hazards_of(before: Any) -> set[str]:
    """The things of a run that harm: its threats, the hazard each
    recommendation names, and every arrow's source that is not a person
    or an animal (a man blocking a road harms nobody). Never raises."""
    from agentic import chain4                   # lazy: chain4 imports this module
    r = clean_result(before)
    labels = {str(o.get("object_id") or ""): str(o.get("label") or "")
              for o in r["detected_objects"]}
    out = {str(t.get("object_id") or "") for t in r["threats"]}
    for rec in r["recommendations"]:
        q = rec.get("structured_reasoning")
        if isinstance(q, dict) and not rec.get("benign"):
            out.add(str(q.get("threat") or "").strip())
    for s in _graph_ends(r)[0]:
        try:
            if not chain4._is_life(s, labels):
                out.add(s)
        except Exception:
            pass
    return {h for h in out if h}


def victims_of(before: Any) -> list[dict]:
    """The things of a run that can be harmed and are not hazards
    themselves: whatever an arrow ends at in either graph, or a
    recommendation lists as affected. A person, an animal, a car, a house:
    anything (Sunny, 2026-11-20: "danger needs two ends... take away
    either end and the danger drops"). [{object_id, label, state}], in the
    order the run lists its objects. Never raises."""
    r = clean_result(before)
    ends = set(_graph_ends(r)[1])
    for rec in r["recommendations"]:
        q = rec.get("structured_reasoning")
        aff = q.get("affected_objects") if isinstance(q, dict) else None
        if not rec.get("benign"):
            ends |= {str(x).strip() for x in (aff if isinstance(aff, list) else [])
                     if isinstance(x, str)}
    haz = hazards_of(r)
    return [{"object_id": str(o.get("object_id")), "label": str(o.get("label") or ""),
             "state": str(o.get("state") or "")}
            for o in r["detected_objects"]
            if str(o.get("object_id") or "") in ends and str(o.get("object_id")) not in haz]


def is_victim(before: Any, oid: Any) -> bool:
    """True when `oid` is one of the run's victims (not a hazard)."""
    return any(v["object_id"] == str(oid or "") for v in victims_of(before))


def measure(before: dict, after: dict, target_oid: str, *,
            modality: str = "both", caption_changed: bool = True,
            image_changed: bool = True, gt_dir: Path | None = None,
            core_basis: str = CORE_BASIS, core_rule: str = "half_max") -> dict:
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
    if result.get("signals") is None and is_victim(before, target_oid):
        # The frozen arm lists only victims in a distress state. Any victim
        # can be taken out of the scene, so the same frozen measuring is
        # run for this one directly (its own `_run_one`, nothing rewritten).
        v = next(x for x in victims_of(before) if x["object_id"] == str(target_oid))
        cand = {**v, "variable_role": "victim", "outgoing_edge_count": 0,
                "edge_count": 0, "rank": 0}
        try:
            core = iv._run_one(baseline, cand,
                               {"target_object_id": v["object_id"], "modality": modality},
                               _after, role="core", core_basis=None, edit=edit)
            enum = result.get("candidates") or iv.enumerate_candidates(
                baseline, core_basis=core_basis, core_rule=core_rule)
            result = {**result, "spec": core.get("spec"), "u_check": core.get("u_check"),
                      "signals": core.get("signals"),
                      "verdict": iv.adjudicate_groundedness(core["spec"], core["signals"], enum),
                      "victim": True}
        except Exception:
            pass
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


def _our_reading(before: dict, after: dict, res: dict) -> dict:
    """The frozen signals of one try with the DIRECTION and SIZE replaced by
    our signed shift (agentic/chain4.py), and the frozen 2x2 cell worked out
    again from them by the frozen `adjudicate_groundedness`. Nothing of the
    frozen logic is rewritten: it is handed different inputs."""
    from agentic import chain4                  # lazy: chain4 imports this module
    signals = dict(_dict(res.get("signals")))
    try:
        sg = chain4.signed_shift(signals, chain4.direction(before, after))
    except Exception:
        return {"signals": signals, "verdict": res.get("verdict")}
    word = {"eased": "de-escalated", "grew": "escalated"}.get(sg["direction"], "unchanged")
    signals.update({"content_shift": abs(sg["value"]), "graph_direction": word,
                    "rec_direction": word, "signed_shift": sg["value"]})
    verdict = res.get("verdict")
    try:
        verdict = _frozen().adjudicate_groundedness(
            _dict(res.get("spec")), signals, _dict(res.get("candidates")))
    except Exception:
        pass
    return {"signals": signals, "verdict": verdict}


ROLES = ("core", "spurious")


def _words(label: Any) -> set[str]:
    return {w for w in str(label or "").strip().lower().replace("-", "_").split("_") if w}


def _gt_to_model(nodes: list, objs: list) -> dict[str, str]:
    """{ground-truth id: model id}, one to one. In order: the same id; the
    same label, when only one free object carries it; one label made of the
    other's words (tanker / tanker_truck), when only one fits; a person for
    a person, when one of each is left. A thing with no match is left out."""
    from agentic import chain4                   # lazy: chain4 imports this module
    free = [str(o.get("object_id") or "") for o in objs if o.get("object_id")]
    label = {str(o.get("object_id") or ""): str(o.get("label") or "") for o in objs}
    out: dict[str, str] = {}

    def life(oid: str, lab: dict) -> bool:
        try:
            return bool(chain4._is_life(oid, lab))
        except Exception:
            return False
    gt_label = {n["id"]: str(n.get("label") or "") for n in nodes}
    rules = (
        lambda g, m: g == m,
        lambda g, m: bool(_words(gt_label[g])) and _words(gt_label[g]) == _words(label[m]),
        lambda g, m: bool(_words(gt_label[g])) and bool(_words(label[m])) and (
            _words(gt_label[g]) <= _words(label[m]) or _words(label[m]) <= _words(gt_label[g])),
        lambda g, m: life(g, gt_label) and life(m, label),
    )
    for rule in rules:
        for n in nodes:
            g = n["id"]
            if g in out:
                continue
            fits = [m for m in free if m not in out.values() and rule(g, m)]
            # as many ground-truth things could claim the same object: only a
            # one-to-one fit is taken
            rivals = [x["id"] for x in nodes if x["id"] not in out
                      and any(rule(x["id"], m) for m in fits)]
            if len(fits) == 1 and rivals == [g]:
                out[g] = fits[0]
    return out


def gt_objects(before: Any) -> dict:
    """What the ground-truth file of the before run says, in the MODEL's ids
    (Sunny, 2026-11-20: "every recognizable instance should be either core
    or spurious"):

      "listed"        every thing the human listed
      "roles"         {id: "core" | "spurious"}. The human's word (`role` on
                      the node) wins; without it: a thing with an arrow out
                      of it, or a person, is core, anything else spurious
                      (a thing can have an arrow INTO it and still be
                      spurious: the shed the fire may reach)
      "same_hazards"  the pairs the human calls one and the same hazard;
                      None when the file does not answer the question
      "graph"         the ground-truth graph ({"nodes", "edges"}); None
                      when there is no readable file

    Dev-side only: this never reaches a prompt. Never raises."""
    out: dict = {"listed": [], "same_hazards": None, "roles": {}, "graph": None}
    gt = _dict(_dict(before).get("gt"))
    try:
        raw = json.loads((Path(str(gt.get("dir"))) /
                          f"{gt.get('image_filename')}.gt.json").read_text())
    except Exception:
        return out
    raw = _dict(raw)
    objs = [o for o in _list(_dict(before).get("detected_objects")) if isinstance(o, dict)]
    nodes = [n for n in _list(raw.get("nodes"))
             if isinstance(n, dict) and isinstance(n.get("id"), str) and n["id"]]
    try:
        to_model = _gt_to_model(nodes, objs)
    except Exception:
        to_model = {}
    out["listed"] = sorted(set(to_model.values()))
    edges = []
    for e in _list(raw.get("edges")):
        if isinstance(e, dict):
            a, b = to_model.get(str(e.get("source"))), to_model.get(str(e.get("target")))
            if a and b:
                edges.append({**e, "source": a, "target": b})
    sources = {e["source"] for e in edges}
    g_nodes = []
    for n in nodes:
        mid = to_model.get(n["id"])
        if not mid:
            continue
        g_nodes.append({**n, "id": mid})
        said = str(n.get("role") or "").strip().lower()
        if said in ROLES:
            out["roles"][mid] = said
        else:
            try:
                from agentic import chain4
                person = bool(chain4._is_life(mid, {mid: str(n.get("label") or "")}))
            except Exception:
                person = False
            out["roles"][mid] = "core" if (mid in sources or person) else "spurious"
    out["graph"] = {"nodes": g_nodes, "edges": edges}
    if isinstance(raw.get("same_hazards"), list):
        pairs = []
        for p in raw["same_hazards"]:
            if (isinstance(p, (list, tuple)) and len(p) == 2
                    and all(isinstance(x, str) for x in p)):
                a, b = to_model.get(p[0]), to_model.get(p[1])
                if a and b and a != b:
                    pairs.append(sorted([a, b]))
        out["same_hazards"] = pairs
    return out


def same_pairs_for_gt(before: Any, extra: Any = None) -> list:
    """The same-hazard pairs the ground-truth comparison goes by: the
    human's when the file answers the question, else the model's own."""
    extra = extra if isinstance(extra, dict) else gt_objects(before)
    if isinstance(extra.get("same_hazards"), list):
        return extra["same_hazards"]
    from agentic import same4                    # lazy: same4 is pure code
    return same4.pairs_of(before)


def counted_as(target: str, after: Any, gt_hazards: Any, pairs: Any) -> str:
    """The hazard a try counts as against ground truth. A try is saved under
    the name that was picked; when ground truth does not have that name as a
    hazard but calls it the same hazard as one it does have, and that one
    went too, the try counts as that hazard (C_tanker ui_2345d3fc: the fire
    was removed, saved as brush_1, and read as a mistake). Never raises."""
    target = str(target or "")
    gt_hazards = {str(x) for x in (gt_hazards if isinstance(gt_hazards, (set, list, tuple))
                                   else [])}
    if not target or target in gt_hazards:
        return target
    from agentic import same4
    after = _dict(after)
    still = {str(t.get("object_id") or "") for t in _list(after.get("threats"))
             if isinstance(t, dict)}
    for m in sorted(same4.groups(pairs if isinstance(pairs, list) else []).get(target, ())):
        if m != target and m in gt_hazards and m not in still:
            return m
    return target


def synthesis(before: dict, tries: list[tuple[dict, dict]], *,
              core_basis: str = CORE_BASIS, core_rule: str = "half_max") -> dict:
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
    frozen_history = []
    extra = gt_objects(before)
    gt_pairs = same_pairs_for_gt(before, extra)
    try:
        gt_haz = {str(c.get("object_id")) for c in
                  _list(_dict(candidates(before, core_basis=core_basis,
                                         core_rule=core_rule)).get("candidates"))
                  if isinstance(c, dict) and c.get("is_gt_hazard")}
    except Exception:
        gt_haz = set()
    out["gt_listed"], out["gt_same_hazards"] = extra["listed"], extra["same_hazards"]
    out["gt_extra"] = extra
    try:
        # the hazards ground truth gives a weight of their own, by the same
        # method as the declared side (agentic/declared4.py)
        from agentic import declared4
        own = {h for h, w in declared4.should_matter(extra).items() if w > 0}
        gt_haz = own or gt_haz
    except Exception:
        pass
    for one, after in tries or []:
        one = _dict(one)
        picked = str(one.get("target") or "")
        one = {**one, "target": counted_as(picked, after, gt_haz, gt_pairs)}
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
        entry = {
            "selection": {"target_object_id": one.get("target")},
            "spec": res.get("spec"), "u_check": res.get("u_check"),
            "verdict": res.get("verdict"), "signals": res.get("signals"),
            "before": {"recommendations": clean_before["recommendations"],
                       "graph_a": clean_before["causal_graph"] or {}}}
        frozen_history.append(entry)
        # OUR reading of the same try (the signed shift): the frozen labels
        # are then computed by the frozen code from it, so the whole screen
        # speaks one reading (Sunny, 2026-11-13: the table said "eased" and
        # the hazard list said "escalated" for the same try).
        ours = _our_reading(before, after, res)
        history.append({**entry, "signals": ours["signals"], "verdict": ours["verdict"]})
        out["tries"].append({"index": one.get("index"), "target": one.get("target"),
                             "picked": picked,
                             "cell": _dict(ours["verdict"]).get("cell"),
                             "frozen_cell": _dict(res.get("verdict")).get("cell")})
    try:
        _frozen()
        import main
        syn = main._synthesis_cell(clean_before, {"tries": history},
                                   core_basis=core_basis, core_rule=core_rule)
        dist = main._distributional_groundedness(syn)
    except Exception as exc:
        out["why"] = f"the frozen synthesis could not run: {exc}"
        return out
    try:
        f_syn = main._synthesis_cell(clean_before, {"tries": frozen_history},
                                     core_basis=core_basis, core_rule=core_rule)
        f_dist = main._distributional_groundedness(f_syn)
        out["frozen"] = {
            "cell": f_syn.get("cell"), "distribution": f_dist.get("verdict"),
            "hazards": {str(h.get("object_id")): h.get("status")
                        for h in (f_syn.get("hazard_status") or []) if isinstance(h, dict)}}
    except Exception:
        out["frozen"] = {}
    out.update({
        "available": True, "core_basis": core_basis, "cell": syn.get("cell"),
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
