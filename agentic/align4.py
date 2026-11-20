"""The same object under the same name in the before run and the edited run.

Each run names its objects by itself, so two runs of nearly the same
picture can hand out the names differently. B_pool ui_b3ada51c (Sunny,
2026-11-20): the struggling child was child_1 and the floating child
child_2; in the edited run the floating child was child_1 and the child
on the deck child_2. Everything that compares the two runs by name then
compared each recommendation with the wrong child, and the table read
"every recommendation dropped".

The rule, in code, with no model call:

    two objects of the same kind (label) that sit in the same place in
    the two pictures are the same object, whatever they are called.

"The same place" = their boxes overlap by at least half (IoU >= 0.5).
An edit changes one thing, so everything else is where it was. What is
left over is paired only when exactly one object of that kind is left on
each side and the two runs have the same number of that kind (the one
that was moved). A "removed" object therefore never takes the name of a
newcomer.

The edited run is then RE-NAMED to the before run's names in a copy: the
objects, the threats, the quads, both graphs, and the ids inside each
action's text. What the model wrote is kept beside it
(`action_as_written`), and the copy carries `renamed` ({its name: the
name used here}) so the screen can say so. Nothing is silent.

Pure code. Never raises on malformed input.
"""
from __future__ import annotations

import copy
import re
from typing import Any

IOU_SAME = 0.5


def _box(o: Any, size: Any = None) -> Any:
    """[x0, y0, x1, y1] as fractions of the picture when its size is known."""
    b = o.get("bbox") if isinstance(o, dict) else None
    if not (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)):
        return None
    x0, y0, x1, y1 = (float(v) for v in b)
    if x1 <= x0 or y1 <= y0:
        return None
    if (isinstance(size, (list, tuple)) and len(size) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
                    for v in size)):
        w, h = float(size[0]), float(size[1])
        return [x0 / w, y0 / h, x1 / w, y1 / h]
    return [x0, y0, x1, y1]


def iou(a: Any, b: Any) -> float:
    """Overlap of two boxes, 0..1 (shared area over joined area)."""
    if not a or not b:
        return 0.0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _objects(result: Any) -> list[dict]:
    objs = result.get("detected_objects") if isinstance(result, dict) else None
    return [o for o in (objs if isinstance(objs, list) else [])
            if isinstance(o, dict) and isinstance(o.get("object_id"), str) and o["object_id"]]


SAME_PLACE, BY_LOOKS, ONE_LEFT = "same place", "by looks", "the only one left"
ONLY_ONE = "the only one of its kind"
LOOKS_SAME = 0.6            # shared part of two colour histograms, 0..1
IOU_OTHER_LABEL = 0.8       # two labels of one family (person / man): only when nearly the same box


def _kind(o: dict) -> str:
    """What two objects must share to be the same object: the family
    (a "person" and a "man" are both people), else the label."""
    return str(o.get("family") or o.get("label") or "")


def _looks(result: Any, o: dict) -> Any:
    """A small colour histogram of one object's own pixels (its outline when
    one was saved, else its box), or None when the picture cannot be read.
    No model: the framework only has to tell two cut-outs apart."""
    try:
        import numpy as np
        from PIL import Image
        from pathlib import Path
        folder = result.get("run_dir") or (Path(str(o.get("mask_path"))).parent
                                           if o.get("mask_path") else None)
        name = result.get("scene_image") or result.get("image_filename")
        box = o.get("bbox")
        if not folder or not name or not _box(o):
            return None
        img = Image.open(Path(str(folder)) / str(name)).convert("RGB")
        x0, y0, x1, y1 = (int(v) for v in box)
        px = np.asarray(img.crop((x0, y0, x1, y1)), dtype=np.uint8).reshape(-1, 3)
        if o.get("mask_path"):
            try:
                m = np.asarray(Image.open(str(o["mask_path"])).convert("L").crop(
                    (x0, y0, x1, y1))).reshape(-1) > 127
                if m.sum() >= 16:
                    px = px[m]
            except Exception:
                pass
        if len(px) < 16:
            return None
        idx = (px[:, 0] // 32).astype(int) * 64 + (px[:, 1] // 32).astype(int) * 8 + (px[:, 2] // 32)
        h = np.bincount(idx, minlength=512).astype(float)
        return h / h.sum()
    except Exception:
        return None


def _alike(h1: Any, h2: Any) -> float:
    if h1 is None or h2 is None:
        return 0.0
    try:
        import numpy as np
        return float(np.minimum(h1, h2).sum())
    except Exception:
        return 0.0


def tracks(before: Any, after: Any) -> dict:
    """Which object of the edited run is which object of the before run.

    {"pairs": [{"before", "after", "how", "sure", "score"}],
     "new": [ids only the edited run has], "gone": [ids only the before run has]}

    In this order, one to one, within one kind (family):
      same place          the boxes overlap by half or more: sure
      by looks            what is left over, when the two cut-outs have much
                          the same colours (something that was moved): sure
      the only one left   one of that kind left on each side, and as many of
                          that kind before as after: a guess, NOT sure
    The framework does this after both runs are over; the model never sees
    it (Sunny, 2026-11-20: "the intervention run has to be independent")."""
    out: dict = {"pairs": [], "new": [], "gone": []}
    b_objs, a_objs = _objects(before), _objects(after)
    if not b_objs or not a_objs:
        out["new"] = [o["object_id"] for o in a_objs]
        out["gone"] = [o["object_id"] for o in b_objs]
        return out
    b_size = before.get("image_size") if isinstance(before, dict) else None
    a_size = after.get("image_size") if isinstance(after, dict) else None
    if not (b_size and a_size):
        b_size = a_size = None                  # compare raw pixels, both or neither
    used_a: set[str] = set()
    used_b: set[str] = set()
    for kind in sorted({_kind(o) for o in a_objs}):
        bs = [o for o in b_objs if _kind(o) == kind]
        as_ = [o for o in a_objs if _kind(o) == kind]
        if not bs or not as_:
            continue
        placed = 0
        # first the same label; then another label of the same family, and
        # that only when the two boxes are nearly the same (a "person" who
        # is a "man" in the other run; not a tree standing where brush was)
        for same_label, least in ((True, IOU_SAME), (False, IOU_OTHER_LABEL)):
            scored = sorted(((iou(_box(x, a_size), _box(y, b_size)),
                              x["object_id"] == y["object_id"], x["object_id"], y["object_id"])
                             for x in as_ for y in bs
                             if (x.get("label") == y.get("label")) == same_label
                             and x["object_id"] not in used_a and y["object_id"] not in used_b),
                            reverse=True)
            for sc, _n, a_id, b_id in scored:
                if sc < least:
                    break
                if a_id in used_a or b_id in used_b:
                    continue
                used_a.add(a_id)
                used_b.add(b_id)
                placed += 1
                out["pairs"].append({"before": b_id, "after": a_id, "how": SAME_PLACE,
                                     "sure": True, "score": round(sc, 3)})
        # one thing of a label in each run, under the same name: the same
        # thing, though its box changed (smoke, a fire, burning brush never
        # keep their shape)
        for lab in sorted({str(o.get("label") or "") for o in as_}):
            la = [o for o in as_ if str(o.get("label") or "") == lab]
            lb = [o for o in bs if str(o.get("label") or "") == lab]
            if (len(la) == 1 and len(lb) == 1 and la[0]["object_id"] == lb[0]["object_id"]
                    and la[0]["object_id"] not in used_a and lb[0]["object_id"] not in used_b):
                used_a.add(la[0]["object_id"])
                used_b.add(lb[0]["object_id"])
                out["pairs"].append({"before": lb[0]["object_id"], "after": la[0]["object_id"],
                                     "how": ONLY_ONE, "sure": True, "score": None})
        left_a = [o for o in as_ if o["object_id"] not in used_a]
        left_b = [o for o in bs if o["object_id"] not in used_b]
        if left_a and left_b:
            looks = sorted(((_alike(_looks(after, x), _looks(before, y)),
                             x["object_id"], y["object_id"]) for x in left_a for y in left_b
                            if x.get("label") == y.get("label")), reverse=True)
            for sc, a_id, b_id in looks:
                if sc < LOOKS_SAME:
                    break
                if a_id in used_a or b_id in used_b:
                    continue
                used_a.add(a_id)
                used_b.add(b_id)
                out["pairs"].append({"before": b_id, "after": a_id, "how": BY_LOOKS,
                                     "sure": True, "score": round(sc, 3)})
            left_a = [o for o in left_a if o["object_id"] not in used_a]
            left_b = [o for o in left_b if o["object_id"] not in used_b]
        # the one that was moved and no longer looks the same: a guess
        if len(left_a) == 1 and len(left_b) == 1 and len(as_) == len(bs) and placed:
            a_id, b_id = left_a[0]["object_id"], left_b[0]["object_id"]
            used_a.add(a_id)
            used_b.add(b_id)
            out["pairs"].append({"before": b_id, "after": a_id, "how": ONE_LEFT,
                                 "sure": False, "score": None})
    out["new"] = [o["object_id"] for o in a_objs if o["object_id"] not in used_a]
    out["gone"] = [o["object_id"] for o in b_objs if o["object_id"] not in used_b]
    return out


def same_place(before: Any, after: Any, found: Any = None) -> dict[str, str]:
    """{name in the edited run: name in the before run}, only where the two
    differ. {} when every name already agrees or nothing can be matched."""
    b_objs, a_objs = _objects(before), _objects(after)
    if not b_objs or not a_objs:
        return {}
    found = found if isinstance(found, dict) else tracks(before, after)
    pairs = {p["after"]: p["before"] for p in found["pairs"]}
    out = {a: b for a, b in pairs.items() if a != b}
    if not out:
        return {}
    # an object of the edited run that keeps its own name must not end up
    # sharing it with one that was just given that name: it gets a free one
    taken = set(out.values()) | {o["object_id"] for o in b_objs}
    for o in a_objs:
        oid = o["object_id"]
        if oid in out or oid in pairs:
            continue
        if oid in set(out.values()):
            stem = re.sub(r"_\d+$", "", oid) or oid
            n = 1
            while f"{stem}_{n}" in taken or f"{stem}_{n}" in {x["object_id"] for x in a_objs}:
                n += 1
            out[oid] = f"{stem}_{n}"
            taken.add(out[oid])
    return out


def _swap(text: Any, mapping: dict[str, str]) -> Any:
    """Every whole-word name in `text` replaced at once (a swap stays a swap)."""
    if not isinstance(text, str) or not mapping:
        return text
    pat = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(
        re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?![A-Za-z0-9_])")
    return pat.sub(lambda m: mapping[m.group(1)], text)


def rename(after: Any, mapping: Any) -> Any:
    """A copy of the edited run with its names replaced. The model's own
    words are kept as `action_as_written`; the copy carries `renamed`."""
    if not isinstance(after, dict) or not isinstance(mapping, dict) or not mapping:
        return after
    m = {str(k): str(v) for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str)}
    if not m:
        return after
    out = copy.deepcopy(after)

    def one(x: Any) -> Any:
        return _swap(x, m)
    for key in ("detected_objects", "threats"):
        for o in out.get(key) if isinstance(out.get(key), list) else []:
            if isinstance(o, dict) and "object_id" in o:
                o["object_id"] = one(o["object_id"])
    for r in out.get("recommendations") if isinstance(out.get("recommendations"), list) else []:
        if not isinstance(r, dict):
            continue
        if isinstance(r.get("action"), str) and one(r["action"]) != r["action"]:
            r["action_as_written"] = r["action"]
            r["action"] = one(r["action"])
        q = r.get("structured_reasoning")
        if isinstance(q, dict):
            if "threat" in q:
                q["threat"] = one(q["threat"])
            if isinstance(q.get("affected_objects"), list):
                q["affected_objects"] = [one(x) for x in q["affected_objects"]]
    for key in ("causal_graph", "graph_b"):
        g = out.get(key)
        if not isinstance(g, dict):
            continue
        for n in g.get("nodes") if isinstance(g.get("nodes"), list) else []:
            if isinstance(n, dict) and "id" in n:
                n["id"] = one(n["id"])
        for e in g.get("edges") if isinstance(g.get("edges"), list) else []:
            if isinstance(e, dict):
                for end in ("source", "target"):
                    if end in e:
                        e[end] = one(e[end])
        for c in (g.get("intervention_candidates")
                  if isinstance(g.get("intervention_candidates"), list) else []):
            if isinstance(c, dict) and "object_id" in c:
                c["object_id"] = one(c["object_id"])
    if isinstance(out.get("same_hazards"), list):
        out["same_hazards"] = [[one(x) for x in p] if isinstance(p, list) else p
                               for p in out["same_hazards"]]
    out["renamed"] = m
    return out


def aligned(before: Any, after: Any) -> Any:
    """The edited run in the before run's names (always a copy, so the saved
    run is never touched). It carries:
      `tracks`       who is who (see `tracks`), in the BEFORE run's names
      `unsure`       the before-run names whose match is a guess
      `renamed`      {its name: the name used here}, when any differ
      `renamed_new`  the names given to objects the before run does not have
    A run with no objects to match is returned as it is."""
    try:
        if not _objects(before) or not _objects(after):
            return after
        found = tracks(before, after)
        m = same_place(before, after, found)
        out = rename(after, m) if m else copy.deepcopy(after)
        known = {o["object_id"] for o in _objects(before)}
        out["renamed_new"] = {a: b for a, b in m.items() if b not in known}
        out["tracks"] = {"pairs": found["pairs"], "gone": found["gone"],
                         "new": [m.get(x, x) for x in found["new"]]}
        out["unsure"] = sorted(p["before"] for p in found["pairs"] if not p["sure"])
        return out
    except Exception:
        return after


def summary(found: Any) -> dict:
    """{"same place": n, "by looks": n, "not sure": n, "new": n, "gone": n}."""
    found = found if isinstance(found, dict) else {}
    pairs = [p for p in (found.get("pairs") if isinstance(found.get("pairs"), list) else [])
             if isinstance(p, dict)]
    return {"same place": sum(1 for p in pairs if p.get("how") in (SAME_PLACE, ONLY_ONE)),
            "by looks": sum(1 for p in pairs if p.get("how") == BY_LOOKS),
            "not sure": sum(1 for p in pairs if not p.get("sure")),
            "new": len(found.get("new") or []) if isinstance(found.get("new"), list) else 0,
            "gone": len(found.get("gone") or []) if isinstance(found.get("gone"), list) else 0}


def note(mapping: Any, new: Any = None) -> str:
    """'matched by its place in the picture: in the edited run the model
    calls child_2 "child_1" and child_1 "child_2"' ('' when nothing). An
    object the before run does not have is said apart: 'its "tree_1" is
    another object, shown as tree_3'."""
    if not isinstance(mapping, dict) or not mapping:
        return ""
    new = new if isinstance(new, dict) else {}
    same = [f'{here} "{its}"' for its, here in sorted(mapping.items(), key=lambda kv: str(kv[1]))
            if isinstance(its, str) and isinstance(here, str) and its not in new]
    other = [f'its "{its}" is another object, shown as {here}'
             for its, here in sorted(new.items(), key=lambda kv: str(kv[1]))
             if isinstance(its, str) and isinstance(here, str)]
    parts = []
    if same:
        parts.append("in the edited run the model calls "
                     + (same[0] if len(same) == 1 else ", ".join(same[:-1]) + " and " + same[-1]))
    parts += other
    return ("matched by its place in the picture: " + "; ".join(parts)) if parts else ""
