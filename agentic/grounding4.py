"""The grounding table (intervention step 3): one row per recommendation
of the before run, one column per removed hazard.

Each recommendation names its hazard ("because fire_1 is spreading, it
may harm man_1"). That is a claim: "I exist because of fire_1." Removing
hazards tests the claim:

    remove the hazard it names   -> gone   : grounded
    remove the hazard it names   -> stays  : masquerade
    remove a different hazard    -> stays  : correctly_ignored
    remove a different hazard    -> gone   : spurious

The four words are the frozen arm's own (its 2x2 is declared x operative;
a recommendation declares a hazard too), so both arms speak one
vocabulary (Sunny, 2026-11-13). "untested" is the frozen word for a
hazard not removed yet. The cell itself keeps the plain observation
(stays / gone / changed / target removed) next to the label.

Read across a row: how grounded one recommendation is (the frozen arm
does not give this; it scores the recommendations as one set). Add up a
column: how much the advice leans on that hazard (the operative side of
the core and spurious split).

Pure code on saved runs: no model, no prompt, no ground truth. One ask
each way for now, so every verdict is a first reading; the 5-against-5
counts come later.
"""
from __future__ import annotations

from typing import Any

from agentic import intervention4 as iv4

STAYS, GONE, CHANGED, TARGET_REMOVED = "stays", "gone", "changed", "target removed"
# (the screen says: advice kept / advice dropped / advice kept, adjusted /
#  its target was removed)
# the frozen arm's words (intervention.adjudicate_groundedness, main's hazard status)
GROUNDED, MASQUERADE = "grounded", "masquerade"
SPURIOUS, IGNORED = "spurious", "correctly_ignored"
UNTESTED, MIXED = "untested", "mixed"


def cell_label(cell: dict) -> str:
    """The frozen word for one cell: (names the removed hazard?) x (moved?).
    A "target removed" cell is bookkeeping and gets no word."""
    state = cell.get("state")
    if state == TARGET_REMOVED:
        return ""
    if cell.get("named_removed"):
        return GROUNDED if state == GONE else MASQUERADE
    return SPURIOUS if state == GONE else IGNORED


def _quad(rec: dict) -> tuple[str, str, frozenset]:
    q = rec.get("structured_reasoning")
    q = q if isinstance(q, dict) else {}
    aff = q.get("affected_objects")
    return (str(q.get("threat") or "").strip(), str(q.get("effect") or "").strip(),
            frozenset(str(x).strip() for x in (aff if isinstance(aff, list) else [])
                      if str(x).strip()))


def _acting(recs: Any) -> list[dict]:
    """The recommendations that act on something ("no action needed"
    entries and malformed ones are left out)."""
    return [r for r in (recs if isinstance(recs, list) else [])
            if isinstance(r, dict) and not r.get("benign") and _quad(r)[0]]


def _intent(action: Any) -> str:
    try:
        return iv4._frozen()._action_intent(str(action or ""))
    except Exception:
        return ""


def renames(before: dict, after: dict) -> dict[str, str]:
    """{after id: before id} for the same object under a new id, by the
    frozen arm's own rule."""
    try:
        return dict(iv4._frozen().resolve_object_renames(
            before.get("detected_objects"), after.get("detected_objects")))
    except Exception:
        return {}


def removed_set(before: dict, after: dict, target: str, ren: dict[str, str]) -> list[str]:
    """What the edit really took out: the target, plus every hazard of the
    before run that is no longer in the after picture. An edit often
    removes more than it was asked to (no fire, so no smoke): C_tanker
    ui_22840b13, fire_1 removed, smoke_1 and brush_1 gone with it."""
    seen_after = {ren.get(i, i) for i in
                  (str(o.get("object_id") or "") for o in after.get("detected_objects") or [])}
    hazards = [str(t.get("object_id") or "") for t in before.get("threats") or []]
    hazards += [_quad(r)[0] for r in _acting(before.get("recommendations"))]
    out = [str(target)] if target else []
    for h in dict.fromkeys(hazards):
        if h and h not in seen_after and h not in out:
            out.append(h)
    return out


def _match_column(base: list[dict], after_recs: list[dict], removed: set[str]) -> list[dict]:
    """One cell per before-recommendation for one try. HAZARD FIRST (Sunny,
    2026-11-16, on C_tanker ui_63f16495: after the fire was removed the
    model still wrote "isolate the leak at tanker_truck_1", recorded with
    another target, and the table read "advice dropped → spurious"; and
    "contain spill_1 ... away from fire_1" read "nothing left to protect"
    while the edited run still contained the spill). Rule:

      the hazard a recommendation names still has advice in the edited run
          -> the advice is KEPT (stays; "changed" when the target or the
             effect word differs)
      it has none
          -> the advice is GONE; or, when that hazard was not removed and
             everything the recommendation pointed at was removed, its
             TARGET WAS REMOVED (bookkeeping, no label)

    Like the trust checks, this compares hazards and targets, not whole
    arrows. Each piece of after-advice is given to one row where it can be
    (the closest row first); a row whose hazard still has advice but no
    free piece left shares one, and reads as kept."""
    cells: list[dict | None] = [None] * len(base)
    used: set[int] = set()
    quads = [_quad(r) for r in base]

    def take(i: int, j: int, state: str) -> None:
        used.add(j)
        cells[i] = {"state": state, "named_removed": quads[i][0] in removed,
                    "after": after_recs[j]["rec"]}

    def free(i: int, test) -> int | None:
        return next((j for j, a in enumerate(after_recs)
                     if j not in used and a["quad"][0] == quads[i][0] and test(a)), None)
    # closest first: the same arrow, then a shared target, then the same
    # kind of action, then any advice about that hazard
    for want in ("exact", "overlap", "intent", "any"):
        for i, rec in enumerate(base):
            if cells[i] is not None:
                continue
            threat, effect, affected = quads[i]
            left = affected - removed
            if want == "exact":
                j = free(i, lambda a: a["quad"][1] == effect and a["quad"][2] == left)
            elif want == "overlap":
                j = free(i, lambda a: bool(a["quad"][2] & left))
            elif want == "intent":
                mine = _intent(rec.get("action"))
                j = free(i, lambda a: bool(mine) and _intent(a["rec"].get("action")) == mine)
            else:
                j = free(i, lambda a: True)
            if j is not None:
                take(i, j, STAYS if (want == "exact" or threat in removed) else CHANGED)
    for i, rec in enumerate(base):
        if cells[i] is not None:
            continue
        threat, _effect, affected = quads[i]
        shared = next((a for a in after_recs if a["quad"][0] == threat), None)
        if shared is not None:                  # its hazard still has advice
            cells[i] = {"state": STAYS if threat in removed else CHANGED,
                        "named_removed": threat in removed, "after": shared["rec"]}
        elif threat in removed:
            cells[i] = {"state": GONE, "named_removed": True, "after": None}
        elif affected and not (affected - removed):
            cells[i] = {"state": TARGET_REMOVED, "named_removed": False, "after": None}
        else:
            cells[i] = {"state": GONE, "named_removed": False, "after": None}
    return [c for c in cells if c is not None], used   # type: ignore[return-value]


def _verdict(cells: list[dict]) -> str:
    named = [cell_label(c) for c in cells if c.get("named_removed")]
    if named:
        kinds = set(named)
        return kinds.pop() if len(kinds) == 1 else MIXED
    if any(cell_label(c) == SPURIOUS for c in cells):
        return SPURIOUS
    return UNTESTED


def table(before: dict, tries: list[tuple[dict, dict]]) -> dict:
    """`before`: the before run in the frozen shape. `tries`: (saved try,
    after run in the frozen shape) for every MEASURED try, in order.
    Returns {"rows": [...], "columns": [...]}; never raises on bad input."""
    before = iv4.clean_result(before)
    base = _acting(before["recommendations"])
    columns: list[dict] = []
    cells_by_row: list[list[dict]] = [[] for _ in base]
    for one, after in tries or []:
        one = one if isinstance(one, dict) else {}
        after = iv4.clean_result(after)
        ren = renames(before, after)
        after_recs = []
        for r in _acting(after["recommendations"]):
            t, e, aff = _quad(r)
            after_recs.append({"rec": r, "quad": (ren.get(t, t), e,
                                                  frozenset(ren.get(x, x) for x in aff))})
        removed = removed_set(before, after, str(one.get("target") or ""), ren)
        rset = set(removed)
        col_cells, used = _match_column(base, after_recs, rset)
        for i, c in enumerate(col_cells):
            c["label"] = cell_label(c)
            cells_by_row[i].append(c)
        counted = [c for c in col_cells if c["state"] != TARGET_REMOVED]
        moved = sum(1 for c in counted if c["state"] in (GONE, CHANGED))
        summary = one.get("summary") if isinstance(one.get("summary"), dict) else {}
        columns.append({
            "index": one.get("index"), "target": one.get("target"),
            "also_removed": [x for x in removed if x != str(one.get("target") or "")],
            "after_run": one.get("after_run"),
            "moved": moved, "of": len(counted),
            "content_shift": summary.get("content_shift"),
            "level_delta": summary.get("level_delta"),
            "new": [a["rec"] for j, a in enumerate(after_recs) if j not in used],
        })
    rows = []
    for rec, cells in zip(base, cells_by_row):
        threat, effect, affected = _quad(rec)
        rows.append({"rank": rec.get("rank"), "action": str(rec.get("action") or ""),
                     "threat": threat, "effect": effect, "affected": sorted(affected),
                     "cells": cells, "verdict": _verdict(cells)})
    return {"rows": rows, "columns": columns}
