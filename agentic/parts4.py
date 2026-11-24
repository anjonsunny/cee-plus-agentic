"""The parts of a try's shift, each with HOW MUCH and WHICH DIRECTION
(Sunny, 2026-11-24).

One row of the feature table is one thing we tried. Its shift is the
average of five parts, and every part answers the same question: after
this one intervention, did the model's sense of danger ease or escalate,
and by how much?

    part                    how much                        direction
    danger level            the level change / 10           down = eased
    recommendations         (action kind, who) pairs that   the urgency totals
                            differ (frozen arm)             (frozen arm)
    graph                   arrows that differ, no state    the weight of the arrows
    state                   threats + at-risk whose state   the severity of the
                            changed                         state words (SEVERITY)
    the tried object itself what the model said about IT    its four lines

The fifth part reads what the model said about the tried object in each
run on its own (the model never sees both pictures; the comparison is
ours):

    a victim                              a hazard
    is it in the at-risk list, which kind is it in the threats list
    which hazards threaten it             who it threatens
    the recommendations that protect it   the recommendations that act on it
    its state                             its state

It is computed only for a try of the kind "dealt with": the thing is still
in the picture, so what the model says about it is the model's choice.
For "removed" every line would read "gone", which our edit forces; the
part is left out and the shift is the average of the other four.

A change with no up or down (the same number of arrows, to other people)
is REDIRECTED; a changed state word with no known severity is CHANGED.
Both add to how much, neither to eased or escalated.

Pure code on saved runs. No model, no prompt, no ground truth.
"""
from __future__ import annotations

from typing import Any

from agentic import chain4, intervention4 as iv4

EASED, ESCALATED = "eased", "escalated"
REDIRECTED, CHANGED, UNCHANGED = "redirected", "changed", "unchanged"
DIRECTIONS = (EASED, ESCALATED, REDIRECTED, CHANGED, UNCHANGED)
ITSELF = "itself"                         # the name of the fifth part
DEALT_WITH = "dealt with"

# How severe a state word is, 0 (normal) to 3 (worst). Hand-made, a prior:
# it only ever compares two states of the SAME object (burning -> burnt),
# never a fire with a person. Every word perception may use is here
# (test_parts4 checks it); a word that is not gets no direction.
SEVERITY: dict[str, int] = {
    # normal
    **{w: 0 for w in (
        "contained", "disengaged", "dissipating", "dry", "healthy", "intact", "relaxed",
        "resting", "sealed", "stable", "standing", "stationary", "steady", "unarmed",
        "uninjured", "upright", "whole", "driving", "parked", "paved", "running",
        "swimming", "walking")},
    # the danger is over or only possible
    "burnt": 1, "fallen": 1, "hazardous_in_context": 1, "cowering": 1, "fleeing": 1,
    # a danger that is there, not at its worst
    "billowing": 2, "seeping": 2, "rising": 2, "collapsed": 2, "crushed": 2, "flooded": 2,
    "armed": 2, "approaching": 2, "coiled": 2, "chemical_spill": 2, "injured": 2,
    "struggling": 2,
    # at its worst
    "burning": 3, "engulfing": 3, "spreading": 3, "escalating": 3, "leaking": 3,
    "collapsing": 3, "striking": 3, "charging": 3, "aiming": 3, "rabid": 3,
    "bleeding": 3, "trapped": 3, "drowning": 3, "suffocating": 3, "unconscious": 3,
}
KIND_RANK = {"proximity": 1, "distress": 2}       # an at-risk kind: distress is worse


def _num(x: Any) -> Any:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def severity(word: Any) -> Any:
    """0..3 for a state word, None when it is not on the list."""
    w = str(word or "").strip().lower().replace(" ", "_")
    return SEVERITY.get(w)


def state_direction(was: Any, now: Any) -> str:
    """One object's state, before against after. `now` None = it is not
    listed after: nothing there to harm or be harmed, eased."""
    if was is None:
        return UNCHANGED
    if now is None:
        return EASED
    a, b = str(was).strip().lower(), str(now).strip().lower()
    if a == b:
        return UNCHANGED
    sa, sb = severity(a), severity(b)
    if sa is None or sb is None or sa == sb:
        return CHANGED
    return EASED if sb < sa else ESCALATED


def vote(items: Any) -> str:
    """The direction several sized items point to together: each eased one
    pulls one way by its size, each escalated one the other way. No pull
    either way but something changed: redirected."""
    net = total = 0.0
    for it in items if isinstance(items, (list, tuple)) else []:
        if not isinstance(it, dict):
            continue
        s = _num(it.get("size"))
        if not s:
            continue
        total += s
        net += s if it.get("direction") == EASED else -s if it.get("direction") == ESCALATED else 0.0
    if net > 1e-9:
        return EASED
    if net < -1e-9:
        return ESCALATED
    return REDIRECTED if total > 0 else UNCHANGED


def part(name: str, size: Any, direction: str, **more: Any) -> dict:
    """One part: its size 0..1, its direction, and `signed` (plus = eased,
    minus = escalated, 0 = no direction) for the older readers."""
    s = max(0.0, min(1.0, _num(size) or 0.0))
    if s <= 0:
        direction = UNCHANGED
    elif direction not in (EASED, ESCALATED, REDIRECTED, CHANGED):
        direction = REDIRECTED
    sign = 1.0 if direction == EASED else -1.0 if direction == ESCALATED else 0.0
    return {"name": name, "size": round(s, 3), "direction": direction,
            "signed": round(s * sign, 3), **more}


def agree(parts: Any) -> dict:
    """{"eased": n, "escalated": n, "redirected": n, "changed": n,
    "unchanged": n, "of": parts counted, "direction": what they point to
    together}."""
    out = {d: 0 for d in DIRECTIONS}
    ok = [p for p in (parts if isinstance(parts, list) else []) if isinstance(p, dict)]
    for p in ok:
        d = p.get("direction")
        if d in out:
            out[d] += 1
    return {**out, "of": len(ok), "direction": vote(ok)}


# ---- the state part -------------------------------------------------------
def _states(result: Any) -> dict[str, str]:
    r = iv4.clean_result(result)
    return {str(o.get("object_id") or ""): str(o.get("state") or "").strip().lower()
            for o in r["detected_objects"] if str(o.get("object_id") or "")}


def state_direction_of(before: Any, after: Any, ids: Any) -> str:
    """Which way the states of `ids` went together: more of them less
    severe = eased, more of them more severe = escalated; changed with no
    known up or down = changed."""
    try:
        was, now = _states(before), _states(after)
    except Exception:
        return UNCHANGED
    n = {EASED: 0, ESCALATED: 0, CHANGED: 0}
    for i in ids if isinstance(ids, (list, tuple, set, frozenset)) else []:
        if i not in was:
            continue
        d = state_direction(was[i], now.get(i))
        if d in n:
            n[d] += 1
    if n[EASED] > n[ESCALATED]:
        return EASED
    if n[ESCALATED] > n[EASED]:
        return ESCALATED
    return CHANGED if sum(n.values()) else UNCHANGED


# ---- the tried object itself ----------------------------------------------
def _line(name: str, size: Any, direction: str, was: Any = "", now: Any = "") -> dict:
    p = part(name, size, direction)
    return {**p, "before": str(was), "after": str(now)}


def _at_risk(result: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for x in (result.get("at_risk") if isinstance(result, dict) else None) or []:
        if isinstance(x, dict) and str(x.get("object_id") or ""):
            out[str(x["object_id"])] = str(x.get("kind") or "").strip().lower()
    return out


def _threats(result: Any) -> set[str]:
    return {str(t.get("object_id") or "") for t in iv4.clean_result(result)["threats"]} - {""}


def _arrows(result: Any, graph: str) -> list[tuple[str, str, str]]:
    try:
        return chain4._edges(iv4.clean_result(result).get(graph))
    except Exception:
        return []


def _weight(result: Any, graph: str, target: str) -> float:
    """The danger on the object's own arrows in one graph of one run (the
    same number as its declared weight)."""
    from agentic import declared4                    # lazy: declared4 imports this module
    try:
        if not any(target in (s, t) for s, _e, t in _arrows(result, graph)):
            return 0.0
        return float(declared4._lost(result, graph, {target}))
    except Exception:
        return 0.0


def _list_line(target: str, before: Any, after: Any, victim: bool) -> dict:
    if victim:
        b, a = _at_risk(before), _at_risk(after)
        was = f"at risk ({b[target]})" if target in b else "not at risk"
        now = f"at risk ({a[target]})" if target in a else "not at risk"
        if (target in b) != (target in a):
            return _line("at risk", 1.0, EASED if target in b else ESCALATED, was, now)
        if target in b and b[target] != a[target]:
            rb, ra = KIND_RANK.get(b[target]), KIND_RANK.get(a[target])
            d = CHANGED if rb is None or ra is None or rb == ra else (
                EASED if ra < rb else ESCALATED)
            return _line("at risk", 0.5, d, was, now)
        return _line("at risk", 0.0, UNCHANGED, was, now)
    b, a = target in _threats(before), target in _threats(after)
    was, now = ("a threat" if b else "not a threat"), ("a threat" if a else "not a threat")
    if b != a:
        return _line("in the threats list", 1.0, EASED if b else ESCALATED, was, now)
    return _line("in the threats list", 0.0, UNCHANGED, was, now)


def _arrow_line(target: str, before: Any, after: Any, victim: bool) -> Any:
    """Who threatens it (a victim) or who it threatens (a hazard): how many
    of its arrows differ, and whether the danger on them fell or rose. Read
    off the model's own graph; off the graph of its recommendations when
    its own graph has no arrow for it in either run. None when neither
    graph has one."""
    name = "hazards that threaten it" if victim else "who it threatens"
    for graph in ("graph_b", "causal_graph"):
        def mine(result: Any) -> set:
            return {(s, e, t) for s, e, t in _arrows(result, graph)
                    if (t == target if victim else s == target)}
        b, a = mine(before), mine(after)
        if not b and not a:
            continue
        size = 1.0 - len(b & a) / len(b | a)
        wb, wa = _weight(before, graph, target), _weight(after, graph, target)
        if abs(wa - wb) < 1e-9:
            wb, wa = float(len(b)), float(len(a))
        d = EASED if wa < wb else ESCALATED if wa > wb else REDIRECTED

        def other(arrows: set) -> str:
            ends = sorted({(s if victim else t) for s, _e, t in arrows})
            return ", ".join(ends) if ends else "nothing"
        return _line(name, size, d, other(b), other(a))
    return None


def _rec_line(target: str, before: Any, pair: Any, victim: bool) -> Any:
    """The recommendations that protect it (a victim) or act on it (a
    hazard): their average shift from the recommendations table; a new one
    that names it counts as a full change, escalated. None when no
    recommendation names it in either run."""
    from agentic import grounding4                   # lazy: it reads this module's callers
    name = "recommendations that protect it" if victim else "recommendations that act on it"
    try:
        tbl = grounding4.table(before, [pair])
        cells = [r["cells"][0] for r in tbl["rows"] if r.get("cells")]
        new = tbl["columns"][0].get("new") or []
    except Exception:
        return None
    named = [c for c in cells if c.get("named_removed")]
    fresh = 0
    for r in new:
        try:
            t, _e, aff = grounding4._quad(r)
        except Exception:
            continue
        if (target in aff) if victim else (t == target):
            fresh += 1
    if not named and not fresh:
        return None
    items = []
    for c in named:
        size = _num(c.get("impact")) or 0.0
        gone = c.get("state") == grounding4.GONE
        d = (EASED if gone or c.get("way") == grounding4.EASED else
             ESCALATED if c.get("way") == grounding4.ESCALATED else REDIRECTED)
        items.append({"size": 1.0 if gone else size, "direction": d})
    items += [{"size": 1.0, "direction": ESCALATED}] * fresh
    size = sum(i["size"] for i in items) / len(items)
    dropped = sum(1 for c in named if c.get("state") == grounding4.GONE)
    now = ", ".join(x for x in (
        f"{dropped} dropped" if dropped else "",
        f"{len(named) - dropped} kept" if len(named) - dropped else "",
        f"{fresh} new" if fresh else "") if x)
    return _line(name, size, vote(items), f"{len(named)}", now)


def _state_line(target: str, before: Any, after: Any) -> Any:
    try:
        was, now = _states(before), _states(after)
    except Exception:
        return None
    if target not in was or target not in now:
        return None                               # not listed after: no state to read
    d = state_direction(was[target], now[target])
    return _line("its state", 0.0 if d == UNCHANGED else 1.0, d, was[target], now[target])


def itself(before: Any, pair: Any) -> Any:
    """The fifth part of one try: what the model said about the tried
    object, before against after. None for a try of the kind "removed",
    for a target the before run does not list, or when nothing about it
    can be read. `listed` False = the edited run does not list it though
    it is still in the picture. Never raises."""
    try:
        one, after = pair
        one = one if isinstance(one, dict) else {}
        target = str(one.get("target") or "")
        if not target or one.get("kind") != DEALT_WITH:
            return None
        was = _states(before)
        if target not in was:
            return None
        victim = bool(iv4.is_victim(before, target))
        lines = [_list_line(target, before, after, victim),
                 _arrow_line(target, before, after, victim),
                 _rec_line(target, before, pair, victim),
                 _state_line(target, before, after)]
        lines = [x for x in lines if isinstance(x, dict)]
        if not lines:
            return None
        size = sum(x["size"] for x in lines) / len(lines)
        return part(ITSELF, size, vote(lines), lines=lines, target=target,
                    victim=victim, listed=target in _states(after))
    except Exception:
        return None
