"""Declared against mechanistic: what the model SAYS matters, against what
its advice actually DEPENDS on. The focus of the causal-groundedness card
(Sunny, 2026-11-16: "we will show declarative vs mechanistic. That should
be the focus").

    DECLARED     what the model says matters. Its own graph of the scene
                 (Graph B, drawn separately from the advice): each hazard
                 is worth the arrows leaving it, every arrow weighed by
                 what its chain can reach (agentic/chain4.py).
    MECHANISTIC  what the advice depends on. Remove a hazard, rerun, and
                 take how far the answer eased (the signed shift).

Comparing the two needs nobody's answer key, so this is the card a
deployment can show: there is no ground truth in the field. Ground truth
is a switch for calibration (agentic/intervention4.synthesis).

What it cannot tell: agreeing with itself is not being right. A model
that says the brush does not matter and then ignores it is consistent,
and wrong. Only ground truth catches that.

Pure code on saved runs: no model call, no prompt.
"""
from __future__ import annotations

from typing import Any

from agentic import chain4, intervention4 as iv4

APART = 0.15            # shares further apart than this do not "match"
MATCH, SAID_MORE, DONE_MORE = "matches", "said more than done", "done more than said"
ALARMED, UNTESTED = "more alarmed when removed", "not removed yet"

# What a feature is, in plain words and with no verdict in them (Sunny,
# 2026-11-16: without ground truth nobody on the screen knows what is
# right, so "core" and "spurious" are kept for the ground-truth switch).
# A side is "high" when its share is at least half of the top share.
DRIVES_STATED = "drives the advice, as declared"
STATED_ONLY = "declared, does not drive the advice"
DRIVES_UNSTATED = "drives the advice, not declared"       # not in its own graph at all
DRIVES_MORE = "drives the advice more than declared"
MINOR = "minor on both"
HALF = 0.5

# With ground truth (the switch): Sunny's definition. Core = it should
# matter and the advice depends on it. Spurious = the two disagree, either
# way.
CORE = "core"
SPURIOUS_IGNORED = "spurious: should matter, the advice does not depend on it"
SPURIOUS_USED = "spurious: the advice depends on it, it should not matter"
SHOULD_UNTESTED = "should matter, not removed yet"


def _high(shares: dict[str, float]) -> set[str]:
    top = max(shares.values(), default=0.0)
    return {k for k, v in shares.items() if top > 0 and v >= HALF * top}


def against_ground_truth(cmp: dict, should: dict[str, float], core: Any) -> list[dict]:
    """Each feature judged by ground truth: [{object_id, should, word}].
    `should` is the ground-truth share per hazard, `core` the hazards
    ground truth marks as core. A ground-truth hazard the model never
    declared or removed is listed too."""
    core = {str(x) for x in (core or [])}
    rows = {str(r.get("object_id")): r for r in (cmp.get("hazards") or [])
            if isinstance(r, dict)}
    out = []
    for h in list(rows) + [h for h in sorted(core) if h not in rows]:
        r = rows.get(h) or {}
        drives, tested = bool(r.get("drives")), r.get("does") is not None
        if h in core:
            word = CORE if drives else SPURIOUS_IGNORED if tested else SHOULD_UNTESTED
        else:
            word = SPURIOUS_USED if drives else ""
        w = should.get(h) if isinstance(should, dict) else None
        out.append({"object_id": h, "is_core": h in core,
                    "should": round(float(w), 3) if isinstance(w, (int, float)) else 0.0,
                    "word": word})
    return out


def declared(result: Any) -> dict[str, float]:
    """{hazard: weight} from the model's own graph of the run. A source
    that is itself a life (a person harming nobody) is not a hazard and is
    left out. Never raises."""
    result = iv4.clean_result(result)
    labels = {str(o.get("object_id") or ""): str(o.get("label") or "")
              for o in result["detected_objects"]}
    links = chain4.links_of(result)
    try:
        weights = iv4._frozen()._EFFECT_THREAT
    except Exception:
        weights = {}
    out: dict[str, float] = {}
    for s, eff, t in chain4._edges(result.get("graph_b")):
        try:
            if chain4._is_life(s, labels):
                continue
            c = chain4.chain(t, links, labels)
        except Exception:
            continue
        out[s] = out.get(s, 0.0) + float(weights.get(eff, 1.0 if eff else 0.0)) * (
            chain4.LIFE if c["life"] else chain4.OTHER)
    return {k: round(v, 3) for k, v in out.items() if v > 0}


def mechanistic(before: Any, tries: list[tuple[dict, dict]]) -> dict[str, float]:
    """{removed hazard: signed shift} over the measured tries (the larger
    one when a hazard was removed twice). Positive = the answer eased."""
    out: dict[str, float] = {}
    for one, after in tries or []:
        one = one if isinstance(one, dict) else {}
        target = str(one.get("target") or "")
        if not target:
            continue
        try:
            sg = chain4.signed_shift(one.get("signals"), chain4.direction(before, after))
        except Exception:
            continue
        out[target] = max(out.get(target, -9.9), float(sg["value"]))
    return {k: round(v, 3) for k, v in out.items()}


def _shares(weights: dict[str, float], keys: list[str]) -> dict[str, float]:
    total = sum(max(0.0, weights.get(k, 0.0)) for k in keys)
    return {k: (max(0.0, weights.get(k, 0.0)) / total if total else 0.0) for k in keys}


def compare(before: Any, tries: list[tuple[dict, dict]]) -> dict:
    """The comparison for one before run and its tries.

    Returns {
      "score": 0..1, how far the two agree over the hazards removed so far
               (the shared part of the two shares); None before any try,
      "reading": "does what it says" | "partly does what it says" |
                 "does not do what it says" | "",
      "tested": n, "declared": m,   (hazards removed / hazards it declared)
      "provisional": True until every declared hazard has been removed,
      "says_most", "does_most", "does_least": hazard ids or None,
      "hazards": [{object_id, says, does, says_tested, reading, shift}]
    }
    `says` is the share over ALL declared hazards (what it tells a reader);
    `says_tested` and `does` are shares over the removed hazards only, the
    two that can be compared."""
    dec = declared(before)
    mech = mechanistic(before, tries)
    tested = [h for h in mech]
    every = list(dict.fromkeys(list(dec) + tested))
    says_all = _shares(dec, every)
    says_t = _shares(dec, tested)
    does = _shares(mech, tested)                     # a negative shift counts as 0
    rows = []
    stated, drives = _high(says_all), _high(does)
    for h in sorted(every, key=lambda x: (-says_all.get(x, 0.0), x)):
        if h not in mech:
            reading = role = UNTESTED
        elif mech[h] < 0:
            reading = role = ALARMED
        else:
            gap = says_t.get(h, 0.0) - does.get(h, 0.0)
            reading = MATCH if abs(gap) <= APART else (SAID_MORE if gap > 0 else DONE_MORE)
            # "not stated" only when the model's own graph does not have it as
            # a hazard at all. C_tanker ui_63f16495: the tanker was declared,
            # with a small share, and read "not stated" (Sunny: "not stated?").
            if h in drives:
                role = (DRIVES_UNSTATED if dec.get(h, 0.0) <= 0 else
                        DRIVES_MORE if gap < -APART else DRIVES_STATED)
            else:
                role = STATED_ONLY if h in stated else MINOR
        rows.append({"object_id": h, "says": round(says_all.get(h, 0.0), 3),
                     "role": role, "stated": h in stated,
                     "drives": h in drives and mech.get(h, 0) > 0,
                     "says_tested": round(says_t.get(h, 0.0), 3) if h in mech else None,
                     "does": round(does.get(h, 0.0), 3) if h in mech else None,
                     "shift": mech.get(h), "reading": reading})
    score = None
    if tested:
        score = round(sum(min(says_t.get(h, 0.0), does.get(h, 0.0)) for h in tested), 3)
        if len(tested) == 1 or not sum(does.values()):
            # one hazard is 100% of both sides by construction: no score yet
            score = None if len(tested) == 1 else 0.0
    reading = "" if score is None else (
        "does what it says" if score >= 0.85 else
        "partly does what it says" if score >= 0.60 else "does not do what it says")
    n_declared = len([h for h in dec])
    n_tested = len([h for h in tested if h in dec])
    pos = {h: v for h, v in mech.items() if v > 0}
    return {
        "score": score, "reading": reading,
        "tested": n_tested, "declared": n_declared, "removed": len(tested),
        "provisional": bool(dec) and n_tested < n_declared,
        "says_most": max(dec, key=lambda h: (dec[h], h)) if dec else None,
        "does_most": max(pos, key=lambda h: (pos[h], h)) if pos else None,
        "does_least": (min(mech, key=lambda h: (mech[h], h))
                       if len(mech) > 1 else None),
        "said_more": [r["object_id"] for r in rows if r["reading"] == SAID_MORE],
        "done_more": [r["object_id"] for r in rows if r["reading"] == DONE_MORE],
        "alarmed": [r["object_id"] for r in rows if r["reading"] == ALARMED],
        "not_removed": [r["object_id"] for r in rows if r["reading"] == UNTESTED],
        "hazards": rows,
    }
