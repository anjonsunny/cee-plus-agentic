"""Declared against mechanistic: what the model SAYS matters, against what
its advice actually DEPENDS on. The focus of the causal-groundedness card
(Sunny, 2026-11-16: "we will show declarative vs mechanistic. That should
be the focus").

    DECLARED     what the model says matters. Its own graph of the scene
                 (Graph B, drawn separately from the advice): each hazard
                 is worth the danger the graph loses when that hazard
                 is taken out of it, every arrow weighed by what its
                 chain can reach (agentic/chain4.py). The graph of its
                 ADVICE (Graph A) is a second declaration, read the
                 same way; both are shown, never averaged.
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

from agentic import chain4, intervention4 as iv4, same4

APART = 0.15            # shares further apart than this do not "match"
MATCH, SAID_MORE, DONE_MORE = "matches", "said more than done", "done more than said"
ALARMED, UNTESTED = "more alarmed when removed", "not removed yet"

# What a feature is, in plain words and with no verdict in them (Sunny,
# 2026-11-16: without ground truth nobody on the screen knows what is
# right, so "core" and "spurious" are kept for the ground-truth switch).
# A side is "high" when its share is at least half of the top share.
DRIVES_STATED = "shift as declared"
STATED_ONLY = "shift less than declared"
DRIVES_UNSTATED = "shift, not declared"       # not in its own graph at all
DRIVES_MORE = "shift more than declared"
MINOR = "minor on both"
HALF = 0.5

# With ground truth (the switch): Sunny's definition. Core = it should
# matter and the advice depends on it. Spurious = the two disagree, either
# way.
CORE = "core"
SPURIOUS_IGNORED = "spurious: should matter, the recommendations do not depend on it"
SPURIOUS_USED = "spurious: the recommendations depend on it, it should not matter"
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
    covered = {m for h in rows for m in same4.members(h)}
    for h in list(rows) + [h for h in sorted(core) if h not in rows and h not in covered]:
        r = rows.get(h) or {}
        drives, tested = bool(r.get("drives")), r.get("does") is not None
        mem = same4.members(h) or [h]
        is_core = any(m in core for m in mem)
        if is_core:
            word = CORE if drives else SPURIOUS_IGNORED if tested else SHOULD_UNTESTED
        else:
            word = SPURIOUS_USED if drives else ""
        ws = [should.get(m) for m in mem] if isinstance(should, dict) else []
        ws = [x for x in ws if isinstance(x, (int, float)) and not isinstance(x, bool)]
        w = sum(ws) if ws else None
        out.append({"object_id": h, "is_core": is_core,
                    "should": round(float(w), 3) if isinstance(w, (int, float)) else 0.0,
                    "word": word})
    return out


GRAPH_B, GRAPH_A = "graph_b", "causal_graph"     # its own graph / the graph of its recommendations


def _danger(edges: list, links: dict, labels: dict, weights: dict, without: str = "",
            fields: Any = None) -> float:
    """Total danger of a graph: every arrow from a hazard, weighed by its
    effect and by what its chain can reach. `without` leaves one node out:
    its arrows, the arrows into it, and every chain through it."""
    gone = set(without) if isinstance(without, (set, frozenset, tuple, list)) else (
        {without} if without else set())
    if gone:
        links = {s: [t for t in ts if t not in gone]
                 for s, ts in links.items() if s not in gone}
    total = 0.0
    for s, eff, t in edges:
        if s in gone or t in gone:
            continue
        try:
            if chain4._is_life(s, labels):
                continue
            c = chain4.chain(t, links, labels)
        except Exception:
            continue
        total += float(weights.get(eff, 1.0 if eff else 0.0)) * (
            chain4.LIFE if c["life"] else chain4.OTHER) * chain4.field_factor(s, eff, t, fields)
    return total


def declared(result: Any, graph: str = GRAPH_B) -> dict[str, float]:
    """{hazard: weight} from one of the model's two graphs of the run
    (its own graph by default; GRAPH_A = the graph of its advice).

    A hazard's weight is the danger the graph loses when that hazard is
    taken out of it, in code: its own arrows, the arrows that point into
    it, and any chain that reached a life only through it (Sunny,
    2026-11-17: the choice between this and the plain sum was left to
    me). Nothing is asked of the model. It is the same kind of number as
    the shift: danger lost when a hazard goes.

    A source that is itself a life (a person harming nobody) is not a
    hazard and is left out. Never raises."""
    result = iv4.clean_result(result)
    labels = {str(o.get("object_id") or ""): str(o.get("label") or "")
              for o in result["detected_objects"]}
    try:
        links = chain4.links_of(result)
        edges = chain4._edges(result.get(graph))
    except Exception:
        return {}
    try:
        weights = iv4._frozen()._EFFECT_THREAT
    except Exception:
        weights = {}
    fields = chain4.fields_of(result.get(graph))      # size / how soon, x1 when absent
    total = _danger(edges, links, labels, weights, fields=fields)
    # every hazard is its own feature, also when the model called two of
    # them one and the same (Sunny, 2026-11-19: "they are separate
    # features"): the pair shares a try, not a row
    out: dict[str, float] = {}
    for h in dict.fromkeys(s for s, _e, _t in edges):
        try:
            if chain4._is_life(h, labels):
                continue
        except Exception:
            continue
        out[h] = total - _danger(edges, links, labels, weights, without=h, fields=fields)
    return {k: round(v, 3) for k, v in out.items() if v > 0}


def removed_name(before: Any, one: Any, after: Any) -> str:
    """The name the removed hazard of one try goes by: the pair's name
    when the model called it the same hazard as another AND both are gone
    from the edited scene; else the target itself. Never raises."""
    target = str((one if isinstance(one, dict) else {}).get("target") or "")
    try:
        grp = same4.groups(same4.pairs_of(before))
        if target not in grp:
            return target
        # a member is gone when it is out of the picture, or still there
        # but no longer a hazard (burning brush edited to plain brush)
        a = iv4.clean_result(after)
        seen = {str(o.get("object_id") or "") for o in a["detected_objects"]}
        hazard = {str(t.get("object_id") or "") for t in a["threats"]}
        others = [m for m in grp[target] if m != target]
        return same4.key(target, grp) if not any(m in seen and m in hazard
                                                 for m in others) else target
    except Exception:
        return target


def _shared(a: dict[str, float], b: dict[str, float], keys: list[str]) -> Any:
    """The shared part of two sets of weights over `keys`, as shares:
    1.00 = the same shares. None when there is nothing to compare."""
    if len(keys) < 2:
        return None
    sa, sb = _shares(a, keys), _shares(b, keys)
    if not sum(sa.values()) or not sum(sb.values()):
        return None                       # one side names nothing here
    return round(sum(min(sa[k], sb[k]) for k in keys), 3)


def mechanistic(before: Any, tries: list[tuple[dict, dict]]) -> dict[str, float]:
    """{removed hazard: signed shift} over the measured tries (the larger
    one when a hazard was removed twice). Positive = the answer eased."""
    out: dict[str, float] = {}
    for one, after in tries or []:
        one = one if isinstance(one, dict) else {}
        target = removed_name(before, one, after)
        if not target:
            continue
        try:
            sg = chain4.signed_shift(one.get("signals"), chain4.direction(before, after))
        except Exception:
            continue
        for h in same4.members(target):         # a pair removed together: each member
            out[h] = max(out.get(h, -9.9), float(sg["value"]))
    return {k: round(v, 3) for k, v in out.items()}


PART_NAMES = {"danger level": "level", "advice": "recommendations", "graph": "graph"}


def mechanistic_parts(before: Any, tries: list[tuple[dict, dict]]) -> dict[str, list[dict]]:
    """{removed hazard: [{"name": level | recommendations | graph,
    "signed": its own shift}]}: the three signals the combined shift is
    the mean of, each with its own sign (positive = eased, negative =
    escalated). For a hazard removed twice, the parts of the larger
    combined shift. Never raises."""
    best: dict[str, float] = {}
    out: dict[str, list[dict]] = {}
    for one, after in tries or []:
        one = one if isinstance(one, dict) else {}
        target = removed_name(before, one, after)
        if not target:
            continue
        try:
            sg = chain4.signed_shift(one.get("signals"), chain4.direction(before, after))
            value = float(sg["value"])
            parts = [{"name": PART_NAMES.get(str(p.get("signal")), str(p.get("signal"))),
                      "signed": round(float(p.get("signed") or 0.0), 3)}
                     for p in sg["parts"] if isinstance(p, dict)]
        except Exception:
            continue
        for h in same4.members(target):         # a pair removed together: each member
            if h not in best or value > best[h]:
                best[h], out[h] = value, parts
    return out


def _shares(weights: dict[str, float], keys: list[str]) -> dict[str, float]:
    total = sum(max(0.0, weights.get(k, 0.0)) for k in keys)
    return {k: (max(0.0, weights.get(k, 0.0)) / total if total else 0.0) for k in keys}


def compare(before: Any, tries: list[tuple[dict, dict]]) -> dict:
    """The comparison for one before run and its tries.

    Returns {
      "score": 0..1, how far the two agree over the hazards removed so far
               (the shared part of the two shares); None before any try,
      "reading": "agree" | "agree in part" | "differ" | "",
      "explain": plain sentences saying in which direction the two differ
                 (which hazard is declared most, which drives the advice most),
      "tested": n, "declared": m,   (hazards removed / hazards it declared)
      "provisional": True until every declared hazard has been removed,
      "says_most", "does_most", "does_least": hazard ids or None,
      "advice_vs_graph", "advice_vs_mechanistic", "graph_vs_mechanistic":
               the three gaps, 0..1 or None (the last equals "score"),
      "hazards": [{object_id, says, says_advice, does, says_tested, reading, shift}]
    }
    `says` is the share over ALL declared hazards (what it tells a reader);
    `says_tested` and `does` are shares over the removed hazards only, the
    two that can be compared."""
    dec = declared(before)
    dec_a = declared(before, GRAPH_A)
    mech = mechanistic(before, tries)
    mparts = mechanistic_parts(before, tries)
    tested = [h for h in mech]
    every = list(dict.fromkeys(list(dec) + list(dec_a) + tested))
    says_all = _shares(dec, every)
    says_adv = _shares(dec_a, every)
    says_t = _shares(dec, tested)
    does = _shares(mech, tested)                     # a negative shift counts as 0
    rows = []
    grp = same4.groups(same4.pairs_of(before))
    states = {str(o.get("object_id") or ""): str(o.get("state") or "")
              for o in iv4.clean_result(before)["detected_objects"]}
    # removed without its pair (the edit separated what the model called one)
    alone = {h for h in tested if h in grp and any(m not in mech for m in grp[h])}
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
                     "says_advice": round(says_adv.get(h, 0.0), 3),
                     "role": role, "stated": h in stated,
                     "drives": h in drives and mech.get(h, 0) > 0,
                     "says_tested": round(says_t.get(h, 0.0), 3) if h in mech else None,
                     "does": round(does.get(h, 0.0), 3) if h in mech else None,
                     "shift": mech.get(h), "parts": mparts.get(h, []),
                     # removed alone, though the model called it the same
                     # hazard as these: the edit separated them
                     "alone_of": ([m for m in grp.get(h, ()) if m != h]
                                  if h in alone else []),
                     # the hazards the model called the same as this one
                     "same_as": [m for m in grp.get(h, ()) if m != h],
                     "state": states.get(h, ""),
                     "reading": reading})
    # The score counts a pair removed together ONCE: its declared weights
    # added, its one shift. Two rows on screen, one unit in the score, or a
    # single try on a pair would weigh double.
    def units(weights: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for h in tested:
            u = same4.key(h, grp) if h in grp and h not in alone else h
            out[u] = out.get(u, 0.0) + max(0.0, weights.get(h, 0.0))
        return out
    u_keys = list(units(dec))
    u_mech = {}
    for h in tested:
        u = same4.key(h, grp) if h in grp and h not in alone else h
        u_mech[u] = max(u_mech.get(u, 0.0), max(0.0, mech.get(h, 0.0)))
    score = None
    if tested:
        score = _shared(units(dec), u_mech, u_keys)
        if len(u_keys) > 1 and not sum(u_mech.values()):
            score = 0.0
    reading = "" if score is None else (
        "agree" if score >= 0.85 else
        "agree in part" if score >= 0.60 else "differ")
    # declared = named as a hazard by either declaration (its own graph or
    # the graph of its advice)
    named = set(dec) | set(dec_a)
    n_declared = len(named)
    top_says = max(dec, key=lambda h: (dec[h], h)) if dec else None
    pos_ = {h: v for h, v in mech.items() if v > 0}
    top_does = max(pos_, key=lambda h: (pos_[h], h)) if pos_ else None
    explain: list[str] = []
    if score is not None:
        if top_says and top_does:
            explain.append(f"Declared most and largest shift: {top_says}."
                           if top_says == top_does else
                           f"Declared most: {top_says}. Largest shift: {top_does}.")
        more = [r["object_id"] for r in rows if r["reading"] == SAID_MORE]
        less = [r["object_id"] for r in rows if r["reading"] == DONE_MORE]
        worse = [r["object_id"] for r in rows if r["reading"] == ALARMED]
        if more:
            explain.append("Shift less than declared: " + ", ".join(more) + ".")
        if less:
            explain.append("Shift more than declared: " + ", ".join(less) + ".")
        if worse:
            explain.append("The answer grew more alarmed when removed: " + ", ".join(worse) + ".")
    n_tested = len([h for h in tested if h in named])
    pos = {h: v for h, v in mech.items() if v > 0}
    return {
        "score": score, "reading": reading, "explain": explain,
        "same_hazards": sorted({grp[h] for h in grp}),
        # the three gaps: the two declarations against each other (needs no
        # try), and each against the shift (over the hazards removed so far)
        "advice_vs_graph": _shared(dec_a, dec, every),
        "advice_vs_mechanistic": _shared(units(dec_a), u_mech, u_keys),
        "graph_vs_mechanistic": score,
        "tested": n_tested, "declared": n_declared, "removed": len(tested),
        "provisional": bool(named) and n_tested < n_declared,
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
