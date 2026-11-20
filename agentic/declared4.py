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
NOT_ALONE = "not tested alone"


def _high(shares: dict[str, float]) -> set[str]:
    top = max(shares.values(), default=0.0)
    return {k for k, v in shares.items() if top > 0 and v >= HALF * top}


def against_ground_truth(cmp: dict, should: dict[str, float], core: Any,
                         listed: Any = None, gt_same: Any = None) -> list[dict]:
    """Each feature judged by ground truth: [{object_id, should, word}].
    `should` is the ground-truth share per hazard, `core` the hazards
    ground truth marks as core. A ground-truth hazard the model never
    declared or removed is listed too."""
    core = {str(x) for x in (core or [])}
    # `listed`: every thing ground truth lists (a listed thing with no arrow
    # should not matter). `gt_same`: the pairs GROUND TRUTH calls the same
    # hazard; None = the file does not say, so the model's own pairs are used.
    listed = {str(x) for x in (listed if isinstance(listed, (list, set, tuple)) else [])}
    gt_grp = same4.groups(gt_same) if isinstance(gt_same, list) else None
    rows = {str(r.get("object_id")): r for r in (cmp.get("hazards") or [])
            if isinstance(r, dict)}
    out = []
    covered = {m for h in rows for m in same4.members(h)}
    for h in list(rows) + [h for h in sorted(core) if h not in rows and h not in covered]:
        r = rows.get(h) or {}
        drives, tested = bool(r.get("drives")), r.get("does") is not None
        mem = same4.members(h) or [h]
        is_core = any(m in core for m in mem)
        # The model called it the same hazard as one ground truth names, and
        # the try took them out together: it is judged as that hazard, not
        # as a stranger to ground truth (C_tanker ui_2345d3fc: fire_1 read
        # core and brush_1, the same fire, read spurious).
        via = []
        own = should.get(h) if isinstance(should, dict) else None
        knows = isinstance(own, (int, float)) and not isinstance(own, bool) and own > 0
        # ...unless ground truth names it itself: then it keeps its own word
        # (also when the human marked it core itself but its weight sits on
        # the other one: the burning brush, weight on the fire)
        if not knows and not r.get("alone_of"):
            partners = (gt_grp.get(h, ()) if gt_grp is not None
                        else (r.get("same_as") if isinstance(r.get("same_as"),
                                                              (list, tuple)) else []))
            via = sorted(str(m) for m in partners if str(m) in core and str(m) != h)
            is_core = is_core or bool(via)
        if tested and r.get("gone_with"):
            # it only went along with another hazard: that try's shift is
            # not its own, so neither core nor spurious can be said
            word = NOT_ALONE
        elif is_core:
            word = CORE if drives else SPURIOUS_IGNORED if tested else SHOULD_UNTESTED
        else:
            word = SPURIOUS_USED if drives else ""
        ws = [should.get(m) for m in mem] if isinstance(should, dict) else []
        ws = [x for x in ws if isinstance(x, (int, float)) and not isinstance(x, bool)]
        w = sum(ws) if ws else None
        out.append({"object_id": h, "is_core": is_core, "via": via,
                    "listed": h in listed or is_core or knows,
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


def declared_together(result: Any, members: Any, graph: str = GRAPH_B) -> float:
    """The danger one of the model's graphs loses when ALL of `members` are
    taken out of it at once. This is the declared weight of a group (the
    hazards one try took out).

    Adding the members' own weights is wrong: an arrow between two of them
    (fire -> brush) goes when either end goes, so it is in both weights
    and would be counted twice. Taken out together it is counted once.
    0.0 for an empty group or a graph that cannot be read. Never raises."""
    gone = {str(m) for m in (members if isinstance(members, (list, tuple, set, frozenset))
                             else []) if isinstance(m, str) and m}
    if not gone:
        return 0.0
    result = iv4.clean_result(result)
    labels = {str(o.get("object_id") or ""): str(o.get("label") or "")
              for o in result["detected_objects"]}
    try:
        links = chain4.links_of(result)
        edges = chain4._edges(result.get(graph))
        # a member that is a life (a person) is not a hazard: it stays
        gone = {m for m in gone if not chain4._is_life(m, labels)}
    except Exception:
        return 0.0
    if not gone:
        return 0.0
    try:
        weights = iv4._frozen()._EFFECT_THREAT
    except Exception:
        weights = {}
    fields = chain4.fields_of(result.get(graph))
    total = _danger(edges, links, labels, weights, fields=fields)
    return round(max(0.0, total - _danger(edges, links, labels, weights,
                                          without=gone, fields=fields)), 3)


def _lost(result: Any, graph: str, gone: set[str]) -> float:
    """The danger one graph of a run loses when the nodes in `gone` are
    taken out of it, whatever they are. 0.0 when it cannot be read."""
    if not gone:
        return 0.0
    result = iv4.clean_result(result)
    labels = {str(o.get("object_id") or ""): str(o.get("label") or "")
              for o in result["detected_objects"]}
    try:
        links = chain4.links_of(result)
        edges = chain4._edges(result.get(graph))
    except Exception:
        return 0.0
    try:
        weights = iv4._frozen()._EFFECT_THREAT
    except Exception:
        weights = {}
    fields = chain4.fields_of(result.get(graph))
    try:
        total = _danger(edges, links, labels, weights, fields=fields)
        return round(max(0.0, total - _danger(edges, links, labels, weights,
                                              without=set(gone), fields=fields)), 3)
    except Exception:
        return 0.0


def victims_declared(result: Any, graph: str = GRAPH_B) -> dict[str, float]:
    """{victim: weight} from one of the model's graphs: the danger that
    graph loses when the victim is taken out of it (every arrow that ends
    at it, and every chain through it). The same measure as a hazard's
    declared weight, from the other end of the arrow (Sunny, 2026-11-20:
    "take away either end and the danger drops"). Victims are
    `intervention4.victims_of`: a person, an animal, a car, a house.
    Never raises."""
    try:
        ids = [v["object_id"] for v in iv4.victims_of(result)]
    except Exception:
        return {}
    return {v: _lost(result, graph, {v}) for v in ids}


def _gt_result(extra: Any) -> dict:
    """The ground-truth graph as a result the weighing code can read."""
    g = extra.get("graph") if isinstance(extra, dict) else None
    g = g if isinstance(g, dict) else {}
    nodes = [n for n in (g.get("nodes") if isinstance(g.get("nodes"), list) else [])
             if isinstance(n, dict)]
    graph = {"nodes": nodes, "edges": [e for e in (g.get("edges") if isinstance(
        g.get("edges"), list) else []) if isinstance(e, dict)]}
    return {"detected_objects": [{"object_id": n.get("id"), "label": n.get("label"),
                                  "state": n.get("state")} for n in nodes],
            "graph_b": graph, "causal_graph": graph, "recommendations": []}


def _core(extra: Any) -> set[str]:
    roles = extra.get("roles") if isinstance(extra, dict) else None
    return {str(k) for k, v in (roles if isinstance(roles, dict) else {}).items()
            if v == "core"}


def should_matter(extra: Any) -> dict[str, float]:
    """{hazard: weight} by GROUND TRUTH, measured the same way as the
    declared side (Sunny, 2026-11-20: "we should be consistent"): the
    danger the ground-truth graph loses when the hazard is taken out of it.
    A thing the human calls spurious gets nothing, arrows or not (the shed
    the fire may reach). `extra` is `intervention4.gt_objects(before)`.
    {} without a readable graph. Never raises."""
    try:
        w = declared(_gt_result(extra))
    except Exception:
        return {}
    core = _core(extra)
    return {h: v for h, v in w.items() if h in core}


def should_matter_together(extra: Any, members: Any) -> float:
    """The ground-truth weight of a group: its core hazards taken out of the
    ground-truth graph together. Never raises."""
    core = _core(extra)
    ms = [m for m in (members if isinstance(members, (list, tuple, set)) else [])
          if isinstance(m, str) and m in core]
    try:
        return declared_together(_gt_result(extra), ms)
    except Exception:
        return 0.0


def victims_should(extra: Any) -> dict[str, float]:
    """{victim: weight} by ground truth, the same measure on the
    ground-truth graph. A victim the human calls spurious gets 0 (the shed
    the fire may reach). Never raises."""
    try:
        r = _gt_result(extra)
        ids = [v["object_id"] for v in iv4.victims_of(r)]
        # a person with no arrow yet is still a victim the human listed
        labels = {str(o.get("object_id") or ""): str(o.get("label") or "")
                  for o in r["detected_objects"]}
        ids += [o for o in labels if o and o not in ids and chain4._is_life(o, labels)]
    except Exception:
        return {}
    core = _core(extra)
    return {v: (_lost(r, GRAPH_B, {v}) if v in core else 0.0) for v in ids}


def match_ground_truth(cmp: Any, extra: Any) -> dict:
    """Ground truth against the shift, the same way declared is compared
    with it: per group (the hazards one try took out), the ground-truth
    weight of the group beside the group's shift.

    Returns {"available", "should": {hazard: share over all core hazards},
    "core": [...], "score": 0..1 or None (needs two groups), "reading":
    agree / agree in part / differ, "removed": n, "of": m (core hazards
    with a weight)}. Never raises."""
    out = {"available": False, "should": {}, "core": sorted(_core(extra)),
           "score": None, "reading": "", "removed": 0, "of": 0}
    w = should_matter(extra)
    if not w:
        return out
    out["available"] = True
    out["should"] = {h: round(v, 3) for h, v in _shares(w, list(w)).items()}
    vs = victims_should(extra)
    out["victims_should"] = {h: round(v, 3) for h, v in _shares(vs, list(vs)).items()}
    out["victims"] = sorted(vs)
    out["core_victims"] = sorted(v for v in vs if v in _core(extra))
    out["core"] = sorted(h for h in _core(extra) if h not in vs)
    groups = [g for g in ((cmp.get("groups") if isinstance(cmp, dict) else None) or [])
              if isinstance(g, dict) and isinstance(g.get("members"), list)]
    keys, gt_w, mech = [], {}, {}
    gone: set[str] = set()
    for i, g in enumerate(groups):
        k = f"group {i}"
        keys.append(k)
        gt_w[k] = should_matter_together(extra, g["members"])
        sh = g.get("shift")
        mech[k] = max(0.0, float(sh)) if isinstance(sh, (int, float)) and not isinstance(
            sh, bool) else 0.0
        gone |= {str(m) for m in g["members"]}
    out["removed"], out["of"] = len([h for h in w if h in gone]), len(w)
    score = _shared(gt_w, mech, keys)
    if score is None and len(keys) > 1 and sum(gt_w.values()) and not sum(mech.values()):
        score = 0.0
    out["score"] = score
    out["reading"] = "" if score is None else (
        "agree" if score >= 0.85 else "agree in part" if score >= 0.60 else "differ")
    return out


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


def _best_tries(before: Any, tries: Any) -> dict[str, dict]:
    """{hazard: {"value", "parts", "try", "target"}}: for every hazard a
    try took out, the try with the larger shift. A try takes out its
    target, the hazards the model called the same as it, and every hazard
    that went with them (no fire, so no smoke): all of them carry that
    try's shift, because the shift cannot be split between them (Sunny,
    2026-11-19: "smoke_1 also should get affected in feature table").
    Never raises."""
    from agentic import grounding4
    out: dict[str, dict] = {}
    for k, pair in enumerate(tries or []):
        try:
            one, after = pair
        except Exception:
            continue
        one = one if isinstance(one, dict) else {}
        target = str(one.get("target") or "")
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
        try:
            victim = iv4.is_victim(before, target)
        except Exception:
            victim = False
        if victim:
            # the other end of the danger: this try is the victim's own
            # test and says nothing about any hazard
            old = out.get(target)
            if old is None or value > old["value"]:
                out[target] = {"value": value, "parts": parts, "try": k, "target": target,
                               "direct": True, "victim": True}
            continue
        gone = same4.members(removed_name(before, one, after))
        aimed = set(gone)                       # the target and its same-hazard pair
        try:
            b, a = iv4.clean_result(before), iv4.clean_result(after)
            gone += [h for h in grounding4.removed_set(b, a, target, grounding4.renames(b, a))
                     if h not in gone]
        except Exception:
            pass
        for h in gone:
            # a try AIMED at this hazard beats one where it only went along
            # (smoke gone with the fire, then smoke removed by itself: the
            # second one is the smoke's own test)
            direct = h in aimed
            old = out.get(h)
            if old is None or (direct and not old["direct"]) or (
                    direct == old["direct"] and value > old["value"]):
                out[h] = {"value": value, "parts": parts, "try": k, "target": target,
                          "direct": direct}
    return out


def mechanistic(before: Any, tries: list[tuple[dict, dict]]) -> dict[str, float]:
    """{removed hazard: signed shift} over the measured tries (the larger
    one when a hazard was removed twice). Positive = the answer eased."""
    return {h: round(v["value"], 3) for h, v in _best_tries(before, tries).items()}


PART_NAMES = {"danger level": "level", "advice": "recommendations", "graph": "graph"}


def mechanistic_parts(before: Any, tries: list[tuple[dict, dict]]) -> dict[str, list[dict]]:
    """{removed hazard: [{"name": level | recommendations | graph,
    "signed": its own shift}]}: the three signals the combined shift is
    the mean of, each with its own sign (positive = eased, negative =
    escalated). For a hazard removed twice, the parts of the larger
    combined shift. Never raises."""
    return {h: v["parts"] for h, v in _best_tries(before, tries).items()}


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
    best = _best_tries(before, tries)
    # tries that took out a victim are their own table: the hazard rows and
    # the hazard scores never count them
    v_best = {h: v for h, v in best.items() if v.get("victim")}
    best = {h: v for h, v in best.items() if not v.get("victim")}
    mech = {h: round(v["value"], 3) for h, v in best.items()}
    mparts = {h: v["parts"] for h, v in best.items()}
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
    alone = {h for h in tested if h in grp
             and any(m not in best or best[m]["try"] != best[h]["try"] for m in grp[h])}

    def unit_of(h: str) -> str:
        """Everything one try took out is one unit in the score."""
        return f"try {best[h]['try']}"
    # the shares a row shows are its UNIT's shares, so every hazard one try
    # took out shows the same number (Sunny: "smoke_1 also should get
    # affected"), and a try that removed three does not read as a third each
    _uk = list(dict.fromkeys(unit_of(h) for h in tested))
    _us: dict[str, float] = {}
    _ud: dict[str, float] = {}
    # the hazards of each unit, and the unit's declared weight: all of them
    # taken out of the graph TOGETHER (an arrow between two of them once)
    _members: dict[str, list[str]] = {}
    for h in tested:
        _members.setdefault(unit_of(h), []).append(h)
    grp_b = {u: declared_together(before, ms, GRAPH_B) for u, ms in _members.items()}
    grp_a = {u: declared_together(before, ms, GRAPH_A) for u, ms in _members.items()}
    for h in tested:
        _us[unit_of(h)] = grp_b[unit_of(h)]
        _ud[unit_of(h)] = max(_ud.get(unit_of(h), 0.0), max(0.0, mech.get(h, 0.0)))
    _us, _ud = _shares(_us, _uk), _shares(_ud, _uk)
    says_t = {h: _us[unit_of(h)] for h in tested}
    does = {h: _ud[unit_of(h)] for h in tested}
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
                     # it went in a try that removed something else: the
                     # other hazards of that try (no fire, so no smoke)
                     "gone_with": ([] if h not in best or best[h]["direct"]
                                   else sorted(m for m in best if m != h
                                               and best[m]["try"] == best[h]["try"]
                                               and best[m]["direct"])),
                     # every other hazard the same try took out: one try is
                     # one unit, so these rows show one and the same share
                     "try_with": ([] if h not in best else
                                  sorted(m for m in best if m != h
                                         and best[m]["try"] == best[h]["try"])),
                     "state": states.get(h, ""),
                     "reading": reading})
    # The score counts everything ONE try took out once: the declared
    # weights added, the one shift. Several rows on screen, one unit in the
    # score, or a single try that removed three hazards would weigh triple.
    def units(weights: dict[str, float]) -> dict[str, float]:
        return dict(grp_a if weights is dec_a else grp_b)
    u_keys = list(_members)
    u_mech = {}
    for h in tested:
        u_mech[unit_of(h)] = max(u_mech.get(unit_of(h), 0.0), max(0.0, mech.get(h, 0.0)))
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
    # ── the victims: what can be harmed, weighed from the other end
    v_b, v_a = victims_declared(before, GRAPH_B), victims_declared(before, GRAPH_A)
    v_ids = list(dict.fromkeys(list(v_b) + list(v_best)))
    v_says, v_adv = _shares(v_b, v_ids), _shares(v_a, v_ids)
    v_mech = {h: round(v["value"], 3) for h, v in v_best.items()}
    v_does = _shares(v_mech, list(v_mech))
    victims = [{"object_id": h, "state": states.get(h, ""), "victim": True,
                "says": round(v_says.get(h, 0.0), 3),
                "says_advice": round(v_adv.get(h, 0.0), 3),
                "does": round(v_does.get(h, 0.0), 3) if h in v_mech else None,
                "shift": v_mech.get(h), "parts": v_best[h]["parts"] if h in v_best else [],
                "drives": v_mech.get(h, 0) > 0}
               for h in sorted(v_ids, key=lambda x: (-v_says.get(x, 0.0), x))]
    n_tested = len([h for h in tested if h in named])
    pos = {h: v for h, v in mech.items() if v > 0}
    return {
        "score": score, "reading": reading, "explain": explain,
        "same_hazards": sorted({grp[h] for h in grp}),
        # tries that took out more than one hazard, in row order: the rows
        # show one share each, the score counts the try once
        "one_try": list({tuple(sorted([r["object_id"]] + r["try_with"])): None
                         for r in rows if r["try_with"]}),
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
        # each try's group: its hazards, the two declared weights of the
        # group (taken out together) and what the members add up to alone
        "groups": [{"members": sorted(ms),
                    "declared_graph": grp_b[u], "declared_recommendations": grp_a[u],
                    "added_graph": round(sum(max(0.0, dec.get(m, 0.0)) for m in ms), 3),
                    "added_recommendations": round(sum(max(0.0, dec_a.get(m, 0.0))
                                                       for m in ms), 3),
                    "shift": max(mech.get(m, 0.0) for m in ms)}
                   for u, ms in _members.items()],
        "hazards": rows,
        "victims": victims,
    }
