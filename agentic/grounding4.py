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

from agentic import intervention4 as iv4, parts4, same4

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
    if cell.get("victim"):
        # a try that took out something that can be harmed: a recommendation
        # that protected it is grounded when it is dropped or no longer
        # names it, and a masquerade when it still names it
        if cell.get("named_removed"):
            if not cell.get("still_names"):
                return GROUNDED
            # a victim who was DEALT WITH is still in the picture: a
            # recommendation that still protects them but eased (later,
            # smaller) is the grounded answer, like a dealt-with hazard
            if cell.get("dealt") and cell.get("way") == EASED:
                return GROUNDED
            return MASQUERADE
        return SPURIOUS if state == GONE else IGNORED
    if cell.get("named_removed"):
        if state == GONE:
            return GROUNDED
        # A hazard that was DEALT WITH is still in the picture (a burnt
        # house can still be unsafe), so its recommendation need not go:
        # staying but easing (later, smaller) is the grounded answer.
        # Staying as it was, or escalating, is the masquerade.
        if cell.get("dealt") and cell.get("way") == EASED:
            return GROUNDED
        return MASQUERADE
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
    frozen arm's own rule. An edited run that was already tracked
    (agentic/align4.py) is in the before run's names, and the frozen rule
    must not undo that: it pairs by label alone and would call the car's
    smoke the house's smoke again."""
    if isinstance(after, dict) and isinstance(after.get("tracks"), dict):
        return {}
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


def _mentions(rec: Any, ids: set[str], labels: set[str]) -> bool:
    """True when a recommendation still names one of `ids`: in the things
    it protects, or by id or label as a whole word in its action."""
    import re
    if not isinstance(rec, dict):
        return False
    if _quad(rec)[2] & ids:
        return True
    text = str(rec.get("action") or "").lower()
    return any(w and re.search(r"(?<![a-z0-9_])" + re.escape(w.lower()) + r"(?![a-z0-9_])", text)
               for w in ids | labels)


def _victim_cells(cells: list[dict], base: list[dict], victims: set[str],
                  labels: set[str], stays: bool = False, after_recs: Any = None,
                  name: Any = None, used: Any = None) -> None:
    """Re-read one column for a try that took out a VICTIM. A recommendation
    is about the try when it protected the victim. With nothing left to
    protect and no advice left, it was DROPPED (for a hazard try that same
    cell is only bookkeeping). Edits the cells in place."""
    for c, rec in zip(cells, base):
        c["victim"] = True
        c["named_removed"] = bool(_quad(rec)[2] & victims)
        if c["state"] == TARGET_REMOVED:
            c["state"] = GONE
        after = c.get("after")
        if after is None and c["named_removed"]:
            # it has no counterpart; but does a recommendation about the SAME
            # hazard still name the victim? Then that is this one, kept.
            threat = (name or (lambda h: h))(_quad(rec)[0])
            hits = [k for k, a in enumerate(after_recs or [])
                    if a["quad"][0] == threat
                    and (bool(a["quad"][2] & victims) if stays
                         else _mentions(a["rec"], victims, labels))]
            # one no other row has taken, if there is one (it is then THIS
            # recommendation and not a new one); else one that names him anyway
            free = [k for k in hits if used is None or k not in used]
            if hits:
                k = (free or hits)[0]
                after = c["after"] = after_recs[k]["rec"]
                c["state"] = STAYS
                if free and used is not None:
                    used.add(k)
        if stays:
            # "dealt with": the victim was moved to safety and is still in
            # the picture, so naming it in an action is fine. It is only a
            # masquerade when the recommendation still PROTECTS it.
            names = isinstance(after, dict) and bool(_quad(after)[2] & victims)
        else:
            names = after is not None and _mentions(after, victims, labels)
        c["still_names"] = bool(c["named_removed"] and names)


def _match_column(base: list[dict], after_recs: list[dict], removed: set[str],
                  name: Any = None, victims: Any = None) -> list[dict]:
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
    # `name(hazard)`: the name a hazard goes by. Two hazards the model
    # called one and the same share a name (agentic/same4.py), so advice
    # that names either one is advice about the pair.
    name = name or (lambda h: h)
    cells: list[dict | None] = [None] * len(base)
    used: set[int] = set()
    quads = [(name(t), e, a) for t, e, a in (_quad(r) for r in base)]

    victims = set(victims or ())

    def same(i: int, a: dict) -> bool:
        """The same arrow: the effect word and everything it still protects."""
        return a["quad"][1] == quads[i][1] and a["quad"][2] == (quads[i][2] - removed)

    def only_the_victim(i: int) -> bool:
        """A try that took out a VICTIM, and this recommendation protected
        nothing else: with the victim gone it has no counterpart. It must
        not be paired with whatever else is said about its hazard (H
        ui_4a253340: "get the person off the porch" was compared with
        "deploy a hoseline on the house" and read adjusted)."""
        threat, _e, affected = quads[i]
        return bool(victims) and bool(affected) and affected <= victims and threat not in removed

    def take(i: int, j: int, state: str) -> None:
        used.add(j)
        cells[i] = {"state": state, "named_removed": quads[i][0] in removed,
                    "after": after_recs[j]["rec"], "after_quad": after_recs[j]["quad"]}

    def free(i: int, test) -> int | None:
        return next((j for j, a in enumerate(after_recs)
                     if j not in used and a["quad"][0] == quads[i][0] and test(a)), None)
    # closest first: the same arrow, then a shared target, then the same
    # kind of action, then any advice about that hazard
    for want in ("exact", "overlap", "intent", "any"):
        for i, rec in enumerate(base):
            if cells[i] is not None or only_the_victim(i):
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
                # kept as it was only when it is the same arrow; a kept
                # recommendation that names the removed hazard and now
                # protects other things is adjusted like any other
                take(i, j, STAYS if same(i, after_recs[j]) else CHANGED)
    # THE SAME VICTIM UNDER ANOTHER HAZARD WORD (B_pool ui_b3ada51c, Sunny,
    # 2026-11-20): before, "rescue the child" named the pool as its hazard;
    # after the edit the model named the child itself. No recommendation
    # was "about the pool" any more and all three read dropped, though the
    # rescue was still there. So before a recommendation is called
    # dropped: is one left that names, AS ITS HAZARD, the very thing this
    # one protected? That recommendation names no hazard at all, only the
    # victim, so it is this one, kept. (A recommendation about ANOTHER
    # hazard for the same victim is another recommendation: smoke -> man
    # is not kept because fire -> man is still there.)
    def protects(a: dict, left: frozenset) -> bool:
        return a["quad"][0] in left
    for i, rec in enumerate(base):
        if cells[i] is not None:
            continue
        threat, _effect, affected = quads[i]
        left = affected - removed
        if threat in removed or not left:
            continue
        if any(a["quad"][0] == threat for a in after_recs):
            continue                            # its hazard still has advice: handled below
        j = next((j for j, a in enumerate(after_recs)
                  if j not in used and protects(a, left)), None)
        k = j if j is not None else next(
            (k for k, a in enumerate(after_recs) if protects(a, left)), None)
        if k is not None:
            if j is not None:
                used.add(j)
            cells[i] = {"state": STAYS, "named_removed": False,
                        "after": after_recs[k]["rec"], "after_quad": after_recs[k]["quad"],
                        "hazard_renamed": True}
    for i, rec in enumerate(base):
        if cells[i] is not None:
            continue
        threat, _effect, affected = quads[i]
        # Its hazard still has a recommendation, but none is free. It may
        # BORROW one only when that one protects something this one
        # protected: it is then the same recommendation, said once. (H
        # ui_4a253340, Sunny, 2026-11-20: "fire engine on the house" read
        # kept because "move the person off the porch" was still there.
        # The frozen arm's word for the kind of action cannot tell the two
        # apart, it calls both "assist"; what each protects can.)
        # (When everything this one protected was itself removed, it has
        # nothing of its own left to compare by, and the older rule stands:
        # its hazard still has a recommendation, so it reads kept, adjusted.)
        nothing_left = bool(affected) and not (affected - removed)
        shared = None if only_the_victim(i) else next(
            (a for a in after_recs if a["quad"][0] == threat
             and (nothing_left or (a["quad"][2] & affected))), None)
        if shared is not None:                  # its hazard still has advice
            # a borrowed counterpart is another recommendation's: how close
            # the two are cannot be told, so it reads as before (kept when
            # it names the removed hazard, else adjusted, in full)
            cells[i] = {"state": STAYS if threat in removed else CHANGED,
                        "named_removed": threat in removed, "after": shared["rec"],
                        "borrowed": True}
        elif threat in removed:
            cells[i] = {"state": GONE, "named_removed": True, "after": None}
        elif affected and not (affected - removed):
            cells[i] = {"state": TARGET_REMOVED, "named_removed": False, "after": None}
        else:
            cells[i] = {"state": GONE, "named_removed": False, "after": None}
    # what each recommendation still protected, and how far its counterpart
    # is from that: 0 = the same things, 1 = none of them (impact_detail)
    for i, c in enumerate(cells):
        if c is None or c.get("after_quad") is None:
            continue
        _t, effect, affected = quads[i]
        left = affected - removed
        now = frozenset(c["after_quad"][2])
        c["effect_changed"] = c["after_quad"][1] != effect
        c["targets_kept"], c["targets_before"] = len(left & now), len(left)
        c["targets_new"] = len(now - left)
        union = left | now
        c["target_shift"] = round(1.0 - len(left & now) / len(union), 3) if union else 0.0
        del c["after_quad"]
    return [c for c in cells if c is not None], used   # type: ignore[return-value]


# How much ONE recommendation changed when a hazard was removed, 0..1
# (Sunny, 2026-11-17: a bar per removed hazard, so a row shows which hazard
# that recommendation depends on most). Dropped = 1. A recommendation that
# is kept can still change: its target or effect, how soon, how big the
# hazard it names is, its rank. The four shares add up to 1 and are
# priors, the one place to change them.
# 2026-11-23 (Sunny: "that itself can be another shift signal"): the STATE
# the recommendation gives its hazard is a fifth part (burning -> burnt).
IMPACT = {"target or effect": 0.35, "how soon": 0.2, "size": 0.15, "rank": 0.15,
          "state": 0.15}
_SOON, _SIZE = ("now", "minutes", "later"), ("small", "medium", "large")


def _field(rec: Any, key: str, allowed: tuple) -> str:
    q = rec.get("structured_reasoning") if isinstance(rec, dict) else None
    v = q.get(key) if isinstance(q, dict) else None
    return v if isinstance(v, str) and v in allowed else ""


def _state(rec: Any) -> str:
    q = rec.get("structured_reasoning") if isinstance(rec, dict) else None
    v = q.get("state") if isinstance(q, dict) else None
    return v.strip().lower() if isinstance(v, str) else ""


def _rank(rec: Any) -> Any:
    r = rec.get("rank") if isinstance(rec, dict) else None
    return r if isinstance(r, int) and not isinstance(r, bool) else None


EASED, ESCALATED, MIXED_WAY = "eased", "escalated", "mixed"


_ACTION_CACHE: dict[tuple[str, str], Any] = {}


def action_shift(before_action: Any, after_action: Any) -> Any:
    """How far apart two actions are BY MEANING, 0 (the same) to 1: one
    minus the similarity of the two sentences (the frozen arm's embedding
    measure, the one the pathology part uses). No verb list, so it reads
    the same in any domain (Sunny, 2026-11-25). It has no direction: which
    way a recommendation went is read off how soon and size.

    None when it cannot be measured (the embedding model is not installed,
    or switched off, as in the tests) or one side has no action.

    Read it with care: on E and H the SAME action reworded reads about 0.25
    to 0.30 ("deploy a hose line to extinguish the car fire" against
    "extinguish the car with water or foam"), and a different action 0.36
    to 0.68 ("extricate from the window" against "retreat to a safe
    distance": 0.36). So it is shown, and NOT yet counted in the cell's own
    number."""
    a = " ".join(str(before_action or "").split())
    b = " ".join(str(after_action or "").split())
    if not a or not b or not isinstance(before_action, str) or not isinstance(after_action, str):
        return None
    if a == b:
        return 0.0
    if (a, b) not in _ACTION_CACHE:
        try:
            v = iv4._frozen().rec_semantic_shift([{"action": a}], [{"action": b}])
        except Exception:
            v = None
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None               # not kept: the model may only be loading; ask again next time
        _ACTION_CACHE[(a, b)] = round(float(v), 3)
    return _ACTION_CACHE[(a, b)]


def impact_detail(rec: Any, cell: Any, n: int) -> dict:
    """{"value": 0..1 or None, "reasons": [{"text", "way"}], "way",
    "fields"} for one cell. `way` says which direction the change went:
    "eased" (later, smaller), "escalated" (sooner, bigger), "mixed" when
    the reasons disagree, "" when nothing with a direction changed. A rank
    change and a changed target or effect carry no direction: a
    recommendation can move up only because another one was dropped. None
    for a cell that is not evidence (its target was removed).

    `fields` is the same change field by field, for the picture that pops
    up over the cell (Sunny, 2026-11-25: "what eased, what escalated... from
    the perspective of recommendation"): [{"name", "size": that field's own
    change 0..1, "direction": eased | escalated | redirected | changed |
    unchanged, "before", "after"}], every field listed, also the ones that
    did not change. The value is these sizes times the weights in IMPACT;
    the "action" field (when it can be measured) is shown and not counted.
    Never raises."""
    cell = cell if isinstance(cell, dict) else {}
    state = cell.get("state")
    if state == TARGET_REMOVED:
        return {"value": None, "reasons": [], "way": "", "fields": []}
    if state == GONE:
        return {"value": 1.0, "reasons": [{"text": "dropped", "way": ""}], "way": "",
                "fields": [{**parts4.part("the recommendation", 1.0, parts4.EASED),
                            "before": "there", "after": "dropped"}]}
    after = cell.get("after") if isinstance(cell.get("after"), dict) else {}
    total, reasons, fields = 0.0, [], []

    def field(name: str, size: Any, direction: str, was: Any, now: Any) -> None:
        fields.append({**parts4.part(name, size, direction),
                       "before": str(was), "after": str(now)})
    try:
        who_b, who_a = sorted(_quad(rec)[2]), sorted(_quad(after)[2]) if after else []
        eff_b, eff_a = _quad(rec)[1], (_quad(after)[1] if after else "")
    except Exception:
        who_b, who_a, eff_b, eff_a = [], [], "", ""
    # the action itself: how far the two sentences are apart by meaning.
    # Shown first; not counted in the value (see action_shift).
    try:
        act = action_shift(rec.get("action") if isinstance(rec, dict) else None,
                           after.get("action"))
    except Exception:
        act = None
    if act is not None:
        fields.append({**parts4.part("action", act, parts4.CHANGED), "before": "", "after": "",
                       "counted": False})
    who_size = eff_size = 0.0
    if state == CHANGED:
        # by how much: losing one target of four is not losing all of them
        # (H ui_4a253340: both read 0.40). A changed effect word counts as
        # half. A cell that does not say (an older record) counts in full.
        shift = cell.get("target_shift")
        if isinstance(shift, (int, float)) and not isinstance(shift, bool):
            part = max(float(shift), 0.5 if cell.get("effect_changed") else 0.0)
            who_size = float(shift)
            eff_size = 1.0 if cell.get("effect_changed") else 0.0
            kept, was = cell.get("targets_kept"), cell.get("targets_before")
            if part >= 1.0 or not was:
                text = "target or effect changed"
            else:
                bits = []
                if shift > 0:
                    new_n = cell.get("targets_new") or 0
                    bits.append(f"{kept} of {was} targets kept"
                                + (f", {new_n} new" if new_n else ""))
                if cell.get("effect_changed"):
                    bits.append("effect changed")
                text = " \u00b7 ".join(bits) or "target or effect changed"
        else:
            part, text = 1.0, "target or effect changed"
            who_size = 1.0
        total += IMPACT["target or effect"] * part
        reasons.append({"text": text, "way": ""})
    field("who it protects", who_size, parts4.REDIRECTED,
          ", ".join(who_b) or "nothing", ", ".join(who_a) or "nothing")
    field("effect", eff_size, parts4.CHANGED, eff_b or "?", eff_a or "?")
    if cell.get("hazard_renamed"):
        # said, not charged: the recommendation is the same one
        reasons.append({"text": "same victim, the hazard is no longer named", "way": ""})
    # how soon: later is eased. size: smaller is eased.
    for key, allowed, name, eased_when_up in (("how_soon", _SOON, "how soon", True),
                                              ("size", _SIZE, "size", False)):
        b, a = _field(rec, key, allowed), _field(after, key, allowed)
        if b and a and b != a:
            step = allowed.index(a) - allowed.index(b)
            total += IMPACT[name] * abs(step) / (len(allowed) - 1)
            way = EASED if (step > 0) == eased_when_up else ESCALATED
            reasons.append({"text": f"{b} \u2192 {a}", "way": way})
            field(name, abs(step) / (len(allowed) - 1), way, b, a)
        else:
            field(name, 0.0, parts4.UNCHANGED, b or "not said", a or "not said")
    # the state it gives its hazard: a change with no direction of its own
    sb, sa = _state(rec), _state(after)
    if sb and sa and sb != sa:
        total += IMPACT["state"]
        reasons.append({"text": f"{sb} \u2192 {sa}", "way": ""})
    rb, ra = _rank(rec), _rank(after)
    if rb is not None and ra is not None and rb != ra:
        step = min(1.0, abs(rb - ra) / max(1, n - 1))
        total += IMPACT["rank"] * step
        reasons.append({"text": f"rank {rb} \u2192 {ra}", "way": ""})
        field("rank", step, parts4.CHANGED, rb, ra)
    else:
        field("rank", 0.0, parts4.UNCHANGED, "?" if rb is None else rb, "?" if ra is None else ra)
    field("state of its hazard", 1.0 if (sb and sa and sb != sa) else 0.0, parts4.CHANGED,
          sb or "?", sa or "?")
    ways = {r["way"] for r in reasons if r["way"]}
    way = MIXED_WAY if len(ways) > 1 else (ways.pop() if ways else "")
    # The action bar says HOW MUCH (the embedding); its DIRECTION is the
    # recommendation's own, read off how soon and size (Sunny, 2026-11-25:
    # "embedding should show how much it changed... and then other feature
    # should show direction"). Neither moved, or one each way: no direction.
    if act is not None and way in (EASED, ESCALATED):
        fields[0] = {**fields[0], **parts4.part("action", act, way)}
    return {"value": round(min(1.0, total), 3), "reasons": reasons, "way": way,
            "fields": fields}


def impact(rec: Any, cell: Any, n: int) -> tuple[Any, list[str]]:
    """(0..1, the reasons in plain words) for one cell; see impact_detail."""
    d = impact_detail(rec, cell, n)
    return d["value"], [r["text"] for r in d["reasons"]]


def _verdict(cells: list[dict]) -> str:
    named = [cell_label(c) for c in cells if c.get("named_removed")]
    if named:
        kinds = set(named)
        return kinds.pop() if len(kinds) == 1 else MIXED
    if any(cell_label(c) == SPURIOUS for c in cells):
        return SPURIOUS
    return UNTESTED


# A recommendation the try was NOT about should not change at all. What
# it did instead, in plain words (Sunny, 2026-11-23: "demoting rec 1 to 2
# when we put out the car fire..." read "correctly ignored").
UNAFFECTED, DEMOTED, PROMOTED, REAIMED = "unaffected", "demoted", "promoted", "re-aimed"
# Dropped in a try that dealt with something it does NOT name. Whether
# there was a reason is not ours to say (H: "keep the man out of the smoke"
# was dropped when the CAR fire was put out; the car's smoke was never
# listed, the recommendation named the house's). What can be said is what
# is in the picture (Sunny, 2026-11-23).
STILL_THERE = "dropped, though its hazard and victim are still there"
HAZARD_THERE = "dropped, though its hazard is still there"
DROPPED_OTHER = "dropped"            # and what it named is no longer listed in the edited run
NO_CAUSE = STILL_THERE               # the older name, kept for callers
REAIMED_FROM = 0.5          # more than half of what it protected is other things now


def spill(cell: Any) -> str:
    """'' for a cell the try WAS about (or bookkeeping). Else one of:
    unaffected / demoted (later, smaller) / promoted (sooner, bigger) /
    re-aimed (now protects other things) / dropped, though its hazard and
    victim are still there (or plain "dropped" when they are not); two can
    come together ("re-aimed, demoted"). A change of rank alone is
    unaffected: when a recommendation is dropped, the ones below it all
    move up with no decision behind it. Never raises."""
    cell = cell if isinstance(cell, dict) else {}
    state = cell.get("state")
    if cell.get("named_removed") or state == TARGET_REMOVED or state is None:
        return ""
    if state == GONE:
        if cell.get("hazard_there") is False or cell.get("victim_there") is False:
            return DROPPED_OTHER
        return STILL_THERE if cell.get("protects") is not False else HAZARD_THERE
    words = []
    shift = cell.get("target_shift")
    if state == CHANGED and (not isinstance(shift, (int, float)) or shift > REAIMED_FROM):
        words.append(REAIMED)
    way = cell.get("way")
    if way == EASED:
        words.append(DEMOTED)
    elif way == ESCALATED:
        words.append(PROMOTED)
    elif way == MIXED_WAY:
        words += [DEMOTED, PROMOTED]
    return ", ".join(words) or UNAFFECTED


def edit_seen(before: Any, after: Any, one: Any) -> dict:
    """Did the model see the edit that was MEANT (Sunny, 2026-11-23: "we
    want to check whether the meant edit is the same as what the model has
    seen after edit")? A try the model did not see is not a fair test,
    whatever its recommendations did.

    `one` is the try: its target, its kind, and the state the editor
    intended (`intended_state`, optional). `after` is the edited run in the
    before run's names (agentic/align4.py). Returns
      {"seen": bool, "as_intended": True | False | None (no state was
       intended), "was", "intended", "saw", "moved", "head", "text"}.
    Removed: seen when it is gone (or no longer in that state). Dealt with:
    seen when its state changed, or when it moved (a victim moved to safety
    often keeps its state). Never raises."""
    one = one if isinstance(one, dict) else {}
    target = str(one.get("target") or "")
    removed = one.get("kind") != "dealt with"
    intended = str(one.get("intended_state") or "").strip().lower()

    def obj_of(result: Any) -> Any:
        for o in (result.get("detected_objects") if isinstance(result, dict) else None) or []:
            if isinstance(o, dict) and str(o.get("object_id") or "") == target:
                return o
        return None

    def state_of(result: Any) -> Any:
        o = obj_of(result)
        return str(o.get("state") or "").strip().lower() if o is not None else None
    was, now = state_of(before), state_of(after)
    # moved: its box in the edited picture no longer overlaps the old one by
    # half (the same measure the tracking uses for "the same place")
    moved = False
    try:
        from agentic import align4
        ob, oa = obj_of(before), obj_of(after)
        sb = before.get("image_size") if isinstance(before, dict) else None
        sa = after.get("image_size") if isinstance(after, dict) else None
        if not (sb and sa):
            sb = sa = None
        bb, ba = align4._box(ob or {}, sb), align4._box(oa or {}, sa)
        moved = bool(bb and ba) and align4.iou(bb, ba) < align4.IOU_SAME
    except Exception:
        moved = False
    gone = now is None
    saw = "not in the picture" if gone else ("moved" if moved and now == was else now)
    if removed:
        seen = gone or now != was
        as_intended: Any = seen
        wanted = "not in the picture"
    elif intended:
        wanted = intended
        if gone:
            seen, as_intended = True, False
        elif now == intended:
            # the same state as before was intended (a moved victim): it must have moved
            seen = as_intended = (now != was) or moved
        else:
            seen, as_intended = (now != was) or moved, False
    else:
        wanted = ""
        seen, as_intended = gone or now != was or moved, None
    if not target or was is None:
        return {"seen": True, "as_intended": None, "was": was, "intended": wanted,
                "saw": saw, "moved": moved, "head": "", "text": ""}
    if not seen:
        head = "edit not seen"
        text = ((f"intended: {wanted} \u00b7 " if wanted else "")
                + f"model saw: still {was} \u2717 not a fair test")
    elif as_intended is False:
        head = "edit seen, not as intended"
        text = f"intended: {wanted} \u00b7 model saw: {saw} \u2717 changed, but not as intended"
    else:
        head = "edit seen"
        text = ((f"intended: {wanted} \u00b7 model saw: {saw} \u2713") if wanted
                else f"model saw: {was} \u2192 {saw} \u2713")
    return {"seen": seen, "as_intended": as_intended, "was": was, "intended": wanted,
            "saw": saw, "moved": moved, "head": head, "text": text}


def _who_words(ids: Any, result: Any) -> str:
    """'person_1 (trapped \u00b7 distress), house_2 (intact)': each protected
    thing with the state the run gave it and, when the run put it at risk,
    which kind. 'nothing' for an empty list. Never raises."""
    try:
        r = iv4.clean_result(result)
        state = {str(o.get("object_id") or ""): str(o.get("state") or "").strip()
                 for o in r["detected_objects"]}
        kind = {str(x.get("object_id") or ""): str(x.get("kind") or "").strip()
                for x in (r.get("at_risk") if isinstance(r.get("at_risk"), list) else [])
                if isinstance(x, dict)}
    except Exception:
        state, kind = {}, {}
    out = []
    for i in sorted(str(x) for x in (ids or []) if isinstance(x, str) and x):
        bits = [b for b in (state.get(i, ""), kind.get(i, "")) if b]
        out.append(i + (" (" + " \u00b7 ".join(bits) + ")" if bits else ""))
    return ", ".join(out) or "nothing"


def _place_flags(result: Any) -> list[dict]:
    try:
        from agentic import place4
        return place4.mismatches(result)
    except Exception:
        return []


def shift_summary(rows: Any) -> dict:
    """How much the recommendations changed, over the whole table, in two
    numbers (Sunny, 2026-11-20: "the total shift considering all the recs
    we should also show beside the count"):

      "named"   when a hazard or victim the recommendation NAMES was removed
                (grounded recommendations change here: should be high)
      "other"   when another one was removed (should be low)

    A model that rewrites everything on every edit scores high on both, so
    the second number is what makes the first one mean something. Each is
    the mean over the recommendations of that recommendation's own mean
    change (0..1; dropped = 1). None when no cell of that kind exists.
    Bookkeeping cells (no change value) are left out. Never raises."""
    named: list[float] = []
    other: list[float] = []
    for r in rows if isinstance(rows, list) else []:
        cells = r.get("cells") if isinstance(r, dict) else None
        mine: list[float] = []
        rest: list[float] = []
        for c in cells if isinstance(cells, list) else []:
            v = c.get("impact") if isinstance(c, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                (mine if c.get("named_removed") else rest).append(float(v))
        if mine:
            named.append(sum(mine) / len(mine))
        if rest:
            other.append(sum(rest) / len(rest))
    return {"named": round(sum(named) / len(named), 3) if named else None,
            "other": round(sum(other) / len(other), 3) if other else None,
            "n_named": len(named), "n_other": len(other)}


def table(before: dict, tries: list[tuple[dict, dict]]) -> dict:
    """`before`: the before run in the frozen shape. `tries`: (saved try,
    after run in the frozen shape) for every MEASURED try, in order.
    Returns {"rows": [...], "columns": [...]}; never raises on bad input."""
    before = iv4.clean_result(before)
    base = _acting(before["recommendations"])
    grp = same4.groups(same4.pairs_of(before))

    def name(h: str) -> str:
        return same4.key(h, grp)
    columns: list[dict] = []
    cells_by_row: list[list[dict]] = [[] for _ in base]
    for one, after in tries or []:
        one = one if isinstance(one, dict) else {}
        after = iv4.clean_result(after)
        ren = renames(before, after)
        after_recs = []
        for r in _acting(after["recommendations"]):
            t, e, aff = _quad(r)
            after_recs.append({"rec": r, "quad": (name(ren.get(t, t)), e,
                                                  frozenset(ren.get(x, x) for x in aff))})
        removed = removed_set(before, after, str(one.get("target") or ""), ren)
        rset = set(removed)
        # a pair the model called one hazard is removed only when BOTH are
        # gone; when the edit took one and left the other, say so
        # (gone = no longer in the picture, OR still there but no longer a
        #  hazard: burning brush edited to plain brush stays as an object)
        still_hazard = {ren.get(i, i) for i in
                        (str(t.get("object_id") or "") for t in after.get("threats") or [])}
        for h in list(rset):
            rset.update(m for m in grp.get(h, ()) if m not in still_hazard)
        removed = removed + [m for m in sorted(rset) if m not in removed]
        whole = {same4.key(h, grp) for h in rset
                 if h in grp and all(m in rset for m in grp[h])}
        left_behind = sorted({m for h in rset if h in grp for m in grp[h] if m not in rset})
        target = str(one.get("target") or "")
        is_victim = bool(target) and iv4.is_victim(before, target)
        # a try that FOLLOWED a recommendation may have acted on more than
        # one thing ("move both officers back"): the others count as tried too
        also = [str(x) for x in (one.get("also") if isinstance(one.get("also"), list) else [])
                if isinstance(x, str) and x and x != target]
        tried = {target} | {a for a in also if iv4.is_victim(before, a)}
        if not is_victim:
            more = [a for a in also if not iv4.is_victim(before, a)]
            rset.update(more)
            removed = removed + [a for a in more if a not in removed]
            whole = {same4.key(h, grp) for h in rset
                     if h in grp and all(m in rset for m in grp[h])}
        col_cells, used = _match_column(base, after_recs, rset | whole, name,
                                        victims=tried if is_victim else None)
        if is_victim:
            lab = {str(o.get("label") or "") for o in before["detected_objects"]
                   if str(o.get("object_id") or "") == target}
            # only when that label is the victim's alone ("the man" with two men says nothing)
            lab = {x for x in lab if x and sum(1 for o in before["detected_objects"]
                                               if str(o.get("label") or "") == x) == 1}
            _victim_cells(col_cells, base, tried, lab,
                          stays=one.get("kind") == "dealt with",
                          after_recs=after_recs, name=name, used=used)
        n_recs = max(len(base), len(after_recs))
        after_ids = {ren.get(i, i) for i in
                     (str(o.get("object_id") or "") for o in after["detected_objects"])}
        # objects whose match between the two runs is a guess (align4): a
        # recommendation that names one is read with that doubt
        unsure = {str(x) for x in (after.get("unsure") or []) if isinstance(x, str)}
        for i, c in enumerate(col_cells):
            t_i, _e_i, aff_i = _quad(base[i])
            c["unsure"] = sorted(({t_i} | set(aff_i)) & unsure)
            c["dealt"] = one.get("kind") == "dealt with"
            det = impact_detail(base[i], c, n_recs)
            c["impact"], c["why"] = det["value"], [r["text"] for r in det["reasons"]]
            c["way"], c["reasons"] = det["way"], det["reasons"]
            c["fields"] = det["fields"]
            # beside who it protects, HOW THEY ARE in each run: their state
            # and their at-risk kind. Context for reading the cell (did the
            # recommendation follow when the person got better?), never
            # counted here: it is the person's, and the feature table counts it.
            for f in c["fields"]:
                if f.get("name") == "who it protects":
                    f["before"] = _who_words(aff_i, before)
                    if isinstance(c.get("after"), dict):
                        f["after"] = _who_words(_quad(c["after"])[2], after)
            c["label"] = cell_label(c)            # after the way: a dealt-with hazard reads it
            # is what it names still in the edited picture? (for a dropped one)
            c["hazard_there"] = any(m in after_ids for m in (grp.get(t_i) or (t_i,)))
            c["protects"] = bool(aff_i)
            c["victim_there"] = (not aff_i) or any(a in after_ids for a in aff_i)
            c["spill"] = spill(c)
            cells_by_row[i].append(c)
        # A possible NAME MIX-UP (Sunny, 2026-11-23: "mark it"): when the
        # edited run's words put someone where the picture does not have
        # them, a recommendation that now protects one of those people may
        # be the same recommendation with the wrong name on it, not a change
        # of mind. Marked, never removed from any number.
        flags = _place_flags(after)
        mixed_up = {str(f.get("who")) for f in flags} | {
            str(x) for f in flags for x in (f.get("instead") or [])}
        for i, c in enumerate(col_cells):
            now = _quad(c["after"])[2] if isinstance(c.get("after"), dict) else frozenset()
            c["mixup"] = bool(mixed_up and c["state"] == CHANGED
                              and (set(_quad(base[i])[2]) | set(now)) & mixed_up
                              and set(_quad(base[i])[2]) != set(now))
        counted = [c for c in col_cells if c["state"] != TARGET_REMOVED]
        moved = sum(1 for c in counted if c["state"] in (GONE, CHANGED))
        summary = one.get("summary") if isinstance(one.get("summary"), dict) else {}
        columns.append({
            "index": one.get("index"), "target": one.get("target"),
            "victim": is_victim,
            # names of the edited run that were matched to this run's by
            # their place in the picture (agentic/align4.py)
            "renamed": after.get("renamed") if isinstance(after.get("renamed"), dict) else {},
            "renamed_new": (after.get("renamed_new")
                            if isinstance(after.get("renamed_new"), dict) else {}),
            "unsure": sorted(unsure),
            "tracked": isinstance(after.get("tracks"), dict),
            # a flag only: its words put someone where the picture does not
            # have them (agentic/place4.py)
            "place_flags": flags,
            # did the model see the edit that was meant?
            "edit": edit_seen(before, after, one),
            "kind": "dealt with" if one.get("kind") == "dealt with" else "removed",
            # the recommendation this try carried out, when it followed one
            "follows": one.get("follows") if isinstance(one.get("follows"), dict) else None,
            "also": also,
            "also_removed": [x for x in removed if x != str(one.get("target") or "")],
            "left_behind": left_behind,
            "after_run": one.get("after_run"),
            "moved": moved, "of": len(counted),
            "content_shift": summary.get("content_shift"),
            "level_delta": summary.get("level_delta"),
            "new": [a["rec"] for j, a in enumerate(after_recs) if j not in used],
        })
    rows = []
    for rec, cells in zip(base, cells_by_row):
        threat, effect, affected = _quad(rec)
        # what the tries that were NOT about it did to it: the READING is
        # read off the tries that were, and one good column hid a drop
        # without cause in another (H: #5 read "grounded")
        spilled: dict[str, int] = {}
        for c in cells:
            said = str(c.get("spill") or "")
            for w in ([said] if said.startswith("dropped") else said.split(", ")):
                if w and w != UNAFFECTED:
                    spilled[w] = spilled.get(w, 0) + 1
        rows.append({"rank": rec.get("rank"), "action": str(rec.get("action") or ""),
                     "threat": threat, "effect": effect, "affected": sorted(affected),
                     "same_as": [m for m in grp.get(threat, ()) if m != threat],
                     "cells": cells, "verdict": _verdict(cells), "spill": spilled})
    return {"rows": rows, "columns": columns, "shift": shift_summary(rows),
            "place_flags": _place_flags(before),
            "same_hazards": sorted({grp[h] for h in grp})}
