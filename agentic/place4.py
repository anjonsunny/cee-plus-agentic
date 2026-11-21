"""A flag: a recommendation's words put someone where the picture does not
have them (Sunny, 2026-11-20: "build both. A flag only.").

H_car_house_fire, two tries in a row: perception had one person on the
porch of the burning house and one in the road. The recommendation said
"Evacuate person_1 from the porch of house_1", and person_1 was the one in
the road. The model had numbered the two people in the order the caption
named them, not by where its own perception put them.

The check, in code, with no model call and no repair:

    the action says  <someone> ... off / out of / from the ... of / inside
                     / leave ...  <a thing>
    so that someone is AT that thing. In the picture, does their box touch
    it? If not, and another of the same kind does, it is flagged.

It says nothing when it cannot tell (no boxes, nobody else there, no such
phrase). It never changes a recommendation and nothing here reaches a
prompt. Pure code; never raises on malformed input.
"""
from __future__ import annotations

import re
from typing import Any

from agentic import align4

# "at": the words that say someone is AT the thing named next. Kept short
# and literal on purpose: a miss is silence, never a false flag.
_AT = (r"(?:off|out of|inside|within|leave|leaves|leaving|exit|exits|exiting|"
       r"from (?:the |its )?(?:\w+ ){0,2}of|"
       r"on (?:the |its )?(?:\w+ ){0,2}of|"
       r"in (?:the |its )?(?:\w+ ){0,2}of|"
       r"(?:off|from|on|in) (?:the |its )?(?:\w+ ){1,2}of)")
TOUCH = 0.25                # share of the person's own box that must lie on the thing


def _ids(result: Any) -> dict[str, dict]:
    return {o["object_id"]: o for o in align4._objects(result)}


def _on(who: dict, thing: dict, size: Any) -> bool:
    """True when `who` is at `thing` in the picture: a fair share of their
    box lies on it, or their feet stand in it."""
    a, b = align4._box(who, size), align4._box(thing, size)
    if not a or not b:
        return False
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    if w > 0 and h > 0 and area > 0 and (w * h) / area >= TOUCH:
        return True
    fx, fy = (a[0] + a[2]) / 2.0, a[3]
    return b[0] <= fx <= b[2] and b[1] <= fy <= b[3]


def _is_life(o: dict) -> bool:
    try:
        from agentic import chain4
        oid = str(o.get("object_id") or "")
        return bool(chain4._is_life(oid, {oid: str(o.get("label") or "")}))
    except Exception:
        return False


def mismatches(result: Any) -> list[dict]:
    """[{rank, who, at, phrase, instead: [others of that kind who ARE there],
    text}] for one run: each place where a recommendation's words put a
    person or an animal at a thing their box does not touch, while another
    of the same kind is there."""
    out: list[dict] = []
    objs = _ids(result)
    if not objs:
        return out
    size = result.get("image_size") if isinstance(result, dict) else None
    names = sorted(objs, key=len, reverse=True)
    any_id = r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9_])"
    recs = result.get("recommendations") if isinstance(result, dict) else None
    for rec in recs if isinstance(recs, list) else []:
        if not isinstance(rec, dict) or rec.get("benign"):
            continue
        action = rec.get("action")
        if not isinstance(action, str) or not action:
            continue
        mentions = [(m.start(), m.group(1)) for m in re.finditer(any_id, action)]
        for m in re.finditer(r"(?<![A-Za-z])(" + _AT + r")\s+(?:the |its )?" + any_id, action,
                             flags=re.IGNORECASE):
            thing = objs.get(m.group(2))
            # the someone: the last person or animal named before the phrase
            who_id = next((n for pos, n in reversed(mentions)
                           if pos < m.start() and _is_life(objs[n])), None)
            if thing is None or who_id is None or who_id == thing["object_id"]:
                continue
            who = objs[who_id]
            if not align4._box(who, size) or not align4._box(thing, size):
                continue                        # cannot tell: say nothing
            if _on(who, thing, size):
                continue
            others = [o["object_id"] for o in objs.values()
                      if o is not who and o is not thing
                      and align4._kind(o) == align4._kind(who) and _on(o, thing, size)]
            if not others:
                continue                        # nobody is there: not a mix-up we can show
            out.append({"rank": rec.get("rank"), "who": who_id, "at": thing["object_id"],
                        "phrase": m.group(0).strip(), "instead": sorted(others),
                        "text": say(who_id, thing["object_id"], m.group(0).strip(),
                                    sorted(others))})
    return out


def say(who: str, at: str, phrase: str, instead: list[str]) -> str:
    """One plain sentence for the screen."""
    there = " and ".join(instead)
    return (f'its words put {who} at {at} ("{phrase}"); in the picture {who} is not there, '
            f'{there} {"is" if len(instead) == 1 else "are"}')
