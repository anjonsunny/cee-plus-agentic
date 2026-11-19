"""Same hazard, entered twice (Sunny, 2026-11-19).

On C_tanker the scene had `fire_1 · spreading` and `brush_1 · burning`:
one fire, two entities. We do not repair that and we do not decide it in
code. The MODEL is asked, inside the recommendation prompt it already
answers, which pairs of hazards are one and the same physical hazard
(`same_hazards`). Its judgment is followed where groundedness is
measured, and shown on screen, so it is also a test of the model:

    it says same  -> the two are one hazard in the groundedness tables:
                     one feature row, and a recommendation that names
                     either one names the pair
    it says nothing -> they stay two, as before

The perception record is never changed. A pair counts as REMOVED only
when both are gone from the edited scene; when the edit took one and
left the other, that is said on screen (the edit separated what the
model called one).

Pure code: no model call, no prompt (the one prompt line lives in
agentic/recommend.py).
"""
from __future__ import annotations

from typing import Any

JOIN = " + "


def _bare(x: Any) -> str:
    return str(x).split("·")[0].strip() if isinstance(x, str) else ""


def parse_pairs(value: Any, notes: list[str]) -> list[list[str]]:
    """The model's raw `same_hazards` -> [[id, id], ...]. A malformed
    entry is dropped with a note, never a crash. Each pair once."""
    if value in (None, "", []):
        return []
    if not isinstance(value, list):
        notes.append(f"same_hazards_not_a_list({type(value).__name__})->[]")
        return []
    out: list[list[str]] = []
    for item in value:
        ids = [_bare(x) for x in item] if isinstance(item, (list, tuple)) else []
        ids = [x for x in ids if x]
        if len(ids) != 2 or ids[0] == ids[1]:
            notes.append(f"same_hazards_entry_dropped({str(item)[:50]!r})")
            continue
        pair = sorted(ids)
        if pair not in out:
            out.append(pair)
    return out


def valid_pairs(pairs: Any, threat_ids: Any, notes: list[str]) -> list[list[str]]:
    """Keep only pairs whose two ids are both hazards of this scene."""
    known = {str(x) for x in (threat_ids or [])}
    out = []
    for p in pairs if isinstance(pairs, list) else []:
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, str) for x in p) \
                and p[0] != p[1] and p[0] in known and p[1] in known:
            out.append(sorted(p))
        else:
            notes.append(f"same_hazards_pair_not_two_hazards({str(p)[:50]!r})")
    return out


def pairs_of(result: Any) -> list[list[str]]:
    """The pairs a run declared, from the frozen-shape result. Never raises."""
    v = result.get("same_hazards") if isinstance(result, dict) else None
    return [sorted(p) for p in (v if isinstance(v, list) else [])
            if isinstance(p, (list, tuple)) and len(p) == 2
            and all(isinstance(x, str) and x for x in p) and p[0] != p[1]]


def groups(pairs: Any) -> dict[str, tuple[str, ...]]:
    """{id: every id the model called the same hazard, itself included}.
    Pairs chain: a=b and b=c make one group of three."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x
    for p in pairs if isinstance(pairs, list) else []:
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, str) for x in p):
            a, b = find(p[0]), find(p[1])
            if a != b:
                parent[max(a, b)] = min(a, b)
    members: dict[str, list[str]] = {}
    for x in list(parent):
        members.setdefault(find(x), []).append(x)
    return {x: tuple(sorted(m)) for m in members.values() if len(m) > 1 for x in m}


def key(oid: Any, grp: dict[str, tuple[str, ...]]) -> str:
    """The name a hazard goes by: itself, or "a + b" for a group."""
    oid = str(oid or "")
    return JOIN.join(grp[oid]) if oid in grp else oid


def members(name: Any) -> list[str]:
    return [x for x in str(name or "").split(JOIN) if x]
