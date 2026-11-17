"""Two extra fields the model declares beside the quad (Sunny, 2026-11-17):

    size       on the HAZARD (node): how big or intense it is right now
               small | medium | large
    how_soon   on the ARROW (edge): how soon the source has this effect
               on the target        now | minutes | later

The model declares, ground truth knows, code only checks. This module
only CARRIES the model's words: it reads them off the raw answer and
puts them back on the parsed quad / graph, because the frozen
normalizers keep the four quad keys and drop everything else.

Backward compatible by one rule: a field that is missing, or is not one
of the three words, is simply absent. The arrow or the quad is always
kept; a bad value is dropped with a note, never silently. Nothing here
changes a score: no weight reads these fields yet.

Pure code: no model call, no prompt.
"""
from __future__ import annotations

from typing import Any

SIZES = ("small", "medium", "large")
HOW_SOON = ("now", "minutes", "later")
SIZE, SOON = "size", "how_soon"


def word(value: Any, allowed: tuple[str, ...]) -> str:
    """The value as one of the allowed words, else ''. Never raises."""
    if not isinstance(value, str):
        return ""
    v = value.strip().lower().replace("-", "_")
    return v if v in allowed else ""


def _bare(x: Any) -> str:
    return str(x or "").split("·")[0].strip()


def _take(src: dict, key: str, allowed: tuple[str, ...], where: str,
          notes: list[str]) -> str:
    """One field off one raw object. '' when absent; '' with a note when
    present but not one of the words."""
    if key not in src or src.get(key) in (None, ""):
        return ""
    w = word(src.get(key), allowed)
    if not w:
        notes.append(f"{key}_dropped_{where}({str(src.get(key))[:30]!r})")
    return w


def attach_quads(raw_recs: Any, recommendations: Any, notes: list[str]) -> None:
    """Put size / how_soon from each raw quad onto the parsed quad at the
    same position. Edits `recommendations` in place. Never raises."""
    if not isinstance(raw_recs, list) or not isinstance(recommendations, list):
        return
    for i, out in enumerate(recommendations):
        if not isinstance(out, dict):
            continue
        src = raw_recs[i] if i < len(raw_recs) and isinstance(raw_recs[i], dict) else {}
        raw_q, q = src.get("structured_reasoning"), out.get("structured_reasoning")
        if not isinstance(raw_q, dict) or not isinstance(q, dict):
            continue
        where = f"rank_{out.get('rank')}"
        for key, allowed in ((SIZE, SIZES), (SOON, HOW_SOON)):
            w = _take(raw_q, key, allowed, where, notes)
            if w:
                q[key] = w


def attach_graph(raw: Any, graph: Any) -> list[str]:
    """Put size on the nodes and how_soon on the edges of a parsed graph,
    read off the model's raw answer. Edits `graph` in place and returns
    the notes (also kept on the graph as `field_notes` when there are
    any). Never raises."""
    notes: list[str] = []
    if not isinstance(graph, dict):
        return notes
    cg = raw.get("causal_graph") if isinstance(raw, dict) else None
    cg = cg if isinstance(cg, dict) else {}
    sizes: dict[str, str] = {}
    for n in cg.get("nodes") or []:
        if isinstance(n, dict):
            w = _take(n, SIZE, SIZES, f"node_{_bare(n.get('id'))}", notes)
            if w:
                sizes[_bare(n.get("id"))] = w
    soon: dict[tuple[str, str, str], str] = {}
    for e in cg.get("edges") or []:
        if isinstance(e, dict):
            key = (_bare(e.get("source")), str(e.get("effect") or ""), _bare(e.get("target")))
            w = _take(e, SOON, HOW_SOON, f"edge_{key[0]}_{key[2]}", notes)
            if w:
                soon[key] = _sooner(soon.get(key, ""), w)
    for n in graph.get("nodes") or []:
        if isinstance(n, dict) and n.get("hazardous") and sizes.get(_bare(n.get("id"))):
            n[SIZE] = sizes[_bare(n.get("id"))]
    for e in graph.get("edges") or []:
        if isinstance(e, dict):
            w = soon.get((_bare(e.get("source")), str(e.get("effect") or ""),
                          _bare(e.get("target"))))
            if w:
                e[SOON] = w
    if notes:
        graph["field_notes"] = notes
    return notes


def _bigger(a: str, b: str) -> str:
    return max((a, b), key=lambda w: SIZES.index(w) if w in SIZES else -1)


def _sooner(a: str, b: str) -> str:
    return min((a, b), key=lambda w: HOW_SOON.index(w) if w in HOW_SOON else 99)


def carry_to_graph_a(graph_a: Any, recommendations: Any) -> list[str]:
    """Graph A is built from the quads by the frozen builder, which knows
    nothing of the two fields. Carry them over: the quad's size onto its
    hazard node, its how_soon onto each arrow it made. When two pieces of
    advice disagree, the larger size and the sooner time are kept and the
    disagreement is noted. Edits in place; returns the notes. Never raises."""
    notes: list[str] = []
    if not isinstance(graph_a, dict) or not isinstance(recommendations, list):
        return notes
    sizes: dict[str, str] = {}
    soon: dict[tuple[str, str, str], str] = {}
    for r in recommendations:
        q = r.get("structured_reasoning") if isinstance(r, dict) else None
        if not isinstance(q, dict) or r.get("benign"):
            continue
        threat = _bare(q.get("threat"))
        s = word(q.get(SIZE), SIZES)
        if s and threat:
            if sizes.get(threat) and sizes[threat] != s:
                notes.append(f"size_conflict_{threat}({sizes[threat]}|{s})")
            sizes[threat] = _bigger(sizes.get(threat, ""), s)
        t = word(q.get(SOON), HOW_SOON)
        aff = q.get("affected_objects")
        for a in (aff if isinstance(aff, (list, tuple)) else []):
            key = (threat, str(q.get("effect") or ""), _bare(a))
            if t and threat and key[2]:
                if soon.get(key) and soon[key] != t:
                    notes.append(f"how_soon_conflict_{threat}_{key[2]}({soon[key]}|{t})")
                soon[key] = _sooner(soon.get(key, ""), t)
    for n in graph_a.get("nodes") or []:
        if isinstance(n, dict) and sizes.get(_bare(n.get("id"))):
            n[SIZE] = sizes[_bare(n.get("id"))]
    for e in graph_a.get("edges") or []:
        if isinstance(e, dict):
            w = soon.get((_bare(e.get("source")), str(e.get("effect") or ""),
                          _bare(e.get("target"))))
            if w:
                e[SOON] = w
    if notes:
        graph_a["field_notes"] = notes
    return notes


def counts(graph: Any) -> dict:
    """How many hazards carry a size and how many arrows a how_soon."""
    g = graph if isinstance(graph, dict) else {}
    return {"sized": sum(1 for n in g.get("nodes") or [] if isinstance(n, dict) and n.get(SIZE)),
            "timed": sum(1 for e in g.get("edges") or [] if isinstance(e, dict) and e.get(SOON))}
