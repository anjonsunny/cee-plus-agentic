"""Direction by chain: after a hazard is removed, did the danger in the
advice ease or grow? Each arrow is weighed by what its CHAIN can do.

The frozen arm answers this by adding a fixed weight per arrow
("may_harm" 2, "may_spread_to" 1.5) and calls a try "escalated" when the
total rises. It counts arrows only (Sunny, 2026-11-13: "that's the
problem"): on C_tanker ui_22840b13, removing spill_1 lowered the danger
level 8 to 6 and calmed the advice, yet two new arrows from the burning
brush to grass and a tree pushed the total from 7.5 to 9.0, "escalated".

Sunny's rule: weigh an arrow by what the chain can do.

    fire ──► tanker ──► spill ──► man

"fire may spread to tanker" stops at the tanker when counted alone. But
the tanker leaks, and the leak reaches the driver, so that arrow is a
threat to a person.

    weight of an arrow = its effect weight x the worst thing its chain reaches
                         (a chain that ends at a life: x2; anything else: x1)

Who makes the chain: CODE. The links are the arrows the model already
wrote in the same run (its own graph and the graph of its advice); code
only follows them. No model call, no prompt, no ground truth. The limit
is honest: a link the model did not write is a link the chain does not
have.

The effect weights, the life labels and the x2 are the frozen arm's own
(`_EFFECT_THREAT`, `_LIFE_LABELS`), imported, so the only thing that
differs from its count is the chain. The frozen label is never changed;
this is shown next to it.
"""
from __future__ import annotations

from typing import Any

from agentic import intervention4 as iv4

LIFE, OTHER = 2.0, 1.0


def _edges(graph: Any) -> list[tuple[str, str, str]]:
    """(source, effect, target) for every well-formed arrow, once each."""
    seen: list[tuple[str, str, str]] = []
    g = graph if isinstance(graph, dict) else {}
    for e in (g.get("edges") if isinstance(g.get("edges"), list) else []):
        if not isinstance(e, dict):
            continue
        s, t = str(e.get("source") or "").strip(), str(e.get("target") or "").strip()
        eff = str(e.get("effect") or "").strip().lower()
        if s and t and s != t and (s, eff, t) not in seen:
            seen.append((s, eff, t))
    return seen


def links_of(result: dict) -> dict[str, list[str]]:
    """What leads to what, in the model's own words for this run: every
    arrow of its own graph and of the graph of its advice."""
    out: dict[str, list[str]] = {}
    for key in ("graph_b", "causal_graph"):
        for s, _eff, t in _edges((result or {}).get(key)):
            if t not in out.setdefault(s, []):
                out[s].append(t)
    return out


def _is_life(oid: str, labels: dict[str, str]) -> bool:
    iv = iv4._frozen()
    label = labels.get(oid) or iv._base_label(oid)
    return iv._canonical_label(label) in iv._LIFE_LABELS or label in iv._LIFE_LABELS


def chain(target: str, links: dict[str, list[str]], labels: dict[str, str]) -> dict:
    """Follow the links from `target`. Returns {"path": the shortest way to
    the worst thing reached, "ends_at": that thing, "life": bool}. A loop
    (fire worsens spill, spill worsens fire) is walked once."""
    prev: dict[str, str | None] = {target: None}
    order = [target]
    i = 0
    while i < len(order):                       # breadth first: shortest path wins
        for nxt in links.get(order[i], []):
            if nxt not in prev:
                prev[nxt] = order[i]
                order.append(nxt)
        i += 1
    end = next((n for n in order if _is_life(n, labels)), target)
    path = [end]
    while prev.get(path[-1]) is not None:
        path.append(prev[path[-1]])             # type: ignore[arg-type]
    return {"path": path[::-1], "ends_at": end, "life": _is_life(end, labels)}


def weigh(result: dict) -> dict:
    """Every arrow of the run's advice graph with its chain and weight.
    Returns {"total": float, "arrows": [{source, effect, target, base,
    ends_at, path, factor, weight}]}."""
    result = iv4.clean_result(result)
    iv = iv4._frozen()
    labels = {str(o.get("object_id") or ""): str(o.get("label") or "")
              for o in result["detected_objects"]}
    links = links_of(result)
    arrows = []
    for s, eff, t in _edges(result.get("causal_graph")):
        base = float(iv._EFFECT_THREAT.get(eff, 1.0 if eff else 0.0))
        c = chain(t, links, labels)
        factor = LIFE if c["life"] else OTHER
        arrows.append({"source": s, "effect": eff, "target": t, "base": base,
                       "ends_at": c["ends_at"], "path": c["path"],
                       "factor": factor, "weight": round(base * factor, 2)})
    return {"total": round(sum(a["weight"] for a in arrows), 2), "arrows": arrows}


def direction(before: dict, after: dict) -> dict:
    """Before against after, by chain. "eased" when the total fell,
    "grew" when it rose, "unchanged" otherwise. Also carries the frozen
    count for the same two graphs, so the two can be read side by side."""
    b, a = weigh(before), weigh(after)
    delta = round(a["total"] - b["total"], 2)
    word = "unchanged" if abs(delta) < 1e-9 else ("eased" if delta < 0 else "grew")
    iv = iv4._frozen()
    try:
        frozen = iv.graph_threat_direction(iv4.clean_result(before).get("causal_graph"),
                                           iv4.clean_result(after).get("causal_graph"))
    except Exception:
        frozen = {}
    return {"before": b["total"], "after": a["total"], "delta": delta,
            "direction": word, "before_arrows": b["arrows"], "after_arrows": a["arrows"],
            "frozen_count": {k: frozen.get(k) for k in ("before", "after", "direction")}}
