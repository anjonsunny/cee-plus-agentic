"""Deterministic bbox geometry: the FREE spatial evidence (Stage 2 merge).

WHY THIS EXISTS
===============
Proximity at-risk ("person_1 standing next to the burning house") is a
spatial judgment. The declared state already contains geometry — every
entity has a bbox — so the CHEAPEST tool runs first: pure math over
declared boxes, no pixels, no model call. Its output feeds the merged
assessment prompt as "spatial hints".

THE G3 DISCIPLINE
=================
2D adjacency is a HINT, not a verdict: image-plane closeness is not
world closeness (a person 50m behind the house overlaps its bbox).
Geometry NOMINATES candidate pairs; the model (and later look_at)
decides. Hints therefore carry the relation, the numbers behind it,
and never the word "at risk".

Thresholds are priors, calibrated later against the six scenes:
  overlap   -> boxes intersect
  adjacent  -> min edge gap <= ADJACENT_FRAC of the image diagonal
"""
from __future__ import annotations

from typing import Any

ADJACENT_FRAC = 0.05     # gap <= 5% of image diagonal counts as adjacent


def _valid(b: Any) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b)
            and b[2] > b[0] and b[3] > b[1])


def box_gap(a: list[float], b: list[float]) -> float:
    """Smallest edge-to-edge distance between two boxes; 0 if they
    intersect or touch."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def boxes_overlap(a: list[float], b: list[float]) -> bool:
    return (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def overlap_fraction(a: list[float], b: list[float]) -> float:
    """Intersection area as a fraction of the SMALLER box. 1.0 = the
    smaller box sits entirely inside the larger; ~0 = corner clip. This
    is the discriminative number "overlap, 0px" hides (A_fire 2026-07-22:
    a whole-street road box and large house boxes made every pair read
    identically as overlap/0px)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]),
                  (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / smaller if smaller > 0 else 0.0


def center_distance(a: list[float], b: list[float]) -> float:
    dx = (a[0] + a[2]) / 2 - (b[0] + b[2]) / 2
    dy = (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
    return (dx * dx + dy * dy) ** 0.5


def spatial_hints(entities: list[Any],
                  image_size: list[int] | None) -> list[dict[str, Any]]:
    """Nominate hazard↔entity pairs by declared-box geometry.

    Input: DetectedObject-likes (objects or dicts). Output hints, one per
    (hazard, other) pair that overlaps or sits adjacent:
      {hazard, other, other_state, relation: overlap|adjacent,
       gap_px, boxes: {hazard: [...], other: [...]}}
    Malformed/missing boxes are skipped (Rule 1a: raw shapes never crash
    the scanner); entities can't be near themselves; hazard-hazard pairs
    are skipped (a fire near a spill is Stage-4 material, not at-risk)."""
    dicts = [e if isinstance(e, dict) else e.model_dump() for e in entities]
    diag = ((image_size[0] ** 2 + image_size[1] ** 2) ** 0.5
            if image_size and len(image_size) == 2 else None)
    hazards = [e for e in dicts if e.get("state_kind") == "hazard_bearing"]
    others = [e for e in dicts if e.get("state_kind") != "hazard_bearing"]
    hints: list[dict[str, Any]] = []
    for hz in hazards:
        hb = hz.get("bbox")
        if not _valid(hb):
            continue
        for ot in others:
            ob = ot.get("bbox")
            if not _valid(ob):
                continue
            if boxes_overlap(hb, ob):
                relation, gap = "overlap", 0.0
            else:
                gap = box_gap(hb, ob)
                if diag is None or gap > ADJACENT_FRAC * diag:
                    continue
                relation = "adjacent"
            hints.append({"hazard": hz.get("object_id"),
                          "other": ot.get("object_id"),
                          "other_state": ot.get("state"),
                          "relation": relation,
                          "gap_px": round(gap, 1),
                          "overlap_frac": round(overlap_fraction(hb, ob), 2),
                          "center_dist_px": round(center_distance(hb, ob), 1),
                          "boxes": {"hazard": list(hb), "other": list(ob)}})
    return hints


def hints_as_prompt_lines(hints: list[dict[str, Any]]) -> str:
    """Render hints for the assessment prompt: derived relation in words
    (what the model reasons over), raw numbers kept for audit. Never says
    'at risk' — nomination only (G3)."""
    if not hints:
        return "  (no entity is near an active hazard by declared geometry)"
    lines = []
    for h in hints:
        if h["relation"] == "overlap":
            frac = h.get("overlap_frac")
            cd = h.get("center_dist_px")
            detail = ""
            if frac is not None and cd is not None:
                detail = (f" ({frac:.0%} of the smaller box; centers "
                          f"≈ {cd:.0f}px apart)")
            rel = f"overlaps{detail}"
        else:
            rel = f"is adjacent to (gap ≈ {h['gap_px']:.0f}px)"
        lines.append(f"  - {h['other']} (state={h['other_state']}) {rel} "
                     f"{h['hazard']}")
    return "\n".join(lines)
