"""Hermetic tests for the deterministic geometry scanner.

Run:  pytest agentic/test_geometry.py -q
"""
from __future__ import annotations

from agentic.geometry import (
    box_gap,
    boxes_overlap,
    hints_as_prompt_lines,
    spatial_hints,
)


def _e(oid, kind, bbox, state="standing"):
    return {"object_id": oid, "state_kind": kind, "bbox": bbox,
            "state": state}


SIZE = [1000, 800]      # diagonal ~1280 -> adjacency threshold ~64px


# ── Primitives ──────────────────────────────────────────────────────────


def test_gap_and_overlap_math():
    assert boxes_overlap([0, 0, 10, 10], [5, 5, 15, 15])
    assert not boxes_overlap([0, 0, 10, 10], [20, 0, 30, 10])
    assert box_gap([0, 0, 10, 10], [20, 0, 30, 10]) == 10.0
    assert box_gap([0, 0, 10, 10], [5, 5, 15, 15]) == 0.0
    # diagonal separation: 30px right, 40px down -> 50px
    assert box_gap([0, 0, 10, 10], [40, 50, 60, 70]) == 50.0


# ── Nomination behavior ─────────────────────────────────────────────────


def test_overlapping_person_is_nominated():
    hints = spatial_hints([
        _e("house_1", "hazard_bearing", [100, 100, 500, 500], "burning"),
        _e("person_1", "normal", [450, 300, 520, 480])], SIZE)
    assert len(hints) == 1
    h = hints[0]
    assert (h["hazard"], h["other"], h["relation"]) == \
        ("house_1", "person_1", "overlap")
    assert h["boxes"]["hazard"] == [100, 100, 500, 500]


def test_adjacent_within_threshold_far_beyond_it():
    near = _e("person_1", "normal", [520, 100, 580, 300])   # gap 20px
    far = _e("dog_1", "normal", [900, 100, 950, 200])       # gap 400px
    hints = spatial_hints([
        _e("house_1", "hazard_bearing", [100, 100, 500, 500], "burning"),
        near, far], SIZE)
    assert [h["other"] for h in hints] == ["person_1"]
    assert hints[0]["relation"] == "adjacent" and hints[0]["gap_px"] == 20.0


def test_hazard_hazard_pairs_and_self_are_skipped():
    hints = spatial_hints([
        _e("fire_1", "hazard_bearing", [0, 0, 100, 100], "spreading"),
        _e("spill_1", "hazard_bearing", [50, 50, 150, 150], "seeping")], SIZE)
    assert hints == []          # hazard near hazard is not an at-risk hint


def test_no_hazard_means_no_hints():
    assert spatial_hints([_e("car_1", "normal", [0, 0, 50, 50])], SIZE) == []


def test_at_risk_entities_are_still_nominated():
    """A distress entity near a hazard still gets a geometry hint (the
    model may cite both distress and proximity evidence)."""
    hints = spatial_hints([
        _e("fire_1", "hazard_bearing", [0, 0, 100, 100], "spreading"),
        _e("child_1", "at_risk", [90, 90, 150, 150], "trapped")], SIZE)
    assert hints and hints[0]["other"] == "child_1"


# ── Rule 1a: malformed shapes never crash the scanner ───────────────────


def test_malformed_boxes_are_skipped():
    hints = spatial_hints([
        _e("house_1", "hazard_bearing", [100], "burning"),      # 1-element
        _e("fire_1", "hazard_bearing", None, "spreading"),      # missing
        _e("spill_1", "hazard_bearing", [10, 10, 5, 5]),        # inverted
        _e("person_1", "normal", ["a", 0, 10, 10])], SIZE)      # strings
    assert hints == []


def test_missing_image_size_disables_adjacency_keeps_overlap():
    ents = [_e("house_1", "hazard_bearing", [0, 0, 100, 100], "burning"),
            _e("person_1", "normal", [90, 90, 150, 150]),       # overlap
            _e("dog_1", "normal", [110, 0, 150, 50])]           # gap 10
    hints = spatial_hints(ents, None)
    assert [h["other"] for h in hints] == ["person_1"]  # no diag -> no adj


# ── Prompt rendering (G3: nominate, never convict) ──────────────────────


def test_prompt_lines_use_words_not_verdicts():
    hints = spatial_hints([
        _e("house_1", "hazard_bearing", [100, 100, 500, 500], "burning"),
        _e("person_1", "normal", [520, 100, 580, 300])], SIZE)
    txt = hints_as_prompt_lines(hints)
    assert "person_1" in txt and "adjacent" in txt and "gap ≈ 20px" in txt
    assert "at risk" not in txt.lower()          # nomination, not verdict
    assert "(no entity is near" in hints_as_prompt_lines([])


# ── Overlap richness (A_fire road case: "overlap, 0px" everywhere) ──────


def test_overlap_hints_carry_fraction_and_center_distance():
    ents = [_e("house_1", "hazard_bearing", [0, 0, 1000, 800], "burning"),
            _e("person_1", "normal", [100, 100, 200, 300], "standing")]
    hints = spatial_hints(ents, [1000, 800])
    h = hints[0]
    assert h["relation"] == "overlap"
    assert h["overlap_frac"] == 1.0          # person fully inside house box
    assert h["center_dist_px"] > 0
    txt = hints_as_prompt_lines(hints)
    assert "100%" in txt and "centers" in txt


def test_corner_clip_reads_small_fraction():
    ents = [_e("house_1", "hazard_bearing", [0, 0, 100, 100], "burning"),
            _e("car_1", "normal", [90, 90, 190, 190], "parked")]
    hints = spatial_hints(ents, [1000, 800])
    assert hints[0]["overlap_frac"] == 0.01   # 10x10 of a 100x100 box
