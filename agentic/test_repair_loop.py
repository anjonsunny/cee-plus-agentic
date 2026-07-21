"""Hermetic tests for Loop 1 (agentic/repair_loop.py). No VLM: the model is
played by fake query functions with scripted behavior.

Covered: each violation kind fires correctly; the semantic boundary (legal
but wrong states never trigger); the three stopping rules (clean, no_change
oscillation guard, cap_reached); trace completeness; integration through
run_perception with the loop's history in the run record.

Run:  pytest agentic/test_repair_loop.py -q
"""
from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from agentic import perception  # noqa: E402
from agentic.repair_loop import (  # noqa: E402
    MAX_REPAIR_ROUNDS,
    build_repair_prompt,
    detect_violations,
    repair_entities,
)

# ── detect_violations: each rule, and the boundary ──────────────────────


def test_family_name_violation_cites_members():
    v = detect_violations([{"label": "vehicle", "state": "intact",
                            "bbox": [0, 0, 10, 10]}])
    assert len(v) == 1 and v[0].kind == "family_name_as_label"
    assert "tanker_truck" in v[0].instruction        # members offered as choices


def test_out_of_vocab_label_violation():
    v = detect_violations([{"label": "zamboni", "state": "stationary",
                            "bbox": [0, 0, 10, 10]}])
    assert [x.kind for x in v] == ["label_out_of_vocab"]


def test_deliberate_other_is_not_a_violation():
    """'other' is the escape hatch; using it on purpose is legal."""
    assert detect_violations([{"label": "other", "state": "intact",
                               "description": "odd machine",
                               "bbox": [0, 0, 10, 10]}]) == []


def test_state_out_of_vocab_violation():
    v = detect_violations([{"label": "person", "state": "floating",
                            "bbox": [0, 0, 10, 10]}])
    assert [x.kind for x in v] == ["state_out_of_vocab"]


def test_missing_bbox_violation():
    v = detect_violations([{"label": "smoke", "state": "billowing"}])
    assert [x.kind for x in v] == ["missing_anchor_bbox"]


def test_semantic_boundary_wrong_but_legal_state_never_fires():
    """THE boundary: a drowning child mislabeled 'swimming' is a legal
    state word. No rule fires. The loop must not coach perception."""
    assert detect_violations([{"label": "child", "state": "swimming",
                               "bbox": [0, 0, 10, 10]}]) == []


def test_clean_entity_no_violations():
    assert detect_violations([{"label": "house", "state": "burning",
                               "bbox": [5, 5, 50, 50]}]) == []


# ── Rule 5: caption-entity completeness ─────────────────────────────────

from agentic.repair_loop import caption_labels  # noqa: E402

C_CAPTION = ("A tanker truck leaks fuel onto a rural road while a brush fire "
             "burns nearby; the driver stands on the road making a phone call.")


def test_caption_labels_resolve_through_vocabulary():
    labels = caption_labels(C_CAPTION)
    # Bigram beats unigram: tanker_truck, not truck. fuel->spill and
    # driver->person resolve through the synonym map.
    assert "tanker_truck" in labels and "spill" in labels
    assert "person" in labels and "fire" in labels and "road" in labels


def test_caption_missing_entity_fires():
    """The C_tanker regression: one entity for a captioned multi-hazard
    scene must fire caption_entity_missing for each named absentee."""
    v = detect_violations(
        [{"label": "fire", "state": "burning", "bbox": [0, 0, 9, 9]}],
        caption=C_CAPTION,
    )
    kinds = {x.raw_label for x in v if x.kind == "caption_entity_missing"}
    assert "tanker_truck" in kinds and "spill" in kinds and "person" in kinds
    assert "fire" not in kinds                    # present, not flagged


def test_caption_person_family_is_lenient():
    """Caption 'driver' is satisfied by man_1 (person family)."""
    v = detect_violations(
        [{"label": "man", "state": "standing", "bbox": [0, 0, 9, 9]}],
        caption="The driver stands on the road.",
    )
    kinds = [x for x in v if x.kind == "caption_entity_missing"]
    assert all(x.raw_label != "person" for x in kinds)


def test_caption_vehicle_needs_exact_label():
    """Caption tanker_truck is NOT satisfied by car_1 (the round 1 bug)."""
    v = detect_violations(
        [{"label": "car", "state": "intact", "bbox": [0, 0, 9, 9]}],
        caption="A tanker truck on the road.",
    )
    assert any(x.raw_label == "tanker_truck" and x.kind == "caption_entity_missing"
               for x in v)


def test_caption_instruction_allows_standing_ground():
    v = detect_violations([], caption="A dog in the park.")
    assert any("leave your list unchanged" in x.instruction for x in v)


def test_no_caption_no_rule5():
    assert detect_violations([{"label": "fire", "state": "burning",
                               "bbox": [0, 0, 9, 9]}], caption="") == []


def test_brush_fire_is_one_demand_not_two():
    """C_tanker artifact: 'brush fire' must resolve to fire alone, never
    demand a separate brush entity."""
    labels = caption_labels("a brush fire burns nearby")
    assert "fire" in labels and "brush" not in labels


def test_plural_synonym_resolves_chemicals_to_spill():
    """D_aerial miss: 'leaking chemicals' must demand a spill."""
    labels = caption_labels("an overturned tanker truck leaking chemicals")
    assert "spill" in labels and "tanker_truck" in labels


def test_operating_state_is_stationary():
    from agentic.perception import state_kind
    assert state_kind("operating") == "normal"


def test_prompt_quotes_entities_and_numbers_problems():
    entities = [{"label": "vehicle", "state": "intact", "bbox": [0, 0, 9, 9]},
                {"label": "person", "state": "floating", "bbox": [1, 1, 5, 5]}]
    prompt = build_repair_prompt(entities, detect_violations(entities))
    assert '"vehicle"' in prompt and "1." in prompt and "2." in prompt
    assert "Fix ONLY the listed problems" in prompt


# ── The loop: stopping rules and trace ──────────────────────────────────

CLEAN = {"label": "house", "state": "burning", "bbox": [5, 5, 50, 50]}
BROKEN = {"label": "vehicle", "state": "intact", "bbox": [0, 0, 9, 9]}
FIXED = {"label": "tanker_truck", "state": "intact", "bbox": [0, 0, 9, 9]}


def test_clean_on_arrival_no_model_call():
    calls = []
    fixed, trace = repair_entities([CLEAN], lambda p: calls.append(p) or [CLEAN])
    assert trace.clean_on_arrival and trace.stopped_reason == "clean"
    assert calls == []                       # a clean answer costs zero calls
    assert fixed == [CLEAN]


def test_one_round_fix():
    fixed, trace = repair_entities([BROKEN], lambda p: [FIXED])
    assert trace.stopped_reason == "clean"
    assert len(trace.rounds) == 1 and trace.rounds[0].changed
    assert fixed == [FIXED]


def test_oscillation_guard_stops_on_identical_answer():
    """A model that returns the same broken list twice ends the loop:
    asking again would add confidence, not information."""
    fixed, trace = repair_entities([BROKEN], lambda p: [dict(BROKEN)])
    assert trace.stopped_reason == "no_change"
    assert len(trace.rounds) == 1
    # The violation is still visible downstream, not hidden.
    assert detect_violations(fixed)


def test_cap_reached_on_ever_changing_broken_answers():
    """A model that keeps producing NEW broken answers hits the hard cap."""
    counter = {"n": 0}

    def churn(prompt):
        counter["n"] += 1
        return [{"label": "vehicle", "state": "intact",
                 "bbox": [counter["n"], 0, counter["n"] + 9, 9]}]

    fixed, trace = repair_entities([BROKEN], churn)
    assert trace.stopped_reason == "cap_reached"
    assert len(trace.rounds) == MAX_REPAIR_ROUNDS == counter["n"]


def test_broken_model_answer_keeps_last_good_list():
    fixed, trace = repair_entities([BROKEN], lambda p: [])
    assert fixed == [BROKEN] and trace.stopped_reason == "no_change"


# ── Integration through run_perception ──────────────────────────────────


@pytest.fixture()
def fake_geometry(monkeypatch):
    def fake_detect(image, entities):
        cands = {}
        for i, e in enumerate(entities):
            p = perception._detector_phrase(e["label"], e.get("description", ""))
            e["_phrase"] = p
            cands.setdefault(p, []).append(
                {"score": 0.8, "bbox": [10 + i * 30, 10, 100 + i * 30, 120]}
            )
        return cands

    monkeypatch.setattr(perception, "detect_candidates", fake_detect)
    monkeypatch.setattr(perception, "mask_for_box",
                        lambda image, bbox: Image.new("L", image.size, 255))


def test_run_perception_records_repair_trace(tmp_path, fake_geometry):
    img = tmp_path / "scene.jpg"
    Image.new("RGB", (300, 200), "gray").save(img)

    # The scripted model fixes its family-name violation when asked once.
    result = perception.run_perception(
        img,
        entities=[{"label": "vehicle", "state": "intact",
                   "description": "tanker", "bbox": [12, 12, 98, 118]}],
        with_masks=False,
        out_dir=tmp_path / "out",
        repair_query_fn=lambda prompt: [
            {"label": "tanker_truck", "state": "intact",
             "description": "tanker", "bbox": [12, 12, 98, 118]}
        ],
    )
    assert result.repair_trace is not None
    assert result.repair_trace["stopped_reason"] == "clean"
    assert result.detected_objects[0].object_id == "tanker_truck_1"
    assert not result.detected_objects[0].family_name_as_label


def test_run_perception_standin_without_repair_fn_skips_loop(tmp_path, fake_geometry):
    img = tmp_path / "scene.jpg"
    Image.new("RGB", (300, 200), "gray").save(img)
    result = perception.run_perception(
        img,
        entities=[{"label": "house", "state": "burning", "bbox": [12, 12, 98, 118]}],
        with_masks=False,
        out_dir=tmp_path / "out",
    )
    assert result.repair_trace is None       # loop not run, honestly recorded
