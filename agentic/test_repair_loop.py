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

# ── F9: fluid-convention-aware caption completeness ─────────────────────
#
# The A_fire regression: "a house on fire" must not demand a fire entity
# when the house already carries state 'burning' — the caption's fire IS
# the house's state (fluid convention). Free-burning phrasings and always-
# diffuse media stay strict.


def test_attached_fire_satisfies_caption_mention():
    """The exact A_fire failure: house·burning present, caption says
    'on fire' -> NO caption_entity_missing for fire."""
    v = detect_violations(
        [{"label": "house", "state": "burning", "bbox": [0, 0, 9, 9]},
         {"label": "person", "state": "standing", "bbox": [1, 1, 9, 9]}],
        caption="A house on fire at night; a person stands nearby.",
    )
    assert not any(x.raw_label == "fire" and x.kind == "caption_entity_missing"
                   for x in v)


def test_caption_fire_still_fires_when_nothing_burns():
    v = detect_violations(
        [{"label": "house", "state": "intact", "bbox": [0, 0, 9, 9]}],
        caption="A fire near the house.",
    )
    assert any(x.raw_label == "fire" and x.kind == "caption_entity_missing"
               for x in v)


def test_free_burning_phrases_stay_strict():
    """'brush fire' is free-burning: a burning house does NOT account for
    it — the fire entity is still owed."""
    v = detect_violations(
        [{"label": "house", "state": "burning", "bbox": [0, 0, 9, 9]}],
        caption="A brush fire burns behind a house.",
    )
    assert any(x.raw_label == "fire" and x.kind == "caption_entity_missing"
               for x in v)


def test_diffuse_media_stay_strict():
    """smoke is always its own entity; a burning house never satisfies a
    caption's smoke mention."""
    v = detect_violations(
        [{"label": "house", "state": "burning", "bbox": [0, 0, 9, 9]}],
        caption="Smoke rises from a house on fire.",
    )
    kinds = {x.raw_label for x in v if x.kind == "caption_entity_missing"}
    assert "smoke" in kinds and "fire" not in kinds


def test_spill_stays_strict_beside_leaking_producer():
    """Producer-and-medium rule: tanker·leaking does NOT satisfy the
    caption's fuel/spill — the spill entity is separately owed."""
    v = detect_violations(
        [{"label": "tanker_truck", "state": "leaking", "bbox": [0, 0, 9, 9]}],
        caption="A tanker truck leaks fuel.",
    )
    assert any(x.raw_label == "spill" and x.kind == "caption_entity_missing"
               for x in v)


def test_flood_satisfied_by_flooded_entity():
    v = detect_violations(
        [{"label": "car", "state": "flooded", "bbox": [0, 0, 9, 9]},
         {"label": "road", "state": "flooded", "bbox": [0, 0, 9, 9]}],
        caption="Floodwater covers the road; a flooded car sits in it.",
    )
    assert not any(x.raw_label == "water" for x in v)


# ── F10 follow-up: de-priming the P3 word list (road·burning autopsy) ───


def test_state_reminder_normal_family_leads():
    """VLM positional bias: NORMAL must be the first group offered, and
    the burning-first colon chain ('what you see: hazard-bearing:
    burning') must be gone."""
    from agentic.repair_loop import _STATE_WORD_REMINDER
    assert _STATE_WORD_REMINDER.index("normal:") \
        < _STATE_WORD_REMINDER.index("at-risk:") \
        < _STATE_WORD_REMINDER.index("hazard-bearing:")
    assert "most entities in most scenes are normal" in _STATE_WORD_REMINDER


def test_p3_instruction_names_entity_own_condition():
    v = detect_violations(
        [{"label": "road", "state": "muddy", "bbox": [0, 0, 9, 9]}],
        caption="")
    ins = next(x.instruction for x in v if x.kind == "state_out_of_vocab")
    assert "THIS entity's own condition" in ins
    assert "not the scene's overall situation" in ins
    # no colon chain: "what you see: hazard-bearing" must not appear
    assert "what you see: hazard-bearing" not in ins


def test_paved_is_now_a_legal_normal_state():
    """The model's honest answer for the road no longer draws a ticket."""
    from agentic import perception
    assert perception.state_kind("paved") == "normal"
    v = detect_violations(
        [{"label": "road", "state": "paved", "bbox": [0, 0, 9, 9]}],
        caption="A house on fire on a residential street.")
    assert not any(x.kind == "state_out_of_vocab" for x in v)


def test_spill_active_draws_no_p3_ticket():
    """Label-aware synonym reaches P3: the model's honest 'active' for a
    spill is legal; a car·'active' still tickets."""
    v = detect_violations(
        [{"label": "spill", "state": "active", "bbox": [0, 0, 9, 9]}],
        caption="")
    assert not any(x.kind == "state_out_of_vocab" for x in v)
    v2 = detect_violations(
        [{"label": "car", "state": "active", "bbox": [0, 0, 9, 9]}],
        caption="")
    assert any(x.kind == "state_out_of_vocab" for x in v2)


# ── P1 fix + P6 duplicate check (E_collapse ui_20fb0754) ────────────────


def test_family_name_ticket_quotes_description_and_opens_full_vocab():
    """The police-car steering bug: the ticket must carry the model's own
    description and permit ANY family, not just the misused one."""
    v = detect_violations(
        [{"label": "infrastructure", "state": "stationary",
          "description": "police car with flashing lights",
          "bbox": [138, 687, 591, 935]}],
        caption="")
    ins = next(x.instruction for x in v if x.kind == "family_name_as_label")
    assert "police car with flashing lights" in ins
    assert "ANY family" in ins
    assert "road" in ins                       # members list stays, as courtesy


def test_duplicate_humans_flagged():
    """Sunny's exact run: person_2 ≈ police_officer_1 (IoU ~0.94)."""
    v = detect_violations(
        [{"label": "person", "state": "standing",
          "description": "police officer behind caution tape",
          "bbox": [443, 660, 558, 941]},
         {"label": "police_officer", "state": "standing",
          "description": "police officer behind caution tape",
          "bbox": [438, 658, 563, 937]}],
        caption="")
    dup = [x for x in v if x.kind == "duplicate_entity"]
    assert len(dup) == 1
    assert "police officer behind caution tape" in dup[0].instruction
    assert "IoU" in dup[0].instruction


def test_distinct_adjacent_officers_not_flagged():
    v = detect_violations(
        [{"label": "police_officer", "state": "standing",
          "bbox": [438, 658, 563, 937]},
         {"label": "police_officer", "state": "standing",
          "bbox": [576, 650, 699, 937]}],       # side by side, IoU ~0
        caption="")
    assert not any(x.kind == "duplicate_entity" for x in v)


def test_cross_group_overlap_not_flagged():
    """A dog overlapping a person (carried dog) is NOT a duplicate."""
    v = detect_violations(
        [{"label": "person", "state": "standing", "bbox": [10, 10, 100, 200]},
         {"label": "dog", "state": "healthy", "bbox": [12, 12, 98, 198]}],
        caption="")
    assert not any(x.kind == "duplicate_entity" for x in v)


def test_duplicate_check_survives_malformed_boxes():
    v = detect_violations(
        [{"label": "person", "state": "standing", "bbox": [7]},
         {"label": "person", "state": "standing", "bbox": None},
         {"label": "responder", "state": "standing"}],
        caption="")
    assert not any(x.kind == "duplicate_entity" for x in v)


# ── P7 / F21: caption CONDITION words, not just caption entities ─────────

def test_caption_danger_states_uses_only_the_closed_vocabulary():
    from agentic.repair_loop import caption_danger_states
    got = caption_danger_states(
        "another child floats motionless and unconscious face down")
    assert "unconscious" in got                  # a legal at_risk state
    assert "motionless" not in got               # not in the vocabulary
    # normal-state words are never evidence of anything
    assert caption_danger_states("a child is swimming in a pool") == {}
    assert caption_danger_states("") == {}
    assert caption_danger_states(None) == {}


def test_p7_fires_when_the_caption_condition_is_not_declared():
    """The B_pool regression: caption says unconscious, list says swimming.
    'swimming' is a legal state and the entity exists, so nothing else fires."""
    entities = [{"label": "child", "state": "swimming",
                 "anchor_bbox": [0, 0, 9, 9]},
                {"label": "child", "state": "drowning",
                 "anchor_bbox": [0, 0, 9, 9]}]
    caption = ("A child is struggling in a swimming pool; another child "
               "floats motionless and unconscious face down farther away.")
    kinds = [v.kind for v in detect_violations(entities, caption)]
    assert "caption_state_contradiction" in kinds


def test_p7_silent_when_the_condition_is_declared():
    entities = [{"label": "child", "state": "unconscious",
                 "anchor_bbox": [0, 0, 9, 9]}]
    kinds = [v.kind for v in detect_violations(entities, "a child is unconscious")]
    assert "caption_state_contradiction" not in kinds


def test_p7_silent_on_the_safe_scene():
    """F_park: no danger word in the caption, so the check must say nothing."""
    entities = [{"label": "person", "state": "standing",
                 "anchor_bbox": [0, 0, 9, 9]}]
    caption = "Families relax in a park on a sunny afternoon."
    kinds = [v.kind for v in detect_violations(entities, caption)]
    assert "caption_state_contradiction" not in kinds


def test_p7_never_names_the_right_state():
    """Iron rule 5: quote the caption, never supply the answer."""
    entities = [{"label": "child", "state": "swimming",
                 "anchor_bbox": [0, 0, 9, 9]}]
    v = [x for x in detect_violations(entities, "a child is unconscious")
         if x.kind == "caption_state_contradiction"][0]
    text = v.instruction.lower()
    assert "unconscious" in text                 # the caption's own word
    assert "should be" not in text and "change it to" not in text
    assert "leave it" in text or "unchanged" in text   # standing is allowed


def test_p7_normalises_both_sides_through_one_function():
    """Both sides go through arm_b_canonical_state: Arm B's declined folds
    first, Arm A's deeper map underneath. Arm A alone is not enough (it folds
    struggling into trapped); Arm B alone is not enough (it cannot resolve
    'down' or 'on fire')."""
    from agentic.perception import arm_b_canonical_state
    from agentic.repair_loop import caption_danger_states
    assert arm_b_canonical_state("down") == "fallen"        # from Arm A
    assert arm_b_canonical_state("on fire") == "burning"    # from Arm A
    assert arm_b_canonical_state("struggling") == "struggling"   # Arm B declines
    assert arm_b_canonical_state("stuck") == "trapped"      # other folds intact
    assert caption_danger_states("a person is struggling")["struggling"] == "struggling"


def test_p7_silent_when_a_synonym_already_covers_the_caption():
    """The declared side is canonicalised too, so an entity carrying a
    SYNONYM of the caption's word must not raise a disagreement."""
    entities = [{"label": "house", "state": "on fire",
                 "anchor_bbox": [0, 0, 9, 9]}]
    kinds = [v.kind for v in detect_violations(entities, "a house is burning")]
    assert "caption_state_contradiction" not in kinds


def test_p7_adds_no_words_to_arm_b_vocabulary():
    """The depth fix must come from importing Arm A, never from extending
    Arm B's own lists."""
    from agentic import vocabulary
    for name in ("trapped", "fallen"):
        assert not any(name in str(getattr(vocabulary, a, ""))
                       for a in dir(vocabulary) if a.startswith("_EXTRA"))


def test_p7_ticket_quotes_the_captions_own_word_not_the_canonical_form():
    """Matching happens on the canonical form; the TICKET must still show the
    word the caption actually used. Showing 'trapped' when the caption said
    'struggling' would put a word in the caption's mouth — we would be
    quoting something the given text never said."""
    entities = [{"label": "person", "state": "standing",
                 "anchor_bbox": [0, 0, 9, 9]}]
    v = [x for x in detect_violations(
        entities, "a person is struggling and another is face down")
        if x.kind == "caption_state_contradiction"][0]
    line = v.instruction.splitlines()[0]      # the template line
    assert '"struggling"' in line and '"down"' in line
    assert "trapped" not in line and "fallen" not in line


def test_arm_b_has_exactly_one_state_normaliser():
    """A declined fold is only real if nothing bypasses it. Any Arm B module
    calling Arm A's canonicalize_state directly would silently re-apply the
    fold, and two checks in the same file could disagree about one word."""
    import pathlib
    import re
    root = pathlib.Path(__file__).parent
    offenders = []
    for f in root.glob("*.py"):
        if f.name.startswith("test_"):
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"\bcanonicalize_state\s*\(", line):
                offenders.append(f"{f.name}:{i}")
    # perception.py defines arm_b_canonical_state and is the ONLY place the
    # frozen Arm A function may be invoked.
    assert all(o.startswith("perception.py") for o in offenders), offenders
