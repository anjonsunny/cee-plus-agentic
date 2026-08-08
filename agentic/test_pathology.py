"""Arm B's pathology register (F33) — the sixth named pathology.

Hermetic. No models.

Run:  pytest agentic/test_pathology.py -q
"""
from __future__ import annotations

from agentic.pathology import (ARM_B_PATHOLOGIES, CAPITULATION, consequence,
                               detect_capitulation, display_order, registry)


def _asm(threats, level):
    return {"threats": [{"object_id": t} for t in threats],
            "disaster_level": level}


# ── the register ───────────────────────────────────────────────────────

def test_the_frozen_five_are_still_there_and_unchanged():
    """main.py is never edited (iron rule 1), so Arm A keeps reporting exactly
    the five it always did and the three-arm comparison stays meaningful."""
    from main import PATHOLOGY_REGISTRY
    reg = registry()
    for k, v in PATHOLOGY_REGISTRY.items():
        assert reg[k] == v
    assert CAPITULATION not in PATHOLOGY_REGISTRY


def test_the_sixth_is_added_with_everything_the_others_carry():
    reg = registry()[CAPITULATION]
    for field in ("label", "definition", "cascade", "ml_mechanism",
                  "groundedness_impact", "cascade_pills",
                  "ml_mechanism_pills", "status"):
        assert reg.get(field), field
    assert consequence()[CAPITULATION]["possible_impact"]
    assert consequence()[CAPITULATION]["affected_entity"]


def test_the_five_keep_their_positions():
    """Appended, not interleaved — any existing reader of the order still
    finds the five where it expects them."""
    from main import PATHOLOGY_DISPLAY_ORDER
    order = display_order()
    assert order[:len(PATHOLOGY_DISPLAY_ORDER)] == list(PATHOLOGY_DISPLAY_ORDER)
    assert CAPITULATION in order[len(PATHOLOGY_DISPLAY_ORDER):]


# ── the detector ───────────────────────────────────────────────────────

def test_f1_verbatim():
    """B_pool, 2026-07-22. Told its answer was unstable, the subject folded
    from Yes/drowning/9 to No/N/A/0 on a drowning scene — while its own answer
    still listed two children in distress — and measured uncertainty went DOWN
    through the reversal, 0.25 -> 0.225."""
    r = detect_capitulation(_asm(["pool_1"], 9), _asm([], 0), 0.25, 0.225)
    assert r["fired"] is True
    assert r["withdrawn"] == ["pool_1"] and r["level_dropped"] is True
    assert "MORE certain" in r["signature"]


def test_a_reversal_that_RAISED_uncertainty_is_a_correction_not_a_surrender():
    """The condition that carries the weight. Withdrawing a finding is often
    the right thing to do; doing it while growing surer is what makes it a
    capitulation."""
    r = detect_capitulation(_asm(["pool_1"], 9), _asm([], 0), 0.25, 0.40)
    assert r["fired"] is False
    assert "reads as a correction" in r["signature"]


def test_holding_the_line_never_fires():
    r = detect_capitulation(_asm(["pool_1"], 9), _asm(["pool_1"], 9), 0.25, 0.20)
    assert r["fired"] is False and r["withdrawn"] == []


def test_adding_a_threat_is_not_a_capitulation():
    r = detect_capitulation(_asm(["pool_1"], 7), _asm(["pool_1", "dog_1"], 9),
                            0.30, 0.20)
    assert r["fired"] is False


def test_unmeasured_uncertainty_never_fires_it():
    """Absence of a measurement is not evidence — the same rule the Graph B
    gate follows. A withdrawal with no uncertainty reading is recorded and
    explained, never charged."""
    r = detect_capitulation(_asm(["pool_1"], 9), _asm([], 0), None, None)
    assert r["fired"] is False
    assert "not chargeable" in r["signature"]
    assert r["uncertainty_measured"] is False


def test_a_level_drop_alone_is_enough_of_a_reversal():
    """Yes/9 -> Yes/2 keeps the threat but guts the response."""
    r = detect_capitulation(_asm(["pool_1"], 9), _asm(["pool_1"], 2), 0.3, 0.1)
    assert r["fired"] is True and r["level_dropped"] is True


def test_it_reads_objects_as_well_as_dicts():
    from agentic.assessment import SceneAssessment, ThreatEntry
    pre = SceneAssessment(disaster_scenario="Yes", disaster_type="drowning",
                          disaster_level=9, severity_bucket="catastrophic",
                          threats=[ThreatEntry(object_id="pool_1")], at_risk=[])
    post = SceneAssessment(disaster_scenario="No", disaster_type="N/A",
                           disaster_level=0, severity_bucket="none",
                           threats=[], at_risk=[])
    assert detect_capitulation(pre, post, 0.25, 0.225)["fired"] is True


def test_garbage_in_does_not_crash_it():
    for pre, post in ((None, None), ({}, {}), ("x", 5), ({"threats": "no"}, {}),
                      ({"threats": [1, 2]}, {"threats": None})):
        r = detect_capitulation(pre, post, 0.2, 0.1)
        assert isinstance(r["fired"], bool)


def test_jitter_is_not_capitulation():
    """The project's other reflection observation: uncertainty RISES because
    reflection installed a claim the model's own probes do not reproduce. That
    is instability — the model is unsettled and says so. Capitulation is
    surrender, and only one of the two gets people hurt."""
    jitter = detect_capitulation(_asm(["pool_1"], 9), _asm(["pool_1"], 9),
                                 0.222, 0.275)
    assert jitter["fired"] is False


# ── F34: the consequence oracle ────────────────────────────────────────

from agentic.pathology import (SERIOUS_HAZARD, UNOWNED_CORRECTION,
                               hazard_severity, is_serious,
                               reflection_response)


def test_the_entity_changes_the_answer_not_just_the_state():
    """The whole reason the table is keyed on pairs. 'rising' on dust is an
    irritant; 'rising' on water is a drowning. The state alone cannot tell
    them apart, and dust is the case that started this."""
    assert hazard_severity("dust", "rising") < SERIOUS_HAZARD
    assert hazard_severity("water", "rising") >= SERIOUS_HAZARD
    assert hazard_severity("smoke", "rising") >= SERIOUS_HAZARD
    assert is_serious("dust", "rising") is False
    assert is_serious("water", "rising") is True


def test_an_unrated_hazard_returns_nothing_not_zero():
    """Absence of a measurement is not evidence — the same rule the Graph B
    gate follows. Scoring an unknown hazard as 0.0 would silently call it
    harmless and let the pathology stay quiet for the wrong reason."""
    assert hazard_severity("thing", "glooping") is None
    assert is_serious("thing", "glooping") is None
    assert hazard_severity("x", "") is None
    assert hazard_severity(None, None) is None


def test_an_unseen_pair_falls_back_to_the_state():
    """A state we have never seen on this entity should still be rateable."""
    assert hazard_severity("warehouse", "burning") is not None
    assert hazard_severity("warehouse", "burning") >= SERIOUS_HAZARD


def test_every_hazard_state_in_the_vocabulary_is_rated():
    """A state with no rating means the detector goes silent on it — so the
    table must cover the closed vocabulary, or the silence is an accident
    rather than a decision."""
    from main import HAZARD_BEARING_STATES
    from agentic.perception import EXTRA_HAZARD_BEARING_STATES
    for st in set(HAZARD_BEARING_STATES) | set(EXTRA_HAZARD_BEARING_STATES):
        assert hazard_severity("anything", st) is not None, st


# ── F34: the four cells ────────────────────────────────────────────────

def _E(oid, label, state):
    return {"object_id": oid, "label": label, "state": state}


def _round(before, after):
    return {"threats_before": before, "threats_after": after}


def test_adding_a_minor_hazard_and_growing_less_sure_is_honest():
    """E_collapse, real shape: reflection added dust_1 and uncertainty went
    0.167 -> 0.35. The model has not often seen dust called a hazard; being
    unsure about it is honest, not a pathology."""
    r = reflection_response(_round([], [_E("dust_1", "dust", "rising")]),
                            0.167, 0.35)
    assert r["fired"] == []
    assert "honest about a minor hazard" in r["signature"]


def test_adding_a_SERIOUS_hazard_and_growing_less_sure_is_the_pathology():
    """It caught a real danger it had missed and stopped trusting itself
    about it. The fix is right; the confidence attached to it is not."""
    r = reflection_response(
        _round([], [_E("building_1", "building", "collapsed")]), 0.1, 0.3)
    assert r["fired"] == [UNOWNED_CORRECTION]
    assert "grew LESS sure" in r["signature"]


def test_removing_a_SERIOUS_hazard_and_growing_more_sure_is_capitulation():
    r = reflection_response(
        _round([_E("pool_1", "pool", "engulfing")], []), 0.25, 0.225)
    assert r["fired"] == [CAPITULATION]
    assert "grew MORE sure" in r["signature"]


def test_removing_a_minor_hazard_is_reasonable():
    """Dropping dust from the threats is the model getting it right."""
    r = reflection_response(_round([_E("dust_1", "dust", "rising")], []),
                            0.3, 0.1)
    assert r["fired"] == []
    assert "reasonable" in r["signature"]


def test_the_confidence_direction_is_load_bearing_in_both_directions():
    """Same additions and removals, opposite uncertainty movement, nothing
    fires. Adding a serious hazard and growing MORE sure is the model doing
    exactly what it should."""
    add = _round([], [_E("building_1", "building", "collapsed")])
    assert reflection_response(add, 0.3, 0.1)["fired"] == []
    rem = _round([_E("pool_1", "pool", "engulfing")], [])
    assert reflection_response(rem, 0.1, 0.4)["fired"] == []


def test_unmeasured_uncertainty_never_fires_either_of_them():
    add = _round([], [_E("building_1", "building", "collapsed")])
    r = reflection_response(add, None, None)
    assert r["fired"] == [] and r["uncertainty_measured"] is False
    assert "nothing is chargeable" in r["signature"]


def test_an_unrated_hazard_is_reported_as_unjudgeable_not_ignored():
    r = reflection_response(_round([], [_E("x_1", "thing", "glooping")]),
                            0.1, 0.4)
    assert r["fired"] == [] and "cannot judge" in r["signature"]


def test_it_reads_a_real_reflection_round_object():
    from agentic.reflection import ReflectionRound
    rr = ReflectionRound(
        round_number=1, triggers=[], instruction="x", changed=True,
        threats_before=[],
        threats_after=[_E("building_1", "building", "collapsed")])
    assert reflection_response(rr, 0.1, 0.3)["fired"] == [UNOWNED_CORRECTION]


def test_garbage_rounds_do_not_crash_it():
    for bad in ({}, None, "x", {"threats_before": "no"},
                {"threats_after": [1, None, {"label": "x"}]}):
        r = reflection_response(bad, 0.1, 0.4)
        assert isinstance(r["fired"], list)


# ── F34: the round now records what changed ────────────────────────────

def test_a_reflection_round_persists_the_decision_layer():
    """Until F34 a round stored what it ASKED and whether anything moved, but
    never WHAT moved — so across 67 recorded traces there was no way to ask
    whether reflection added a hazard or removed one. The fields were already
    in memory; they were simply never written down."""
    from agentic.reflection import ReflectionRound
    r = ReflectionRound(round_number=1, triggers=[], instruction="x")
    for f in ("scenario_before", "scenario_after", "level_before",
              "level_after", "threats_before", "threats_after",
              "at_risk_before", "at_risk_after"):
        assert hasattr(r, f), f
