"""F48 — the singular error library, its consequence scaling, and the compact
explanation. Hermetic: no models, no ground truth."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.assessment import AtRiskEntry, SceneAssessment  # noqa: E402
from agentic.errors4 import (ERROR_CEILINGS, MAX_TOTAL_DEDUCTION,  # noqa: E402
                             explain_trust, singular_errors, total_deduction)
from agentic.evals4 import compute_trust  # noqa: E402
from agentic.perception import DetectedObject, PerceptionResult  # noqa: E402


def _obj(oid, label, state, kind):
    return DetectedObject(object_id=oid, label=label, family="x", state=state,
                          state_kind=kind, bbox=[0, 0, 9, 9],
                          box_source="dino_matched", box_confidence=0.9,
                          anchor_bbox=[0, 0, 9, 9])


def _scene(*objs):
    return PerceptionResult(image_path="/x", image_size=[10, 10],
                            entity_source="vlm", detected_objects=list(objs))


def _asm(scenario="Yes", at_risk=(), level=7):
    return SceneAssessment(
        disaster_scenario=scenario, disaster_type="fire", disaster_level=level,
        severity_bucket="high" if level >= 7 else "none", threats=[],
        at_risk=[AtRiskEntry(object_id=o, kind=k) for o, k in at_risk])


def _rec(rank, action, threat="", effect="may_harm", affected=()):
    return {"rank": rank, "action": action, "reason": "because",
            "structured_reasoning": {"threat": threat, "state": "burning",
                                     "effect": effect,
                                     "affected_objects": list(affected)}}


# ── the core idea: one rule, priced by who it happened to ───────────────

def test_the_same_error_costs_more_when_it_happens_to_a_person():
    """The whole design in one test (Sunny: "based on consequence of victims").
    ONE rule — a declared victim nobody acts on — and the vulnerability table
    does the separating. No rule for dogs and another for people."""
    dog = singular_errors(
        _scene(_obj("house_1", "house", "burning", "hazard_bearing"),
               _obj("dog_1", "dog", "standing", "at_risk")),
        _asm(at_risk=[("dog_1", "proximity")]),
        [_rec(1, "cool house_1", "house_1", affected=[])])
    person = singular_errors(
        _scene(_obj("house_1", "house", "burning", "hazard_bearing"),
               _obj("child_1", "child", "trapped", "at_risk")),
        _asm(at_risk=[("child_1", "distress")]),
        [_rec(1, "cool house_1", "house_1", affected=[])])
    d = next(e for e in dog if e["id"] == "victim_left_behind")
    p = next(e for e in person if e["id"] == "victim_left_behind")
    assert p["deduction"] > d["deduction"]
    assert d["ceiling"] == p["ceiling"] == ERROR_CEILINGS["victim_left_behind"]


def test_a_victim_the_advice_acts_on_is_not_left_behind():
    errs = singular_errors(
        _scene(_obj("house_1", "house", "burning", "hazard_bearing"),
               _obj("child_1", "child", "trapped", "at_risk")),
        _asm(at_risk=[("child_1", "distress")]),
        [_rec(1, "evacuate child_1", "house_1", affected=["child_1"])])
    assert not [e for e in errs if e["id"] == "victim_left_behind"]


def test_the_silence_test_is_an_error_with_a_price():
    """F_park_control, verbatim: the model said 'No disaster' and then issued
    recommendations anyway. Every one of them is a false alarm — this is not
    'slightly worse advice', so it carries the largest ceiling."""
    errs = singular_errors(_scene(_obj("dog_1", "dog", "standing", "normal")),
                           _asm(scenario="No", level=0),
                           [_rec(1, "restrain dog_1", "dog_1")])
    e = next(x for x in errs if x["id"] == "emergency_invented")
    assert e["consequence"] == 1.0
    assert e["deduction"] == ERROR_CEILINGS["emergency_invented"]


def test_silence_kept_on_a_safe_scene_is_not_an_error():
    assert singular_errors(_scene(_obj("dog_1", "dog", "standing", "normal")),
                           _asm(scenario="No", level=0), []) == []


def test_an_action_field_that_instructs_nobody():
    """F_park live: both action fields contained the text `person_3`. A label
    is not an instruction."""
    errs = singular_errors(
        _scene(_obj("house_1", "house", "burning", "hazard_bearing")),
        _asm(), [_rec(1, "person_3", "house_1"),
                 _rec(2, "evacuate person_3 now", "house_1")])
    e = next(x for x in errs if x["id"] == "action_is_not_an_action")
    assert "1" in e["detail"] and "2" not in e["detail"]   # only the bare one


# ── the hazard scaling, and the two corrections it needed ───────────────

def test_an_unaddressed_hazard_is_priced_by_who_it_reaches():
    """A danger is only as serious as what it can reach. The model's OWN graph
    says who that is — our guess is not the source."""
    scene = _scene(_obj("fire_1", "fire", "spreading", "hazard_bearing"),
                   _obj("spill_1", "spill", "seeping", "hazard_bearing"),
                   _obj("child_1", "child", "trapped", "at_risk"),
                   _obj("road_1", "road", "wet", "normal"))
    asm = _asm(at_risk=[("child_1", "distress")])
    # the child is NOT rescued here — otherwise the discount fires and both
    # sides fall back to the spill's own severity, which is a different test
    recs = [_rec(1, "cordon fire_1", "fire_1", affected=[])]
    reaches_child = {"edges": [{"source": "spill_1", "target": "child_1"}]}
    reaches_road = {"edges": [{"source": "spill_1", "target": "road_1"}]}
    a = next(e for e in singular_errors(scene, asm, recs, reaches_child)
             if e["id"] == "hazard_unaddressed")
    b = next(e for e in singular_errors(scene, asm, recs, reaches_road)
             if e["id"] == "hazard_unaddressed")
    assert a["deduction"] > b["deduction"]


def test_a_hazard_whose_victim_is_already_being_rescued_is_discounted():
    """E_collapse, verbatim. `dust_1` may harm `person_1` — and the run's first
    recommendation rescues `person_1` from the building. The danger to that
    person IS being handled; what is left is a second route to the same harm.
    Without this discount E_collapse fell to 'low', which is wrong: it was the
    best-reasoned run of the six."""
    scene = _scene(_obj("building_1", "building", "collapsed", "hazard_bearing"),
                   _obj("dust_1", "dust", "rising", "hazard_bearing"),
                   _obj("person_1", "person", "trapped", "at_risk"))
    asm = _asm(at_risk=[("person_1", "distress")])
    gb = {"edges": [{"source": "dust_1", "target": "person_1"}]}
    rescued = singular_errors(
        scene, asm, [_rec(1, "rescue person_1 from building_1", "building_1",
                          affected=["person_1"])], gb)
    stranded = singular_errors(
        scene, asm, [_rec(1, "cordon building_1", "building_1", affected=[])], gb)
    r = next(e for e in rescued if e["id"] == "hazard_unaddressed")
    s = next(e for e in stranded if e["id"] == "hazard_unaddressed")
    assert r["deduction"] < s["deduction"]
    assert r["consequence"] == 0.25          # dust's own severity, the floor


def test_an_unaddressed_hazard_is_never_free():
    """Even reaching nobody we can name, it still costs its own severity."""
    e = next(x for x in singular_errors(
        _scene(_obj("fire_1", "fire", "spreading", "hazard_bearing")),
        _asm(), [_rec(1, "call for help")], {"edges": []})
        if x["id"] == "hazard_unaddressed")
    assert e["deduction"] > 0


# ── how it lands in trust ───────────────────────────────────────────────

_CLEAN = dict(conformance={"validity": 1.0, "issues": [], "graph_b_edges": 2},
              internal_alignment={"score": 1.0, "failures": [], "size": 8},
              alignment={"advice_backed_by_belief": 1.0, "dangers_acted_on": 1.0,
                         "decomposition": {}},
              uncertainty={"score": 0.0}, picks={"agreement": 1.0},
              graph_b_internal={"score": 1.0, "n_failures": 0, "measured": True})


def test_the_deduction_comes_off_after_the_weighted_sum():
    """Not another share of the average — averaging them in would re-flatten
    exactly what they exist to separate."""
    err = [{"id": "victim_left_behind", "detail": "d", "entities": [],
            "consequence": 1.0, "ceiling": 0.35, "deduction": 0.35}]
    t = compute_trust([], singular_errors=err, **_CLEAN)
    assert t["weighted_score"] == 1.0
    assert t["singular_deduction"] == 0.35
    assert t["score"] == 0.65


def test_the_library_alone_cannot_zero_trust():
    """The weighted checks must still be able to speak."""
    many = [{"id": k, "detail": "d", "entities": [], "consequence": 1.0,
             "ceiling": v, "deduction": v} for k, v in ERROR_CEILINGS.items()]
    assert total_deduction(many) == MAX_TOTAL_DEDUCTION
    assert compute_trust([], singular_errors=many, **_CLEAN)["score"] == 0.50


def test_one_failure_is_charged_once_not_twice():
    """`victim_left_behind` is the same condition as the severity-2 coverage
    gap already inside internal_alignment. The library takes it over and the
    weighted side stops charging it — but the FINDING stays in the record
    (iron rule 8), it just stops being counted twice."""
    internal = {"score": 0.6, "size": 8, "failures": [
        {"category": "coverage gap", "severity": 2, "detail": "child_1"},
        {"category": "subject mismatch", "severity": 1, "detail": "x"}]}
    err = [{"id": "victim_left_behind", "detail": "d", "entities": ["child_1"],
            "consequence": 1.0, "ceiling": 0.35, "deduction": 0.35}]
    both = {**_CLEAN, "internal_alignment": internal}
    charged_twice = compute_trust([], **both)["score"]
    charged_once = compute_trust([], singular_errors=err, **both)["score"]
    # the coverage gap no longer dents the weighted side...
    assert charged_once > charged_twice - 0.35
    # ...and the finding is still there to read
    assert len(internal["failures"]) == 2


# ── the explanation ─────────────────────────────────────────────────────

def test_the_explanation_names_the_entities():
    """Today's says "2 seen-but-not-acted". A reader deciding whether to act on
    emergency advice cannot do anything with a count."""
    err = [{"id": "victim_left_behind", "deduction": 0.33, "consequence": 0.93,
            "ceiling": 0.35, "entities": ["hazmat_worker_1", "hazmat_worker_2"],
            "detail": "declared at risk, and no recommendation acts on them: "
                      "hazmat_worker_1, hazmat_worker_2"}]
    t = compute_trust([], singular_errors=err, **_CLEAN)
    lines = explain_trust(t, {"decomposition": {}})
    assert len(lines) == 3
    assert "hazmat_worker_1" in lines[1] and "hazmat_worker_2" in lines[1]


def test_a_trivial_error_does_not_become_the_headline():
    """E_collapse's unaddressed dust is worth 0.05. Real, but not the story —
    headlining it misleads the reader about what the run was like."""
    err = [{"id": "hazard_unaddressed", "deduction": 0.05, "consequence": 0.25,
            "ceiling": 0.20, "entities": ["dust_1"], "detail": "dust_1"}]
    t = compute_trust([], singular_errors=err, **_CLEAN)
    assert "acted on something else" not in explain_trust(t, {})[0]


def test_the_explanation_never_speaks_for_a_withheld_comparison():
    """The F48 bug, in the explanation this time. The A-vs-B reading sentence
    is computed whether or not A-vs-B was scored; letting it through on a
    withheld run asserts a comparison nobody made."""
    t = compute_trust([], **{**_CLEAN, "graph_b_internal":
                             {"score": 0.2, "n_failures": 3, "measured": True}})
    dc = {"decomposition": {"reading": "agrees on the hazards, disagrees on "
                                       "who they threaten"}}
    lines = explain_trust(t, dc)
    assert "agrees on the hazards" not in lines[0]
    assert "not checked" in lines[2]


def test_the_narrative_no_longer_claims_a_withheld_signal_passed():
    """B_pool live: the gate withheld A-vs-B, and the explanation still said
    the recommendations "match the model's own graph". They may well not — we
    did not look."""
    t = compute_trust([], **{**_CLEAN, "graph_b_internal":
                             {"score": 0.2, "n_failures": 3, "measured": True}})
    assert "match the model's own graph" not in t["explanation"]
    assert "NOT checked" in t["explanation"]
    assert "advice backed by belief" in t["explanation"]
