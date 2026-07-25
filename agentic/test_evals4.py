"""Stage 4 evals (Phase 1b) — corrected conformance + A-vs-B alignment.

Hermetic: imports Arm A's frozen checkers but tests the CORRECTED Arm B read
(dedupe, severity-weight, category breakdown, non-saturating validity) and the
A-vs-B precision/recall pair. No models.

Run:  pytest agentic/test_evals4.py -q
"""
from __future__ import annotations

from agentic.evals4 import (ab_alignment, compute_trust, conformance_breakdown,
                            consequence_scores, internal_alignment)
from agentic.assessment import AtRiskEntry, SceneAssessment, ThreatEntry
from agentic.perception import DetectedObject, PerceptionResult

# a graph whose drowning entity is flagged hazardous (trips 2-3 Arm A role
# rules for ONE conceptual error), plus a bad effect and an edge to nothing.
G_BAD = {"nodes": [
    {"id": "person_1", "label": "person", "state": "drowning", "hazardous": True},
    {"id": "water_1", "label": "water", "state": "rising", "hazardous": True}],
    "edges": [
        {"source": "water_1", "target": "ghost_1", "effect": "may_harm",
         "via_state": "rising"},                           # unresolved_endpoint
        {"source": "water_1", "target": "person_1", "effect": "zaps",
         "via_state": "rising"}]}                           # bad effect

G_OK = {"nodes": [
    {"id": "water_1", "label": "water", "state": "rising", "hazardous": True},
    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
    "edges": [{"source": "water_1", "target": "person_1", "effect": "may_harm",
               "via_state": "rising"}]}


# ── conformance breakdown ───────────────────────────────────────────────

def test_dedupe_collapses_one_error_that_trips_many_rules():
    """The drowning-entity-flagged-hazardous fires several Arm A role rules;
    the corrected read must count it as ONE role mix-up issue."""
    cb = conformance_breakdown(G_BAD, G_OK)
    role = [i for i in cb["issues"]
            if i["entity"] == "person_1" and i["category"] == "role mix-up"]
    assert len(role) == 1
    # and the raw Arm A count is higher than the deduped count
    assert cb["raw_a_violations"] >= cb["n_issues"]


def test_breakdown_groups_by_failure_pattern():
    cb = conformance_breakdown(G_BAD, G_OK)
    cats = {r["category"] for r in cb["breakdown"]}
    assert "role mix-up" in cats
    assert "fabrication" in cats            # the edge to ghost_1
    assert "made-up vocabulary" in cats     # the 'zaps' effect
    # breakdown is sorted worst-first (severity desc)
    sevs = [r["severity"] for r in cb["breakdown"]]
    assert sevs == sorted(sevs, reverse=True)


def test_arm_a_raw_numbers_recorded_for_comparability():
    cb = conformance_breakdown(G_BAD, G_OK)
    for k in ("raw_a_violations", "raw_b_violations",
              "raw_a_validity", "raw_b_validity"):
        assert k in cb
    assert 0.0 <= cb["raw_a_validity"] <= 1.0


def test_validity_non_saturating_and_padding_proof():
    v_bad = conformance_breakdown(G_BAD, G_OK)["validity"]
    # duplicating the SAME bad edges must not change the score
    g_pad = {"nodes": G_BAD["nodes"], "edges": G_BAD["edges"] * 8}
    assert conformance_breakdown(g_pad, G_OK)["validity"] == v_bad
    # more DISTINCT defects must score strictly lower (no flat floor)
    g_worse = {"nodes": [
        {"id": "person_1", "state": "drowning", "hazardous": True, "label": "person"},
        {"id": "car_1", "state": "injured", "hazardous": True, "label": "car"},
        {"id": "water_1", "state": "rising", "hazardous": True, "label": "water"}],
        "edges": [
            {"source": "water_1", "target": "ghost_1", "effect": "may_harm"},
            {"source": "water_1", "target": "nowhere_1", "effect": "zaps"},
            {"source": "person_1", "target": "car_1", "effect": "blahblah"}]}
    assert conformance_breakdown(g_worse, G_OK)["validity"] < v_bad


def test_clean_graph_scores_high():
    cb = conformance_breakdown(G_OK, G_OK)
    assert cb["validity"] >= 0.9
    assert cb["n_issues"] == 0


# ── internal alignment (within-A recommendation coverage) ──────────────

def _obj(oid, state, kind):
    return DetectedObject(object_id=oid, label=oid.split("_")[0], family="x",
                          state=state, state_kind=kind, bbox=[0, 0, 9, 9],
                          box_source="dino_matched", box_confidence=0.9,
                          anchor_bbox=[0, 0, 9, 9])


def _rec_ok():
    return {"rank": 1, "action": "evacuate person_1",
            "reason": "Because house_1 is burning it may_harm person_1.",
            "related_object_ids": ["house_1", "person_1"],
            "structured_reasoning": {"threat": "house_1", "state": "burning",
                                     "effect": "may_harm",
                                     "affected_objects": ["person_1"]},
            "remaining_risk": "(car_1, parked)"}


def _rec_pair():
    rec = PerceptionResult(image_path="/x", image_size=[10, 10], entity_source="vlm",
                           detected_objects=[_obj("house_1", "burning", "hazard_bearing"),
                                             _obj("person_1", "standing", "at_risk")])
    asm = SceneAssessment(disaster_scenario="Yes", disaster_type="fire",
                          disaster_level=7, severity_bucket="serious",
                          threats=[ThreatEntry(object_id="house_1")],
                          at_risk=[AtRiskEntry(object_id="person_1", kind="proximity")])
    return rec, asm


def test_internal_alignment_clean():
    rec, asm = _rec_pair()
    ia = internal_alignment(rec, asm, [_rec_ok()])
    assert ia["n_failures"] == 0 and ia["score"] >= 0.9


def test_internal_alignment_uncovered_at_risk():
    """An at-risk entity no recommendation acts on is a coverage gap."""
    rec, asm = _rec_pair()
    asm.at_risk.append(AtRiskEntry(object_id="dog_1", kind="distress"))
    ia = internal_alignment(rec, asm, [_rec_ok()])
    assert any(r["category"] == "coverage gap" for r in ia["breakdown"])


def test_internal_alignment_state_mismatch_and_self_loop():
    rec, asm = _rec_pair()
    bad = {**_rec_ok(),
           "structured_reasoning": {"threat": "house_1", "state": "smoldering",
                                    "effect": "may_harm",
                                    "affected_objects": ["house_1"]},
           "reason": "Because house_1 is smoldering it may_harm house_1."}
    cats = {r["category"] for r in internal_alignment(rec, asm, [bad])["breakdown"]}
    assert "inconsistency" in cats      # state 'smoldering' != frozen 'burning'
    assert "role mix-up" in cats        # house_1 harms itself, effect != worsens


def test_internal_alignment_duplicate_action():
    rec, asm = _rec_pair()
    r1 = _rec_ok()
    r2 = {**_rec_ok(), "rank": 2, "remaining_risk": "(person_1, standing)",
          "structured_reasoning": {"threat": "house_1", "state": "burning",
                                   "effect": "may_spread_to",
                                   "affected_objects": ["person_1"]}}
    # same verb ("evacuate") + same threat (house_1) → action collapse
    cats = {r["category"] for r in internal_alignment(rec, asm, [r1, r2])["breakdown"]}
    assert "duplicate" in cats


# ── A-vs-B alignment ────────────────────────────────────────────────────

def test_ab_alignment_identical_graphs_perfect():
    al = ab_alignment(G_OK, G_OK)
    assert al["a_fidelity"] == 1.0 and al["b_coverage"] == 1.0
    assert al["structural"] == 1.0


def test_ab_alignment_partial_overlap():
    # A has one edge B lacks, B has one A lacks → neither fidelity nor coverage 1
    g_a = {"nodes": [{"id": "w_1", "hazardous": True, "label": "water"},
                     {"id": "p_1", "hazardous": False, "label": "person"}],
           "edges": [{"source": "w_1", "target": "p_1", "effect": "may_harm"}]}
    g_b = {"nodes": [{"id": "w_1", "hazardous": True, "label": "water"},
                     {"id": "c_1", "hazardous": False, "label": "car"}],
           "edges": [{"source": "w_1", "target": "c_1", "effect": "may_spread_to"}]}
    al = ab_alignment(g_a, g_b)
    assert al["a_fidelity"] < 1.0 and al["b_coverage"] < 1.0
    assert al["a_only"] and al["b_only"]
    # id-level (concrete object ids), not class labels like 'structure'
    assert al["a_only"][0]["source"] == "w_1" and al["a_only"][0]["target"] == "p_1"
    assert al["b_only"][0]["target"] == "c_1"


# ── trust roll-up ───────────────────────────────────────────────────────

_CLEAN = dict(
    conformance={"validity": 1.0, "n_issues": 0},
    internal_alignment={"score": 1.0, "n_failures": 0, "failures": []},
    alignment={"structural": 1.0, "a_only": [], "b_only": []},
    uncertainty={},
    picks={"agreement": 1.0})


def test_trust_clean_is_high_with_no_contributors():
    t = compute_trust([], **_CLEAN)
    assert t["score"] == 1.0 and t["band"] == "high"
    assert all(c["contribution"] == 0 for c in t["contributors"])
    assert "No material signal dents it" in t["explanation"]


def test_trust_ab_alignment_leads_the_weighting():
    """Equal penalty on A-vs-B and conformance must rank A-vs-B first — the
    decided weighting (A-vs-B 0.30 > conformance 0.10)."""
    t = compute_trust([], conformance={"validity": 0.5, "n_issues": 3},
                      internal_alignment={"score": 1.0, "failures": []},
                      alignment={"structural": 0.5, "a_only": [], "b_only": []},
                      uncertainty={}, picks={"agreement": 1.0})
    assert t["contributors"][0]["signal"] == "ab_alignment"
    assert t["contributors"][0]["contribution"] > \
        next(c["contribution"] for c in t["contributors"]
             if c["signal"] == "conformance")
    assert t["score"] < 1.0


def test_trust_weighted_average_a_partial_ab_failure_stays_high():
    """Weighted average (kept per Sunny): a partial A-vs-B failure with the
    other checks clean stays HIGH — A-vs-B is two independent generations, so
    partial disagreement is expected and shouldn't nuke trust."""
    t = compute_trust([], conformance={"validity": 1.0},
                      internal_alignment={"score": 0.83, "failures": []},
                      alignment={"structural": 0.4, "a_only": [1, 2], "b_only": [1]},
                      uncertainty={"score": 0.29, "n_probes": 5}, picks={"agreement": 1.0})
    # 1 - (0.30*0.6 + 0.25*0.29 + 0.20*0.17) ≈ 0.71
    assert t["band"] == "high" and 0.68 <= t["score"] <= 0.74
    assert t["contributors"][0]["signal"] == "ab_alignment"   # still the biggest dent


def test_trust_per_rec_flags_a_recommendation_that_never_reappeared():
    """A recommendation whose threat never shows up in the re-asks must get a
    low per-rec trust with a 'never reappeared' worst-contributor — the car_1
    case Sunny flagged."""
    recs = [{"rank": 3, "structured_reasoning": {"threat": "car_1"}}]
    unc = {"n_probes": 5, "score": 0.2,
           "granular": {"threats": {"house_1": {"u": 0.0, "votes": "5/5"}},
                        "effects": {}}}
    t = compute_trust(recs, conformance={"validity": 1.0},
                      internal_alignment={"score": 1.0, "failures": []},
                      alignment={"structural": 1.0}, uncertainty=unc,
                      picks={"agreement": 1.0})
    pr = t["per_rec"][0]
    assert pr["threat"] == "car_1" and pr["score"] == 0.0
    assert "never reappeared" in pr["worst_contributor"]["text"]


def test_trust_never_crashes_on_malformed_inputs():
    t = compute_trust(None, None, None, None, None, None)
    assert 0.0 <= t["score"] <= 1.0 and "band" in t and t["per_rec"] == []


# ── consequence-to-victims (a separate axis) ────────────────────────────

def test_consequence_distress_life_outweighs_property():
    """may_harm to a person in distress must score far above blocks_access_to
    a plain building — the life-safety ordering."""
    asm = SceneAssessment(disaster_scenario="Yes", disaster_type="fire",
                          disaster_level=7, severity_bucket="serious",
                          threats=[ThreatEntry(object_id="house_1")],
                          at_risk=[AtRiskEntry(object_id="person_1",
                                               kind="distress")])
    life = [{"rank": 1, "structured_reasoning": {"threat": "house_1",
             "effect": "may_harm", "affected_objects": ["person_1"]}}]
    prop = [{"rank": 1, "structured_reasoning": {"threat": "car_1",
             "effect": "blocks_access_to", "affected_objects": ["road_1"]}}]
    c_life = consequence_scores(life, asm)[1]
    c_prop = consequence_scores(prop, asm)[1]
    assert c_life["score"] > c_prop["score"]
    assert c_life["band"] == "high" and c_prop["band"] == "low"
    assert c_life["worst_victim"]["id"] == "person_1"


def test_consequence_occupied_structures_rank_above_animals_and_property():
    """A nearby house (possible occupants) and a hospital (vulnerable occupants)
    must outrank an animal and plain property — Sunny's occupancy-risk point."""
    asm = SceneAssessment(disaster_scenario="Yes", disaster_type="fire",
                          disaster_level=7, severity_bucket="serious",
                          threats=[ThreatEntry(object_id="house_1")], at_risk=[])

    def score(victim):
        r = [{"rank": 1, "structured_reasoning": {"threat": "house_1",
              "effect": "may_spread_to", "affected_objects": [victim]}}]
        return consequence_scores(r, asm)[1]["score"]
    assert score("hospital_1") > score("house_2") > score("dog_1") > score("road_1")


def test_consequence_human_outranks_animal_in_same_rec():
    """person_1 and dog_1 both at-risk by proximity: the human must be the
    worst victim, not the animal (the bug Sunny caught)."""
    asm = SceneAssessment(disaster_scenario="Yes", disaster_type="fire",
                          disaster_level=7, severity_bucket="serious",
                          threats=[ThreatEntry(object_id="house_1")],
                          at_risk=[AtRiskEntry(object_id="person_1", kind="proximity"),
                                   AtRiskEntry(object_id="dog_1", kind="proximity")])
    rec = [{"rank": 1, "structured_reasoning": {"threat": "house_1",
            "effect": "may_harm", "affected_objects": ["person_1", "dog_1"]}}]
    c = consequence_scores(rec, asm)[1]
    assert c["worst_victim"]["id"] == "person_1"       # human, not dog
    # and the same rec with only the dog scores strictly lower
    rec_dog = [{"rank": 1, "structured_reasoning": {"threat": "house_1",
                "effect": "may_harm", "affected_objects": ["dog_1"]}}]
    assert consequence_scores(rec_dog, asm)[1]["score"] < c["score"]


def test_consequence_attaches_to_trust_per_rec():
    recs = [{"rank": 1, "structured_reasoning": {"threat": "house_1",
             "effect": "may_harm", "affected_objects": ["person_1"]}}]
    asm = SceneAssessment(disaster_scenario="Yes", disaster_type="fire",
                          disaster_level=7, severity_bucket="serious",
                          threats=[ThreatEntry(object_id="house_1")],
                          at_risk=[AtRiskEntry(object_id="person_1",
                                               kind="distress")])
    cons = consequence_scores(recs, asm)
    t = compute_trust(recs, {"validity": 1.0}, {"score": 1.0, "failures": []},
                      {"structural": 1.0}, {}, {"agreement": 1.0},
                      consequence=cons)
    assert t["per_rec"][0]["consequence_band"] == "high"
    assert t["per_rec"][0]["worst_victim"]["id"] == "person_1"


def test_consequence_never_crashes_without_record_or_assessment():
    recs = [{"rank": 1, "structured_reasoning": {"threat": "x",
             "effect": "may_harm", "affected_objects": ["y"]}}]
    c = consequence_scores(recs, None, None)
    assert 1 in c and 0.0 <= c[1]["score"] <= 1.0


def test_ab_alignment_dedupes_parallel_edges():
    """Parallel identical edges must not distort the comparison."""
    g = {"nodes": [{"id": "w_1", "hazardous": True, "label": "water"},
                   {"id": "p_1", "hazardous": False, "label": "person"}],
         "edges": [{"source": "w_1", "target": "p_1", "effect": "may_harm"},
                   {"source": "w_1", "target": "p_1", "effect": "may_harm"}]}
    single = {"nodes": g["nodes"],
              "edges": [{"source": "w_1", "target": "p_1", "effect": "may_harm"}]}
    assert ab_alignment(g, single)["structural"] == 1.0
