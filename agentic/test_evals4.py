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
    """F18/F20: the self-loop is still a role mix-up at the RECOMMENDATION
    layer (the model wrote a quad whose threat harms itself), now at the
    raised severity 3. What F18 changes is the GRAPH layer — the placeholder
    edge is re-filed as a node annotation there, not here."""
    rec, asm = _rec_pair()
    bad = {**_rec_ok(),
           "structured_reasoning": {"threat": "house_1", "state": "smoldering",
                                    "effect": "may_harm",
                                    "affected_objects": ["house_1"]},
           "reason": "Because house_1 is smoldering it may_harm house_1."}
    ia = internal_alignment(rec, asm, [bad])
    cats = {r["category"] for r in ia["breakdown"]}
    assert "inconsistency" in cats      # state 'smoldering' != frozen 'burning'
    assert "role mix-up" in cats        # house_1 harms itself, effect != worsens
    assert max(f["severity"] for f in ia["failures"]
               if f["category"] == "role mix-up") == 3


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


# ── F23: the id regex must read multi-word ids whole ─────────────────────

def test_id_regex_reads_multiword_ids_whole():
    """F23: the old pattern had no left anchor, so 'tanker_truck_1' matched as
    'truck_1' and every reason naming a multi-word entity was reported as not
    naming it. C_tanker's entire internal-alignment penalty was this bug."""
    from agentic.evals4 import _ID_RE
    reason = ("Because fire_1 is spreading, it may harm tanker_truck_1 and "
              "person_1 if not contained.")
    assert set(_ID_RE.findall(reason)) == {"fire_1", "tanker_truck_1", "person_1"}
    for oid in ("lifeguard_chair_1", "hazmat_worker_1", "police_officer_2"):
        assert _ID_RE.findall(oid) == [oid]


def test_id_regex_still_reads_presumed_ids_whole():
    """The presumed_<noun>_in_<id> form must survive the anchoring."""
    from agentic.evals4 import _ID_RE
    for oid in ("presumed_residents_in_house_1", "presumed_driver_in_car_1"):
        assert _ID_RE.findall(oid) == [oid]


def test_id_regex_ignores_non_ids():
    """Malformed / adjacent text must not manufacture ids."""
    from agentic.evals4 import _ID_RE
    assert _ID_RE.findall("no ids here at all") == []
    assert _ID_RE.findall("") == []
    assert _ID_RE.findall("Rule 4 says 12 things about 3 items") == []


def test_multiword_id_named_in_reason_is_not_a_failure():
    """The C_tanker regression: a reason that names a multi-word entity in
    plain English must not be reported as omitting it."""
    rec = PerceptionResult(
        image_path="/x", image_size=[10, 10], entity_source="vlm",
        detected_objects=[_obj("fire_1", "spreading", "hazard_bearing"),
                          _obj("tanker_truck_1", "leaking", "hazard_bearing"),
                          _obj("person_1", "standing", "at_risk")])
    asm = SceneAssessment(disaster_scenario="Yes", disaster_type="fire",
                          disaster_level=7, severity_bucket="serious",
                          threats=[ThreatEntry(object_id="fire_1")],
                          at_risk=[AtRiskEntry(object_id="person_1",
                                               kind="proximity")])
    r = {"rank": 1, "action": "contain fire_1",
         "reason": ("Because fire_1 is spreading, it may_harm tanker_truck_1 "
                    "and person_1."),
         "related_object_ids": ["fire_1", "tanker_truck_1", "person_1"],
         "structured_reasoning": {"threat": "fire_1", "state": "spreading",
                                  "effect": "may_harm",
                                  "affected_objects": ["tanker_truck_1",
                                                       "person_1"]}}
    ia = internal_alignment(rec, asm, [r])
    assert ia["n_failures"] == 0, ia["failures"]
    assert ia["score"] == 1.0


# ── F20: the case split, and the widened scan domain ─────────────────────

def test_case_a_quad_not_mirrored_is_severity_zero():
    """Case A — the quad carries the id, related_object_ids does not. A
    bookkeeping slip against our own spec: the causal claim is intact, so it
    is recorded but must not dent the score."""
    rec, asm = _rec_pair()
    r = {**_rec_ok(), "related_object_ids": ["house_1"]}   # person_1 dropped
    ia = internal_alignment(rec, asm, [r])
    hits = [f for f in ia["failures"] if f["category"] == "bookkeeping"]
    assert len(hits) == 1 and hits[0]["severity"] == 0
    assert "person_1" in hits[0]["detail"]
    assert ia["score"] == 1.0          # severity 0 => no score impact


def test_case_b_declared_related_but_no_quad_covers_it():
    """Case B — the D_aerial/spill_1 shape. Declared in related, absent from
    the quad AND from the reason, so the old reason-only scan never saw it."""
    rec, asm = _rec_pair()
    rec.detected_objects.append(_obj("spill_1", "spreading", "hazard_bearing"))
    r = {**_rec_ok(),
         "related_object_ids": ["house_1", "person_1", "spill_1"]}
    ia = internal_alignment(rec, asm, [r])
    hits = [f for f in ia["failures"]
            if f["category"] == "coverage gap" and "spill_1" in f["detail"]]
    assert len(hits) == 1 and hits[0]["severity"] == 1


def test_case_b_was_invisible_to_a_reason_only_scan():
    """Guard on the widening itself: the case-B id appears in NEITHER the
    reason nor the quad, so a reason-only scan is structurally blind to it."""
    import re
    from agentic.evals4 import _ID_RE
    r = {**_rec_ok(), "related_object_ids": ["house_1", "person_1", "spill_1"]}
    reason_ids = set(_ID_RE.findall(r["reason"]))
    quad = {"house_1", "person_1"}
    assert "spill_1" not in reason_ids and "spill_1" not in quad


def test_case_c_named_in_reason_only():
    """Case C — in the reason, in neither the quad nor related."""
    rec, asm = _rec_pair()
    rec.detected_objects.append(_obj("car_1", "parked", "normal"))
    r = {**_rec_ok(),
         "reason": "Because house_1 is burning it may_harm person_1 near car_1."}
    ia = internal_alignment(rec, asm, [r])
    hits = [f for f in ia["failures"]
            if f["category"] == "coverage gap" and "car_1" in f["detail"]]
    assert len(hits) == 1 and hits[0]["severity"] == 1
    assert "neither" in hits[0]["detail"]


def test_case_split_tolerates_malformed_related_field():
    """related_object_ids arriving as None, a bare string, or junk must not
    crash the check (boundary that eats raw model output)."""
    rec, asm = _rec_pair()
    for junk in (None, [], ["", None], "house_1", 17):
        r = {**_rec_ok(), "related_object_ids": junk}
        ia = internal_alignment(rec, asm, [r])
        assert isinstance(ia["n_failures"], int)


# ── F16: a stray dot must never manufacture a violation ──────────────────

def test_dot_suffixed_ids_still_resolve_in_internal_alignment():
    """Belt to the prompt's braces: even if the model emits 'x·proximity'
    anyway, the id must match itself and raise no failure."""
    rec, asm = _rec_pair()
    r = {**_rec_ok(),
         "structured_reasoning": {"threat": "house_1·burning",
                                  "state": "burning", "effect": "may_harm",
                                  "affected_objects": ["person_1·proximity"]},
         "related_object_ids": ["house_1·burning", "person_1·proximity"]}
    ia = internal_alignment(rec, asm, [r])
    assert ia["n_failures"] == 0, ia["failures"]


def test_dot_suffixed_edges_match_clean_edges_in_ab_alignment():
    """D_aerial's collapse: A's 'x·proximity' target could never match B's
    'x', so structural alignment read 0.0 on a genuine agreement."""
    ga = {"nodes": [], "edges": [{"source": "tanker_truck_1",
                                  "target": "hazmat_worker_1·proximity",
                                  "effect": "may_spread_to"}]}
    gb = {"nodes": [], "edges": [{"source": "tanker_truck_1",
                                  "target": "hazmat_worker_1",
                                  "effect": "may_spread_to"}]}
    assert ab_alignment(ga, gb)["structural"] == 1.0


def test_normalisation_does_not_hide_a_real_fabrication():
    """Guard against the fix concealing what it was built to expose: an id
    the record never declared is still a failure after normalising."""
    rec, asm = _rec_pair()
    r = {**_rec_ok(),
         "reason": "Because house_1 is burning it may_harm ghost_9.",
         "related_object_ids": ["house_1", "ghost_9"],
         "structured_reasoning": {"threat": "house_1", "state": "burning",
                                  "effect": "may_harm",
                                  "affected_objects": ["ghost_9"]}}
    ia = internal_alignment(rec, asm, [r])
    assert ia["n_failures"] > 0


# ── O1: the paired-arm guard ─────────────────────────────────────────────

def _u(*probe_edge_sets):
    return {"candidates": [{"votes": 1, "edges": e} for e in probe_edge_sets]}


def test_paired_guard_ships_when_only_noise_moves():
    from agentic.evals4 import paired_arm_guard
    arm = _u([["h_1", "may_harm", ["p_1"]]], [["h_1", "may_harm", ["p_1"]]],
             [["h_1", "may_harm", ["p_1"]]], [["h_1", "may_harm", ["p_1"]]],
             [["h_1", "may_harm", ["p_1"]]])
    assert paired_arm_guard(arm, arm)["verdict"] == "ship"


def test_paired_guard_holds_when_the_clause_reshapes_output():
    """The failure we are guarding against: the clause suppresses a claim on a
    hazard scene rather than only permitting silence on a safe one."""
    from agentic.evals4 import paired_arm_guard
    off = _u(*[[["h_1", "may_harm", ["p_1"]]]] * 5)          # 5/5
    on = _u([], [], [], [], [])                               # 0/5
    g = paired_arm_guard(off, on)
    assert g["verdict"] == "hold"
    assert g["moves"] and g["moves"][0]["delta"] == -1.0


def test_paired_guard_does_not_fire_on_probe_noise_alone():
    """B_pool's shape — five distinct sets, one vote each — must not read as
    the clause doing something. Set identity would fail here; rates must not."""
    from agentic.evals4 import paired_arm_guard
    off = _u([["a_1", "may_harm", ["p_1"]]], [["b_1", "may_harm", ["p_1"]]],
             [["c_1", "may_harm", ["p_1"]]], [["d_1", "may_harm", ["p_1"]]],
             [["e_1", "may_harm", ["p_1"]]])
    on = _u([["e_1", "may_harm", ["p_1"]]], [["d_1", "may_harm", ["p_1"]]],
            [["c_1", "may_harm", ["p_1"]]], [["b_1", "may_harm", ["p_1"]]],
            [["a_1", "may_harm", ["p_1"]]])
    assert paired_arm_guard(off, on)["verdict"] == "ship"


def test_paired_guard_handles_empty_and_malformed_arms():
    from agentic.evals4 import paired_arm_guard
    assert paired_arm_guard({}, {})["verdict"] == "insufficient"
    assert paired_arm_guard(None, None)["verdict"] == "insufficient"
    assert paired_arm_guard(_u([["h_1", "may_harm", ["p_1"]]]),
                            {"candidates": [{"edges": ["junk", None, 7]}]}
                            )["verdict"] in ("ship", "hold")


def test_a_score_built_on_fewer_signals_says_so():
    """F17's hole: if perception misses the hazard AND stage 2 says No, the
    null path opens on a dangerous scene and a blind run scores 1.0. Trust
    cannot distinguish that from a genuinely safe scene — so it must at least
    report how little it measured. The number never travels alone."""
    t = compute_trust([], conformance={"validity": 1.0},
                      internal_alignment={"score": 1.0},
                      alignment={"structural": 0.0, "a_total": 0, "b_total": 0},
                      uncertainty={"score": 0.0},
                      picks={"agreement": 0.333, "a_pick": {}, "b_pick": {},
                             "llm_pick": {}},
                      no_hazards=True)
    assert t["score"] == 1.0
    assert t["signals_measured"] == "3/5"
    assert "3 of 5 signals" in t["explanation"]
    assert {x["signal"] for x in t["not_applicable"]} == {"ab_alignment",
                                                          "pick_agreement"}


def test_full_measurement_reports_all_five():
    t = compute_trust([], conformance={"validity": 1.0},
                      internal_alignment={"score": 1.0},
                      alignment={"structural": 1.0}, uncertainty={"score": 0.0},
                      picks={"agreement": 1.0})
    assert t["signals_measured"] == "5/5"
    assert "signals —" not in t["explanation"]
