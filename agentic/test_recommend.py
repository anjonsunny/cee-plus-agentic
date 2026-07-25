"""Stage 4 recommendation — parsing + node cores, hermetic (no models).

Covers the deterministic promises of the generation spine and the malformed-
model-output cases at every boundary that eats a raw VLM answer (Iron Rule 2 +
1a): non-dict answers, non-list recommendations, empty/missing quads, malformed
advisory entries. The model is a scripted pure function of the prompt.

Run:  pytest agentic/test_recommend.py -q
"""
from __future__ import annotations

from agentic.assessment import AtRiskEntry, SceneAssessment, ThreatEntry
from agentic.perception import DetectedObject, PerceptionResult
from agentic.recommend import (build_graph_a, parse_pick, parse_recommend,
                               pick_targets, run_graph_b, run_recommend,
                               run_stage4)


# ── fixtures ────────────────────────────────────────────────────────────

def _obj(oid, label, state, kind):
    return DetectedObject(object_id=oid, label=label, family=label, state=state,
                          state_kind=kind, bbox=[0, 0, 9, 9],
                          box_source="dino_matched", box_confidence=0.9,
                          anchor_bbox=[0, 0, 9, 9])


def _record():
    return PerceptionResult(
        image_path="/x/E.jpg", image_size=[100, 80], caption="collapse",
        entity_source="vlm", detected_objects=[
            _obj("building_1", "building", "collapsed", "hazard_bearing"),
            _obj("dust_1", "dust", "rising", "hazard_bearing"),
            _obj("person_1", "person", "trapped", "at_risk")])


def _asm():
    return SceneAssessment(
        disaster_scenario="Yes", disaster_type="Building Collapse",
        disaster_level=8, severity_bucket="catastrophic",
        threats=[ThreatEntry(object_id="building_1"),
                 ThreatEntry(object_id="dust_1")],
        at_risk=[AtRiskEntry(object_id="person_1", kind="distress")])


def _rec(threat, state, effect, targets):
    return {"rank": 1, "action": f"act on {threat}",
            "reason": f"Because {threat} is {state}, it {effect} {targets}.",
            "related_object_ids": [threat] + list(targets),
            "structured_reasoning": {"threat": threat, "state": state,
                                     "effect": effect, "affected_objects": targets},
            "expected_consequence": "ok", "remaining_risk": f"({threat}, {state})",
            "possible_follow_up_action": "next"}


REC_OK = {
    "scene_summary": "a collapse", "key_observations": ["person_1 up high"],
    "assumptions": ["conscious"], "uncertainty_notes": ["collapse extent unclear"],
    "recommendations": [_rec("building_1", "collapsed", "may_harm", ["person_1"]),
                        {**_rec("dust_1", "rising", "may_harm", ["person_1"]),
                         "rank": 2}],
    "assumptions_advisory": [{"suspected": "occupants inside building_1",
                              "anchor_object_id": "building_1",
                              "cue": "residential", "suggested_action": "search"}]}

GRAPH_B_OK = {"causal_graph": {"nodes": [
    {"id": "building_1", "label": "building", "state": "collapsed",
     "hazardous": True, "inferred": False},
    {"id": "person_1", "label": "person", "state": "trapped",
     "hazardous": False, "inferred": False}],
    "edges": [{"source": "building_1", "target": "person_1",
               "effect": "may_harm", "via_state": "collapsed"}]},
    "suppression_pick": {"threat": "building_1", "state": "collapsed",
                         "reason": "main hazard"}}

PICK_OK = {"threat": "building_1", "state": "collapsed", "reason": "impact"}


def _script(rec=REC_OK, gb=GRAPH_B_OK, pick=PICK_OK):
    def q(prompt):
        if "assumptions_advisory" in prompt:
            return rec
        if "neutralize exactly ONE" in prompt:
            return pick
        return gb
    return q


# ── parse_recommend ─────────────────────────────────────────────────────

def test_parse_recommend_wellformed():
    frame, recs, advisory, notes = parse_recommend(REC_OK)
    assert frame["scene_summary"] == "a collapse"
    assert frame["key_observations"] == ["person_1 up high"]
    assert len(recs) == 2 and recs[0]["structured_reasoning"]["threat"] == "building_1"
    assert len(advisory) == 1 and advisory[0]["anchor_object_id"] == "building_1"
    assert notes == []


def test_parse_recommend_non_dict_never_crashes():
    for bad in (None, "oops", 42, ["a"]):
        frame, recs, advisory, notes = parse_recommend(bad)
        assert recs == [] and advisory == []
        assert any("recommend_raw_not_dict" in n for n in notes)


def test_parse_recommend_recommendations_not_a_list():
    frame, recs, advisory, notes = parse_recommend(
        {"recommendations": "not a list"})
    assert recs == []


def test_parse_advisory_malformed_entries_dropped():
    raw = {"recommendations": [], "assumptions_advisory": [
        {"suspected": "ok", "anchor_object_id": "building_1"},  # kept
        "not a dict",                                           # dropped + note
        123]}                                                   # dropped + note
    _f, _r, advisory, notes = parse_recommend(raw)
    assert len(advisory) == 1
    assert sum("advisory_" in n and "malformed" in n for n in notes) == 2


def test_parse_advisory_not_a_list():
    _f, _r, advisory, notes = parse_recommend(
        {"recommendations": [], "assumptions_advisory": "nope"})
    assert advisory == [] and any("advisory_not_a_list" in n for n in notes)


def test_parse_pick_non_dict():
    pick, notes = parse_pick("garbage")
    assert pick["threat"] == "" and any("pick_raw_not_dict" in n for n in notes)


# ── run_recommend: two layers, events ───────────────────────────────────

def test_run_recommend_two_layers_and_events():
    events = []
    out = run_recommend(_record(), _asm(), query_fn=_script(),
                        on_event=events.append)
    assert len(out["recommendations"]) == 2         # HARD layer
    assert len(out["advisory"]) == 1                # ADVISORY layer
    types = [e["type"] for e in events]
    assert types[0] == "stage_started"
    assert "recommendations_ready" in types
    ready = next(e for e in events if e["type"] == "recommendations_ready")
    assert ready["n_recs"] == 2 and ready["n_advisory"] == 1


# ── build_graph_a: quads -> graph (code) ────────────────────────────────

def test_build_graph_a_from_quads():
    events = []
    _f, recs, _a, _n = parse_recommend(REC_OK)
    g = build_graph_a(_record(), _asm(), recs, on_event=events.append)
    assert len(g["nodes"]) == 3                     # all frozen entities
    assert len(g["edges"]) == 2                     # one per quad target
    cands = {c["threat"]: c["outgoing_edge_count"]
             for c in g["intervention_candidates"]}
    assert cands == {"building_1": 1, "dust_1": 1}
    assert any(e["type"] == "graph_a_built" for e in events)


def test_build_graph_a_empty_affected_objects_warns_no_edge():
    """A malformed quad (empty affected_objects) must not crash and must not
    emit an edge — Arm A records a graph_warning instead."""
    bad = _rec("building_1", "collapsed", "may_harm", [])   # empty targets
    g = build_graph_a(_record(), _asm(), [bad])
    assert g["edges"] == []
    assert g["graph_warnings"]                               # warned, not crashed


# ── run_graph_b ─────────────────────────────────────────────────────────

def test_graph_b_prompt_feeds_the_frozen_ids():
    """Graph B is independent of the RECOMMENDATIONS, not of the entities:
    it must be handed the frozen detected_objects (with ids) + threats, or it
    invents its own entity names (the 'burning_house' bug)."""
    from agentic.recommend import _graph_b_prompt
    p = _graph_b_prompt(_record(), _asm())
    assert "building_1" in p and "dust_1" in p           # frozen ids present
    assert "detected_objects" in p and "threats" in p
    assert "recommendations withheld" in p               # still independent of recs


def test_run_graph_b_parses_graph_and_pick():
    events = []
    g = run_graph_b(_record(), _asm(), query_fn=_script(), on_event=events.append)
    assert g["suppression_pick"]["threat"] == "building_1"
    assert len(g["edges"]) == 1
    assert any(e["type"] == "graph_b_built" for e in events)


def test_run_graph_b_garbage_answer_never_crashes():
    g = run_graph_b(_record(), _asm(), query_fn=lambda p: "not a dict")
    assert isinstance(g.get("nodes"), list)          # normalized to empty-ish


# ── pick_targets: three ways + agreement ────────────────────────────────

def test_pick_targets_three_ways_disagree():
    """Algorithm picks the acute hazard (dust, 'rising'); the model picks the
    building both ways → 2/3 agreement, not unanimous."""
    _f, recs, _a, _n = parse_recommend(REC_OK)
    g_a = build_graph_a(_record(), _asm(), recs)
    g_b = run_graph_b(_record(), _asm(), query_fn=_script())
    picks = pick_targets(_record(), g_a, g_b, recs, query_fn=_script())
    assert picks["a_pick"]["threat"] == "dust_1"        # acute beats stable
    assert picks["b_pick"]["threat"] == "building_1"
    assert picks["llm_pick"]["threat"] == "building_1"
    assert picks["agreement"] == round(2 / 3, 3)
    assert picks["unanimous"] is False


def test_pick_targets_resolves_dirty_ids_to_frozen():
    """A pick that arrives as 'building_1·collapsed' must resolve to the frozen
    id building_1 so it can agree with the Graph B pick. Without resolution the
    dirty string would be a distinct key (0.333); with it, b_pick and llm_pick
    both read building_1 → 0.667. (a_pick is dust_1: the algorithm favors the
    acute 'rising' state.)"""
    gb = {**GRAPH_B_OK, "suppression_pick": {"threat": "building_1",
                                             "state": "collapsed", "reason": "x"}}
    pick = {"threat": "building_1·collapsed", "state": "", "reason": "x"}  # dirty
    _f, recs, _a, _n = parse_recommend(REC_OK)
    g_a = build_graph_a(_record(), _asm(), recs)
    g_b = run_graph_b(_record(), _asm(), query_fn=_script(gb=gb))
    picks = pick_targets(_record(), g_a, g_b, recs,
                        query_fn=_script(gb=gb, pick=pick))
    assert picks["b_pick"]["object_id"] == "building_1"
    assert picks["llm_pick"]["object_id"] == "building_1"   # ·collapsed stripped
    assert picks["agreement"] == round(2 / 3, 3)            # b + llm agree
    assert picks["unanimous"] is False


def test_pick_targets_unanimous():
    """When all three name the same threat, agreement is 1.0 / unanimous."""
    gb = {**GRAPH_B_OK, "suppression_pick": {"threat": "dust_1",
                                             "state": "rising", "reason": "x"}}
    pick = {"threat": "dust_1", "state": "rising", "reason": "x"}
    _f, recs, _a, _n = parse_recommend(REC_OK)
    g_a = build_graph_a(_record(), _asm(), recs)
    g_b = run_graph_b(_record(), _asm(), query_fn=_script(gb=gb))
    picks = pick_targets(_record(), g_a, g_b, recs,
                        query_fn=_script(gb=gb, pick=pick))
    assert picks["a_pick"]["threat"] == "dust_1"
    assert picks["unanimous"] is True and picks["agreement"] == 1.0


# ── full Python spine ───────────────────────────────────────────────────

def test_run_stage4_end_to_end():
    events = []
    res = run_stage4(_record(), _asm(), "/x/E.jpg", query_fn=_script(),
                    on_event=events.append)
    assert len(res.recommendations) == 2
    assert len(res.advisory) == 1
    assert len(res.graph_a["nodes"]) == 3
    assert res.picks["b_pick"]["threat"] == "building_1"
    assert [e["type"] for e in events] == [
        "stage_started", "recommendations_ready", "graph_a_built",
        "graph_b_built", "targets_picked", "conformance_ready",
        "internal_alignment_ready", "alignment_ready", "trust_ready",
        "stage_done"]


# ── measured uncertainty over the recommend step (channel 2) ────────────

def _rec_answer(recs):
    """A full recommend answer carrying a given recommendations list."""
    return {"scene_summary": "s", "key_observations": [], "assumptions": [],
            "uncertainty_notes": [], "recommendations": recs,
            "assumptions_advisory": []}


def test_run_stage4_no_probes_leaves_uncertainty_empty():
    res = run_stage4(_record(), _asm(), query_fn=_script())   # n_probes=0
    assert res.uncertainty == {}


def test_run_stage4_probes_measure_advice_dispersion():
    """5 probes that flip the top target and the mechanism must surface a
    non-zero score, per-entity granularity, and the actual disagreeing pieces."""
    # 3 probes recommend building_1 first; 2 recommend dust_1 first — and the
    # mechanism for dust_1 splits may_harm/worsens.
    probe_seq = [
        [_rec("building_1", "collapsed", "may_harm", ["person_1"])],
        [_rec("building_1", "collapsed", "may_harm", ["person_1"])],
        [_rec("building_1", "collapsed", "may_harm", ["person_1"])],
        [_rec("dust_1", "rising", "may_harm", ["person_1"])],
        [_rec("dust_1", "rising", "worsens", ["dust_1"])],
    ]
    box = {"i": 0}

    def probe_fn(prompt):
        a = _rec_answer(probe_seq[box["i"] % len(probe_seq)])
        box["i"] += 1
        return a

    events = []
    res = run_stage4(_record(), _asm(), query_fn=_script(), probe_fn=probe_fn,
                    n_probes=5, on_event=events.append)
    u = res.uncertainty
    assert u["n_probes"] == 5 and u["score"] > 0.0
    # top-priority target flipped building_1 (×3) vs dust_1 (×2)
    assert u["granular"]["fields"]["top_priority_target"]["u"] > 0.0
    # two distinct advice candidates surfaced, ranked by votes
    assert len(u["candidates"]) >= 2 and u["candidates"][0]["votes"] == 3
    kinds = {d["kind"] for d in u["drivers"]}
    assert "top_target_flip" in kinds
    assert "recommend_probe" in [e["type"] for e in events]
    assert "recommend_uncertainty_ready" in [e["type"] for e in events]


def test_probes_all_fail_reads_as_unmeasured():
    def boom(prompt):
        raise RuntimeError("endpoint down")

    from agentic.recommend import measure_recommend_uncertainty
    mu = measure_recommend_uncertainty("prompt", 5, probe_fn=boom)
    assert mu.score == 1.0
    assert any(d.kind == "probes_failed" for d in mu.drivers)


def test_probe_garbage_answer_counts_as_empty_reading():
    """A malformed probe answer must parse to [] and count as instability, not
    crash the measurement (Iron Rule 1a at the probe boundary)."""
    from agentic.recommend import measure_recommend_uncertainty
    seq = [_rec_answer([_rec("building_1", "collapsed", "may_harm", ["person_1"])]),
           "not a dict", 42, None,
           _rec_answer([_rec("building_1", "collapsed", "may_harm", ["person_1"])])]
    box = {"i": 0}

    def probe_fn(prompt):
        a = seq[box["i"] % len(seq)]
        box["i"] += 1
        return a

    mu = measure_recommend_uncertainty("prompt", 5, probe_fn=probe_fn)
    # 5 readings recorded (3 empty from garbage), count wobble surfaced
    assert mu.n_probes == 5
    assert mu.granular["fields"]["recommendation_count"]["u"] > 0.0
