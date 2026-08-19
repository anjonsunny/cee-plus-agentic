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
    invents its own entity names (the 'burning_house' bug).

    F54-amended: independence is enforced by ABSENCE — no recommendation
    text ever enters the context — not by a sentence announcing the absence
    (Sunny: "if you don't tell about recommendations, how would the model
    know about the recommendations?")."""
    from agentic.recommend import _graph_b_prompt
    p = _graph_b_prompt(_record(), _asm())
    assert "building_1" in p and "dust_1" in p           # frozen ids present
    assert "detected_objects" in p and "threats" in p
    # independence by absence: no recommendation content in the context
    assert "evacuate" not in p.lower()                    # no advice leaked


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
        "internal_alignment_ready", "alignment_ready",
        "graph_b_internal_ready", "explanation_alignment_ready",
        "set_report_ready", "trust_ready",
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
    mu, _recs = measure_recommend_uncertainty("prompt", 5, probe_fn=boom)
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

    mu, _recs = measure_recommend_uncertainty("prompt", 5, probe_fn=probe_fn)
    # 5 readings recorded (3 empty from garbage), count wobble surfaced
    assert mu.n_probes == 5
    assert mu.granular["fields"]["recommendation_count"]["u"] > 0.0


# ── F16: one separator, one meaning ──────────────────────────────────────

def test_bare_id_normalises_every_shape():
    from agentic.recommend import bare_id
    assert bare_id("ambulance_1·proximity") == "ambulance_1"
    assert bare_id("house_1·burning") == "house_1"
    assert bare_id("house_1") == "house_1"
    assert bare_id("  spill_1 ") == "spill_1"
    assert bare_id(None) == "" and bare_id("") == "" and bare_id(17) == "17"


def test_scene_block_never_emits_a_second_dot_meaning():
    """The prompt taught 'id·role' beside 'id·state'; the model copied it.
    at-risk roles now travel under their own key."""
    from agentic.recommend import _scene_block
    block = _scene_block(_record(), _asm())
    assert "·" not in block
    assert "at_risk_as=" in block


# ── F18: one-ended claims become node annotations ────────────────────────

def _g(nodes, edges):
    return {"nodes": nodes, "edges": edges}


def test_placeholder_self_loop_leaves_the_edge_set():
    """B_pool's shape: the victim named as its own threat is a placeholder for
    a missing half, not a causal claim. It must not survive as an edge."""
    from agentic.recommend import annotate_one_ended
    g = annotate_one_ended(_g(
        [{"id": "child_1", "at_risk": True}, {"id": "chair_1"}],
        [{"source": "child_1", "effect": "may_harm", "target": "child_1"},
         {"source": "chair_1", "effect": "may_harm", "target": "child_1"}]))
    assert len(g["edges"]) == 1
    assert g["edges"][0]["source"] == "chair_1"
    assert any("F18" in w for w in g.get("graph_warnings", []))


def test_no_self_loop_survives_not_even_worsens():
    """Sunny, 2026-07-28: exclude self-loops COMPLETELY, so there is no
    confusion. A 'worsens' self-loop looks identical to a placeholder one, and
    two identical shapes with opposite meanings is the F16 ambiguity again.
    A hazard that threatens nothing is 'unattached', said one way only."""
    from agentic.recommend import annotate_one_ended
    g = annotate_one_ended(_g(
        [{"id": "fire_1", "hazardous": True}],
        [{"source": "fire_1", "effect": "worsens", "target": "fire_1"}]))
    assert g["edges"] == []
    assert g["nodes"][0]["unattached"] is True


def test_unattached_hazard_is_flagged():
    from agentic.recommend import annotate_one_ended
    g = annotate_one_ended(_g([{"id": "spill_1", "hazardous": True}], []))
    assert g["nodes"][0]["unattached"] is True
    assert "no target" in g["nodes"][0]["annotation_note"]


def test_unattributed_ignores_a_self_loop_as_an_incoming_edge():
    """Cowork's catch: 'no incoming edge' would NOT fire on B_pool, because
    child_1 has one — from itself. It must be 'from a DIFFERENT node'."""
    from agentic.recommend import annotate_one_ended
    g = annotate_one_ended(_g(
        [{"id": "child_1", "at_risk": True}],
        [{"source": "child_1", "effect": "may_harm", "target": "child_1"}]))
    assert g["nodes"][0]["unattributed"] is True


def test_a_victim_with_a_real_source_is_not_flagged():
    from agentic.recommend import annotate_one_ended
    g = annotate_one_ended(_g(
        [{"id": "pool_1", "hazardous": True}, {"id": "child_1", "at_risk": True}],
        [{"source": "pool_1", "effect": "may_harm", "target": "child_1"}]))
    by = {n["id"]: n for n in g["nodes"]}
    assert not by["child_1"].get("unattributed")
    assert not by["pool_1"].get("unattached")


def test_annotate_tolerates_malformed_graphs():
    from agentic.recommend import annotate_one_ended
    for junk in (None, {}, {"nodes": None, "edges": None},
                 {"nodes": ["x", 3], "edges": ["y", None]}):
        g = annotate_one_ended(junk)
        assert isinstance(g.get("nodes"), list) and isinstance(g.get("edges"), list)


# ── O1: both permissions ride one flag ───────────────────────────────────

def _prompt_with(flag: bool) -> str:
    import agentic.recommend as R
    old = R.RECS_MAY_BE_EMPTY
    R.RECS_MAY_BE_EMPTY = flag
    try:
        return R.RECOMMEND_PROMPT.format(
            scene_block=R._scene_block(_record(), _asm()),
            effects=R._EFFECT_LINE,
            empty_clause=R.EMPTY_RECS_CLAUSE if flag else "",
            affected_clause=(R.AFFECTED_OPTIONAL if flag
                             else R.AFFECTED_REQUIRED))
    finally:
        R.RECS_MAY_BE_EMPTY = old


def test_permissions_are_off_by_default():
    """O1 ships OFF. The clause-off arm must be the prompt we already ran, or
    the paired experiment has no baseline."""
    import agentic.recommend as R
    assert R.RECS_MAY_BE_EMPTY is False
    off = _prompt_with(False)
    assert "NON-EMPTY list of object_ids harmed." in off
    assert "MAY BE EMPTY" not in off


def test_both_permissions_flip_together():
    """The two permissions are one experiment. If they could flip separately a
    paired run could not say which one moved the output."""
    on = _prompt_with(True)
    assert "MAY BE EMPTY if no entity in the scene is hazardous" in on
    assert "MAY BE EMPTY if this hazard threatens nothing in particular" in on
    assert "NON-EMPTY" not in on


def test_the_permission_grants_and_never_steers():
    """Iron rule 5: it may say the array CAN be empty; it must never say the
    scene is safe, or to be cautious, or not to invent hazards — that teaches
    the answer and destroys the measurement we are taking."""
    on = _prompt_with(True).lower()
    for banned in ("do not invent", "be conservative", "only if you are sure",
                   "avoid", "no hazards are present", "be careful"):
        assert banned not in on


# ── The two prompt lines that manufactured B_pool's bad quads ────────────

def test_prompt_scopes_the_threat_slot_to_declared_threats():
    """Round 2, B_pool: the model wrote `threat: child_1, state: drowning` and
    was RIGHT by our own rules — nothing said the threat must be a source of
    harm, and 'drowning' really is child_1's declared state. It also wrote
    `child_2 --exposes--> child_1`, a swimmer as a hazard. Both are the same
    missing constraint."""
    import re

    import agentic.recommend as R
    p = re.sub(r"\s+", " ", R.RECOMMEND_PROMPT)      # the prompt is wrapped
    assert "one of the object_ids on the `threats:` line" in p
    assert "never its own threat" in p
    assert "at-risk entity is never the threat" in p


def test_prompt_allows_one_hazard_to_justify_several_actions():
    """The old line — 'one entry per distinct (threat, state) causal logic' —
    told the model each recommendation needed its OWN threat. With one pool
    and two children it had to invent a second hazard, so it nominated a
    child. Two actions, one hazard, must be legal."""
    import re

    import agentic.recommend as R
    p = re.sub(r"\s+", " ", R.RECOMMEND_PROMPT)
    assert "one entry per ACTION" in p
    assert "may rest on the SAME (threat, state)" in p
    assert "one entry per distinct (threat, state)" not in p


def test_prompt_still_renders_in_both_o1_arms():
    for flag in (False, True):
        import re
        assert "threats:` line" in re.sub(r"\s+", " ", _prompt_with(flag))


# ── Graph B direction example (the victim-as-source inversion) ───────────

def test_graph_b_prompt_has_direction_example():
    """B_pool, twice: `child_1 · drowning --may_harm--> pool_1`. The threat
    reason we hand Graph B ends on the victim, and the model read word order
    as arrow direction."""
    from agentic.recommend import _graph_b_prompt
    p = _graph_b_prompt(_record(), _asm())
    assert "may_harm--> worker" in p


def test_graph_b_prompt_stays_neutral():
    """Iron rule 5: the example must carry no calibration scene and no
    id-shaped token. Electricity appears in none of the six scenes."""
    import re

    from main import GRAPH_B_PROMPT
    from agentic.recommend import GRAPH_B_DIRECTION_EXAMPLE, _graph_b_prompt
    example = GRAPH_B_DIRECTION_EXAMPLE
    for scene in ("fire", "pool", "tanker", "spill", "collapse", "park"):
        assert scene not in example.lower(), f"example names {scene}"
    assert re.search(r"\b\w+_\d+\b", example) is None
    # The example must be the only thing we ADD to the frozen block that could
    # name a scene. The caption and detected_objects legitimately carry scene
    # words — they are the run's data, not prompt — so the check is scoped to
    # the block we author.
    added = _graph_b_prompt(_record(), _asm()).replace(GRAPH_B_PROMPT, "")
    assert example in added
    assert "wire --may_harm--> worker" in added


def test_main_untouched():
    """The example lives in Arm B. main.py stays byte-identical."""
    from main import GRAPH_B_PROMPT
    assert "wire" not in GRAPH_B_PROMPT


# ── Instruction 2: is Graph B's causal belief stable? ────────────────────

def _gb(edges, pick=""):
    """A raw Graph B answer in the shape the model actually returns:
    normalize_graph_b reads raw['causal_graph'], and the pick names a
    'threat'. Nodes must be declared or the edges are dropped."""
    ids = {x for s, _e, t in edges for x in (s, t)}
    return {"causal_graph": {
        "nodes": [{"id": i, "state": "collapsed" if i != "person_1"
                   else "trapped", "hazardous": i != "person_1"}
                  for i in sorted(ids)],
        "edges": [{"source": s, "effect": e, "target": t,
                   "via_state": "collapsed" if s != "person_1" else "trapped"}
                  for s, e, t in edges]},
        "suppression_pick": {"threat": pick, "state": "collapsed"}}


def _probe_seq(graphs):
    """A probe_fn that returns each scripted graph in turn."""
    box = {"i": 0}

    def fn(_p):
        g = graphs[box["i"] % len(graphs)]
        box["i"] += 1
        if isinstance(g, Exception):
            raise g
        return g
    return fn, box


def test_graph_b_uncertainty_stable_when_every_probe_agrees():
    from agentic.recommend import measure_graph_b_uncertainty
    g = _gb([("building_1", "may_harm", "person_1")], pick="building_1")
    fn, _ = _probe_seq([g])
    u = measure_graph_b_uncertainty(_record(), _asm(), 5, probe_fn=fn)
    assert u["score"] == 0.0
    assert u["direction_instability"] == 0.0
    assert u["n_probes"] == 5


def test_graph_b_uncertainty_catches_a_reversed_edge():
    """The whole point: 3 probes one way, 2 the other. A count of edges could
    never see this — only the direction can."""
    from agentic.recommend import measure_graph_b_uncertainty
    fwd = _gb([("building_1", "may_harm", "person_1")], pick="building_1")
    rev = _gb([("person_1", "may_harm", "building_1")], pick="person_1")
    fn, _ = _probe_seq([fwd, fwd, fwd, rev, rev])
    u = measure_graph_b_uncertainty(_record(), _asm(), 5, probe_fn=fn)
    assert u["direction_instability"] == 0.4
    # the modal set keeps the majority direction
    assert ["building_1", "may_harm", "person_1"] in u["modal_edges"]


def test_graph_b_uncertainty_survives_malformed_answers():
    from agentic.recommend import measure_graph_b_uncertainty
    junk = [{"edges": "not a list"}, {}, {"nodes": None, "edges": None},
            "a bare string", 17]
    fn, _ = _probe_seq(junk)
    u = measure_graph_b_uncertainty(_record(), _asm(), 5, probe_fn=fn)
    assert isinstance(u["score"], float)


def test_graph_b_uncertainty_all_probes_failed_reads_maximally_unstable():
    """Silence must never read as agreement."""
    from agentic.recommend import measure_graph_b_uncertainty
    fn, _ = _probe_seq([RuntimeError("endpoint down")])
    u = measure_graph_b_uncertainty(_record(), _asm(), 5, probe_fn=fn)
    assert u["score"] == 1.0
    assert u["flags"] and u["flags"][0]["kind"] == "graph_b_probes_failed"


def test_graph_b_uncertainty_off_by_default_never_calls_the_model():
    from agentic.recommend import measure_graph_b_uncertainty
    fn, box = _probe_seq([_gb([("building_1", "may_harm", "person_1")])])
    u = measure_graph_b_uncertainty(_record(), _asm(), 0, probe_fn=fn)
    assert u == {} and box["i"] == 0


def test_graph_b_probe_events_carry_the_edges_not_just_a_count():
    """The recommend probes log only n_recs, which is why the reversed-edge
    run was undiagnosable after the fact."""
    from agentic.recommend import measure_graph_b_uncertainty
    seen = []
    fn, _ = _probe_seq([_gb([("building_1", "may_harm", "person_1")],
                            pick="building_1")])
    measure_graph_b_uncertainty(_record(), _asm(), 2, probe_fn=fn,
                                on_event=lambda e: seen.append(e))
    probes = [e for e in seen if e["type"] == "graph_b_probe"]
    assert probes and probes[0]["edges"] == [["building_1", "may_harm",
                                              "person_1"]]
    assert probes[0]["pick"] == "building_1"


# ── Prompt neutrality across the Stage 4 prompts (iron rule 5) ───────────

def test_stage4_prompts_name_no_scene_and_no_id_token():
    """The rulebook has had this guard since the P7 leak; the Stage 4 prompts
    never did, and a proposed line for this very change carried a calibration
    entity AND an id-shaped token. Anything we author and send to the model is
    covered now — the injected scene DATA is exempt, it is the run, not prompt."""
    import re

    import agentic.recommend as R
    scenes = re.compile(r"tanker|collapse\b|swimming pool|brush fire",
                        re.I)
    idtok = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d+\b")
    authored = {
        "RECOMMEND_PROMPT": R.RECOMMEND_PROMPT,
        "GRAPH_B_DIRECTION_EXAMPLE": R.GRAPH_B_DIRECTION_EXAMPLE,
        "LLM_PICK_PROMPT": R.LLM_PICK_PROMPT,
        "EMPTY_RECS_CLAUSE": R.EMPTY_RECS_CLAUSE,
        "AFFECTED_OPTIONAL": R.AFFECTED_OPTIONAL,
    }
    for name, text in authored.items():
        assert not scenes.search(text), f"{name} names a calibration scene"
        assert idtok.search(text) is None, f"{name} carries an id-shaped token"


def test_the_action_must_carry_ids_and_the_quad_must_cover_them():
    """Across three scenes the quad stayed anchored to the action in 3/3 cases
    where the action named ids, and drifted in 3/3 where it was prose only.
    Naming the entity is what ties the quad to the action."""
    import re

    import agentic.recommend as R
    p = re.sub(r"\s+", " ", R.RECOMMEND_PROMPT)
    assert "name it by its object_id, exactly as listed" in p
    assert "never by a prose description" in p
    assert "must also appear in the quad below" in p


# ── F25: the quad arrives in whatever shape the model felt like ─────────

def test_a_quad_written_as_an_arrow_string_is_recovered():
    """A_fire round 4, verbatim. The model's causal claim was CORRECT; the
    frozen normalizer takes any non-dict quad to the all-N/A placeholder, so
    all three recommendations lost their entire claim — silently."""
    from agentic.recommend import parse_recommend
    raw = {"recommendations": [
        {"rank": 1, "action": "Evacuate person_1 and dog_1.",
         "reason": "Because house_1 is burning, it may harm person_1.",
         "structured_reasoning":
             "house_1 -> burning -> may_harm -> person_1, dog_1"}]}
    _f, recs, _a, notes = parse_recommend(raw)
    q = recs[0]["structured_reasoning"]
    assert q["threat"] == "house_1" and q["state"] == "burning"
    assert q["effect"] == "may_harm"
    assert q["affected_objects"] == ["person_1", "dog_1"]
    assert any("recovered_from_string" in n for n in notes)


def test_every_arrow_dialect_the_model_has_used():
    from agentic.recommend import coerce_quad
    for text, want in (
            ("a_1 -> burning -> may_harm -> b_1", ("a_1", "burning", "may_harm")),
            ("a_1 · burning --may_spread_to--> b_1",
             ("a_1", "burning", "may_spread_to")),
            ("a_1 → burning → worsens → b_1",
             ("a_1", "burning", "worsens")),
            ("a_1 -> may_harm -> b_1", ("a_1", "", "may_harm"))):
        q = coerce_quad(text, [], 1)
        assert (q["threat"], q["state"], q["effect"]) == want, text
        assert q["affected_objects"] == ["b_1"]


def test_an_unreadable_quad_is_noted_not_silently_zeroed():
    from agentic.recommend import coerce_quad
    notes: list = []
    assert coerce_quad("total nonsense", notes, 2) == "total nonsense"
    assert any("unreadable" in n for n in notes)


def test_a_quad_that_arrived_and_left_empty_is_always_noted():
    """The general guard, and the one that actually matters. Whatever shape
    comes next, a recommendation that ARRIVED with a quad and LEFT without one
    is a parse loss — and it must never again be silent. The old behaviour
    reported an empty parse_notes while destroying the claim Stage 4 exists to
    measure, and the card checks then charged the model for our deletion."""
    from agentic.recommend import parse_recommend
    raw = {"recommendations": [
        {"rank": 1, "action": "x", "reason": "y",
         "structured_reasoning": ["house_1", "burning", "may_harm"]}]}
    _f, recs, _a, notes = parse_recommend(raw)
    assert recs[0]["structured_reasoning"]["threat"] == "N/A"
    assert any("LOST_IN_PARSE" in n for n in notes)


def test_a_well_formed_quad_is_left_alone_and_notes_nothing():
    from agentic.recommend import parse_recommend
    raw = {"recommendations": [
        {"rank": 1, "action": "x", "reason": "y",
         "structured_reasoning": {"threat": "house_1", "state": "burning",
                                  "effect": "may_harm",
                                  "affected_objects": ["person_1"]}}]}
    _f, recs, _a, notes = parse_recommend(raw)
    assert recs[0]["structured_reasoning"]["threat"] == "house_1"
    assert not [n for n in notes if "quad" in n]


def test_the_prompt_asks_for_an_object_not_a_sentence():
    """The regression was ours: 'the SAME sentence as the reason, with its
    slots filled in' made a JSON field read like prose, and the model wrote
    prose. The restatement meaning has to survive without that reading."""
    from agentic.recommend import RECOMMEND_PROMPT
    assert "JSON OBJECT" in RECOMMEND_PROMPT
    assert "never a sentence, never an arrow string" in RECOMMEND_PROMPT
    assert "SAME sentence as the reason" not in RECOMMEND_PROMPT


# ── F27: the instruction was stricter than the rule it described ────────

def test_the_action_clause_permits_naming_what_is_not_in_the_scene():
    """A_fire regression. The clause read as an absolute ban on any noun that
    is not an object_id — so the model stopped recommending anything that
    brings in an outside resource, and every hazard-directed action vanished:
    15 prior runs all extinguished the hazard, the two after the edit only
    rescued victims.

    Nothing in code ever forbade it — the checks read the scene ids, classify
    the action, and never look at the other words. The INSTRUCTION was stricter
    than the rule it described, and the model believes the instruction."""
    from agentic.recommend import RECOMMEND_PROMPT
    assert "describes the SCENE, not the response" in RECOMMEND_PROMPT
    assert "name those in ordinary words" in RECOMMEND_PROMPT
    # and the real constraint survives: a scene entity must be named by id
    assert "name it by its object_id" in RECOMMEND_PROMPT


def test_the_action_clause_names_no_disaster_type():
    """Prompt neutrality (iron rule 5). The permission covers fire, flood,
    quake, spill, collapse alike, so it may not carry an example from any of
    them — an instruction that reaches for one domain teaches that domain."""
    from agentic.recommend import RECOMMEND_PROMPT
    low = RECOMMEND_PROMPT.lower()
    for word in ("firefighter", "fire ", "flood", "hurricane", "earthquake",
                 "spill", "drown", "collapse", "ambulance", "hazmat",
                 "police", "paramedic"):
        assert word not in low, word


def test_an_action_naming_an_outside_resource_is_still_hazard_directed():
    """The behaviour the clause was blocking. This card must pass clean and be
    classified as suppression-testable — it is exactly what the intervention
    gate needs to exist."""
    from agentic.evals4 import explanation_alignment
    from agentic.test_evals4 import _obj, _rec_pair
    rec, asm = _rec_pair()
    card = {"rank": 1,
            "action": "Deploy a response team to extinguish house_1.",
            "reason": "Because house_1 is burning it may_harm person_1.",
            "structured_reasoning": {"threat": "house_1", "state": "burning",
                                     "effect": "may_harm",
                                     "affected_objects": ["person_1"]},
            "remaining_risk": "(car_1, parked)"}
    r = explanation_alignment(rec, asm, [card])
    assert r["modes"][0]["mode"] == "hazard_directed"
    assert r["failures"] == []


def test_probe_events_carry_the_full_prose_not_just_the_skeleton():
    """F49 / JUDGES.md 9.9. The five probe answers to the SAME prompt are the
    preference-pair corpus for DPO. The loop used to keep only the quad
    skeleton (a count and one entity id per probe), so every run before the
    fix permanently lost the action/reason prose — the half of the answer a
    subject model would actually be trained on."""
    def probe_fn(prompt):
        return _rec_answer([_rec("building_1", "collapsed", "may_harm",
                                 ["person_1"])])

    events = []
    run_stage4(_record(), _asm(), query_fn=_script(), probe_fn=probe_fn,
               n_probes=2, on_event=events.append)
    probes = [e for e in events if e["type"] == "recommend_probe"]
    assert len(probes) == 2
    for e in probes:
        recs = e["recs"]
        assert recs, "the full parsed recommendations must ride in the event"
        assert recs[0]["action"], "prose action lost — the capture bug is back"
        assert recs[0]["reason"], "prose reason lost — the capture bug is back"
        q = recs[0]["structured_reasoning"]
        assert q["threat"] == "building_1" and q["state"] == "collapsed"


def test_graph_b_opening_is_sunnys_and_threats_are_back():
    """F54 AMENDED in the section-by-section prompt review (2026-08-19):
    threats deliberately return — Graph B's job is causal STRUCTURE, not
    hazard detection — while the at-risk register stays unseeded, so the
    victim side remains the earned half of A-vs-B. The opening paragraph is
    Sunny's, approved verbatim: the first prompt text shipped under the
    inspection rule."""
    from agentic.recommend import _graph_b_prompt
    prompt = _graph_b_prompt(_record(), _asm())
    assert prompt.startswith("You are extracting the causal graph")
    assert "from the perspective of an emergency response analyst" in prompt
    assert '"threats":' in prompt                        # seeded again, chosen
    assert "at_risk" not in prompt.split("Prior analysis")[1]  # victims earned
    # the two frozen sentences Sunny cut stay cut
    assert "Recommendations are deliberately withheld" not in prompt
    assert "regardless of which a responder would address first" not in prompt
    # and the frozen body still follows the new opening intact
    assert "## State vocabulary" in prompt and "## Rules" in prompt
    # section 2: the instancing paragraph is cut to its one live sentence —
    # Stage 1 fixes the entity list upstream, so the ten-node budgeting and
    # people-counting rules had nothing to govern here (Sunny: "That's it.")
    assert "Representative instancing" not in prompt
    assert "Do not add nodes beyond the detected_objects supplied." in prompt
    # the preamble (Sunny, verbatim): anatomy defined BEFORE the law refers
    # to it, and the three state lists announced as one decoder
    assert "has two parts: NODES" in prompt
    assert "FROM the entity doing the harm" in prompt
    assert "role comes ONLY from its `state` field" in prompt
    i_pre = prompt.index("has two parts: NODES")
    assert i_pre < prompt.index("Hazardous (entity is a SOURCE")  # defs first
    # sections 3-7 (Sunny, verbatim, the prompt review): engulfing gone
    # (perception remaps it away), collapse tie-breaker and behavioral
    # families gone (Stage-1 state-choosing), living-beings carve-out and
    # proximity clause folded into the per-entity classifier, "Normal states"
    # list dead with its category, categories named by the schema flags.
    assert "engulfing" not in prompt
    assert "`collapsing` vs `collapsed`" not in prompt
    assert "Behavioral families" not in prompt
    assert "**Living beings only.**" not in prompt
    assert "Normal-state entities that are nonetheless" not in prompt
    assert "Normal states:" not in prompt
    assert "classify it in this order" in prompt
    assert "deliberate claim that it is safe" in prompt
    flat = prompt.replace("\n", " ")
    assert "hazardous_in_context. `hazardous_in_context` is the last-resort" \
        in flat
    assert "suffocating, unconscious." in flat
    # fluid area (Sunny): emission half dead (Stage 1 emits objects), routing
    # core folded into the effect table; provenance says what suppression
    # actually buys (the fluid stops being FED; released fluid may persist)
    assert "Fluid / gaseous hazards" not in prompt
    assert "run FROM the fluid, not from an entity it has inundated" in flat
    assert "what has already been released may persist" in flat
    # self-loops: not banned, just NEVER MENTIONED (Sunny: "remove it and
    # don't talk about it" — a ban teaches the concept in order to forbid it,
    # and mentioning a thing invites it, F2). Standalone nodes are allowed.
    assert "self-loop" not in flat
    assert "Self-reference" not in flat
    import re
    squashed = re.sub(r"\s+", " ", prompt)
    assert "or no edges at all when nothing in the scene is affected" \
        in squashed
    assert "5. Do NOT produce" in flat          # rules renumbered, none lost
    # harm channels: compressed to the one non-redundant law
    assert "Independent harm channels" not in prompt
    assert "one edge PER hazard that reaches it" in flat
    # effect vocabulary: ordered checklist, first match wins (Sunny) — the
    # truth-conditions list, the fluid verb table, and the tense clause all
    # died as duplicates or consumerless commentary; no examples (F2)
    assert "take the FIRST that fits" in flat
    assert "worsens, emitted in BOTH directions" in flat
    assert "most specific applicable" not in flat
    assert "truth conditions" not in flat.lower()
    assert "conversion pending" not in flat
    assert "harm is happening NOW" not in flat
    assert "harm actualized" not in flat
    # distance Part B (Sunny): purpose first, one rule per hazard type, and
    # the compound-hazard exception that closes the C hole
    assert "never claimed to injure someone it cannot physically reach" \
        in re.sub(r"\s+", " ", prompt)
    assert "ignition radius, not contact distance" in flat
    assert "mid-yard is the boundary" not in prompt   # old threshold prose gone
    # distance Part C (Sunny): obstruction edges only when blocking endangers
    assert "when the blocking itself endangers them" in flat
    assert "Obstruction coupling rule:" not in prompt
    assert "ENTRAPMENT" not in prompt
    assert "Distress state" not in prompt             # dead taxonomy word
    assert "TOWARD a hazard" not in prompt            # closer removed (Sunny)
    # mutual-hazard: two live laws only
    assert "Mutual-hazard rule" not in prompt
    assert "never use may_spread_to" in flat
    assert "draw the edges from that cause to each" in flat
    assert "6." not in squashed.split("## Rules")[1].split("Return valid")[0]
    assert "on the SAME entity" not in flat
