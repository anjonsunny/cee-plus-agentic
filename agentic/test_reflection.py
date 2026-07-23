"""Hermetic tests for the Stage 2 reflection loop (increment 2).

Scripted subject model, no network. Covered: trigger collection with the
U threshold, prompt composition (states + geometry words + raw boxes for
the spatial question + rulebook citations), repair, STOOD, cap, model
death, malformed reflection answers, and the U0->U1 trajectory.

Run:  pytest agentic/test_reflection.py -q
"""
from __future__ import annotations

from agentic.assessment import parse_assessment, run_assessment
from agentic.perception import DetectedObject, PerceptionResult
from agentic.reflection import (
    collect_triggers,
    compose_reflection,
    run_reflection,
)
from agentic.uncertainty import measure_merged

# ── Fixtures ────────────────────────────────────────────────────────────


def _obj(oid, label, state, kind, bbox=(0, 0, 10, 10)):
    return DetectedObject(
        object_id=oid, label=label, family="x", state=state,
        state_kind=kind, description="", bbox=list(bbox),
        box_source="dino_matched", box_confidence=0.9,
        anchor_bbox=list(bbox), mask_path=None, label_note="",
        vocab_extension=False, family_name_as_label=False)


POOL = PerceptionResult(
    image_path="/x/B.jpg", image_size=[1000, 800],
    caption="A child struggles in a pool.", entity_source="vlm",
    detected_objects=[
        _obj("child_1", "child", "drowning", "at_risk", (200, 200, 300, 300)),
        _obj("pool_1", "pool", "engulfing", "hazard_bearing",
             (100, 150, 600, 500)),
        _obj("adult_1", "adult", "standing", "normal", (610, 200, 700, 450)),
    ])

BAD_ANSWER = {"disaster_scenario": "Yes", "disaster_type": "drowning",
              "disaster_level": 8, "confidence": 0.9,
              "threats": [{"object_id": "child_1", "reason": "drowning"}],
              "at_risk": [{"object_id": "child_1", "kind": "distress"}]}
GOOD_ANSWER = {"disaster_scenario": "Yes", "disaster_type": "drowning",
               "disaster_level": 8, "confidence": 0.9,
               "threats": [{"object_id": "pool_1", "reason": "engulfing"}],
               "at_risk": [{"object_id": "child_1", "kind": "distress"}]}


def scripted(answers):
    state = {"i": 0}

    def fn(prompt):
        a = answers[min(state["i"], len(answers) - 1)]
        state["i"] += 1
        return a

    return fn


# ── Triggers ────────────────────────────────────────────────────────────


def test_membership_triggers_respect_threshold():
    def probe(at_risk_ids):
        return {"scenario": "Yes", "disaster_type": "fire", "level": 7,
                "bucket": "catastrophic", "threat_ids": ["pool_1"],
                "at_risk_ids": at_risk_ids}
    mu = measure_merged([probe(["child_1", "adult_1"]), probe(["child_1"]),
                         probe(["child_1"]), probe(["child_1"]),
                         probe(["child_1"])])
    t = collect_triggers([], mu)
    # child_1 5/5 -> U 0.0 (no trigger); adult_1 1/5 -> U 0.8 (trigger)
    memb = [x for x in t if x["type"] == "membership_split"]
    assert [m["object_id"] for m in memb] == ["adult_1"]
    assert memb[0]["votes"] == "1/5"


def test_violation_triggers_always_count():
    t = collect_triggers([{"kind": "scenario_no_level_gt0",
                           "evidence": "No but level=3"}], None)
    assert t == [{"type": "violation", "kind": "scenario_no_level_gt0",
                  "evidence": "No but level=3"}]


# ── Prompt composition ──────────────────────────────────────────────────


def test_compose_quotes_rulebook_and_context_policy():
    a, _ = parse_assessment(BAD_ANSWER)
    prompt = compose_reflection(a, POOL, [
        {"type": "violation", "kind": "threat_state_not_hazardous",
         "evidence": "child_1 listed as threat but state 'drowning' is "
                     "at_risk"},
        {"type": "membership_split", "list": "at_risk",
         "object_id": "adult_1", "votes": "2/5"},
    ])
    assert "RULE S5" in prompt                       # violation citation
    assert "RULE G1" in prompt and "RULE G3" in prompt
    assert "standing (normal)" in prompt             # states always
    assert "adjacent" in prompt or "overlap" in prompt  # geometry in words
    assert "[610, 200, 700, 450]" in prompt          # raw boxes: spatial q
    assert "YOUR PREVIOUS ANSWER" in prompt
    assert "return it unchanged" in prompt           # standing is allowed


# ── The loop ────────────────────────────────────────────────────────────


def test_reflection_repairs_a_violation():
    """B_pool's real case: the drowning child listed as a threat. The
    loop quotes S5; the scripted model moves the threat to pool_1."""
    a, _ = parse_assessment(BAD_ANSWER)
    from agentic.assessment import enforce_kinds, internal_check
    v = enforce_kinds(a, POOL) + internal_check(a, POOL)
    assert sorted(x["kind"] for x in v) == ["hazard_not_in_threats",
                                            "threat_state_not_hazardous"]

    events = []
    a2, v2, notes, trace = run_reflection(
        POOL, a, v, None, scripted([GOOD_ANSWER]),
        emit=lambda t, **d: events.append({"type": t, **d}))
    assert v2 == []
    assert [t.object_id for t in a2.threats] == ["pool_1"]
    assert trace.stopped_reason == "clean"
    assert trace.rounds[0].changed
    kinds = [e["type"] for e in events]
    assert kinds == ["reflect_round_started", "reflect_round_done"]


def test_reflection_stood_ground_records_and_stops():
    a, _ = parse_assessment(BAD_ANSWER)
    from agentic.assessment import enforce_kinds, internal_check
    v = enforce_kinds(a, POOL) + internal_check(a, POOL)
    a2, v2, _, trace = run_reflection(POOL, a, v, None,
                                      scripted([BAD_ANSWER]))
    assert trace.stopped_reason == "no_change"
    assert len(trace.rounds) == 1 and not trace.rounds[0].changed
    assert sorted(x["kind"] for x in v2) == ["hazard_not_in_threats",
                                             "threat_state_not_hazardous"]


def test_reflection_cap_reached_on_oscillating_answers():
    """A model that keeps changing but never fixes: cap ends it."""
    a, _ = parse_assessment(BAD_ANSWER)
    from agentic.assessment import enforce_kinds, internal_check
    v = enforce_kinds(a, POOL) + internal_check(a, POOL)
    other_bad = dict(BAD_ANSWER, disaster_level=9)   # changed, still bad
    third_bad = dict(BAD_ANSWER, disaster_level=7)
    a2, v2, _, trace = run_reflection(POOL, a, v, None,
                                      scripted([other_bad, third_bad]),
                                      cap=2)
    assert trace.stopped_reason == "cap_reached"
    assert len(trace.rounds) == 2
    assert any(x["kind"] == "threat_state_not_hazardous" for x in v2)


def test_reflection_survives_malformed_and_dead_model():
    a, _ = parse_assessment(BAD_ANSWER)
    from agentic.assessment import enforce_kinds, internal_check
    v = enforce_kinds(a, POOL) + internal_check(a, POOL)
    # malformed reflection answer: parses to conservative default (a
    # CHANGED answer), gets re-checked, loop continues within cap
    a2, v2, notes, trace = run_reflection(POOL, a, v, None,
                                          scripted(["garbage"]), cap=1)
    assert trace.stopped_reason == "cap_reached"
    assert any("reflect_r1:answer_not_object" in n for n in notes)

    def dead(prompt):
        raise ConnectionError("ollama down")
    a3, v3, _, trace2 = run_reflection(POOL, a, v, None, dead)
    assert trace2.stopped_reason == "model_error"
    assert a3.model_dump() == a.model_dump()         # answer untouched


# ── End-to-end through run_assessment: U0 -> U1 ─────────────────────────


def test_u0_u1_trajectory_recorded():
    """Canonical answer is bad -> probes measure U0 -> reflection fixes
    -> re-probe measures U1 -> both recorded in the trace."""
    canonical_then_reflection = scripted([BAD_ANSWER, GOOD_ANSWER])
    probes = scripted([BAD_ANSWER, GOOD_ANSWER, BAD_ANSWER,   # U0 probes
                       GOOD_ANSWER, GOOD_ANSWER, GOOD_ANSWER])  # U1 probes
    events = []
    out = run_assessment(POOL, query_fn=canonical_then_reflection,
                         n_probes=3, probe_fn=probes,
                         on_event=events.append)
    trace = out.reflection_trace
    assert trace["stopped_reason"] == "clean"
    assert trace["u_before"] is not None and trace["u_after"] is not None
    assert trace["u_after"] <= trace["u_before"]     # stabilized here
    # final record carries the REPAIRED verdict
    assert out.violations == []
    assert [t.object_id for t in out.assessment.threats] == ["pool_1"]
    phases = [e.get("phase") for e in events
              if e["type"] == "assess_uncertainty"]
    assert phases == ["initial", "post_reflection"]
    stop = next(e for e in events if e["type"] == "reflect_stopped")
    assert stop["u_before"] == trace["u_before"]
    # two verdict events: the original and the corrected one
    assert sum(1 for e in events if e["type"] == "assess_verdict") == 2


def test_field_instability_prompt_avoids_capitulation_framing():
    """B_pool lesson: the scenario question must present EVIDENCE, never
    the model's reliability. 'unstable' framing invited folding to No."""
    a, _ = parse_assessment(GOOD_ANSWER)
    prompt = compose_reflection(a, POOL, [
        {"type": "field_instability", "driver": "scenario_flip",
         "evidence": "probes split Yes×3, No×2"}])
    assert "DECLARED DANGER STATES" in prompt
    assert "child_1·drowning" in prompt and "pool_1·engulfing" in prompt
    assert "cannot be 'No'" in prompt
    assert "independently of any previous answer" in prompt
    assert "UNSTABLE FIELD" not in prompt        # the old framing is gone


def test_capitulation_is_now_caught_in_round():
    """Replay B_pool's failure with the new checks: the model capitulates
    to No/0 mid-reflection; the round's re-check rejects it, and with the
    cap spent the loop ends STOOD-style with violations recorded — the
    capitulated answer can no longer end as 'clean'."""
    a, _ = parse_assessment(BAD_ANSWER)
    from agentic.assessment import enforce_kinds, internal_check
    v = enforce_kinds(a, POOL) + internal_check(a, POOL)
    capitulated = {"disaster_scenario": "No", "disaster_type": "N/A",
                   "disaster_level": 0, "confidence": 0.9,
                   "at_risk": [{"object_id": "child_1", "kind": "distress"}]}
    a2, v2, _, trace = run_reflection(POOL, a, v, None,
                                      scripted([capitulated, capitulated]),
                                      cap=2)
    kinds = [x["kind"] for x in v2]
    assert "missed_disaster_incoherence" in kinds
    assert "scenario_no_with_entities" in kinds
    assert trace.stopped_reason in ("no_change", "cap_reached")
    assert trace.stopped_reason != "clean"       # the B_pool bug is dead


# ── Top-2 runoff (Sunny: high U -> blind-judge the top two readings) ────


def test_probe_candidates_group_and_rank():
    from agentic.uncertainty import probe_candidates
    l7 = {"disaster_scenario": "Yes", "severity_bucket": "catastrophic",
          "disaster_level": 7, "disaster_type": "fire",
          "threats": [{"object_id": "house_1", "reason": "a"}],
          "at_risk": [{"object_id": "person_1", "kind": "proximity"}]}
    l7b = dict(l7, threats=[{"object_id": "house_1", "reason": "WORDED "
                             "differently — same candidate"}])
    l4 = dict(l7, severity_bucket="serious", disaster_level=4)
    cands = probe_candidates([l7, l4, l7b, l7])
    assert [c["votes"] for c in cands] == [3, 1]     # reasons never split
    assert cands[0]["answer"]["severity_bucket"] == "catastrophic"


TANKERISH = PerceptionResult(
    image_path="/x/C.jpg", image_size=[1000, 800],
    caption="tanker leak with fire", entity_source="vlm",
    detected_objects=[
        _obj("tanker_truck_1", "tanker_truck", "leaking", "hazard_bearing",
             (0, 0, 300, 200)),
        _obj("spill_1", "spill", "seeping", "hazard_bearing",
             (100, 180, 500, 300)),
        _obj("fire_1", "fire", "spreading", "hazard_bearing",
             (600, 100, 900, 400)),
        _obj("person_1", "person", "standing", "normal",
             (320, 100, 380, 260)),
    ])


def test_runoff_fires_only_above_threshold_with_two_candidates():
    from agentic.assessment import run_assessment
    l7 = {"disaster_scenario": "Yes", "disaster_type": "fire",
          "disaster_level": 7, "confidence": 0.9,
          "threats": [{"object_id": "fire_1", "reason": "spreading"},
                      {"object_id": "tanker_truck_1", "reason": "leaking"},
                      {"object_id": "spill_1", "reason": "seeping"}],
          "at_risk": [{"object_id": "person_1", "kind": "proximity",
                       "reason": "adjacent"}]}
    l3 = dict(l7, disaster_level=3, threats=[
        {"object_id": "fire_1", "reason": "small fire"}])
    judged = {"n": 0}

    def judge(prompt):
        judged["n"] += 1
        assert "CANDIDATE A" in prompt and "DECLARED STATES" in prompt
        return "A — the full threat set matches the declared states."

    events = []
    out = run_assessment(
        TANKERISH, query_fn=scripted([l7, l7]),  # canonical + reflection
        n_probes=5, probe_fn=scripted([l7, l3, l7, l3, l3]),
        runoff_judge_fn=judge, on_event=lambda e: events.append(e))
    mu = out.measured_uncertainty
    # NOTE: final mu may be the post-reflection re-probe; the runoff
    # happened on the INITIAL measurement — its event proves it ran.
    runoff_events = [e for e in events if e["type"] == "assess_runoff"]
    assert judged["n"] >= 1 and len(runoff_events) == 1
    ev = runoff_events[0]
    assert ev["winner"] in ("top1", "top2")
    assert ev["top1_votes"] in ("3/5", "2/5")

    # unanimous probes -> no runoff, judge never called
    judged["n"] = 0
    run_assessment(TANKERISH, query_fn=scripted([l7]),
                   n_probes=3, probe_fn=scripted([l7, l7, l7]),
                   runoff_judge_fn=judge)
    assert judged["n"] == 0


def test_runoff_trigger_reaches_the_reflection_prompt():
    from agentic.uncertainty import MeasuredUncertainty
    mu = MeasuredUncertainty(n_probes=5, score=0.3)
    mu.runoff = {"winner": "top1", "raw": "A cites burning states.",
                 "top1_votes": "3/5", "top2_votes": "2/5",
                 "top1": {"disaster_scenario": "Yes",
                          "disaster_type": "fire", "disaster_level": 7,
                          "threats": [{"object_id": "house_1"}]},
                 "top2": {"disaster_scenario": "Yes",
                          "disaster_type": "fire", "disaster_level": 4,
                          "threats": []}}
    t = collect_triggers([], mu)
    assert any(x["type"] == "candidate_runoff" for x in t)
    a, _ = parse_assessment(GOOD_ANSWER)
    prompt = compose_reflection(a, POOL, t)
    assert "RUNOFF VERDICT" in prompt
    assert "PREFERRED: Yes · fire · level 7" in prompt
    assert "JUDGE'S REASON: A cites burning states." in prompt
    assert "advice, not an order" in prompt


def test_runoff_dead_judge_never_breaks_the_run():
    from agentic.assessment import run_assessment
    l7 = {"disaster_scenario": "Yes", "disaster_type": "fire",
          "disaster_level": 7, "confidence": 0.9,
          "threats": [{"object_id": "fire_1", "reason": "x"},
                      {"object_id": "tanker_truck_1", "reason": "y"},
                      {"object_id": "spill_1", "reason": "z"}]}
    # differ enough to push U above the runoff threshold
    l3 = dict(l7, disaster_level=3,
              threats=[{"object_id": "fire_1", "reason": "x"}])

    def dead(prompt):
        raise ConnectionError("no judge model")

    events = []
    out = run_assessment(TANKERISH, query_fn=scripted([l7, l7]),
                         n_probes=4, probe_fn=scripted([l7, l3, l3, l7]),
                         runoff_judge_fn=dead,
                         on_event=lambda e: events.append(e))
    assert out.assessment.disaster_level == 7        # run completed
    assert any(e["type"] == "assess_runoff_error" for e in events)


def test_probe_candidates_coarse_fallback_on_full_scatter():
    """Your run: five singleton candidates made 'top-2' a coin flip.
    Full scatter now regroups on scenario+bucket so the runoff compares
    genuinely opposed READINGS."""
    from agentic.uncertainty import probe_candidates

    def ans(level, bucket, tid):
        return {"disaster_scenario": "Yes", "severity_bucket": bucket,
                "disaster_level": level,
                "threats": [{"object_id": tid, "reason": "x"}],
                "at_risk": []}
    scattered = [ans(7, "catastrophic", "a"), ans(7, "catastrophic", "b"),
                 ans(8, "catastrophic", "c"), ans(4, "serious", "d"),
                 ans(5, "serious", "e")]
    cands = probe_candidates(scattered)
    assert [c["votes"] for c in cands] == [3, 2]      # coarse regroup
    assert cands[0]["answer"]["severity_bucket"] == "catastrophic"


# ── Weak-reason triggers (talking points feeding reflection) ────────────


def test_weak_reasons_trigger_and_compose_expectations():
    from agentic.reflection import weak_reason_triggers
    vague = {"disaster_scenario": "Yes", "disaster_type": "drowning",
             "disaster_level": 8, "confidence": 0.9,
             "threats": [{"object_id": "pool_1",
                          "reason": "poses a significant risk"}],
             "at_risk": [{"object_id": "child_1", "kind": "distress",
                          "reason": "the child is drowning"}]}
    a, _ = parse_assessment(vague)
    t = weak_reason_triggers(a, POOL)
    assert [(x["object_id"], x["list"]) for x in t] == [("pool_1", "threats")]
    # child_1's reason cites 'drowning' -> no trigger
    prompt = compose_reflection(a, POOL, t)
    assert "WEAK REASON" in prompt
    assert "'engulfing'" in prompt                # the declared state named
    assert "poses a significant risk" in prompt   # the offending text quoted
    assert "specific causal mechanism" in prompt  # the expectation
    assert "reconsider whether the entry belongs" in prompt


def test_weak_reasons_flow_through_collect_triggers():
    vague = {"disaster_scenario": "Yes", "disaster_type": "drowning",
             "disaster_level": 8, "confidence": 0.9,
             "threats": [{"object_id": "pool_1", "reason": "dangerous"}]}
    a, _ = parse_assessment(vague)
    t = collect_triggers([], None, assessment=a, record=POOL)
    assert any(x["type"] == "weak_reason" for x in t)
