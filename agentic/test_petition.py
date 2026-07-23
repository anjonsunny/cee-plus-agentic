"""Hermetic tests for increment 3 — contextual re-perception petitions.

Scripted models everywhere. Covered: GT-free trigger logic (fires on
B_pool's shape, silent on clean runs), the non-leading prompt contract
(context + reason + previous answer, NEVER what to look for), provenance
tagging, diffs, failure honesty, and the full orchestrator cascade.

Run:  pytest agentic/test_petition.py -q
"""
from __future__ import annotations

from agentic.perception import DetectedObject, PerceptionResult
from agentic.petition import (
    Petition,
    assess_with_petition,
    compose_petition,
    detect_petition,
    diff_records,
    mark_provenance,
)

# ── Fixtures: the B_pool shape (no hazard declared) ─────────────────────


def _obj(oid, label, state, kind, bbox=(0, 0, 10, 10)):
    return DetectedObject(
        object_id=oid, label=label, family="person", state=state,
        state_kind=kind, description="", bbox=list(bbox),
        box_source="dino_matched", box_confidence=0.9,
        anchor_bbox=list(bbox), mask_path=None, label_note="",
        vocab_extension=False, family_name_as_label=False)


def pool_record(with_pool=False):
    objs = [_obj("child_1", "child", "drowning", "at_risk"),
            _obj("child_2", "child", "trapped", "at_risk", (20, 0, 30, 10))]
    if with_pool:
        o = _obj("pool_1", "pool", "engulfing", "hazard_bearing",
                 (0, 0, 90, 60))
        o.family = "structure"
        objs.append(o)
    return PerceptionResult(
        image_path="/x/B.jpg", image_size=[100, 80],
        caption="A child struggles in a swimming pool.",
        entity_source="vlm", detected_objects=objs)


S5_SURVIVED = [{"kind": "threat_state_not_hazardous",
                "evidence": "child_2 listed as threat but its state "
                            "'trapped' is at_risk, not hazard_bearing"}]
STOOD_TRACE = {"stopped_reason": "cap_reached", "rounds": [],
               "u_before": 0.2, "u_after": None}


# ── Trigger logic ───────────────────────────────────────────────────────


def test_petition_fires_on_bpool_shape():
    p = detect_petition(STOOD_TRACE, S5_SURVIVED, pool_record())
    assert p is not None
    r = p.reasons[0]
    assert r["kind"] == "threat_state_not_hazardous"
    assert r["locates"] == "child_2"          # problem located in prev answer
    assert "may believe a source of harm exists" in r["implication"]


def test_no_petition_when_reflection_ended_clean():
    clean = {"stopped_reason": "clean", "rounds": []}
    assert detect_petition(clean, S5_SURVIVED, pool_record()) is None
    assert detect_petition(STOOD_TRACE, [], pool_record()) is None


def test_non_petitionable_violations_do_not_fire():
    v = [{"kind": "scenario_no_level_gt0", "evidence": "No but level=3"}]
    assert detect_petition(STOOD_TRACE, v, pool_record()) is None


def test_stood_caption_ticket_fires():
    rec = pool_record()
    rec.repair_trace = {"stopped_reason": "no_change", "rounds": [
        {"violations": [{"kind": "caption_entity_missing",
                         "raw_label": "swimming pool"}]}]}
    p = detect_petition({"stopped_reason": "clean"}, [], rec)
    assert p is not None
    assert p.reasons[0]["kind"] == "caption_entity_missing"
    assert "swimming pool" in p.reasons[0]["evidence"]


# ── The prompt contract: context + reason, NEVER what to look for ───────


def test_petition_prompt_is_context_only_and_locates_problems():
    p = detect_petition(STOOD_TRACE, S5_SURVIVED, pool_record())
    prompt = compose_petition(pool_record(), p)
    # previous answer, with the problem pointed at
    assert "YOUR PREVIOUS ANSWER" in prompt
    assert "child_2" in prompt
    assert "a downstream problem involves this entity" in prompt
    # the reason, quoted as evidence
    assert "WHY YOU ARE BEING ASKED TO LOOK AGAIN" in prompt
    assert "threat_state_not_hazardous" in prompt
    # standing is allowed; no invention
    assert "return it unchanged" in prompt
    assert "do not add anything you cannot see" in prompt
    # THE SUNNY RULE: no steering — the petition ADDENDUM must not name
    # what the downstream stage wishes existed. (The base prompt's state
    # vocabulary legitimately lists every state word; only the addendum
    # is petition-authored.)
    addendum = prompt.split("YOU HAVE ANALYZED")[1].lower()
    for leading in ("look for the pool", "look for water", "engulfing",
                    "hazard is missing", "add a hazard", "find the",
                    "pool", "water"):
        assert leading not in addendum, leading


# ── Provenance + diff ───────────────────────────────────────────────────


def test_mark_provenance_tags_only_new_entities():
    old, new = pool_record(), pool_record(with_pool=True)
    mark_provenance(old, new)
    tags = {o.object_id: o.provenance for o in new.detected_objects}
    assert tags["pool_1"] == "petition"
    assert tags["child_1"] == "original" and tags["child_2"] == "original"


def test_diff_records_by_label_state():
    d = diff_records(pool_record(), pool_record(with_pool=True))
    assert d["added"] == ["pool·engulfing"] and d["removed"] == []


# ── The orchestrator cascade ────────────────────────────────────────────

NO_HAZARD_ANSWER = {
    "disaster_scenario": "Yes", "disaster_type": "drowning",
    "disaster_level": 9, "confidence": 0.9,
    "threats": [{"object_id": "child_2", "reason": "trapped"}],
    "at_risk": [{"object_id": "child_1", "kind": "distress", "reason": "d"},
                {"object_id": "child_2", "kind": "distress", "reason": "t"}]}
WITH_POOL_ANSWER = {
    "disaster_scenario": "Yes", "disaster_type": "drowning",
    "disaster_level": 9, "confidence": 0.9,
    "threats": [{"object_id": "pool_1", "reason": "engulfing the child"}],
    "at_risk": [{"object_id": "child_1", "kind": "distress", "reason": "d"},
                {"object_id": "child_2", "kind": "distress", "reason": "t"}]}


def test_full_cascade_petition_resolves_the_pressure():
    """B_pool end-to-end: stubborn victim-as-threat -> STOOD -> petition
    -> merged pool entity -> re-assessment cites the pool -> pressure
    gone, outcome resolved."""
    calls = {"n": 0}

    def stubborn_then_good(prompt):
        calls["n"] += 1
        return NO_HAZARD_ANSWER if calls["n"] <= 3 else WITH_POOL_ANSWER

    def fake_perceive(image_path, caption, prompt):
        assert "WHY YOU ARE BEING ASKED TO LOOK AGAIN" in prompt
        return pool_record(with_pool=True)

    events = []
    rec, result, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(),
        on_event=events.append, perceive_fn=fake_perceive,
        query_fn=stubborn_then_good)
    assert petitioned
    assert any(o.object_id == "pool_1" and o.provenance == "petition"
               for o in rec.detected_objects)
    assert [t.object_id for t in result.assessment.threats] == ["pool_1"]
    assert not any(v["kind"] == "threat_state_not_hazardous"
                   for v in result.violations)
    kinds = [e["type"] for e in events]
    assert "petition_started" in kinds and "petition_done" in kinds
    outcome = next(e for e in events if e["type"] == "petition_outcome")
    assert outcome["resolved"] is True
    assert outcome["violations_before"] == ["threat_state_not_hazardous"]


def test_cascade_skips_petition_on_clean_runs():
    def good(prompt):
        return WITH_POOL_ANSWER

    rec, result, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(with_pool=True),
        perceive_fn=lambda *a: (_ for _ in ()).throw(AssertionError(
            "perceive_fn must not be called on a clean run")),
        query_fn=good)
    assert not petitioned


def test_failed_petition_keeps_original_record():
    def stubborn(prompt):
        return NO_HAZARD_ANSWER

    events = []
    rec, result, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(),
        on_event=events.append,
        perceive_fn=lambda *a: None,          # re-perception dies
        query_fn=stubborn)
    assert not petitioned
    assert [o.object_id for o in rec.detected_objects] == \
        ["child_1", "child_2"]                # original record intact
    assert any(e["type"] == "petition_failed" for e in events)


# ── The two-witness rule: petitioned entities need the detector ─────────


def _fallback_pool_record():
    rec = pool_record(with_pool=True)
    pool = next(o for o in rec.detected_objects if o.object_id == "pool_1")
    pool.box_source = "vlm_sam_fallback"      # only the claimant vouches
    return rec


def test_petitioned_entity_without_dino_is_rejected():
    """A hallucinated entity arrives with only the VLM's own box. The
    two-witness rule refuses it, and with nothing else changed the whole
    petition FAILS honestly — it cannot slip in as low-confidence."""
    events = []
    rec, result, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(),
        on_event=events.append,
        perceive_fn=lambda *a: _fallback_pool_record(),
        query_fn=lambda p: NO_HAZARD_ANSWER)
    assert not petitioned
    assert [o.object_id for o in rec.detected_objects] == \
        ["child_1", "child_2"]                        # nothing merged
    fail = next(e for e in events if e["type"] == "petition_failed")
    assert "survived detector grounding" in fail["error"]
    assert "pool·engulfing" in fail["error"]


def test_grounded_petition_still_merges():
    """dino_matched petitioned entities pass the two-witness rule."""
    calls = {"n": 0}

    def model(prompt):
        calls["n"] += 1
        return NO_HAZARD_ANSWER if calls["n"] <= 3 else WITH_POOL_ANSWER

    rec, result, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(),
        perceive_fn=lambda *a: pool_record(with_pool=True),  # dino_matched
        query_fn=model)
    assert petitioned
    assert any(o.object_id == "pool_1" for o in rec.detected_objects)


def test_model_standing_by_its_perception_is_not_a_failure():
    """The second look returns the same list: legitimate, recorded as
    petition_done with a note, no wasteful re-assessment."""
    events = []
    rec, result, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(),
        on_event=events.append,
        perceive_fn=lambda *a: pool_record(),         # unchanged
        query_fn=lambda p: NO_HAZARD_ANSWER)
    assert not petitioned
    done = next(e for e in events if e["type"] == "petition_done")
    assert done["note"] == "model stood by its original perception"
    assert not any(e["type"] == "petition_outcome" for e in events)


def test_petition_never_erases_originals():
    """B_pool live-run regression: the second look DROPPED child_2 (a
    drowning victim). The merge must preserve every original, record the
    omission as a DISPUTE, and still admit the grounded addition."""
    def second_look(*a):
        rec = pool_record(with_pool=True)
        # the re-perception omits child_2 entirely
        rec.detected_objects = [o for o in rec.detected_objects
                                if o.object_id != "child_2"]
        return rec

    calls = {"n": 0}

    def model(prompt):
        calls["n"] += 1
        return NO_HAZARD_ANSWER if calls["n"] <= 3 else WITH_POOL_ANSWER

    events = []
    rec, result, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(), on_event=events.append,
        perceive_fn=second_look, query_fn=model)
    assert petitioned
    ids = [o.object_id for o in rec.detected_objects]
    assert "child_2" in ids                      # victim PRESERVED
    assert any(o.label == "pool" for o in rec.detected_objects)  # added
    done = next(e for e in events if e["type"] == "petition_done")
    assert done["disputed"] == ["child·trapped"]
    assert done["removed"] == []                 # nothing erased, ever
    assert any("petition dispute" in n for n in rec.notes)


def test_petition_id_collision_renumbers():
    """A petitioned entity whose fresh-run id collides with an original
    gets renumbered instead of clobbering."""
    def second_look(*a):
        rec = pool_record(with_pool=True)
        pool = next(o for o in rec.detected_objects if o.label == "pool")
        pool.object_id = "child_1"               # id churn collision
        return rec

    calls = {"n": 0}

    def model(prompt):
        calls["n"] += 1
        return NO_HAZARD_ANSWER if calls["n"] <= 3 else WITH_POOL_ANSWER

    rec, _r, petitioned = assess_with_petition(
        "/x/B.jpg", pool_record(), perceive_fn=second_look, query_fn=model)
    assert petitioned
    pool = next(o for o in rec.detected_objects if o.label == "pool")
    assert pool.object_id != "child_1"           # renumbered
    assert len([o for o in rec.detected_objects
                if o.object_id == "child_1"]) == 1


# ── Petition routing: image look vs fresh re-ask (E_collapse) ───────────

from agentic.assessment import run_assessment  # noqa: E402
from agentic.petition import compose_stage2_petition, run_stage2_petition  # noqa: E402


def _record(objs, caption="A scene."):
    return PerceptionResult(
        image_path="/x/E.jpg", image_size=[100, 80], caption=caption,
        entity_source="vlm", detected_objects=objs)


def _trace_uncleaned():
    return {"stopped_reason": "cap_reached"}


def test_routing_no_hazard_goes_to_stage1():
    """B_pool shape: victim-as-threat with NO legal hazard declared —
    the entity list may be missing the source -> image look."""
    rec = _record([_obj("child_1", "child", "drowning", "at_risk")])
    p = detect_petition(_trace_uncleaned(),
                        [{"kind": "threat_state_not_hazardous",
                          "evidence": "child_1 listed as threat"}], rec)
    assert p is not None and p.target == "stage1"


def test_routing_with_hazard_goes_to_stage2():
    """E_collapse shape: hazards exist, states are right — the sorting
    is wrong -> re-ask the question, don't touch the image."""
    rec = _record([_obj("person_1", "person", "trapped", "at_risk"),
                   _obj("building_1", "building", "collapsed",
                        "hazard_bearing")])
    p = detect_petition(_trace_uncleaned(),
                        [{"kind": "threat_state_not_hazardous",
                          "evidence": "person_1 listed as threat but its "
                                      "state 'trapped' is at_risk"}], rec)
    assert p is not None and p.target == "stage2"


def test_routing_caption_ticket_always_stage1():
    """A standing caption ticket is an existence question — stage 1 wins
    even when hazards exist."""
    rec = _record([_obj("building_1", "building", "collapsed",
                        "hazard_bearing")])
    rec.repair_trace = {"stopped_reason": "no_change", "rounds": [
        {"round_number": 1, "changed": False, "violations": [
            {"kind": "caption_entity_missing", "raw_label": "spill"}]}]}
    p = detect_petition(_trace_uncleaned(),
                        [{"kind": "threat_state_not_hazardous",
                          "evidence": "x listed as threat"}], rec)
    assert p is not None and p.target == "stage1"


def test_stage2_petition_prompt_is_non_leading():
    from agentic.petition import Petition
    rec = _record([_obj("person_1", "person", "trapped", "at_risk"),
                   _obj("building_1", "building", "collapsed",
                        "hazard_bearing")])
    from agentic.assessment import parse_assessment
    prev, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_type": "collapse",
        "disaster_level": 9,
        "threats": [{"object_id": "person_1", "reason": "trapped"}],
        "at_risk": []})
    p = Petition(target="stage2", reasons=[
        {"kind": "threat_state_not_hazardous",
         "evidence": "person_1 listed as threat but its state 'trapped' "
                     "is at_risk", "implication": "x", "locates": "person_1"}])
    prompt = compose_stage2_petition(rec, prev, p)
    assert "YOUR PREVIOUS ANSWER" in prompt
    assert "person_1 listed as threat" in prompt          # problem quoted
    assert "return it unchanged" in prompt                # right to stand
    # never names the correct answer
    assert "move person_1" not in prompt.lower()


def test_stage2_petition_fresh_answer_resolves():
    """The fresh re-ask fixes the sorting -> resolved, record untouched."""
    rec = _record([_obj("person_1", "person", "trapped", "at_risk"),
                   _obj("building_1", "building", "collapsed",
                        "hazard_bearing")])
    events = []
    result = run_assessment(
        rec,
        query_fn=lambda p: {
            "disaster_scenario": "Yes", "disaster_type": "collapse",
            "disaster_level": 9, "confidence": 0.9,
            "threats": [
                {"object_id": "building_1", "reason": "collapsed structure"},
                {"object_id": "person_1", "reason": "trapped"}],
            "at_risk": [{"object_id": "person_1", "kind": "distress",
                         "reason": "state is trapped"}]},
        reflect=False)
    assert any(v["kind"] == "threat_state_not_hazardous"
               for v in result.violations)
    from agentic.petition import Petition
    pet = Petition(target="stage2", reasons=[
        {"kind": "threat_state_not_hazardous",
         "evidence": "person_1 listed as threat", "implication": "x",
         "locates": "person_1"}])
    fixed = {"disaster_scenario": "Yes", "disaster_type": "collapse",
             "disaster_level": 9, "confidence": 0.9,
             "threats": [{"object_id": "building_1",
                          "reason": "collapsed structure"}],
             "at_risk": [{"object_id": "person_1", "kind": "distress",
                          "reason": "state is trapped"}]}
    out = run_stage2_petition(rec, result, pet,
                              on_event=events.append,
                              query_fn=lambda p: fixed)
    assert not [v for v in out.violations
                if v["kind"] == "threat_state_not_hazardous"]
    started = next(e for e in events if e["type"] == "petition_started")
    assert started["target"] == "stage2"
    oc = next(e for e in events if e["type"] == "petition_outcome")
    assert oc["resolved"] is True
    done = next(e for e in events if e["type"] == "petition_done")
    assert "CHANGED" in done["note"]
    # the shared record was never touched
    assert [o.object_id for o in rec.detected_objects] == \
        ["person_1", "building_1"]


def test_stage2_petition_stood_and_garbage_answer():
    rec = _record([_obj("person_1", "person", "trapped", "at_risk"),
                   _obj("building_1", "building", "collapsed",
                        "hazard_bearing")])
    bad = {"disaster_scenario": "Yes", "disaster_type": "collapse",
           "disaster_level": 9, "confidence": 0.9,
           "threats": [{"object_id": "building_1", "reason": "collapsed"},
                       {"object_id": "person_1", "reason": "trapped"}],
           "at_risk": []}
    result = run_assessment(rec, query_fn=lambda p: bad, reflect=False)
    from agentic.petition import Petition
    pet = Petition(target="stage2", reasons=[
        {"kind": "threat_state_not_hazardous", "evidence": "person_1",
         "implication": "x", "locates": "person_1"}])
    events = []
    # model stands (same answer back)
    out = run_stage2_petition(rec, result, pet, on_event=events.append,
                              query_fn=lambda p: bad)
    done = next(e for e in events if e["type"] == "petition_done")
    assert "stood" in done["note"]
    oc = next(e for e in events if e["type"] == "petition_outcome")
    assert oc["resolved"] is False
    # garbage answer -> parse coerces, never crashes (Rule 1a)
    out2 = run_stage2_petition(rec, result, pet, on_event=events.append,
                               query_fn=lambda p: "total garbage")
    assert out2 is not None
