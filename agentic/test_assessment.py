"""Hermetic tests for Stage 2 scene assessment (increment 1).

No model, no network: query_fn is scripted. Per the standing rules, every
boundary that consumes raw model output is fed malformed shapes (level
"high", level 47, missing keys, non-dict answers).

Run:  pytest agentic/test_assessment.py -q
"""
from __future__ import annotations

import pytest

from agentic.assessment import (
    AssessmentResult,
    build_assessment_prompt,
    internal_check,
    parse_assessment,
    run_assessment,
    severity_bucket,
)
from agentic.perception import DetectedObject, PerceptionResult

# ── Fixture: a mini C_tanker_fire-shaped record ─────────────────────────


def _obj(oid: str, label: str, state: str, kind: str) -> DetectedObject:
    return DetectedObject(
        object_id=oid, label=label, family="vehicle", state=state,
        state_kind=kind, description="", bbox=[0, 0, 10, 10],
        box_source="dino_matched", box_confidence=0.9,
        anchor_bbox=[0, 0, 10, 10], mask_path=None, label_note="",
        vocab_extension=False, family_name_as_label=False)


def record(objs, caption="A tanker truck leaks fuel."):
    return PerceptionResult(
        image_path="/x/C.jpg", image_size=[100, 100], caption=caption,
        entity_source="vlm", detected_objects=objs)


TANKER = record([
    _obj("tanker_truck_1", "tanker_truck", "leaking", "hazard_bearing"),
    _obj("spill_1", "spill", "seeping", "hazard_bearing"),
    _obj("fire_1", "fire", "spreading", "hazard_bearing"),
    _obj("person_1", "person", "standing", "normal"),
])
CONTROL = record([_obj("car_1", "car", "stationary", "normal")],
                 caption="A quiet street.")


# ── Buckets ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("level,bucket", [
    (0, "none"), (1, "minor"), (3, "minor"), (4, "serious"),
    (6, "serious"), (7, "catastrophic"), (10, "catastrophic")])
def test_bucket_boundaries(level, bucket):
    assert severity_bucket(level) == bucket


# ── Parsing: the malformed-model-output gauntlet (Rule 1a) ──────────────


def test_parse_clean_answer():
    a, notes = parse_assessment({"disaster_scenario": "Yes",
                                 "disaster_type": "hazmat fire",
                                 "disaster_level": 7, "reasoning": "fuel+fire",
                                 "confidence": 0.9})
    assert (a.disaster_scenario, a.disaster_level, a.severity_bucket) == \
        ("Yes", 7, "catastrophic")
    assert notes == []


@pytest.mark.parametrize("raw,scenario", [
    ({"disaster_scenario": "YES"}, "Yes"),
    ({"disaster_scenario": "true"}, "Yes"),
    ({"disaster_scenario": "no"}, "No"),
    ({"disaster_scenario": "N/A"}, "No"),
])
def test_parse_scenario_normalization(raw, scenario):
    a, _ = parse_assessment(raw)
    assert a.disaster_scenario == scenario


def test_parse_garbage_is_coerced_and_logged():
    a, notes = parse_assessment({"disaster_scenario": "maybe?",
                                 "disaster_level": "high"})
    assert a.disaster_scenario == "No" and a.disaster_level == 0
    assert any("scenario_unparseable" in n for n in notes)
    assert any("level_unparseable" in n for n in notes)


def test_parse_out_of_range_level_clamped_with_note():
    a, notes = parse_assessment({"disaster_scenario": "Yes",
                                 "disaster_level": 47})
    assert a.disaster_level == 10 and a.severity_bucket == "catastrophic"
    assert any("level_clamped" in n for n in notes)
    b, notes2 = parse_assessment({"disaster_scenario": "No",
                                  "disaster_level": -3})
    assert b.disaster_level == 0 and any("level_clamped" in n for n in notes2)


@pytest.mark.parametrize("raw", [None, "yes", 7, ["Yes"]])
def test_parse_non_dict_answers(raw):
    a, notes = parse_assessment(raw)
    assert a.disaster_scenario == "No" and a.disaster_level == 0
    assert any("answer_not_object" in n for n in notes)


def test_parse_missing_type_defaults_na():
    a, _ = parse_assessment({"disaster_scenario": "Yes", "disaster_level": 5})
    assert a.disaster_type == "N/A"


# ── Internal check ──────────────────────────────────────────────────────


def test_internal_check_fires_both_directions():
    no_but_level, _ = parse_assessment({"disaster_scenario": "No",
                                        "disaster_level": 3})
    kinds = [v["kind"] for v in internal_check(no_but_level)]
    assert kinds == ["scenario_no_level_gt0"]

    yes_but_zero, _ = parse_assessment({"disaster_scenario": "Yes",
                                        "disaster_level": 0})
    kinds = [v["kind"] for v in internal_check(yes_but_zero)]
    assert kinds == ["scenario_yes_level_0"]


def test_internal_check_clean_cases():
    for raw in ({"disaster_scenario": "Yes", "disaster_level": 6},
                {"disaster_scenario": "No", "disaster_level": 0}):
        a, _ = parse_assessment(raw)
        assert internal_check(a) == []


# ── The prompt: text-only, state-first ──────────────────────────────────


def test_prompt_carries_states_and_no_image():
    p = build_assessment_prompt(TANKER)
    assert "tanker_truck_1" in p and "leaking" in p and "hazard_bearing" in p
    assert "A tanker truck leaks fuel." in p
    assert "do NOT get the image" in p.replace("You do\nNOT", "You do NOT")
    assert "image_url" not in p and "base64" not in p


def test_prompt_empty_scene():
    p = build_assessment_prompt(record([], caption=""))
    assert "(no entities perceived)" in p and "(none)" in p


# ── The node ────────────────────────────────────────────────────────────


def test_run_assessment_happy_path_and_events():
    events: list[dict] = []
    out = run_assessment(
        TANKER,
        query_fn=lambda prompt: {"disaster_scenario": "Yes",
                                 "disaster_type": "hazmat fire",
                                 "disaster_level": 7, "confidence": 0.9,
                                 "reasoning": "leaking + spreading fire",
                                 "threats": [
                                     {"object_id": "tanker_truck_1",
                                      "reason": "leaking"},
                                     {"object_id": "spill_1",
                                      "reason": "seeping"},
                                     {"object_id": "fire_1",
                                      "reason": "spreading"}],
                                 "at_risk": [
                                     {"object_id": "person_1",
                                      "kind": "proximity",
                                      "reason": "standing near the spill"}]},
        on_event=events.append)
    assert isinstance(out, AssessmentResult)
    assert out.assessment.severity_bucket == "catastrophic"
    assert out.violations == []
    kinds = [e["type"] for e in events]
    assert kinds == ["stage_started", "assess_context", "assess_verdict",
                     "reflect_stopped", "stage_done"]
    stop = next(e for e in events if e["type"] == "reflect_stopped")
    assert stop["reason"] == "clean" and stop["rounds"] == 0
    ctx = events[1]
    assert set(ctx["hazard_ids"]) == {"tanker_truck_1", "spill_1", "fire_1"}
    assert events[2]["bucket"] == "catastrophic"


def test_run_assessment_records_violation_without_fixing():
    """Increment 1 DETECTS and RECORDS; it must not silently repair —
    corrections belong to the visible loop (increment 2)."""
    events: list[dict] = []
    out = run_assessment(
        TANKER,
        query_fn=lambda p: {"disaster_scenario": "No", "disaster_level": 4},
        on_event=events.append, reflect=False)
    assert out.assessment.disaster_scenario == "No"       # NOT patched
    assert out.assessment.disaster_level == 4             # NOT patched
    assert [v["kind"] for v in out.violations] == ["scenario_no_level_gt0"]
    assert any(e["type"] == "assess_violation" for e in events)


def test_run_assessment_survives_malformed_model_answer():
    out = run_assessment(CONTROL, query_fn=lambda p: "total garbage")
    assert out.assessment.disaster_scenario == "No"
    assert out.raw_answer is None
    assert any("answer_not_object" in n for n in out.parse_notes)


def test_run_assessment_parse_notes_become_events():
    events: list[dict] = []
    run_assessment(CONTROL,
                   query_fn=lambda p: {"disaster_scenario": "Yes",
                                       "disaster_level": "high"},
                   on_event=events.append)
    notes = [e for e in events if e["type"] == "assess_parse_note"]
    assert notes and "level_unparseable" in notes[0]["note"]


# ── Channel 1: self-reported confidence (subject data) ──────────────────


def test_self_confidence_parsed_and_clamped():
    a, _ = parse_assessment({"disaster_scenario": "Yes", "disaster_level": 5,
                             "confidence": 0.85})
    assert a.self_confidence == 0.85
    b, notes = parse_assessment({"disaster_scenario": "Yes",
                                 "disaster_level": 5, "confidence": 3})
    assert b.self_confidence == 1.0
    assert any("confidence_clamped" in n for n in notes)


def test_self_confidence_missing_or_garbage_noted():
    a, notes = parse_assessment({"disaster_scenario": "No",
                                 "disaster_level": 0})
    assert a.self_confidence is None
    assert any("no_self_confidence_reported" in n for n in notes)
    b, notes2 = parse_assessment({"disaster_scenario": "No",
                                  "disaster_level": 0, "confidence": "very"})
    assert b.self_confidence is None
    assert any("confidence_unparseable" in n for n in notes2)


def test_prompt_asks_for_confidence():
    assert '"confidence"' in build_assessment_prompt(TANKER)


# ── Channel 2: probes wired into the node ───────────────────────────────


def _probe_script(answers):
    state = {"i": 0}

    def fn(prompt):
        a = answers[min(state["i"], len(answers) - 1)]
        state["i"] += 1
        return a

    return fn


def test_probes_measure_dispersion_and_emit_events():
    events: list[dict] = []
    out = run_assessment(
        TANKER,
        query_fn=lambda p: {"disaster_scenario": "Yes",
                            "disaster_type": "hazmat fire",
                            "disaster_level": 7, "confidence": 0.9},
        n_probes=3,
        probe_fn=_probe_script([
            {"disaster_scenario": "Yes", "disaster_type": "hazmat fire",
             "disaster_level": 7},
            {"disaster_scenario": "Yes", "disaster_type": "hazmat fire",
             "disaster_level": 8},
            {"disaster_scenario": "Yes", "disaster_type": "fuel fire",
             "disaster_level": 7}]),
        on_event=events.append)
    mu = out.measured_uncertainty
    assert mu is not None and mu.n_probes == 3
    assert mu.scenario_agreement == 1.0
    # 'hazmat fire' and 'fuel fire' fold to the same family: wording
    # wobble is no longer measured disagreement (six-scene calibration)
    assert mu.type_agreement == 1.0
    assert not any(d.kind == "type_split" for d in mu.drivers)
    kinds = [e["type"] for e in events]
    assert kinds.count("assess_probe") == 3
    assert "assess_uncertainty" in kinds
    unc = next(e for e in events if e["type"] == "assess_uncertainty")
    assert unc["explanation"] and unc["phase"] == "initial"


def test_no_probes_means_no_uncertainty_block():
    out = run_assessment(CONTROL, query_fn=lambda p: {
        "disaster_scenario": "No", "disaster_level": 0})
    assert out.measured_uncertainty is None


def test_probe_garbage_counts_as_instability_not_crash():
    """A probe returning garbage parses to the conservative default —
    that instability is measured, not discarded (Rule 1a)."""
    out = run_assessment(
        TANKER,
        query_fn=lambda p: {"disaster_scenario": "Yes",
                            "disaster_type": "hazmat fire",
                            "disaster_level": 7},
        n_probes=2,
        probe_fn=_probe_script([
            {"disaster_scenario": "Yes", "disaster_type": "hazmat fire",
             "disaster_level": 7},
            "utter garbage"]))
    mu = out.measured_uncertainty
    assert mu.n_probes == 2 and mu.scenario_agreement == 0.5
    assert any(d.kind == "scenario_flip" for d in mu.drivers)


def test_all_probes_failing_reads_as_unmeasured_not_unanimous():
    def dead(prompt):
        raise ConnectionError("ollama down")
    events: list[dict] = []
    out = run_assessment(
        TANKER, query_fn=lambda p: {"disaster_scenario": "Yes",
                                    "disaster_level": 7},
        n_probes=3, probe_fn=dead, on_event=events.append)
    mu = out.measured_uncertainty
    assert mu.score == 1.0
    assert any(d.kind == "probes_failed" for d in mu.drivers)
    assert sum(1 for e in events if e["type"] == "assess_probe_error") == 3


def test_explainer_receives_drivers_and_result_records_llm():
    out = run_assessment(
        TANKER,
        query_fn=lambda p: {"disaster_scenario": "Yes",
                            "disaster_type": "hazmat fire",
                            "disaster_level": 7},
        n_probes=2,
        probe_fn=_probe_script([
            {"disaster_scenario": "Yes", "disaster_type": "hazmat fire",
             "disaster_level": 6},
            {"disaster_scenario": "Yes", "disaster_type": "hazmat fire",
             "disaster_level": 7}]),
        explain_fn=lambda prompt: "Severity straddles the serious/"
                                  "catastrophic boundary; use the bucket "
                                  "with caution at Stage 12.")
    mu = out.measured_uncertainty
    assert mu.explainer == "llm" and "Stage 12" in mu.explanation


# ── The merged stage (Stage 2+3): threats + at_risk in one verdict ──────


def test_parse_merged_lists_clean():
    a, notes = parse_assessment({
        "disaster_scenario": "Yes", "disaster_type": "house fire",
        "disaster_level": 7, "confidence": 0.9,
        "threats": [{"object_id": "house_1", "reason": "burning"}],
        "at_risk": [{"object_id": "person_1", "kind": "proximity",
                     "reason": "adjacent to burning house"},
                    {"object_id": "child_1", "kind": "distress",
                     "reason": "state is drowning"}]})
    assert [t.object_id for t in a.threats] == ["house_1"]
    assert [(r.object_id, r.kind) for r in a.at_risk] == \
        [("person_1", "proximity"), ("child_1", "distress")]
    assert notes == []


def test_parse_merged_lists_malformed_entries():
    """Rule 1a gauntlet: bare strings kept with a note, garbage dropped
    with a note, bogus kind coerced with a note."""
    a, notes = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 5, "confidence": 1,
        "threats": ["house_1", 42, {"reason": "no id"}],
        "at_risk": [{"object_id": "p_1", "kind": "endangered"}]})
    assert [t.object_id for t in a.threats] == ["house_1"]
    assert a.at_risk[0].kind == "proximity"
    assert any("threat_0_bare_string" in n for n in notes)
    assert any("threat_1_malformed" in n for n in notes)
    assert any("threat_2_malformed" in n for n in notes)
    assert any("at_risk_0_kind_coerced" in n for n in notes)


def test_check_id_must_resolve_to_perception():
    a, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 7, "confidence": 1,
        "threats": [{"object_id": "ghost_9", "reason": "?"}]})
    kinds = [v["kind"] for v in internal_check(a, TANKER)]
    assert "id_not_in_perception" in kinds


def test_check_threat_needs_hazardous_state():
    a, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 7, "confidence": 1,
        "threats": [{"object_id": "person_1", "reason": "menacing"}]})
    v = internal_check(a, TANKER)
    kinds = [x["kind"] for x in v]
    assert "threat_state_not_hazardous" in kinds
    # the three declared hazards missing from threats are also flagged
    assert kinds.count("hazard_not_in_threats") == 3
    tv = next(x for x in v if x["kind"] == "threat_state_not_hazardous")
    assert "standing" in tv["evidence"]


def test_enforce_kinds_schema_strict_both_directions():
    """G4 rebuilt after C_tanker: the STATE alone decides the kind
    (baseline main.py:26). Both directions violate; the record carries
    the derived kind either way."""
    from agentic.assessment import enforce_kinds
    rec = record([
        _obj("fire_1", "fire", "spreading", "hazard_bearing"),
        _obj("child_1", "child", "drowning", "at_risk"),
        _obj("person_1", "person", "standing", "normal")])
    a, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 8, "confidence": 1,
        "at_risk": [
            {"object_id": "child_1", "kind": "proximity"},   # downgrade
            {"object_id": "person_1", "kind": "distress"}]}) # promote
    v = enforce_kinds(a, rec)
    assert [x["kind"] for x in v] == ["at_risk_kind_mismatch"] * 2
    # the RECORD carries the derived truth; the claim lives in evidence
    assert a.at_risk[0].kind == "distress"
    assert a.at_risk[1].kind == "proximity"
    assert "state alone decides" in v[0]["evidence"]


def test_check_proximity_needs_a_hazard_g2():
    a, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 3, "confidence": 1,
        "at_risk": [{"object_id": "car_1", "kind": "proximity"}]})
    kinds = [v["kind"] for v in internal_check(a, CONTROL)]
    assert "proximity_without_hazard" in kinds


def test_prompt_carries_spatial_hints_and_merged_schema():
    rec = record([
        _obj("house_1", "house", "burning", "hazard_bearing"),
        _obj("person_1", "person", "standing", "normal")])
    rec.image_size = [1000, 800]      # small fixture diag would veto adjacency
    rec.detected_objects[0].bbox = [100, 100, 500, 500]
    rec.detected_objects[1].bbox = [510, 200, 560, 400]
    p = build_assessment_prompt(rec)
    assert "Spatial hints" in p and "person_1" in p and "adjacent" in p
    assert '"threats"' in p and '"at_risk"' in p
    assert "distress" in p and "proximity" in p


def test_granular_uncertainty_pinpoints_flickering_entity():
    def probe(at_risk_ids):
        return {"disaster_scenario": "Yes", "disaster_type": "house fire",
                "disaster_level": 7, "confidence": 0.9,
                "threats": [{"object_id": "house_1", "reason": "burning"}],
                "at_risk": [{"object_id": i, "kind": "proximity"}
                            for i in at_risk_ids]}
    answers = [probe(["person_1"]), probe(["person_1"]),
               probe(["person_1", "dog_1"]), probe([]), probe(["person_1"])]
    out = run_assessment(
        TANKER, query_fn=lambda p: probe(["person_1"]),
        n_probes=5, probe_fn=_probe_script(answers))
    g = out.measured_uncertainty.granular
    assert g["threats"]["house_1"]["u"] == 0.0        # rock solid
    assert g["at_risk"]["person_1"]["votes"] == "4/5"  # pinpointed
    assert g["at_risk"]["dog_1"]["u"] == 0.8           # very unstable
    assert g["fields"]["disaster_scenario"]["u"] == 0.0
    kinds = [d.kind for d in out.measured_uncertainty.drivers]
    assert "at_risk_membership_split" in kinds
    split = next(d for d in out.measured_uncertainty.drivers
                 if d.kind == "at_risk_membership_split"
                 and "dog_1" in d.evidence)
    assert "1/5" in split.evidence and "G1-G4" in split.action


# ── The B_pool-batch checks (S2/S3 in code, emptiness, coverage) ────────


def test_s2_no_verdict_with_danger_states_fires():
    """The push_06 / B_pool-capitulation catch, now in code."""
    rec = record([_obj("child_1", "child", "drowning", "at_risk"),
                  _obj("pool_1", "pool", "engulfing", "hazard_bearing")])
    a, _ = parse_assessment({"disaster_scenario": "No", "disaster_level": 0})
    kinds = [v["kind"] for v in internal_check(a, rec)]
    assert "missed_disaster_incoherence" in kinds
    ev = next(v for v in internal_check(a, rec)
              if v["kind"] == "missed_disaster_incoherence")
    assert "child_1·drowning" in ev["evidence"]


def test_s3_yes_verdict_without_support_fires():
    a, _ = parse_assessment({"disaster_scenario": "Yes",
                             "disaster_level": 4, "disaster_type": "fire"})
    kinds = [v["kind"] for v in internal_check(a, CONTROL)]
    assert "false_alarm_incoherence" in kinds


def test_no_verdict_with_populated_lists_fires():
    """B_pool's exact final shape: No · 0 with two at-risk children."""
    rec = record([_obj("child_1", "child", "drowning", "at_risk")])
    a, _ = parse_assessment({
        "disaster_scenario": "No", "disaster_level": 0,
        "at_risk": [{"object_id": "child_1", "kind": "distress"}]})
    kinds = [v["kind"] for v in internal_check(a, rec)]
    assert "scenario_no_with_entities" in kinds
    assert "missed_disaster_incoherence" in kinds


def test_hazard_coverage_flags_forgotten_hazard():
    """C_tanker's spill_1: declared hazard absent from threats."""
    a, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 7, "confidence": 1,
        "threats": [{"object_id": "tanker_truck_1", "reason": "leak"},
                    {"object_id": "fire_1", "reason": "fire"}]})
    v = [x for x in internal_check(a, TANKER)
         if x["kind"] == "hazard_not_in_threats"]
    assert len(v) == 1 and "spill_1" in v[0]["evidence"]
    # No-verdict scenes don't get coverage flags (nothing should be listed)
    b, _ = parse_assessment({"disaster_scenario": "No", "disaster_level": 0})
    assert not [x for x in internal_check(b, CONTROL)
                if x["kind"] == "hazard_not_in_threats"]


def test_s7_hazard_cannot_be_its_own_victim():
    """Live-run find (A_fire): house_1·burning ended in threats AND
    at_risk and nothing objected. S7 objects."""
    a, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 7, "confidence": 1,
        "threats": [{"object_id": "fire_1", "reason": "spreading"},
                    {"object_id": "tanker_truck_1", "reason": "leaking"},
                    {"object_id": "spill_1", "reason": "seeping"}],
        "at_risk": [{"object_id": "fire_1", "kind": "proximity",
                     "reason": "near itself, somehow"}]})
    kinds = [v["kind"] for v in internal_check(a, TANKER)]
    assert "hazard_as_at_risk" in kinds


def test_burning_person_is_dual_role_no_violation():
    """Sunny: a burning person is a threat AND at risk. Living beings in
    hazard states are automatic dual-role: both lists, kind distress,
    S7 silent. A burning HOUSE double-listed still violates."""
    from agentic.assessment import enforce_kinds

    def _living(oid, label, family, state, kind):
        o = _obj(oid, label, state, kind)
        o.family = family
        return o

    rec = record([
        _living("person_1", "person", "person", "burning", "hazard_bearing"),
        _living("house_1", "house", "structure", "burning", "hazard_bearing")])
    a, _ = parse_assessment({
        "disaster_scenario": "Yes", "disaster_level": 8, "confidence": 1,
        "threats": [{"object_id": "person_1", "reason": "burning, may ignite"},
                    {"object_id": "house_1", "reason": "burning"}],
        "at_risk": [{"object_id": "person_1", "kind": "proximity",
                     "reason": "on fire"},
                    {"object_id": "house_1", "kind": "proximity",
                     "reason": "at risk of the fire"}]})
    v = enforce_kinds(a, rec) + internal_check(a, rec)
    kinds = [x["kind"] for x in v]
    # burning person: kind derived to DISTRESS (mismatch recorded once,
    # record corrected), and NO S7
    assert a.at_risk[0].kind == "distress"
    s7 = [x for x in v if x["kind"] == "hazard_as_at_risk"]
    assert len(s7) == 1 and "house_1" in s7[0]["evidence"]  # house only


# ── Medium-bound derivation at the assessment boundary ──────────────────


def test_assessment_derives_pool_hazard_before_judging():
    """F7's root cause closed: a frozen/petitioned record with
    child·drowning + pool·normal enters run_assessment -> the pool is
    derived hazardous BEFORE the prompt is built, so the assessor finally
    has a legal hazard and geometry/S6 see it too."""
    events: list[dict] = []
    seen_prompt = {}

    def scripted(prompt):
        seen_prompt["p"] = prompt
        return {"disaster_scenario": "Yes", "disaster_type": "drowning",
                "disaster_level": 7, "confidence": 0.9,
                "reasoning": "child_1 is drowning in pool_1",
                "threats": [{"object_id": "pool_1",
                             "reason": "the pool is engulfing child_1"}],
                "at_risk": [{"object_id": "child_1", "kind": "distress",
                             "reason": "state is drowning"}]}

    rec = record([_obj("child_1", "child", "drowning", "at_risk"),
                  _obj("pool_1", "pool", "normal", "normal")],
                 caption="A child struggles in a swimming pool.")
    out = run_assessment(rec, query_fn=scripted, on_event=events.append,
                         reflect=False)
    derived = [e for e in events if e["type"] == "hazard_derived"]
    assert derived == [{"type": "hazard_derived", "medium": "pool_1",
                        "was": "normal", "now": "engulfing",
                        "victim": "child_1", "victim_state": "drowning"}]
    # The record the whole stage reads was updated in place...
    pool = rec.detected_objects[1]
    assert pool.state == "engulfing" and pool.state_kind == "hazard_bearing"
    # ...the prompt saw the derived state, not the model's 'normal'...
    assert "engulfing" in seen_prompt["p"]
    # ...the context event counts the pool as a hazard...
    ctx = next(e for e in events if e["type"] == "assess_context")
    assert "pool_1" in ctx["hazard_ids"]
    # ...and pool-as-threat is now LEGAL: no S5 fires for it.
    assert not [v for v in out.violations
                if v["kind"] == "threat_state_not_hazardous"]


def test_assessment_no_derivation_without_medium():
    """B_pool's actual frozen record (no pool perceived at all): nothing
    is invented, empty threats stay legal."""
    events: list[dict] = []
    rec = record([_obj("child_1", "child", "drowning", "at_risk")],
                 caption="A child struggles in the water.")
    run_assessment(
        rec,
        query_fn=lambda p: {"disaster_scenario": "Yes",
                            "disaster_type": "drowning",
                            "disaster_level": 7, "confidence": 0.9,
                            "threats": [],
                            "at_risk": [{"object_id": "child_1",
                                         "kind": "distress",
                                         "reason": "state is drowning"}]},
        on_event=events.append, reflect=False)
    assert not [e for e in events if e["type"] == "hazard_derived"]


# ── S8: victim-shaped reasons in the threat slot ────────────────────────

from agentic.assessment import victim_shaped  # noqa: E402


def test_victim_shaped_detector_direction_sensitive():
    # The A_fire road case + F5's car_1: RECEIVING harm -> True
    assert victim_shaped("Road is in proximity to an intact house and "
                         "could be at risk if the fire spreads")
    assert victim_shaped("the car is in danger from the flames")
    assert victim_shaped("it appears to be vulnerable to the spill")
    # CAUSING harm -> False (a real threat reason must never match)
    assert not victim_shaped("puts person_1 at risk of burns")
    assert not victim_shaped("poses a danger to everyone nearby")
    assert not victim_shaped("the fire is spreading toward the house")
    assert not victim_shaped("places car_1 in harm's way")
    # Rule 1a: garbage never crashes
    assert not victim_shaped(None) and not victim_shaped(42)


def test_s8_fires_on_victim_shaped_threat_reason():
    """The exact ui_34d8177e shape: road_1·burning (coerced) promoted to
    threats with a victim-shaped reason -> S8 violation recorded."""
    rec = record([_obj("house_1", "house", "burning", "hazard_bearing"),
                  _obj("road_1", "road", "burning", "hazard_bearing")])
    out = run_assessment(
        rec,
        query_fn=lambda p: {
            "disaster_scenario": "Yes", "disaster_type": "Fire",
            "disaster_level": 7, "confidence": 0.9,
            "threats": [
                {"object_id": "house_1",
                 "reason": "burning — direct flame contact harms anyone near"},
                {"object_id": "road_1",
                 "reason": "could be at risk if the fire spreads"}],
            "at_risk": []},
        reflect=False)
    kinds = [v["kind"] for v in out.violations]
    assert "threat_reason_victim_shaped" in kinds
    ev = next(v["evidence"] for v in out.violations
              if v["kind"] == "threat_reason_victim_shaped")
    assert "road_1" in ev and "RECEIVING" in ev
    # the honest threat is untouched
    assert kinds.count("threat_reason_victim_shaped") == 1


def test_s8_silent_on_causal_reasons():
    rec = record([_obj("house_1", "house", "burning", "hazard_bearing")])
    out = run_assessment(
        rec,
        query_fn=lambda p: {
            "disaster_scenario": "Yes", "disaster_type": "Fire",
            "disaster_level": 6, "confidence": 0.9,
            "threats": [{"object_id": "house_1",
                         "reason": "burning house puts person_1 at risk"}],
            "at_risk": []},
        reflect=False)
    assert not [v for v in out.violations
                if v["kind"] == "threat_reason_victim_shaped"]


def test_victim_shaped_catches_in_distress():
    """E_collapse ui_e45e9956: 'appears to be in distress due to the
    building's condition' sat in a threat slot and slipped the old
    phrase set."""
    assert victim_shaped("The person is trapped at an upper window and "
                         "appears to be in distress due to the "
                         "building's condition")
    # causing-direction still never matches
    assert not victim_shaped("the collapse puts person_1 in distress")
