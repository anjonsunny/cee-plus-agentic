"""Hermetic tests for the eval battery (GT harness, quadrant, citation
counts, blind pairwise judge, reason rubric). Scripted judges, tmp GT.

Run:  pytest agentic/test_evals.py -q
"""
from __future__ import annotations

import json

from agentic.evals import (
    citation_counts,
    eval_stage1,
    eval_stage2,
    evaluate_scene,
    judge_pairwise,
    load_gt,
    quadrant,
    rubric_reasons,
)

# ── Fixtures: a mini B_pool world ───────────────────────────────────────

PERCEPTION = {
    "image_path": "/x/B.jpg", "image_size": [100, 100], "caption": "cap",
    "entity_source": "vlm",
    "detected_objects": [
        {"object_id": "child_1", "label": "child", "family": "person",
         "state": "drowning", "state_kind": "at_risk", "description": "",
         "bbox": [0, 0, 10, 10], "box_source": "dino_matched",
         "box_confidence": 0.9, "anchor_bbox": [0, 0, 10, 10],
         "mask_path": None, "label_note": "", "vocab_extension": False,
         "family_name_as_label": False},
        {"object_id": "child_2", "label": "child", "family": "person",
         "state": "trapped", "state_kind": "at_risk", "description": "",
         "bbox": [20, 0, 30, 10], "box_source": "dino_matched",
         "box_confidence": 0.9, "anchor_bbox": [20, 0, 30, 10],
         "mask_path": None, "label_note": "", "vocab_extension": False,
         "family_name_as_label": False}],
    "unlocalized": [], "notes": [], "repair_trace": None,
}

GT1 = {"required": [
    {"label": "child", "state": "drowning", "state_kind": "at_risk"},
    {"label": "child", "state_kind": "at_risk"},
    {"label": "pool", "state_kind": "hazard_bearing"}],
    "optional": [], "expect_hazard": True}

GT2 = {"disaster_scenario": "Yes", "type_families": ["water"],
       "bucket": "catastrophic", "bucket_acceptable": ["catastrophic"],
       "threats": [], "threats_optional": [],
       "at_risk": {"child_1": "distress", "child_2": "distress"},
       "at_risk_optional": {}}

GOOD = {"disaster_scenario": "Yes", "disaster_type": "drowning",
        "disaster_level": 9, "severity_bucket": "catastrophic",
        "reasoning": "child_1 drowning; child_2 trapped",
        "threats": [],
        "at_risk": [{"object_id": "child_1", "kind": "distress",
                     "reason": "the child is drowning in the pool"},
                    {"object_id": "child_2", "kind": "distress",
                     "reason": "trapped under the pool cover"}]}

CAPITULATED = {"disaster_scenario": "No", "disaster_type": "N/A",
               "disaster_level": 0, "severity_bucket": "none",
               "reasoning": "", "threats": [],
               "at_risk": [{"object_id": "child_1", "kind": "distress",
                            "reason": "drowning"}]}


# ── Stage 1 ─────────────────────────────────────────────────────────────


def test_stage1_finds_the_pool_miss():
    """B_pool's known perception gap: the water hazard was never
    declared. This miss IS the re-perception charge sheet."""
    r = eval_stage1(PERCEPTION, GT1)
    assert r["required_recall"] == round(2 / 3, 3)
    assert r["missed_entities"] == [{"label": "pool",
                                     "state_kind": "hazard_bearing"}]
    assert r["missing_hazard"] is True


def test_stage1_duplicate_labels_matched_once():
    """Two GT 'child' entries must consume two DISTINCT objects."""
    one_child = dict(PERCEPTION,
                     detected_objects=[PERCEPTION["detected_objects"][0]])
    r = eval_stage1(one_child, GT1)
    assert r["required_recall"] == round(1 / 3, 3)


def test_stage1_false_hazard_on_control():
    control = dict(PERCEPTION, detected_objects=[
        dict(PERCEPTION["detected_objects"][0], state="resting",
             state_kind="normal")])
    r = eval_stage1(control, {"required": [], "optional": [],
                              "expect_hazard": False})
    assert not r["false_hazard"]
    hazard = dict(PERCEPTION["detected_objects"][0],
                  state="burning", state_kind="hazard_bearing")
    r2 = eval_stage1(dict(PERCEPTION, detected_objects=[hazard]),
                     {"required": [], "expect_hazard": False})
    assert r2["false_hazard"] is True


# ── Stage 2 ─────────────────────────────────────────────────────────────


def test_stage2_perfect_answer_scores_zero():
    e = eval_stage2(GOOD, GT2)
    assert e["error_score"] == 0.0
    assert e["verdict_ok"] and e["bucket_ok"] and e["at_risk_recall"] == 1.0


def test_stage2_capitulation_scores_heavily():
    e = eval_stage2(CAPITULATED, GT2)
    # verdict 3 + wrong type family 1 + bucket distance 3 + missed
    # at-risk 1 (child_2) = 8.0
    assert not e["verdict_ok"]
    assert e["bucket_distance"] == 3
    assert e["error_score"] == 8.0


def test_stage2_spurious_and_kind_errors_counted():
    noisy = dict(GOOD, threats=[{"object_id": "child_1", "reason": "x"}],
                 at_risk=[{"object_id": "child_1", "kind": "proximity",
                           "reason": "y"}])
    e = eval_stage2(noisy, GT2)
    assert e["threat_spurious"] == ["child_1"]
    assert e["kind_errors"] == ["child_1"]
    # spurious threat 1 + missed child_2 1 + kind 0.5 = 2.5
    assert e["error_score"] == 2.5


def test_stage2_optional_entries_not_spurious():
    gt = dict(GT2, at_risk_optional={"adult_1": "proximity"})
    extra = dict(GOOD, at_risk=GOOD["at_risk"]
                 + [{"object_id": "adult_1", "kind": "proximity",
                     "reason": "near"}])
    e = eval_stage2(extra, gt)
    assert e["at_risk_spurious"] == [] and e["error_score"] == 0.0


# ── Quadrant ────────────────────────────────────────────────────────────


def test_quadrant_cells():
    assert quadrant(5, 1, 0.3, 0.1) == "REFINEMENT"
    assert quadrant(1, 5, 0.3, 0.1) == "FALSE CERTAINTY"
    assert quadrant(1, 5, 0.1, 0.4) == "DESTABILIZATION"
    assert quadrant(5, 1, 0.1, 0.4) == "noisy improvement"
    assert quadrant(3, 3, 0.2, 0.2) == "unchanged"
    assert quadrant(3, 3, None, None) == "unchanged"
    assert quadrant(5, 1, None, None) == "improvement (U flat)"


# ── Citation counts ─────────────────────────────────────────────────────


def test_citation_counts_state_words():
    c = citation_counts(GOOD, PERCEPTION)
    # child_1's reason contains "drowning"; child_2's contains "trapped"
    assert c["state_cited"] == 2 and c["state_citation_rate"] == 1.0
    assert c["reasoning_names_entities"] == 2
    vague = dict(GOOD, at_risk=[{"object_id": "child_1", "kind": "distress",
                                 "reason": "is in grave danger"}])
    assert citation_counts(vague, PERCEPTION)["state_cited"] == 0
    assert citation_counts(dict(GOOD, threats=[], at_risk=[]),
                           PERCEPTION)["state_citation_rate"] is None


# ── Judge (scripted) ────────────────────────────────────────────────────


def test_judge_pairwise_blind_mapping_both_orders():
    """Whatever the shuffle, 'the answer with two at-risk children wins'
    must map back to the right pre/post label."""
    def judge(prompt):
        # pick the assessment that mentions child_2 (the GOOD one)
        blocks = prompt.split("ASSESSMENT A:")[1]
        a_block = blocks.split("ASSESSMENT B:")[0]
        return "A because it covers both children." \
            if "child_2" in a_block else "B because it covers both children."

    for scene in ("B_pool", "A_fire", "C_tanker_fire", "X", "Y", "Z"):
        v = judge_pairwise(CAPITULATED, GOOD, PERCEPTION, scene,
                           judge_fn=judge)
        assert v["winner"] == "post", scene    # GOOD is always post here


def test_judge_pairwise_tie_and_garbage():
    v = judge_pairwise(GOOD, GOOD, PERCEPTION, "B_pool",
                       judge_fn=lambda p: "TIE — identical.")
    assert v["winner"] == "tie"
    v2 = judge_pairwise(GOOD, GOOD, PERCEPTION, "B_pool",
                        judge_fn=lambda p: "hmm not sure")
    assert v2["winner"] == "unparseable"


def test_rubric_parses_scores_and_survives_garbage():
    rows = rubric_reasons(GOOD, PERCEPTION,
                          judge_fn=lambda p: "R1: 1\nR2: 1\nSCORE: 2/2")
    assert len(rows) == 2 and all(r["score"] == 2 for r in rows)
    rows2 = rubric_reasons(GOOD, PERCEPTION,
                           judge_fn=lambda p: "no rubric here")
    assert all(r["score"] is None for r in rows2)


# ── End-to-end scene evaluation with the real GT file ───────────────────


def test_gt_file_loads_and_covers_six_scenes():
    gt = load_gt()
    assert set(gt) == {"A_fire", "B_pool", "C_tanker_fire",
                       "D_aerial_spill", "E_collapse", "F_park_control"}
    for name, g in gt.items():
        assert g["stage2"]["disaster_scenario"] in ("Yes", "No")
        assert g["stage2"]["bucket"] in g["stage2"]["bucket_acceptable"]


def test_evaluate_scene_attributes_bpool_to_stage1():
    gt = {"B_pool": {"stage1": GT1, "stage2": GT2}}
    rec = {"assessment": CAPITULATED, "raw_answer": CAPITULATED,
           "reflection_trace": {"u_before": 0.25, "u_after": 0.22,
                                "rounds": [], "stopped_reason": "clean"},
           "parse_notes": [], "violations": []}
    r = evaluate_scene("B_pool", gt, PERCEPTION, rec)
    assert r["stage1"]["missing_hazard"]
    assert "stage1" in r["attribution"]
    assert r["post"]["error_score"] >= 7.0


def test_judges_see_the_assessors_full_evidence_basis():
    """Pairwise, runoff, and rubric prompts carry caption + states +
    geometry hints — the same (and ONLY the same) evidence the assessor
    had. Never the image."""
    from agentic.evals import judge_runoff
    seen = {}

    def capture(prompt):
        seen["p"] = prompt
        return "TIE — equal."

    judge_pairwise(GOOD, CAPITULATED, PERCEPTION, "B_pool", judge_fn=capture)
    assert "CAPTION: cap" in seen["p"]
    assert "DECLARED STATES:" in seen["p"]
    assert "SPATIAL HINTS" in seen["p"]
    assert "image" not in seen["p"].split("CAPTION")[0].lower()

    judge_runoff(GOOD, CAPITULATED, PERCEPTION, "B_pool", judge_fn=capture)
    assert "CAPTION: cap" in seen["p"] and "SPATIAL HINTS" in seen["p"]

    rubric_reasons(GOOD, PERCEPTION, judge_fn=capture)
    assert "CAPTION: cap" in seen["p"]


def test_citation_counts_names_the_vague_reasons():
    vague = dict(GOOD, at_risk=[
        {"object_id": "child_1", "kind": "distress",
         "reason": "the child is drowning"},
        {"object_id": "child_2", "kind": "distress",
         "reason": "in grave danger"}])          # no state word
    cc = citation_counts(vague, PERCEPTION)
    assert cc["uncited"] == ["child_2"]


# ── Substantive-change gate for auto-pairwise (C_tanker ui_529ce417) ────

from agentic.evals import substantive_key  # noqa: E402


def _ans(**kw):
    base = {"disaster_scenario": "Yes", "disaster_type": "hazmat fire",
            "disaster_level": 7,
            "threats": [{"object_id": "tanker_truck_1", "reason": "x"},
                        {"object_id": "fire_1", "reason": "y"}],
            "at_risk": [{"object_id": "person_1", "kind": "proximity",
                         "reason": "z"}]}
    base.update(kw)
    return base


def test_substantive_key_ignores_narration():
    a = _ans()
    b = _ans()
    b["reasoning"] = "totally different prose"
    b["threats"] = [{"object_id": "fire_1", "reason": "reworded!"},
                    {"object_id": "tanker_truck_1", "reason": "also new"}]
    assert substantive_key(a) == substantive_key(b)   # sets + folds equal


def test_substantive_key_folds_wording_and_level():
    a = _ans()
    b = _ans(disaster_type="Fire and Hazardous Material Spill",
             disaster_level=8)              # same family, same bucket
    assert substantive_key(a) == substantive_key(b)


def test_substantive_key_sees_real_changes():
    a = _ans()
    assert substantive_key(a) != substantive_key(_ans(disaster_level=4))
    assert substantive_key(a) != substantive_key(_ans(disaster_scenario="No"))
    assert substantive_key(a) != substantive_key(
        _ans(threats=[{"object_id": "tanker_truck_1", "reason": "x"}]))
    assert substantive_key(a) != substantive_key(
        _ans(at_risk=[{"object_id": "person_1", "kind": "distress",
                       "reason": "z"}]))    # kind flip IS substantive


def test_substantive_key_survives_garbage():
    assert substantive_key({})              # no crash on empty/malformed
    assert substantive_key({"disaster_level": None, "threats": None,
                            "at_risk": None})
