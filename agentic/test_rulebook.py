"""Hermetic tests for the rulebook (the Loop 1 RAG seam, v0 = exact lookup).

Run:  pytest agentic/test_rulebook.py -q
"""
from __future__ import annotations

from agentic.repair_loop import detect_violations
from agentic.rulebook import RULES, instruction_for, retrieve


def test_every_rule_chunk_is_whole():
    """Chunking on rule boundaries: every chunk carries statement,
    rationale, worked example, and template, none empty."""
    assert set(RULES) == {"family_name_as_label", "label_out_of_vocab",
                          "state_out_of_vocab", "missing_anchor_bbox",
                          "caption_entity_missing", "duplicate_entity",
                          # S-family: merged-stage assessment rules
                          "scenario_level_incoherent",
                          "missed_disaster_incoherence",
                          "false_alarm_incoherence",
                          "id_not_in_perception",
                          "threat_state_not_hazardous",
                          "hazard_not_in_threats",
                          "hazard_as_at_risk",
                          "threat_reason_victim_shaped",
                          # G-family: geometry rules (Stage 2+3 merge)
                          "geometry_adjacency",
                          "proximity_without_hazard",
                          "geometry_is_a_hint",
                          "at_risk_kind_mismatch"}
    for kind, chunk in RULES.items():
        assert chunk.rule and chunk.rationale and chunk.example and chunk.template, kind
        assert chunk.rule_id[0] in "PSG"


def test_retrieve_exact_lookup():
    assert retrieve("family_name_as_label").rule_id == "P1"
    assert retrieve("nonexistent_kind") is None


def test_instruction_carries_rule_rationale_example():
    text = instruction_for("family_name_as_label",
                           index=0, raw_label="vehicle", members="car, truck",
                           description="fuel tanker on the road")
    assert "'vehicle' is a family name" in text        # evidence
    assert "car, truck" in text                        # legal options
    assert "Rule P1:" in text and "Why:" in text       # retrieved rule + rationale
    assert "Example:" in text and "tanker_truck" in text  # worked example


def test_detect_violations_speaks_through_the_rulebook():
    """Loop 1's instructions now come from retrieval, not ad-hoc strings."""
    v = detect_violations([{"label": "vehicle", "state": "intact",
                            "bbox": [0, 0, 9, 9]}])
    assert len(v) == 1
    assert "Rule P1:" in v[0].instruction and "Why:" in v[0].instruction

    v = detect_violations([], caption="A dog in the park.")
    assert any("Rule P5:" in x.instruction and
               "leave your list unchanged" in x.instruction for x in v)


def test_unknown_kind_degrades_gracefully():
    assert "unknown_kind" in instruction_for("unknown_kind", foo="bar")
