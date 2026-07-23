"""Hermetic tests for channel-2 uncertainty + causal explanations.

Run:  pytest agentic/test_uncertainty.py -q
"""
from __future__ import annotations

import pytest

from agentic.uncertainty import (
    MeasuredUncertainty,
    agreement,
    entity_uncertainty,
    explain,
    measure_verdicts,
    spread,
    stood_entity_ids,
)

# ── Dispersion math ─────────────────────────────────────────────────────


def test_agreement_math():
    assert agreement(["Yes"] * 5) == 1.0
    assert agreement(["Yes", "Yes", "Yes", "No", "No"]) == 0.6
    assert agreement([]) == 1.0            # nothing measured ≠ disagreement


def test_spread_math():
    s = spread([5, 5, 5])
    assert (s["min"], s["max"], s["std"]) == (5.0, 5.0, 0.0)
    s = spread([3, 8])
    assert s["min"] == 3.0 and s["max"] == 8.0 and s["std"] == 2.5


def _probe(scenario="Yes", dtype="hazmat fire", level=7, bucket="catastrophic"):
    return {"scenario": scenario, "disaster_type": dtype,
            "level": level, "bucket": bucket}


def test_unanimous_probes_score_zero_no_drivers():
    mu = measure_verdicts([_probe()] * 5)
    assert mu.score == 0.0 and mu.drivers == []
    assert mu.scenario_agreement == 1.0


def test_scenario_flip_driver_with_split_evidence_and_action():
    mu = measure_verdicts([_probe(), _probe(), _probe(),
                           _probe(scenario="No", level=0, bucket="none",
                                  dtype="n/a"),
                           _probe(scenario="No", level=0, bucket="none",
                                  dtype="n/a")])
    kinds = [d.kind for d in mu.drivers]
    assert "scenario_flip" in kinds
    flip = next(d for d in mu.drivers if d.kind == "scenario_flip")
    assert "Yes×3" in flip.evidence and "No×2" in flip.evidence
    assert "S2/S3" in flip.action          # actionable: names the check


def test_type_split_folds_cosmetic_variants():
    """'Hazmat Fire' vs 'hazmat-fire' is not disagreement; 'flood' is."""
    same = measure_verdicts([_probe(dtype="Hazmat Fire"),
                             _probe(dtype="hazmat-fire"),
                             _probe(dtype="hazmat fire")])
    assert same.type_agreement == 1.0
    mixed = measure_verdicts([_probe(dtype="hazmat fire"),
                              _probe(dtype="flood"), _probe(dtype="flood")])
    assert mixed.type_agreement < 1.0
    assert any(d.kind == "type_split" for d in mixed.drivers)


def test_bucket_split_vs_level_wobble():
    straddle = measure_verdicts([_probe(level=6, bucket="serious"),
                                 _probe(level=7, bucket="catastrophic")])
    assert any(d.kind == "bucket_split" for d in straddle.drivers)
    wobble = measure_verdicts([_probe(level=7), _probe(level=10),
                               _probe(level=7)])
    kinds = [d.kind for d in wobble.drivers]
    assert "level_wobble" in kinds and "bucket_split" not in kinds
    w = next(d for d in wobble.drivers if d.kind == "level_wobble")
    assert "bucket is stable" in w.action


# ── Structural entity uncertainty (Stage 1) ─────────────────────────────

CLEAN = {"object_id": "car_1", "label": "car", "box_source": "dino_matched",
         "box_confidence": 0.9, "vocab_extension": False, "label_note": "",
         "state": "stationary", "state_kind": "normal"}


def test_clean_entity_scores_zero():
    u = entity_uncertainty(CLEAN)
    assert u == {"score": 0.0, "factors": []}


def test_factors_compose_with_evidence():
    shaky = dict(CLEAN, box_source="vlm_sam_fallback", box_confidence=0.3,
                 state="glorping", state_kind="unknown")
    u = entity_uncertainty(shaky)
    names = [f["factor"] for f in u["factors"]]
    assert names == ["fallback_box", "low_box_conf", "unknown_state"]
    assert u["score"] == pytest.approx(0.7)
    assert "vlm_sam_fallback" in u["factors"][0]["evidence"]


def test_score_clamped_and_stood_ticket_counts():
    worst = dict(CLEAN, box_source="none", box_confidence=0.1,
                 vocab_extension=True, label_note="coerced",
                 state_kind="unknown")
    u = entity_uncertainty(worst, stood_ids={"car_1"})
    assert u["score"] == 1.0               # sum would exceed 1; clamped
    assert any(f["factor"] == "stood_ticket" for f in u["factors"])


def test_entity_uncertainty_accepts_malformed_dicts():
    """Rule 1a: raw dicts with missing keys must not crash the scorer."""
    u = entity_uncertainty({"object_id": "x_1"})
    assert 0.0 <= u["score"] <= 1.0
    assert any(f["factor"] == "fallback_box" for f in u["factors"])


def test_stood_ids_from_trace_shapes():
    assert stood_entity_ids(None) == set()
    assert stood_entity_ids({"rounds": []}) == set()
    trace = {"rounds": [{"stood": [{"object_id": "spill_1"}, {}]},
                        {"stood": None}]}
    assert stood_entity_ids(trace) == {"spill_1"}


# ── The causal narrative ────────────────────────────────────────────────


def _mu_with_driver() -> MeasuredUncertainty:
    return measure_verdicts([_probe(), _probe(scenario="No", level=0,
                                              bucket="none", dtype="n/a")])


def test_no_drivers_explains_unanimity():
    mu = explain(measure_verdicts([_probe()] * 3), "Yes · fire · 7")
    assert "unanimous" in mu.explanation and mu.explainer == "deterministic"


def test_deterministic_fallback_carries_evidence_and_action():
    mu = explain(_mu_with_driver(), "Yes · fire · 7")
    assert "scenario_flip" in mu.explanation
    assert "->" in mu.explanation          # evidence -> action, actionable


def test_llm_narrative_used_when_available():
    mu = explain(_mu_with_driver(), "Yes · fire · 7",
                 explain_fn=lambda p: "Probes split 1-1 on scenario; "
                                      "run S2/S3 before trusting.")
    assert mu.explainer == "llm" and "S2/S3" in mu.explanation


def test_explainer_failure_falls_back_silently():
    def broken(prompt: str) -> str:
        raise RuntimeError("ollama down")
    mu = explain(_mu_with_driver(), "Yes · fire · 7", explain_fn=broken)
    assert mu.explainer == "deterministic" and "scenario_flip" in mu.explanation


def test_explainer_prompt_contains_only_driver_material():
    """The narrator gets drivers + verdict, never raw scene access: the
    narrative cannot introduce causes the code did not find."""
    seen = {}

    def capture(prompt: str) -> str:
        seen["prompt"] = prompt
        return "ok"

    explain(_mu_with_driver(), "Yes · fire · 7", explain_fn=capture)
    assert "scenario_flip" in seen["prompt"]
    assert "do not invent causes" in seen["prompt"]
