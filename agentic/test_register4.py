"""The Stage 4 ticket register (2026-09-21): code and judge findings as
tickets, Stage 1/2 grammar, all OPEN. Hermetic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.register4 import MIN_CODE_SEVERITY, stage4_register  # noqa: E402


def _s4(**kw):
    base = {"conformance": {"issues": []}, "internal_alignment": {"failures": []},
            "trust": {"singular_errors": []}, "card_judge": {}, "graph_judge": {},
            "runoff_judge": {}}
    base.update(kw)
    return base


def test_empty_record_gives_an_empty_register_without_raising():
    reg = stage4_register({})
    assert reg["rule"] == [] and reg["pathology"] == []
    assert reg["counts"] == {"rule": 0, "pathology": 0, "judge": 0, "code": 0}


def test_code_findings_become_tickets_above_the_bookkeeping_floor():
    reg = stage4_register(_s4(
        conformance={"issues": [
            {"graph": "card", "rule": "remaining_risk_not_a_pair", "severity": 0,
             "detail": "stringified list", "rank": 1},
            {"graph": "graph_b", "rule": "hazard_flag_state_mismatch", "severity": 2,
             "detail": "spill_1: seeping vs hazardous=False"}]},
        internal_alignment={"failures": [
            {"category": "role mix-up", "severity": 3,
             "detail": "at-risk person_1 used as a threat", "rank": None}]},
        trust={"singular_errors": [
            {"id": "emergency_invented", "deduction": 0.4, "detail": "acted on a safe scene"},
            {"id": "hazard_unaddressed", "deduction": 0.0, "waived": True,
             "detail": "smoke covered through its victims"}]}))
    kinds = [(t["source"], t["kind"]) for t in reg["rule"]]
    assert ("code · graph_b", "hazard_flag_state_mismatch") in kinds
    assert ("code · internal", "role mix-up") in kinds
    assert ("code · error library", "emergency_invented") in kinds
    # severity-0 bookkeeping and WAIVED errors earn no ticket
    assert all(t["kind"] != "remaining_risk_not_a_pair" for t in reg["rule"])
    assert all(t["kind"] != "hazard_unaddressed" for t in reg["rule"])
    assert all(t["status"] == "open" and t["round"] == 0 for t in reg["rule"])
    assert MIN_CODE_SEVERITY == 1


def test_judge_verdicts_become_advisory_tickets_only_when_they_ask_for_something():
    reg = stage4_register(_s4(
        card_judge={"rollup": {"findings": [
            {"rank": 3, "kind": "not_aligned", "votes": 3, "n": 3, "thin": False,
             "text": "rec 3: the quad is not causally aligned with its action"}]}},
        graph_judge={"text": {"account": {"verdict": "account_b", "votes": 3, "n": 3},
                              "victims": {"verdict": "account_b"}},
                     "sets": {"graph_a": ["car_1"], "graph_b": ["person_1"]},
                     "twins_agree": False},
        runoff_judge={"recommendations": {
            "text": {"verdict": "answer_b"}, "twins_agree": True,
            "candidate_a": "  1. action: A", "candidate_b": "  1. action: Evacuate person_1"}}))
    judge = [t for t in reg["rule"] if t["advisory"]]
    kinds = {t["kind"] for t in judge}
    assert kinds == {"not_aligned", "advice_on_the_weaker_account",
                     "advice_protects_the_lesser_set",
                     "model_stands_behind_another_answer"}
    weaker = next(t for t in judge if t["kind"] == "advice_on_the_weaker_account")
    assert "twins disagree" in weaker["evidence"]
    stands = next(t for t in judge if t["kind"] == "model_stands_behind_another_answer")
    assert "Evacuate person_1" in stands["evidence"]
    assert reg["counts"]["judge"] == 4 and reg["counts"]["code"] == 0
    # a clean verdict is not a finding
    clean = stage4_register(_s4(
        graph_judge={"text": {"account": {"verdict": "account_a"}}},
        runoff_judge={"graph_b": {"text": {"verdict": "equally_good"}}}))
    assert clean["rule"] == []


def test_stage4result_carries_the_register_identically_in_both_controls():
    from agentic.recommend import Stage4Result
    s4 = _s4(card_judge={"rollup": {"findings": [
        {"rank": 1, "kind": "not_aligned", "votes": 3, "n": 3, "text": "rec 1: x"}]}})
    a = Stage4Result(**s4).model_dump()
    b = Stage4Result(**json.loads(json.dumps(s4))).model_dump()
    assert a["tickets"] == b["tickets"]
    assert a["tickets"]["counts"]["judge"] == 1


def test_register_panel_renders_old_and_new_records():
    from agentic import ui
    assert ui._register_panel({})                     # empty record: placeholders
    panel = ui._register_panel(_s4(card_judge={"rollup": {"findings": [
        {"rank": 2, "kind": "not_aligned", "votes": 2, "n": 3, "thin": True,
         "text": "rec 2: y"}]}}))
    assert any(getattr(x, "className", "") == "ticket open" for x in panel)
