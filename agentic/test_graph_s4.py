"""Stage 4 control — Python vs LangGraph equivalence, hermetic.

The two controls for the Stage-4 spine must be BYTE-IDENTICAL: same
Stage4Result (model_dump) and same event stream in the same order, under
scripted models. Same guarantee test_graph_live.py makes for Stage 2. Plus the
control-flag / override tests, and an equivalence check on malformed model
output (garbage must still drive both paths identically).

Run:  pytest agentic/test_graph_s4.py -q
"""
from __future__ import annotations

from agentic.assessment import AtRiskEntry, SceneAssessment, ThreatEntry
from agentic.graph_s4 import (control_flag, run_s4_graph, set_control,
                              stage4_with_control)
from agentic.perception import DetectedObject, PerceptionResult
from agentic.recommend import run_stage4
from agentic.test_recommend import (GRAPH_B_OK, PICK_OK, REC_OK, _asm, _record,
                                    _script)


# ── equivalence harness (mirrors test_graph_live._run_both) ─────────────

def _run_both(record, assessment, query_fn):
    ev_py: list = []
    r_py = run_stage4(record.model_copy(deep=True),
                     assessment.model_copy(deep=True), "/x/E.jpg",
                     query_fn=query_fn, on_event=ev_py.append)
    ev_lg: list = []
    r_lg = run_s4_graph(record.model_copy(deep=True),
                       assessment.model_copy(deep=True), "/x/E.jpg",
                       query_fn=query_fn, on_event=ev_lg.append)
    return r_py, r_lg, ev_py, ev_lg


def _assert_equivalent(r_py, r_lg, ev_py, ev_lg):
    assert r_py.model_dump() == r_lg.model_dump(), "Stage4Result differs"
    assert ev_py == ev_lg, "event streams differ"


# ── the spine, byte-identical ───────────────────────────────────────────

def test_equivalence_straight_line():
    r_py, r_lg, ev_py, ev_lg = _run_both(_record(), _asm(), _script())
    _assert_equivalent(r_py, r_lg, ev_py, ev_lg)
    assert [e["type"] for e in ev_lg] == [
        "stage_started", "recommendations_ready", "graph_a_built",
        "graph_b_built", "targets_picked", "conformance_ready",
        "internal_alignment_ready", "alignment_ready", "trust_ready",
        "stage_done"]


def test_equivalence_unanimous_pick():
    gb = {**GRAPH_B_OK, "suppression_pick": {"threat": "dust_1",
                                             "state": "rising", "reason": "x"}}
    pick = {"threat": "dust_1", "state": "rising", "reason": "x"}
    r_py, r_lg, ev_py, ev_lg = _run_both(
        _record(), _asm(), _script(gb=gb, pick=pick))
    _assert_equivalent(r_py, r_lg, ev_py, ev_lg)
    assert r_lg.picks["unanimous"] is True


def test_equivalence_with_probes():
    """The measured-uncertainty node must be byte-identical across controls too.
    Each control gets a FRESH probe sequence (a shared counter would let the
    python run consume it before the langgraph run)."""
    from agentic.test_recommend import _rec, _rec_answer
    seq = [_rec_answer([_rec("building_1", "collapsed", "may_harm", ["person_1"])]),
           _rec_answer([_rec("building_1", "collapsed", "may_harm", ["person_1"])]),
           _rec_answer([_rec("dust_1", "rising", "worsens", ["dust_1"])])]

    def make_probe():
        box = {"i": 0}

        def p(prompt):
            a = seq[box["i"] % len(seq)]
            box["i"] += 1
            return a
        return p

    ev_py: list = []
    r_py = run_stage4(_record().model_copy(deep=True), _asm(), "/x/E.jpg",
                     query_fn=_script(), probe_fn=make_probe(), n_probes=3,
                     on_event=ev_py.append)
    ev_lg: list = []
    r_lg = run_s4_graph(_record().model_copy(deep=True), _asm(), "/x/E.jpg",
                       query_fn=_script(), probe_fn=make_probe(), n_probes=3,
                       on_event=ev_lg.append)
    assert r_py.model_dump() == r_lg.model_dump(), "uncertainty differs"
    assert ev_py == ev_lg, "probe event streams differ"
    assert r_lg.uncertainty["n_probes"] == 3 and r_lg.uncertainty["score"] > 0.0
    assert "recommend_probe" in [e["type"] for e in ev_lg]
    assert "recommend_uncertainty_ready" in [e["type"] for e in ev_lg]


def test_equivalence_on_malformed_recommend():
    """Garbage from the recommend call must drive BOTH controls identically —
    the equivalence guarantee has to hold on the error paths too."""
    def q(prompt):
        if "assumptions_advisory" in prompt:
            return "total garbage"                  # non-dict recommend answer
        if "neutralize exactly ONE" in prompt:
            return PICK_OK
        return GRAPH_B_OK
    r_py, r_lg, ev_py, ev_lg = _run_both(_record(), _asm(), q)
    _assert_equivalent(r_py, r_lg, ev_py, ev_lg)
    assert r_lg.recommendations == []                # both degraded the same way


# ── control flag / dispatcher / override ────────────────────────────────

def test_control_flag_default(monkeypatch):
    set_control(None)
    monkeypatch.delenv("AGENTIC_CONTROL", raising=False)
    assert control_flag() == "python"
    monkeypatch.setenv("AGENTIC_CONTROL", "LangGraph")
    assert control_flag() == "langgraph"             # case-folded


def test_dispatcher_routes_by_flag(monkeypatch):
    monkeypatch.delenv("AGENTIC_CONTROL", raising=False)
    set_control(None)
    # both branches must return the same Stage4Result under the same script
    ev_p: list = []
    set_control("python")
    r_p = stage4_with_control(_record(), _asm(), "/x/E.jpg",
                             query_fn=_script(), on_event=ev_p.append)
    ev_l: list = []
    set_control("langgraph")
    r_l = stage4_with_control(_record(), _asm(), "/x/E.jpg",
                             query_fn=_script(), on_event=ev_l.append)
    set_control(None)
    assert r_p.model_dump() == r_l.model_dump()
    assert ev_p == ev_l


def test_ui_override_beats_env(monkeypatch):
    monkeypatch.setenv("AGENTIC_CONTROL", "python")
    set_control("langgraph")
    assert control_flag() == "langgraph"
    set_control(None)
    assert control_flag() == "python"
