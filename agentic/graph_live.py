"""LangGraph control for the live run — a drop-in alternative to the plain-
Python orchestration in assess_with_petition (agentic/petition.py).

WHY THIS EXISTS
===============
Today the control flow — assess -> detect petition -> route (re-look image
vs re-ask question) -> cascade -> done — is a sequence of Python calls
inside `assess_with_petition`. This module expresses the SAME control as an
explicit LangGraph StateGraph: each step is a node, the petition route is a
conditional edge, the cascade is a back-edge. Nothing about the model calls
or the checks changes; only the control layer moves from implicit call-order
to an inspectable graph.

IRON RULE (Sunny, 2026-07-23): built ALONGSIDE the Python path, never
replacing it until proven identical. `run_graph` must return byte-identical
(record, result, petitioned) to `assess_with_petition` given the same model
answers. That equivalence is asserted hermetically in test_graph_live.py.

Selected with the env flag AGENTIC_CONTROL=python (default: langgraph).

SCOPE (v1): this graph replaces the STAGE-2 + PETITION control (the rich
part: the 3-way router, the cascade, run_assessment's own reflection loop
stays inside its node for now — decomposing the two inner loops into their
own self-edges is a clean v2, guarded by the same equivalence test).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic.petition import (
    _PETITIONABLE,
    assess_with_petition,
    detect_petition,
    run_petition,
    run_stage2_petition,
)


# ── The state that travels through the graph ────────────────────────────
#
# A TypedDict (not a pydantic model) so the existing PerceptionResult /
# AssessmentResult / Petition objects ride through untouched, no
# serialization. Each node returns only the keys it changed; LangGraph
# merges them (last-write-wins — no reducers needed here).

class GLState(TypedDict, total=False):
    record: Any            # PerceptionResult (evolves on a stage-1 petition)
    result: Any            # AssessmentResult (evolves on any petition)
    petition: Any          # the detected Petition, or None
    petition_target: str   # "" | "none" | "stage1" | "stage2"
    petitioned: bool       # the third element of the return tuple
    relook_ok: bool        # did the stage-1 re-perception yield a merged record?


# A UI toggle can override the env var for the current process (the UI
# sets this when a run launches); env/default is the fallback.
_CONTROL_OVERRIDE: Optional[str] = None


def set_control(mode: str | None) -> None:
    """Set (or clear, with None) the in-process control choice."""
    global _CONTROL_OVERRIDE
    _CONTROL_OVERRIDE = (mode or "").strip().lower() or None


def control_flag() -> str:
    """python | langgraph (default) — UI override, else AGENTIC_CONTROL.

    LangGraph is the default from 2026-07-28 (Sunny). The two controls are
    proven byte-identical under scripted models, so the default is a choice
    about which one we exercise in anger — and the one we ship should be the
    one every run tests."""
    if _CONTROL_OVERRIDE:
        return _CONTROL_OVERRIDE
    return os.getenv("AGENTIC_CONTROL", "langgraph").strip().lower()


# ── Build the graph (nodes close over the run config) ───────────────────

def build_assess_graph(
    *,
    image_path: str,
    on_event: Any = None,
    perceive_fn: Optional[Callable[..., Any]] = None,
    assess_kwargs: dict[str, Any],
):
    """Compile the Stage-2 + petition StateGraph. The model config
    (query_fn, probe_fn, n_probes, ...) is baked into the node closures so
    the state only carries data — mirroring assess_with_petition's args."""
    from agentic.assessment import run_assessment

    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"type": event_type, **data})

    # node: the canonical assessment (verdict -> checks -> probes ->
    # runoff -> reflection loop, all inside run_assessment, unchanged).
    def assess(state: GLState) -> dict[str, Any]:
        result = run_assessment(state["record"], on_event=on_event,
                                **assess_kwargs)
        return {"result": result}

    # node: decide whether to petition and by which route.
    def router(state: GLState) -> dict[str, Any]:
        petition = detect_petition(state["result"].reflection_trace,
                                   state["result"].violations,
                                   state["record"])
        if petition is None:
            return {"petition": None, "petition_target": "none"}
        return {"petition": petition, "petition_target": petition.target}

    # node: Route B — re-ask the question, record untouched.
    def reask(state: GLState) -> dict[str, Any]:
        result2 = run_stage2_petition(
            state["record"], state["result"], state["petition"],
            on_event=on_event, query_fn=assess_kwargs.get("query_fn"))
        return {"result": result2, "petitioned": False}

    # node: Route A — re-look the image (two-witness / no-erasure merge).
    def relook(state: GLState) -> dict[str, Any]:
        new_record = run_petition(image_path, state["record"],
                                  state["petition"], on_event=on_event,
                                  perceive_fn=perceive_fn)
        if new_record is None:
            return {"relook_ok": False, "petitioned": False}
        return {"record": new_record, "relook_ok": True}

    # node: the cascade — a merged record re-runs Stage 2 in full.
    def reassess(state: GLState) -> dict[str, Any]:
        result2 = run_assessment(state["record"], on_event=on_event,
                                 **assess_kwargs)
        before = sorted(v["kind"] for v in state["result"].violations
                        if v["kind"] in _PETITIONABLE)
        after = sorted(v["kind"] for v in result2.violations
                       if v["kind"] in _PETITIONABLE)
        emit("petition_outcome", resolved=(not after),
             violations_before=before, violations_after=after)
        return {"result": result2, "petitioned": True}

    # conditional edge out of the router (the 3-way branch)
    def route_from_router(state: GLState) -> str:
        t = state.get("petition_target", "none")
        if t == "stage1":
            return "relook"
        if t == "stage2":
            return "reask"
        return END

    # conditional edge out of relook (cascade only if the merge produced one)
    def route_from_relook(state: GLState) -> str:
        return "reassess" if state.get("relook_ok") else END

    g = StateGraph(GLState)
    g.add_node("assess", assess)
    g.add_node("router", router)
    g.add_node("reask", reask)
    g.add_node("relook", relook)
    g.add_node("reassess", reassess)

    g.add_edge(START, "assess")
    g.add_edge("assess", "router")
    g.add_conditional_edges("router", route_from_router,
                            {"relook": "relook", "reask": "reask", END: END})
    g.add_edge("reask", END)
    g.add_conditional_edges("relook", route_from_relook,
                            {"reassess": "reassess", END: END})
    g.add_edge("reassess", END)
    return g.compile()


# ── The public entry — same contract as assess_with_petition ────────────

def run_graph(image_path: str, record: Any,
              on_event: Any = None,
              perceive_fn: Optional[Callable[..., Any]] = None,
              **assess_kwargs) -> tuple[Any, Any, bool]:
    """LangGraph twin of assess_with_petition. Returns
    (final_record, final_result, petitioned) — must match the Python path
    field-for-field given the same model answers."""
    graph = build_assess_graph(image_path=image_path, on_event=on_event,
                               perceive_fn=perceive_fn,
                               assess_kwargs=assess_kwargs)
    final: GLState = graph.invoke({"record": record})
    return final["record"], final["result"], final.get("petitioned", False)


# ── The dispatcher live callers use ─────────────────────────────────────

def assess_with_control(image_path: str, record: Any,
                        on_event: Any = None,
                        perceive_fn: Optional[Callable[..., Any]] = None,
                        **assess_kwargs) -> tuple[Any, Any, bool]:
    """Route to the LangGraph or Python control by the AGENTIC_CONTROL flag.
    Both are proven output-identical (test_graph_live.py), so this is a safe
    swap: default python; set AGENTIC_CONTROL=langgraph to run the graph."""
    if control_flag() == "langgraph":
        return run_graph(image_path, record, on_event=on_event,
                         perceive_fn=perceive_fn, **assess_kwargs)
    return assess_with_petition(image_path, record, on_event=on_event,
                                perceive_fn=perceive_fn, **assess_kwargs)
