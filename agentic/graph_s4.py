"""Stage 4 control — LangGraph twin of the Python straight line (Phase 1a).

Same story as graph_live.py for Stage 2: there are TWO controls for the Stage-4
spine — a plain-Python one (`recommend.run_stage4`) and this LangGraph one — and
they are proven BYTE-IDENTICAL under scripted models (same Stage4Result, same
event stream, same order). The UI/env toggle picks which runs, through the SAME
`AGENTIC_CONTROL` flag the rest of the pipeline uses (reused from graph_live, so
one switch governs the whole run — not a second knob).

The spine is linear — no branches yet (Phase 1a):

    START -> recommend -> graph_a -> graph_b -> picks -> END

Each node is a 1:1 lift of one segment of `run_stage4`, calling the SAME node
core from recommend.py in the SAME order, so the two paths cannot diverge. The
conditional branches (reflection loop, petition) arrive in Phase 2 and slot onto
this graph without rewiring the spine.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

# Reuse the SINGLE pipeline control flag (python | langgraph). One toggle for
# the whole run — Stage 2 and Stage 4 switch together.
from agentic.graph_live import control_flag, set_control  # noqa: F401
from agentic.recommend import (QueryFn, Stage4Result, build_graph_a,
                               run_card_judge, run_graph_judge,
                               build_graph_a_probes, measure_graph_b_uncertainty,
                               pick_targets, run_evals,
                               run_graph_b, run_graph_b_probes,
                               run_recommend, run_recommend_uncertainty,
                               run_stage4, run_trust, _emitter)


class S4State(TypedDict, total=False):
    record: Any             # frozen PerceptionResult (Stage 1) — input, unchanged
    assessment: Any         # frozen assessment (Stage 2) — input, unchanged
    frame: dict             # reasoning frame (scene_summary, observations, ...)
    recommendations: list   # the model's recommendations (quads → Graph A)
    advisory: list          # the assumptions advisory (recorded, not in graph)
    recommend_notes: list   # parse notes from the recommend boundary
    recommend_raw: Any       # the raw recommend answer, kept verbatim
    uncertainty: dict       # measured probe U over the recommend step (1b)
    graph_a: dict           # built from the quads (code)
    graph_b: dict           # the model's independent graph + suppression_pick
    graphs_a: list          # F19: one Graph A per probe (free, from candidates)
    graphs_b: list          # F19: one Graph B per probe (n_probes calls)
    graph_b_uncertainty: dict  # is the model's causal BELIEF stable?
    graph_b_internal: dict     # does Graph B agree with its own declarations?
    picks: dict             # A_pick / B_pick / llm_pick + agreement
    conformance: dict       # corrected conformance breakdown (Phase 1b)
    internal_alignment: dict  # within-A recommendation coverage (Phase 1b)
    explanation_alignment: dict  # F24: action / reason / quad, one law
    set_report: dict        # F29: findings about the SET, not one card
    card_judge: dict        # F24: the advisory judge (display-only)
    graph_judge: dict       # F38: the graph judge (display-only)
    alignment: dict         # A-vs-B declared-vs-structured (Phase 1b)
    trust: dict             # folded trust score + breakdown (Phase 1b)


def build_s4_graph(*, query_fn: QueryFn, probe_fn: QueryFn | None = None,
                   explain_fn: Any = None, judge_fn: Any = None,
                   n_probes: int = 0, on_event: Any = None):
    """Compile the Stage-4 spine. Model config (query_fn, probe_fn, n_probes,
    on_event) is baked into the node closures, so the state only carries data —
    mirroring graph_live.build_assess_graph."""

    def recommend(state: S4State) -> dict[str, Any]:
        out = run_recommend(state["record"], state["assessment"],
                            query_fn=query_fn, on_event=on_event)
        return out

    def uncertainty(state: S4State) -> dict[str, Any]:
        from agentic.recommend import _canonical_threats
        return run_recommend_uncertainty(
            state["record"], state["assessment"], probe_fn=probe_fn,
            explain_fn=explain_fn, n_probes=n_probes,
            canonical_threats=_canonical_threats(state["recommendations"]),
            on_event=on_event)

    def graph_a(state: S4State) -> dict[str, Any]:
        g = build_graph_a(state["record"], state["assessment"],
                          state["recommendations"], on_event=on_event)
        return {"graph_a": g}

    def graph_b(state: S4State) -> dict[str, Any]:
        g = run_graph_b(state["record"], state["assessment"],
                        query_fn=query_fn, on_event=on_event)
        # F19: probe both sides — A is free (rebuilt from the recorded probe
        # sets), B costs n_probes calls. Same place, same order as the Python
        # control, so the twin stays byte-identical.
        # SAME ORDER as the Python control: A probes (free) then B probes.
        # The twin is proven byte-identical on the event stream, so order is
        # part of the contract, not a detail.
        ga = build_graph_a_probes(state["record"], state["assessment"],
                                  state["uncertainty"], on_event=on_event)
        gbu = measure_graph_b_uncertainty(state["record"], state["assessment"],
                                          n_probes, probe_fn=probe_fn,
                                          on_event=on_event)
        return {"graph_b": g, "graphs_a": ga,
                "graphs_b": gbu.get("graphs") or [],
                "graph_b_uncertainty": gbu}

    def picks(state: S4State) -> dict[str, Any]:
        p = pick_targets(state["record"], state["graph_a"], state["graph_b"],
                         state["recommendations"], query_fn=query_fn,
                         on_event=on_event)
        return {"picks": p}

    def evals(state: S4State) -> dict[str, Any]:
        e = run_evals(state["record"], state["assessment"],
                      state["recommendations"], state["graph_a"],
                      state["graph_b"], on_event=on_event,
                      graphs_a=state.get("graphs_a"),
                      graphs_b=state.get("graphs_b"))
        _gbi = e.get("graph_b_internal")
        return {"conformance": e["conformance"],
                "internal_alignment": e["internal_alignment"],
                "graph_b_internal": _gbi,
                "explanation_alignment": e["explanation_alignment"],
                "set_report": e["set_report"],
                "alignment": e["alignment"]}

    def card_judge(state: S4State) -> dict[str, Any]:
        return run_card_judge(state["record"], state["assessment"],
                              state["recommendations"],
                              state.get("explanation_alignment") or {},
                              judge_fn=judge_fn, on_event=on_event)

    def graph_judge(state: S4State) -> dict[str, Any]:
        return run_graph_judge(state["record"], state["assessment"],
                               state["graph_a"], state["graph_b"],
                               state.get("alignment") or {},
                               judge_fn=judge_fn, on_event=on_event)

    def trust(state: S4State) -> dict[str, Any]:
        t = run_trust(state["recommendations"], state["conformance"],
                      state["internal_alignment"], state["alignment"],
                      state.get("uncertainty", {}), state["picks"],
                      record=state["record"], assessment=state["assessment"],
                      graph_b_internal=state.get("graph_b_internal"),
                      graph_b_uncertainty=state.get("graph_b_uncertainty"),
                      on_event=on_event)
        return {"trust": t["trust"]}

    g = StateGraph(S4State)
    g.add_node("recommend", recommend)
    g.add_node("uncertainty", uncertainty)
    g.add_node("graph_a", graph_a)
    g.add_node("graph_b", graph_b)
    g.add_node("picks", picks)
    g.add_node("evals", evals)
    g.add_node("card_judge", card_judge)
    g.add_node("graph_judge", graph_judge)
    g.add_node("trust", trust)

    g.add_edge(START, "recommend")
    g.add_edge("recommend", "uncertainty")
    g.add_edge("uncertainty", "graph_a")
    g.add_edge("graph_a", "graph_b")
    g.add_edge("graph_b", "picks")
    g.add_edge("picks", "evals")
    g.add_edge("evals", "card_judge")
    g.add_edge("card_judge", "graph_judge")
    g.add_edge("graph_judge", "trust")
    g.add_edge("trust", END)
    return g.compile()


def run_s4_graph(record: Any, assessment: Any, image_path: str = "",
                 *, query_fn: QueryFn | None = None,
                 probe_fn: QueryFn | None = None, explain_fn: Any = None,
                 judge_fn: Any = None,
                 n_probes: int = 0, on_event: Any = None) -> Stage4Result:
    """LangGraph twin of run_stage4 — identical positional signature and return
    type. Assembles the same Stage4Result from the final state, and emits the
    terminal `stage_done` in the same place the Python control does (after the
    picks node), so the two event streams match exactly."""
    from agentic.recommend import _query_vlm
    query_fn = query_fn or (lambda p: _query_vlm(p, temperature=0.0))

    graph = build_s4_graph(query_fn=query_fn, probe_fn=probe_fn,
                           explain_fn=explain_fn, judge_fn=judge_fn,
                           n_probes=n_probes, on_event=on_event)
    final: S4State = graph.invoke({"record": record, "assessment": assessment})

    _emitter(on_event)("stage_done", stage="recommend")
    return Stage4Result(
        frame=final.get("frame", {}),
        recommendations=final.get("recommendations", []),
        advisory=final.get("advisory", []),
        graph_a=final.get("graph_a", {}),
        graph_b=final.get("graph_b", {}),
        picks=final.get("picks", {}),
        conformance=final.get("conformance", {}),
        internal_alignment=final.get("internal_alignment", {}),
        explanation_alignment=final.get("explanation_alignment", {}) or {},
        set_report=final.get("set_report", {}) or {},
        card_judge=final.get("card_judge", {}) or {},
        graph_judge=final.get("graph_judge", {}) or {},
        alignment=final.get("alignment", {}),
        uncertainty=final.get("uncertainty", {}),
        trust=final.get("trust", {}),
        graph_b_uncertainty=final.get("graph_b_uncertainty", {}),
        graph_b_internal=final.get("graph_b_internal", {}) or {},
        parse_notes=final.get("recommend_notes", []),
        raw_answer=final.get("recommend_raw"))


def stage4_with_control(record: Any, assessment: Any, image_path: str = "",
                        *, query_fn: QueryFn | None = None,
                        probe_fn: QueryFn | None = None, explain_fn: Any = None,
                        judge_fn: Any = None, n_probes: int = 0,
                        on_event: Any = None) -> Stage4Result:
    """Dispatch the Stage-4 spine by the pipeline control flag. Identical
    contract on both branches — the whole point of the equivalence tests."""
    if control_flag() == "langgraph":
        return run_s4_graph(record, assessment, image_path,
                            query_fn=query_fn, probe_fn=probe_fn,
                            explain_fn=explain_fn, judge_fn=judge_fn,
                            n_probes=n_probes, on_event=on_event)
    return run_stage4(record, assessment, image_path,
                     query_fn=query_fn, probe_fn=probe_fn,
                     explain_fn=explain_fn, judge_fn=judge_fn,
                     n_probes=n_probes, on_event=on_event)


# ── Live runner (Phase 1a has no UI yet — this is the live check) ────────

def _image_data_url(path: str) -> Optional[str]:
    import base64
    import mimetypes
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def run_dir_live(run_dir: str) -> Stage4Result:
    """Load a frozen run's perception + assessment (a ui_* export dir) and run
    the Stage-4 spine LIVE against the local model, with the scene image bound
    in as context. Prints a short summary. Needs Ollama running."""
    import os
    from pathlib import Path

    from agentic.assessment import AssessmentResult
    from agentic.perception import PerceptionResult
    from agentic.recommend import _query_vlm

    d = Path(run_dir)
    perc = next(d.glob("*__perception.json"))
    record = PerceptionResult.model_validate_json(perc.read_text())
    assessment = AssessmentResult.model_validate_json(
        (d / "assessment.json").read_text()).assessment
    image = None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        for img in sorted(d.glob(f"*{ext}")):
            if "__" not in img.name:
                image = str(img)
                break
        if image:
            break
    data_url = _image_data_url(image) if image else None

    def q(prompt: str) -> dict:
        return _query_vlm(prompt, image_contents=data_url, temperature=0.0)

    from agentic.recommend import (DEFAULT_REC_N_PROBES,
                                   REC_PROBE_TEMPERATURE)

    def probe(prompt: str) -> dict:
        return _query_vlm(prompt, image_contents=data_url,
                          temperature=REC_PROBE_TEMPERATURE)

    n_probes = int(os.getenv("REC_N_PROBES", str(DEFAULT_REC_N_PROBES)))
    events: list = []
    res = stage4_with_control(record, assessment, image or "",
                             query_fn=q, probe_fn=probe, n_probes=n_probes,
                             on_event=events.append)
    print(f"control        : {control_flag()}")
    print(f"recommendations: {len(res.recommendations)}  "
          f"(ranks {[r.get('rank') for r in res.recommendations]})")
    print(f"advisory flags : {len(res.advisory)}")
    print(f"graph A        : {len(res.graph_a.get('nodes', []))} nodes, "
          f"{len(res.graph_a.get('edges', []))} edges")
    print(f"graph B        : {len(res.graph_b.get('nodes', []))} nodes, "
          f"{len(res.graph_b.get('edges', []))} edges")
    p = res.picks
    print(f"A_pick         : {p.get('a_pick', {}).get('threat')}")
    print(f"B_pick         : {p.get('b_pick', {}).get('threat')}")
    print(f"llm_pick       : {p.get('llm_pick', {}).get('threat')}")
    print(f"pick agreement : {p.get('agreement')}  unanimous={p.get('unanimous')}")
    u = res.uncertainty or {}
    if u:
        print(f"measured U     : {u.get('score')}  ({u.get('n_probes')} probes, "
              f"{len(u.get('drivers', []))} drivers)")
        for drv in u.get("drivers", [])[:4]:
            print(f"   - {drv.get('kind')}: {drv.get('evidence')}")
    t = res.trust or {}
    if t:
        print(f"trust          : {t.get('score')}  ({t.get('band')})")
        for c in t.get("contributors", [])[:3]:
            if c.get("contribution", 0) > 0:
                print(f"   - {c.get('signal')}: {c.get('text')} "
                      f"(−{c.get('contribution')})")
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--run":
        run_dir_live(sys.argv[2])
    else:
        print("usage: python -m agentic.graph_s4 --run <exports/agentic_runs/ui_XXXX>")

