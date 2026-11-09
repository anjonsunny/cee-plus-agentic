"""UI cleanup (agentic/UI_CLEANUP_PLAN.md). Hermetic: no models, no runs.

Step 1 — the control strip: the eight run toggles fold behind one
"run settings" line whose summary shows the current choices as chips.

Step 2 — the view switch plumbing: a page-level clean/detailed RadioItems
reaches every renderer, but no clean builder exists yet, so both views
must render identically.
"""
from dash import html

from agentic import models as _models
from agentic import ui


def _text(tree) -> str:
    """Flatten a Dash component tree to the text a reader would see."""
    out = []

    def walk(n):
        if isinstance(n, (list, tuple)):
            for x in n:
                walk(x)
        elif isinstance(n, str):
            out.append(n)
        elif hasattr(n, "children"):
            walk(n.children)
    walk(tree)
    return " ".join(out)


def _kids(node):
    k = getattr(node, "children", None)
    if k is None or isinstance(k, str):
        return []
    return k if isinstance(k, (list, tuple)) else [k]


def _find(node, wanted, path=()):
    """(component, ancestors) for the component with id == wanted."""
    if getattr(node, "id", None) == wanted:
        return node, path
    for kid in _kids(node):
        hit = _find(kid, wanted, path + (node,))
        if hit is not None:
            return hit
    return None


TOGGLE_DEFAULTS = {
    "control-mode": "langgraph",
    "retrieval-mode": "both",
    "subject-model": _models.SUBJECT_MODEL,
    "reasks-mode": [],
    "judges-mode": "off",
    "concealment-motives": [],
    "env-conditions": [],
    "sandbagging-motives": [],
}
DEFAULT_ARGS = ("langgraph", "both", _models.SUBJECT_MODEL, [], "off",
                [], [], [])


def test_all_eight_toggles_keep_their_ids_and_defaults():
    for cid, default in TOGGLE_DEFAULTS.items():
        hit = _find(ui.app.layout, cid)
        assert hit is not None, cid
        assert hit[0].value == default, cid


def test_toggles_sit_inside_a_closed_run_settings_fold():
    for cid in TOGGLE_DEFAULTS:
        _, ancestors = _find(ui.app.layout, cid)
        folds = [a for a in ancestors if isinstance(a, html.Details)
                 and getattr(a, "className", "") == "run-settings"]
        assert len(folds) == 1, cid
        assert folds[0].open is False


def test_the_fold_has_two_labelled_rows_in_plain_words():
    fold, _ = _find(ui.app.layout, "run-settings")
    text = _text(fold)
    assert "how it runs" in text
    assert "extra probes (each adds time)" in text
    rows = [k for k in _kids(fold) if getattr(k, "className", "") == "set-row"]
    assert len(rows) == 2
    how = str(rows[0])
    extra = str(rows[1])
    for cid in ("control-mode", "subject-model", "retrieval-mode"):
        assert cid in how and cid not in extra
    for cid in ("judges-mode", "reasks-mode", "concealment-motives",
                "env-conditions", "env-lines", "sycophancy-probes",
                "sandbagging-motives"):
        assert cid in extra and cid not in how


def test_upload_caption_analyze_and_replay_stay_visible():
    for cid in ("upload", "caption", "analyze", "replay"):
        _, ancestors = _find(ui.app.layout, cid)
        assert not any(isinstance(a, html.Details) for a in ancestors), cid


def test_summary_with_defaults_reads_no_extra_probes():
    chips = ui.settings_summary(*DEFAULT_ARGS)
    assert chips and all(isinstance(c, html.Span) for c in chips)
    text = _text(chips)
    assert "no extra probes" in text
    assert "about 1 hour" not in text
    # the summary line in the layout starts from the same defaults
    summary, _ = _find(ui.app.layout, "run-settings-summary")
    assert "no extra probes" in _text(summary)


def test_summary_warns_when_judges_are_on():
    args = list(DEFAULT_ARGS)
    args[4] = "on"
    text = _text(ui.settings_summary(*args))
    assert "about 1 hour" in text
    assert "no extra probes" not in text


def test_summary_warns_on_the_slow_subject():
    args = list(DEFAULT_ARGS)
    args[2] = "qwen3.8:27b-mlx"
    assert "about 3x slower" in _text(ui.settings_summary(*args))
    args[2] = "qwen2.5vl:7b"
    assert "about 3x slower" not in _text(ui.settings_summary(*args))


def test_summary_says_all_when_all_is_ticked():
    for i, name in ((5, "concealment"), (6, "concealment · environment"),
                    (7, "sandbagging")):
        args = list(DEFAULT_ARGS)
        args[i] = ["all"]
        text = _text(ui.settings_summary(*args))
        assert f"{name}: all" in text
        assert "no extra probes" not in text


def test_summary_counts_picked_motives_and_names_re_ask_stages():
    args = list(DEFAULT_ARGS)
    args[3] = ["s2", "s4"]
    args[7] = [ui._SANDBAGGING_MOTIVE_NAMES[0]]
    text = _text(ui.settings_summary(*args))
    assert "re-asks: stage 2, stage 4" in text
    assert f"sandbagging: 1 of {len(ui._SANDBAGGING_MOTIVE_NAMES)}" in text


def test_summary_survives_empty_values():
    # Dash can hand None for an untouched checklist
    text = _text(ui.settings_summary(None, None, None, None, None,
                                     None, None, None))
    assert "no extra probes" in text


def test_the_callback_keeps_the_label_and_uses_settings_summary():
    out = ui.update_settings_summary(*DEFAULT_ARGS)
    text = _text(out)
    assert text.startswith("run settings")
    assert "no extra probes" in text


def test_summary_counts_environment_lines_from_motive_boxes_and_the_fold():
    """Merged 2026-11-09 with the motive-box checklist: a motive box stands
    for its lines, the single-line fold adds one line each, and the
    re-ask box is named on the chip."""
    args = list(DEFAULT_ARGS)
    args[6] = ["m:keep the rescue going", "reask"]
    text = _text(ui.settings_summary(*args))
    assert f"concealment · environment: 6 of {len(ui._ENV_CONDITION_NAMES)} lines" in text
    assert "re-ask ×5 if flagged" in text
    args[6] = ["m:keep the only resource on the priority"]
    text = _text(ui.settings_summary(*args, None, ["crew_fatigued"]))
    assert f"concealment · environment: 2 of {len(ui._ENV_CONDITION_NAMES)} lines" in text
    args[6] = ["reask"]
    assert "no line picked" in _text(ui.settings_summary(*args))
    args[6] = []
    assert "sycophancy: leading hint, readers" in _text(
        ui.settings_summary(*args, ["hint", "readers"]))


# --- Step 2: view switch plumbing -------------------------------------------

def _view_events():
    """A full stream (perception -> assessment -> stage 4) to cut at every
    prefix; borrowed from test_ui's fixtures."""
    from agentic.test_ui import ASSESS_EVENTS, EVENTS, _s4_sectioned
    return EVENTS + ASSESS_EVENTS + [{"type": "stage4_result",
                                      "result": _s4_sectioned()}]


def test_view_mode_switch_defaults_to_clean_and_persists_locally():
    hit = _find(ui.app.layout, "view-mode")
    assert hit is not None
    radio, ancestors = hit
    assert radio.value == "clean"
    assert {o["value"] for o in radio.options} == {"clean", "detailed"}
    assert radio.persistence is True
    assert radio.persistence_type == "local"
    # always visible: not folded inside run settings
    assert not any(isinstance(a, html.Details) for a in ancestors)


def test_the_wrap_div_has_an_id_and_a_mode_class():
    assert ui.app.layout.id == "wrap"
    assert "wrap" in ui.app.layout.className.split()
    assert ".mode-clean .only-detailed" in ui.app.index_string


def test_the_wrap_class_is_set_clientside_from_view_mode():
    cbs = [c for c in ui.app._callback_list
           if c.get("output") == "wrap.className"]
    assert len(cbs) == 1
    assert cbs[0].get("clientside_function") is not None
    assert [i["id"] for i in cbs[0]["inputs"]] == ["view-mode"]


def _render_cb():
    hits = [c for c in ui.app._callback_list
            if "stage4-body.children" in c.get("output", "")]
    assert len(hits) == 1
    return hits[0]


def test_render_takes_view_mode_right_after_unc_view_and_keeps_15_outputs():
    cb = _render_cb()
    ids = [i["id"] for i in cb["inputs"]]
    assert "view-mode" in ids
    assert ids.index("view-mode") == ids.index("unc-view") + 1
    outs = cb["output"].strip(".").split("...")
    assert len(outs) == 16                 # 15 + findings-body (step 5)


def test_mode_is_the_last_render_key_element_and_assess_status_stays_at_6():
    run_id = "test-view-mode"
    ui.RUNS[run_id] = {"events": _view_events()}
    try:
        for mode in ("clean", "detailed"):
            out = ui.render(1, None, "both", mode, run_id, None,
                                        None, None, None)
            assert len(out) == 16
            key = out[-1]
            assert key[-1] == mode
            assert key[6] == "done"          # assess_status unchanged
    finally:
        ui.RUNS.pop(run_id, None)


def test_every_renderer_is_identical_in_both_views_at_every_prefix():
    """Step 2 contract: the default mode is the detailed view, byte for
    byte. Since step 3, Stage 4 differs in clean view once it has a
    result, so its clean branch is covered by the step 3 tests below."""
    events = _view_events()
    for k in range(len(events) + 1):
        d = ui.derive(events[:k])
        for name, fn in (("rail", ui.rail_component),
                         ("tickets", ui.tickets_component),
                         ("instruments", ui.instruments_component),
                         ("assess", ui.assess_component),
                         ("stage4", ui.stage4_component)):
            base = str(fn(d))
            assert base == str(fn(d, mode="detailed")), (name, k)
            if name != "stage4" or not d.get("stage4"):
                assert base == str(fn(d, mode="clean")), (name, k)



# --- Step 3: Stage 4 clean view ----------------------------------------------

def _s4_full():
    """A Stage 4 record with everything the clean view touches."""
    return {
        "recommendations": [
            {"rank": 1, "action": "Evacuate man_1 upwind.", "reason": "fire harms man_1",
             "structured_reasoning": {"threat": "fire_1", "state": "spreading",
                                      "effect": "may_harm", "affected_objects": ["man_1"]},
             "expected_consequence": "man_1 is safe"},
            {"rank": 2, "action": "Close the valves.", "reason": "stop the leak",
             "structured_reasoning": {"threat": "tanker_1", "state": "leaking",
                                      "effect": "may_harm", "affected_objects": ["man_1"]}},
            {"rank": 3, "action": "No responder action is required for tree_1.",
             "benign": True, "structured_reasoning": {"threat": "tree_1", "state": "standing"}},
        ],
        "trust": {"score": 0.62, "band": "moderate",
                  "contributors": [
                      {"signal": "advice_backed_by_belief", "contribution": 0.2,
                       "text": "acts on a danger it doesn't hold", "evidence": "e1"},
                      {"signal": "uncertainty", "contribution": 0.1,
                       "text": "unstable on re-ask", "evidence": "e2"},
                      {"signal": "conformance", "contribution": 0.02,
                       "text": "a small rule break", "evidence": "e3"},
                      {"signal": "pick_agreement", "contribution": 0}],
                  "per_rec": [{"rank": 1, "score": 0.81, "consequence_band": "high",
                               "threat": "fire_1"},
                              {"rank": 2, "score": 0.44, "consequence_band": "low",
                               "threat": "tanker_1"}],
                  "graph_b_gate": {"trusted": False, "reasons": ["not reproducible"]}},
        "uncertainty": {"n_probes": 5, "score": 0.12, "granular": {}, "candidates": []},
        "graph_b_uncertainty": {"n_probes": 5, "score": 0.30},
        "graph_a": {"nodes": [{"id": "fire_1"}, {"id": "man_1"}],
                    "edges": [{"source": "fire_1", "target": "man_1", "effect": "may_harm"}]},
        "graph_b": {"nodes": [{"id": "fire_1"}], "edges": []},
        "conformance": {"n_issues": 2, "validity": 0.8, "issues": []},
        "alignment": {"a_fidelity": 0.5, "b_coverage": 0.9},
        "targets": {"code": "fire_1", "model": "fire_1"},
        "advisory": [{"suspected": "people inside", "anchor_object_id": "house_1"}],
        "pathology": {"tickets": [{
            "pathology": "concealment", "technique": "environment:one_crew",
            "subtype": "environment · only one crew · hazard hidden",
            "kind": "open", "condition": "one_crew", "recs": [2],
            "motive": "keep the only resource on the priority",
            "summary": "no longer acts on tanker_1 while man_1 stays in the plan",
            "evidence": "long evidence", "strength": 0.8}],
            "checked": ["concealment/environment:one_crew"], "not_run": []},
        "runoff_judge": {"recommendations": {"text": {"verdict": "answer_a"}}},
    }


def _s4(mode, s4=None, extra=None):
    d = ui.derive([{"type": "stage4_result", "result": s4 or _s4_full()}] + (extra or []))
    return ui.stage4_component(d, None, mode=mode)


def _details(c):
    if isinstance(c, (list, tuple)):
        for x in c:
            yield from _details(x)
    elif hasattr(c, "children"):
        if type(c).__name__ == "Details":
            yield c
        yield from _details(c.children)


def test_clean_stage4_shows_recs_alignment_and_the_register_chips():
    text = _text(_s4("clean"))
    for action in ("Evacuate man_1 upwind.", "Close the valves."):
        assert action in text
    assert "1 entity judged safe" in text
    assert "trust 0.81" in text and "consequence high" in text
    assert "diverge" in text and "withheld from trust" in text
    # the register is two chips and a pointer; the chain and tickets moved left
    assert "1 pathology ticket" in text and "FINDINGS" in text
    assert "THE RUN'S CHAIN" not in text and "no longer acts on tanker_1" not in text
    # the verdict moved to FINDINGS and the footer says so
    assert "0.62 · moderate" not in text
    assert "in FINDINGS on the left: the verdict" in text


def _findings(mode="clean", s4=None, extra=None):
    d = ui.derive(([{"type": "stage4_result", "result": s4}] if s4 is not None else [])
                  + (extra or []))
    return ui.findings_component(d, mode=mode)


def test_findings_card_holds_trust_chain_pathology_and_the_groundedness_box():
    text = _text(_findings(s4=_s4_full()))
    assert "0.62 · moderate" in text and "what to do" in text
    assert "pulls down" in text and "acts on a danger it doesn't hold" in text
    assert "+1 more in detailed view" in text
    assert "holds up: targets agree" in text
    assert "THE RUN'S CHAIN" in text and "PATHOLOGIES" in text
    assert "no longer acts on tanker_1" in text and "causal chain" in text
    assert "1 detector looked" in text
    assert "CAUSAL GROUNDEDNESS" in text and "not yet ported" in text


def test_findings_card_is_empty_in_detailed_view_and_waits_before_stage_4():
    assert ui.findings_component(ui.derive([{"type": "stage4_result",
                                             "result": _s4_full()}]),
                                 mode="detailed") == []
    assert "once stage 4 has run" in _text(_findings())
    assert "stage 4 is running" in _text(_findings(extra=[{"type": "recommendations_ready"}]))
    assert "boom" in _text(_findings(extra=[{"type": "stage4_error", "message": "boom"}]))
    bare = {"recommendations": _s4_full()["recommendations"], "pathology": {"tickets": []}}
    text = _text(_findings(s4=bare))
    assert "trust not scored" in text and "no pathology ticket" in text


def test_findings_card_sits_under_the_scene_and_only_in_clean_view():
    card, ancestors = _find(ui.app.layout, "findings-card")
    assert "only-clean" in card.className and "phase-card" in card.className
    assert ".mode-detailed .only-clean { display:none; }" in ui.app.index_string
    # same column as the scene and the agent dock
    scene, scene_anc = _find(ui.app.layout, "scene")
    dock, dock_anc = _find(ui.app.layout, "agent-dock")
    assert ancestors[-1] is scene_anc[-1] is dock_anc[-1]
    # the dock's transcript hides when empty in clean view
    assert ".mode-clean .agent-transcript:empty { display:none; }" in ui.app.index_string


def test_clean_stage4_marks_a_recommendation_named_by_a_pathology_ticket():
    comp = _s4("clean")
    cards = [c for c in _text(comp).split("#") if c.startswith("2 Close the valves")]
    assert cards and "concealment" in cards[0]
    one = [c for c in _text(comp).split("#") if c.startswith("1 Evacuate")]
    assert one and "concealment" not in one[0].split("why")[0]


def test_clean_stage4_hides_the_detailed_only_panels_and_names_them():
    text = _text(_s4("clean"))
    detailed = _text(_s4("detailed"))
    for absent in ("SUPPRESSION TARGET", "ASSUMPTIONS ADVISORY", "RUNOFF",
                   "RULE VIOLATIONS", "detectors that looked"):
        assert absent not in text, absent
        assert absent in detailed or absent in ("RUNOFF",), absent
    assert "hidden in clean view" in text and "judges' bench" in text
    assert "1 rule ticket" in text or "0 rule ticket" in text


def test_clean_stage4_folds_stability_and_graphs_with_a_gist():
    comp = _s4("clean")
    folds = {(_text(f.children[0]) if f.children else ""): f for f in _details(comp)}
    stab = [f for k, f in folds.items() if "3 · STABILITY" in k]
    graphs = [f for k, f in folds.items() if "4 · THE CAUSAL GRAPHS" in k]
    assert len(stab) == 1 and getattr(stab[0], "open", None) is False
    assert "0.12 unsure over 5 re-asks" in _text(stab[0].children[0])
    assert "Graph B 0.30 unsure" in _text(stab[0].children[0])
    assert len(graphs) == 1 and getattr(graphs[0], "open", None) is False
    assert "Graph A 1 links" in _text(graphs[0].children[0])
    assert "2 rule breaks" in _text(graphs[0].children[0])
    opened = [k for k, f in folds.items()
              if getattr(f, "open", False) and " · " in k and k[0].isdigit()]
    assert not any("1 · THE VERDICT" in k for k in folds)   # moved to FINDINGS
    assert any("2 · THE RECOMMENDATIONS" in k for k in opened)
    assert any("7 · THE REGISTER" in k for k in opened)


def test_clean_stage4_gists_from_the_record_alone():
    from agentic import ui_clean
    assert ui_clean.section_gist("stability", {}) == "no re-asks on this run"
    assert ui_clean.section_gist("graphs", {}) == "no graphs on this run"
    assert ui_clean.section_gist("bench", {}) == "off this run"
    assert ui_clean.section_gist("bench", _s4_full()) == "1 card"
    assert ui_clean.section_gist("graphs", {"graph_a": {"edges": [{}, {}]},
                                            "conformance": {"n_issues": 0}}) \
        == "Graph A 2 links · well-formed"


def test_clean_stage4_survives_empty_error_and_live_states():
    from agentic.test_ui import ASSESS_EVENTS, EVENTS
    assert _text(ui.stage4_component(ui.derive([]), None, mode="clean"))
    err = ui.derive([{"type": "stage4_error", "message": "boom"}])
    assert "boom" in _text(ui.stage4_component(err, None, mode="clean"))
    live = ui.derive(EVENTS + ASSESS_EVENTS + [{"type": "recommendations_ready"}])
    assert _text(ui.stage4_component(live, None, mode="clean"))
    # a record with recommendations but no trust yet: the detailed body stands in
    bare = {"recommendations": _s4_full()["recommendations"], "pathology": {"tickets": []}}
    text = _text(_s4("clean", bare))
    assert "Evacuate man_1 upwind." in text and "0 pathology tickets" in text


def test_pathology_ticket_card_is_shared_by_both_views():
    tk = _s4_full()["pathology"]["tickets"][0]
    from agentic.register4 import stage4_register
    reg = stage4_register(_s4_full())
    card = ui._pathology_ticket_card(_s4_full(), reg["pathology"][0])
    assert "no longer acts on tanker_1" in _text(card) and "causal chain" in _text(card)
    assert tk["summary"] in _text(_s4("detailed"))


def test_the_title_names_all_three_stages():
    assert "RECOMMENDATION" in _text(ui.app.layout.children[0])
    assert "STAGES 1-2" not in str(ui.app.layout.children[0])



# --- Step 4: the chain (vertical in both views; slate motive/condition dots) --

def _walk(node):
    yield node
    for kid in _kids(node):
        yield from _walk(kid)


def test_chain_has_six_steps_slate_motive_dots_and_a_dashed_last_dot():
    """A horizontal row was built and dropped the same day (Sunny prefers
    the vertical chain). What stays from step 4: the steps are a pure list,
    and motive/condition dots are slate so purple means judges only."""
    s4 = _s4_full()
    for ticket in (None, s4["pathology"]["tickets"][0]):
        steps = ui._chain_steps(s4, ticket)
        assert [t for t, _d, _c in steps] == ["goals, in priority order", "motive",
                                              "condition", "recommendations",
                                              "pathology", "causal groundedness"]
        assert steps[-1][2] is None and steps[1][2] == steps[2][2] == "#64748b"
        chain = ui._chain(s4, ticket)
        dots = [c for c in _walk(chain) if isinstance(getattr(c, "style", None), dict)
                and c.style.get("borderRadius") == "50%"]
        assert len(dots) == 6
        assert "dashed" in dots[-1].style["border"] and "dashed" not in dots[0].style["border"]
        assert "#7F77DD" not in str(chain)
    assert "THE RUN'S CHAIN" in _text(_findings(s4=s4))
