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
                "env-conditions", "sandbagging-motives"):
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
    assert len(outs) == 15


def test_mode_is_the_last_render_key_element_and_assess_status_stays_at_6():
    run_id = "test-view-mode"
    ui.RUNS[run_id] = {"events": _view_events()}
    try:
        for mode in ("clean", "detailed"):
            out = ui.render(1, None, "both", mode, run_id, None,
                                        None, None, None)
            assert len(out) == 15
            key = out[-1]
            assert key[-1] == mode
            assert key[6] == "done"          # assess_status unchanged
    finally:
        ui.RUNS.pop(run_id, None)


def test_every_renderer_is_identical_in_both_views_at_every_prefix():
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
            assert base == str(fn(d, mode="clean")), (name, k)
