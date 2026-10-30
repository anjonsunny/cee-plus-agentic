"""UI cleanup (agentic/UI_CLEANUP_PLAN.md). Hermetic: no models, no runs.

Step 1 — the control strip: the eight run toggles fold behind one
"run settings" line whose summary shows the current choices as chips.
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
