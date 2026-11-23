"""The progress bar of a live run (agentic/progress4.py). Hermetic."""
import json
import os

import pytest

from agentic import progress4 as p4
from agentic.test_ui_intervene import _by_class, _flat

SET = {"type": "run_settings", "control": "python", "subject": "m", "judges": False}


def _run(t0, marks, settings=SET, done=True):
    """Events of one run: `marks` = [(seconds, type, stage)]."""
    ev = [{"t": t0, **settings}, {"t": t0, "type": "run_started"}]
    ev += [{"t": t0 + s, "type": ty, **({"stage": st} if st else {})} for s, ty, st in marks]
    if done:
        ev.append({"t": t0 + marks[-1][0], "type": p4.DONE_EVENT})
    return ev


REF = [(0, "stage_started", "Perceive"), (50, "stage_done", "Perceive"),
       (50, "stage_started", "assess"), (100, "assess_verdict", ""),
       (100, "stage_started", "recommend"), (300, "recommendations_ready", ""),
       (400, "graph_b_built", ""), (400, "stage_done", "recommend")]


def _save(tmp_path, name, events, mtime):
    d = tmp_path / name
    d.mkdir()
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    os.utime(d / "events.jsonl", (mtime, mtime))


def test_the_bar_is_sized_by_the_time_the_last_same_settings_run_took():
    ref = p4.timeline(_run(1000, REF))
    assert ref["finished"] and ref["total"] == 400
    # 20 s into the recommendation step (which took 200 s of 400 last time)
    live = _run(5000, REF[:5], done=False)
    est = p4.estimate(live, ref, now=5000 + 120)
    assert est["fraction"] == pytest.approx(120 / 400)
    assert est["left"] == pytest.approx(280) and est["elapsed"] == pytest.approx(120)
    assert est["step"] == "writing the recommendations"
    # this run is slower: it reached the same mark later, and is placed by the MARK
    slow = _run(5000, [(s * 2, ty, st) for s, ty, st in REF[:5]], done=False)
    est = p4.estimate(slow, ref, now=5000 + 220)                # 20 s after the mark
    assert est["fraction"] == pytest.approx(120 / 400)
    # a long silent call: the bar waits just before the next mark, never past it
    est = p4.estimate(live, ref, now=5000 + 5000)
    assert 100 / 400 < est["fraction"] < 300 / 400
    assert p4.estimate(_run(5000, REF[:7], done=False), ref, now=99999)["fraction"] <= 0.99


def test_with_nothing_to_time_it_by_only_the_step_and_the_time_are_shown():
    live = _run(5000, REF[:3], done=False)
    est = p4.estimate(live, None, now=5065)
    assert est["fraction"] is None and est["left"] is None
    assert est["elapsed"] == pytest.approx(65) and est["step"] == "assessing the scene"
    comp = p4.component(est)
    words = _flat(_by_class(comp, "run-progress-words"))
    assert "assessing the scene" in words and "1:05 passed" in words
    assert "no earlier run with these settings" in words and "%" not in words
    assert _by_class(comp, "run-progress-track")[0].to_plotly_json()["props"]["data-known"] == "no"
    # with a reference: a percent, and how long is left
    est = p4.estimate(_run(5000, REF[:5], done=False), p4.timeline(_run(0, REF)), now=5120)
    words = _flat(_by_class(p4.component(est, "edited scene: removed man_1"),
                            "run-progress-words"))
    assert words.startswith("30% · edited scene: removed man_1 · writing the")
    assert "about 5 min left" in words and "2:00 passed" in words
    done = p4.estimate(_run(5000, REF), None, done=True)
    assert done["fraction"] == 1.0 and done["step"] == "done" and done["elapsed"] == 400


def test_the_reference_is_the_newest_finished_run_with_the_same_settings(tmp_path):
    other = {**SET, "judges": True}
    _save(tmp_path, "ui_old", _run(100, REF), 1000)
    _save(tmp_path, "ui_newer", _run(200, [(s * 2, t, st) for s, t, st in REF]), 2000)
    _save(tmp_path, "ui_other", _run(300, REF, settings=other), 3000)        # other settings
    _save(tmp_path, "ui_torn", _run(400, REF[:4], done=False), 4000)         # never finished
    (tmp_path / "ui_junk").mkdir()
    (tmp_path / "ui_junk" / "events.jsonl").write_text("{ torn\n[1]\n")
    key = p4.settings_of(_run(0, REF))
    p4._cache.clear()
    ref = p4.reference(tmp_path, key, now=10.0)
    assert ref["total"] == 800                                   # ui_newer, not the unfinished one
    assert p4.reference(tmp_path, key, exclude=("ui_newer",), now=10.0)["total"] == 400
    assert p4.reference(tmp_path, p4.settings_of(_run(0, REF, settings=other)),
                        now=10.0)["total"] == 400
    assert p4.reference(tmp_path, "nothing like it", now=10.0) is None
    assert p4.reference(tmp_path, "", now=10.0) is None
    assert p4.reference(None, key, now=10.0) is None
    assert p4.reference(tmp_path / "missing", key, now=10.0) is None
    # looked at once a minute at most: a newer run is not seen until then
    _save(tmp_path, "ui_newest", _run(500, [(s * 3, t, st) for s, t, st in REF]), 5000)
    assert p4.reference(tmp_path, key, now=20.0)["total"] == 800
    assert p4.reference(tmp_path, key, now=100.0)["total"] == 1200


def test_odd_input_never_breaks_the_bar():
    for bad in (None, "x", [], [None, 3, "e"], [{"t": "soon", "type": 7, "stage": 9}],
                [{"type": "run_started"}]):
        est = p4.estimate(bad, {"marks": "zz", "total": "long"}, now=10.0)
        assert est["fraction"] is None and est["elapsed"] == 0.0
        assert p4.step_label(bad) == "starting" and p4.settings_of(bad) == ""
        assert p4.timeline(bad)["total"] == 0.0
        assert p4.component(est) is not None
    assert p4.component(None) is not None and p4.component({"fraction": True}) is not None
    assert p4.estimate([{"t": 5.0, "type": "x"}], {"marks": {("x", ""): "a"}, "total": 0},
                       now=9.0)["fraction"] is None
    assert p4.clock(None) == "0:00" and p4.clock(3725) == "1:02:05" and p4.clock(-4) == "0:00"
    assert p4.about("x") == "" and p4.about(20) == "under a minute left"
    assert p4.key(None) == ("", "") and p4.key({"type": 3, "stage": None}) == ("", "")


def test_the_page_has_a_bar_above_the_picture_and_one_for_a_running_try(tmp_path, monkeypatch):
    from agentic import ui
    from agentic.test_ui_views import _find
    bar, anc = _find(ui.app.layout, "run-progress")
    scene, scene_anc = _find(ui.app.layout, "scene")
    assert anc[-1] is scene_anc[-1]
    kids = anc[-1].children
    assert kids.index(bar) < kids.index(scene)                    # above the picture
    panel = _find(ui.app.layout, "iv-panel")[0]
    assert _find(panel, "iv-progress") is not None
    # the try's bar sits with the NEW TRY form: right under it, above the tables
    order = [getattr(k, "id", None) for k in panel.children]
    assert order.index("iv-progress") == order.index("iv-controls") + 1
    assert order.index("iv-progress") < order.index("iv-evidence")
    assert order.index("iv-features-fold") < order.index("iv-controls")
    # a live run shows a bar; a finished one, a replay or an unknown run shows none
    monkeypatch.setattr(ui, "UI_RUNS_DIR", tmp_path)
    p4._cache.clear()
    ui.RUNS["live1"] = {"events": _run(1000, REF[:3], done=False), "done": False,
                        "record_name": "ui_live1"}
    try:
        out = ui.run_progress_component("live1", "edited scene: removed man_1")
        assert "removed man_1" in _flat(out) and "assessing the scene" in _flat(out)
        assert ui.run_progress("x", "live1", None) == out or _flat(ui.run_progress(
            0, "live1", None)).startswith("edited") is False
        ui.RUNS["live1"]["done"] = True
        assert ui.run_progress_component("live1") == []
        assert ui.run_progress(0, "live1", None) is ui.dash.no_update    # nothing shown, nothing to clear
        assert ui.run_progress(0, "live1", ["old bar"]) == []            # the bar goes when the run ends
        assert ui.run_progress_component("nope") == [] and ui.run_progress_component(None) == []
        ui.RUNS["live1"] = "broken"
        assert ui.run_progress_component("live1") == []
        assert ui.iv_progress(0, "nope", None) is ui.dash.no_update
    finally:
        ui.RUNS.pop("live1", None)


def test_findings_and_new_try_are_titled_like_the_two_groundedness_panels():
    from agentic import ui, ui_intervene as ui_iv
    from agentic.test_ui_views import _find
    fold, anc = _find(ui.app.layout, "findings-fold")
    assert fold.open is True and any(getattr(a, "id", None) == "findings-card" for a in anc)
    head = fold.children[0]
    assert head.children[0] == "FINDINGS" and head.className == "iv-title"
    assert head.style["fontSize"] == ui_iv.TITLE_STYLE["fontSize"]
    # it folds only its own content: the groundedness panel is beside it, not inside
    assert _find(fold, "findings-body") is not None and _find(fold, "iv-panel") is None
    card, _ = _find(ui.app.layout, "findings-card")
    assert _find(card, "iv-panel") is not None
    new_try = _find(ui.app.layout, "iv-controls")[0].children[0]
    assert new_try.children == "NEW TRY"
    assert new_try.style["fontSize"] == ui_iv.TITLE_STYLE["fontSize"]
    feats = _find(ui.app.layout, "iv-features-fold")[0].children[0]
    assert feats.style["fontSize"] == new_try.style["fontSize"] == head.style["fontSize"]
