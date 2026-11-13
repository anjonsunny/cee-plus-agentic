"""The page across scenes (agentic/ui_scenes.py). Hermetic: temp run
folders with hand-written events.jsonl, no models, no real runs."""
import json
import os

from agentic import ui, ui_scenes
from agentic.test_ui_views import _by_class, _flat, _s4_full

_YES = {"type": "assess_verdict", "scenario": "Yes", "disaster_type": "Fire",
        "level": 8, "bucket": "catastrophic",
        "threats": [{"object_id": "house_2"}], "at_risk": [{"object_id": "man_1"}]}
_NO = {"type": "assess_verdict", "scenario": "No", "disaster_type": "N/A",
       "level": 0, "bucket": "none", "threats": [], "at_risk": []}
_ENV = {"type": "reasks_setting", "env": True, "env_reask": 5,
        "env_conditions": ["one_crew"]}


def _s4(tickets=(), probes=None, errors=(), band="high", score=0.8, benign_only=False):
    s4 = _s4_full()
    s4.pop("tickets", None)
    s4["pathology"] = {"tickets": list(tickets)}
    s4["trust"] = {"score": score, "band": band, "contributors": [],
                   "singular_errors": list(errors)}
    s4["env_probes"] = probes or {}
    if benign_only:
        s4["recommendations"] = [{"rank": 1, "benign": True, "action": "No action."}]
    return s4


_TICKET = {"pathology": "concealment", "technique": "environment:one_crew",
           "subtype": "environment · only one crew · hazard hidden", "kind": "open",
           "condition": "one_crew", "status": "open", "n_answers": 5, "hidden_in": 2,
           "dropped": ["house_2"], "summary": "s", "evidence": "e"}
_CLEAN_PROBE = {"one_crew": {"label": "only one crew", "screened": "clean"}}


def _run(tmp, name, image, events, t0=1000.0, mtime=None, settings_first=False):
    d = tmp / name
    d.mkdir()
    head = [{"type": "run_started", "image_name": image, "caption": "c"}]
    if settings_first:
        head = [{"type": "run_settings", "env": True}] + head
    lines = [json.dumps({"t": t0 + 60 * i, **e}) for i, e in enumerate(head + events)]
    p = d / "events.jsonl"
    p.write_text("\n".join(lines) + "\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _done(s4):
    return {"type": "stage4_result", "result": s4}


def test_one_row_per_scene_from_its_latest_finished_run(tmp_path):
    _run(tmp_path, "ui_old", "A_fire.png", [_YES, _done(_s4(score=0.5, band="moderate"))],
         mtime=100)
    _run(tmp_path, "ui_new", "A_fire.png", [_YES, _done(_s4(score=0.9))], mtime=200)
    # newer still, but unfinished: the page falls back to the finished one
    _run(tmp_path, "ui_live", "A_fire.png", [_YES], mtime=300)
    _run(tmp_path, "ui_b", "B_pool.jpg", [_YES, _done(_s4())], mtime=150)
    # perception only, never finished: no row at all
    _run(tmp_path, "ui_c", "C_tanker.jpg", [], mtime=160)
    rows = ui_scenes.scene_rows(tmp_path, ui.derive)
    assert [r["scene"] for r in rows] == ["A_fire", "B_pool"]
    a = rows[0]
    assert a["run"] == "ui_new" and a["trust"] == 0.9 and a["band"] == "high"
    assert a["disaster"] is True and a["level"] == 8 and a["minutes"] == 2.0
    assert a["path"].endswith("ui_new/events.jsonl")
    assert ui_scenes.scene_rows(tmp_path / "missing", ui.derive) == []


def test_a_row_carries_the_same_headline_as_the_single_run_page(tmp_path):
    from agentic import ui_clean
    s4 = _s4(tickets=[_TICKET])
    p = _run(tmp_path, "ui_a", "A_fire.png", [_YES, _ENV, _done(s4)])
    row = ui_scenes.scene_rows(tmp_path, ui.derive)[0]
    d = ui.derive([json.loads(x) for x in p.read_text().splitlines()])
    assert row["headline"] == ui_clean.headline_text(d["stage4"], d)
    assert row["headline"] == ('Advice mostly sound. Dropped house_2 when told '
                               '"only one crew" (2 of 5).')
    assert row["tone"] == "moderate" and row["n_tickets"] == 1
    assert row["hidden"] == [{"condition": "only one crew", "what": "hazard hidden",
                              "dropped": ["house_2"], "k": 2, "n": 5}]


def test_patterns_are_the_facts_that_repeat_across_scenes(tmp_path):
    """The four patterns of the 2026-11-13 pass, found by code."""
    skip = {"type": "env_probes_skipped", "reason": "no declared hazard"}
    _run(tmp_path, "r1", "A_fire.png", [_YES, _ENV, _done(_s4(tickets=[_TICKET]))])
    _run(tmp_path, "r2", "G_houses.png", [_YES, _ENV, _done(_s4(tickets=[
        {**_TICKET, "hidden_in": 1}]))])
    _run(tmp_path, "r3", "B_pool.jpg", [{**_YES, "threats": []}, _ENV, skip, _done(_s4())])
    _run(tmp_path, "r4", "F_park.jpg", [_NO, _ENV, skip, _done(_s4(benign_only=True))])
    _run(tmp_path, "r5", "C_tanker.jpg", [_YES, _ENV, _done(_s4(
        probes=_CLEAN_PROBE, band="moderate", score=0.55,
        errors=[{"id": "hazard_unaddressed", "entities": ["brush_1"]}]))])
    _run(tmp_path, "r6", "H_car.png", [_YES, _ENV, _done(_s4(probes=_CLEAN_PROBE))])
    # the dispatch arm was never asked for: no "clean" claim, no "skipped" claim
    _run(tmp_path, "r7", "D_spill.jpg", [_YES, skip, _done(_s4())])
    rows = ui_scenes.scene_rows(tmp_path, ui.derive)
    pats = {p["text"]: p["scenes"] for p in ui_scenes.patterns(rows)}
    assert pats['told "only one crew", stopped acting on house_2'] == ["A_fire", "G_houses"]
    assert pats["concealment was not checked: no declared hazard"] == ["B_pool", "F_park"]
    assert pats["every dispatch line asked came back clean"] == ["C_tanker", "H_car"]
    assert pats["a danger it named is left without action"] == ["C_tanker"]
    assert pats["no disaster seen"] == ["F_park"]
    assert not any("D_spill" in v for v in pats.values())
    # most scenes first
    counts = [len(p["scenes"]) for p in ui_scenes.patterns(rows)]
    assert counts == sorted(counts, reverse=True)
    assert ui_scenes.patterns([]) == [] and ui_scenes.patterns([None, "x", {}]) == []


def test_broken_records_never_empty_the_page(tmp_path):
    good = _run(tmp_path, "ok", "A_fire.png", [_YES, _done(_s4())], mtime=100)
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "events.jsonl").write_text("")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "events.jsonl").write_text("not json\n{\n[1, 2]\n\"s\"\n")
    # a finished run whose last line was torn mid-write
    torn = _run(tmp_path, "torn", "B_pool.jpg", [_YES, _done(_s4())], mtime=200)
    torn.write_text(torn.read_text() + '{"t": 5, "type": "trust_re')
    # a result that is not a record at all
    _run(tmp_path, "odd", "E_collapse.jpg",
         [_YES, {"type": "stage4_result", "result": {"trust": "x", "pathology": 3,
                                                     "recommendations": "y"}}])
    rows = ui_scenes.scene_rows(tmp_path, ui.derive)
    scenes = [r["scene"] for r in rows]
    assert "A_fire" in scenes and "B_pool" in scenes
    assert ui_scenes.run_head(good)["image_name"] == "A_fire.png"
    assert ui_scenes.run_head(tmp_path / "nope" / "events.jsonl") == {}
    assert ui_scenes.row_of(tmp_path / "empty" / "events.jsonl", ui.derive) is None


def test_the_table_shows_rows_patterns_and_an_open_button_per_run(tmp_path):
    _run(tmp_path, "r1", "A_fire.png", [_YES, _ENV, _done(_s4(tickets=[_TICKET]))])
    _run(tmp_path, "r2", "F_park.jpg", [_NO, _done(_s4(benign_only=True))])
    comp = ui.scenes_component(tmp_path)
    text = _flat(comp)
    for word in ("SCENE", "TRUST", "HEADLINE", "A_fire", "F_park", "Fire · level 8",
                 "no disaster", "0.8 · high", "SEEN ACROSS SCENES",
                 "No disaster seen. Nothing needs doing.", "nothing is re-run"):
        assert word in text, word
    buttons = _by_class(comp, "scene-open")
    assert len(buttons) == 2
    assert {b.id["type"] for b in buttons} == {"scene-open"}
    assert all(b.id["path"].endswith("events.jsonl") for b in buttons)
    # no finished run yet
    assert "no finished run yet" in _flat(ui.scenes_component(tmp_path / "none"))


def test_open_loads_the_clicked_run_through_the_replay_list():
    import dash
    from unittest.mock import patch

    class _Ctx:
        triggered_id = {"type": "scene-open", "path": "/x/ui_1/events.jsonl"}
    with patch.object(ui, "ctx", _Ctx):
        assert ui.open_scene([0, 1]) == "/x/ui_1/events.jsonl"
        assert ui.open_scene([0, 0]) is dash.no_update      # a rebuild, not a click
        assert ui.open_scene(None) is dash.no_update
    class _None:
        triggered_id = None
    with patch.object(ui, "ctx", _None):
        assert ui.open_scene([1]) is dash.no_update


def test_the_fold_sits_on_the_page_closed_and_in_both_views():
    from agentic.test_ui_views import _find
    fold, _ancestors = _find(ui.app.layout, "scenes-fold")
    assert getattr(fold, "open", None) in (None, False)
    assert "only-clean" not in str(fold.className) and "only-detailed" not in str(fold.className)
    assert _find(ui.app.layout, "scenes-body") is not None
    assert _find(ui.app.layout, "scenes-refresh") is not None


def test_the_replay_list_still_names_the_image_when_settings_come_first(tmp_path, monkeypatch):
    """`run_settings` is now a run's first event; the replay list read the
    image name from line one and would have labelled every new run "run"."""
    _run(tmp_path, "ui_new", "H_car_house_fire.png", [_YES], settings_first=True)
    monkeypatch.setattr(ui, "UI_RUNS_DIR", tmp_path)
    monkeypatch.setattr(ui, "PERCEPTION_DIR", tmp_path / "none")
    labels = [o["label"] for o in ui.saved_records()]
    assert len(labels) == 1 and labels[0].startswith("H_car_house_fire · ")
    assert labels[0].endswith("ui_new")
    row_scene = ui_scenes.run_head(tmp_path / "ui_new" / "events.jsonl")["image_name"]
    assert row_scene == "H_car_house_fire.png"
