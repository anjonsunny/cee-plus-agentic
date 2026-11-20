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
    assert row["headline"] == ('Trust is high. Dropped house_2 when told '
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


# ── trust AND causal groundedness per scene (Sunny, 2026-11-16: "currently
#    we only have trust headline") ───────────────────────────────────────

def _full_run(tmp, name, image, s4, record, assess, mtime, image_bytes=b"img"):
    """A run folder with its events and the three saved files a try reads."""
    p = _run(tmp, name, image, [
        {"type": "assembled", "result": record},
        {**_YES, "threats": [], "at_risk": []}, _done(s4)], mtime=mtime)
    d = p.parent
    (d / f"{image.rsplit('.', 1)[0]}__perception.json").write_text(json.dumps(record))
    (d / "assessment.json").write_text(json.dumps({"assessment": assess}))
    (d / "stage4.json").write_text(json.dumps(s4))
    (d / image).write_bytes(image_bytes)
    return d


def test_a_row_shows_groundedness_from_the_tries_and_prefers_the_run_that_has_them(tmp_path):
    from agentic import ui_intervene as ui_iv
    from agentic.test_intervention4 import ASSESS, RECORD, S4, _after_without_fire
    base = _full_run(tmp_path, "ui_old", "C_tanker.jpg", S4, RECORD, ASSESS, mtime=100)
    rec, a, s4 = _after_without_fire()
    after = _full_run(tmp_path, "ui_after", "C_tanker.jpg", s4, rec, a, mtime=150,
                      image_bytes=b"edited")
    (after / "after_of.json").write_text(json.dumps({"before_run": "ui_old"}))
    pending = {"target": "fire_1", "modality": "both", "caption": "no fire",
               "caption_changed": True, "image_changed": True}
    ui_iv.save_try(base, ui_iv.finish_try(base, after, pending))
    ui_iv.save_try(base, ui_iv.finish_try(base, base, {**pending, "target": "tanker_1"}))
    # a NEWER run of the same scene, with no try
    _full_run(tmp_path, "ui_new", "C_tanker.jpg", S4, RECORD, ASSESS, mtime=300)
    _full_run(tmp_path, "ui_b", "B_pool.jpg", S4, RECORD, ASSESS, mtime=200)
    rows = {r["scene"]: r for r in ui_scenes.scene_rows(tmp_path, ui.derive)}
    c = rows["C_tanker"]
    assert c["run"] == "ui_old"                          # the run with the tries, not the newest
    g = c["grounded"]
    assert g["n"] == 2 and g["targets"] == ["fire_1", "tanker_1"]
    assert g["grounded"] == 1 and g["judged"] == 2
    assert rows["B_pool"]["grounded"] is None
    comp = ui.scenes_component(tmp_path)
    text = _flat(comp)
    for word in ("RECOMMENDATIONS GROUNDED", "DECLARED VS MECHANISTIC", "HAZARDS REMOVED",
                 "not tested"):
        assert word in text, word
    cells = {_flat(r).split()[0]: _flat(_by_class(r, "scene-grounded"))
             for r in _by_class(comp, "scene-row")}
    assert cells["C_tanker"] == "1 of 2" and cells["B_pool"] == "not tested"
    # the headline column stays about trust: groundedness has its own columns
    assert "removed" not in c["headline"] and "depends" not in c["headline"]


def test_the_groundedness_cells_cover_every_state():
    quiet = ui_scenes._grounded_cells(None)
    assert _flat(quiet) == "not tested" and len(quiet) == 3
    one = ui_scenes._grounded_cells({"n": 1, "grounded": 2, "judged": 2, "score": None,
                                     "tested": 1, "declared": 4, "targets": ["fire_1"]})
    assert _flat(one) == "2 of 2 needs a second removed hazard 1 of 4"
    full = ui_scenes._grounded_cells({"n": 3, "grounded": 5, "judged": 5, "score": 0.954,
                                      "reading": "agree", "tested": 3,
                                      "declared": 5, "targets": ["a", "b", "c"]})
    assert _flat(full) == "5 of 5 0.95 · agree 3 of 5"
    assert "a, b, c" in full[2].title
    for bad in ("x", {}, {"n": 2}, {"n": 1, "judged": "z", "score": "q"}):
        assert len(ui_scenes._grounded_cells(bad)) == 3


# ── the replay list: a run, and its edited runs under it (Sunny, 2026-11-20:
#    "simplify the runs... easily recognizable") ──────────────────────────

def _listed(tmp_path, monkeypatch, runs):
    """runs: [(name, image, t, origin or None)] -> the option texts."""
    import json as _json, os as _os
    for name, image, t, origin in runs:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "events.jsonl").write_text(_json.dumps(
            {"t": t, "type": "run_started", "image_name": image}) + "\n")
        _os.utime(d / "events.jsonl", (t, t))
        if origin is not None:
            (d / "after_of.json").write_text(origin if isinstance(origin, str)
                                             else _json.dumps(origin))
    monkeypatch.setattr(ui, "UI_RUNS_DIR", tmp_path)
    monkeypatch.setattr(ui, "PERCEPTION_DIR", tmp_path / "none")
    return ui.saved_records()


def test_edited_runs_sit_under_their_run_and_say_what_was_removed(tmp_path, monkeypatch):
    import json as _json
    opts = _listed(tmp_path, monkeypatch, [
        ("ui_old", "A_fire.png", 1000.0, None),
        ("ui_base", "C_tanker_fire.jpg", 2000.0, None),
        ("ui_e1", "C_tanker_fire.jpg", 3000.0, {"before_run": "ui_base", "target": "fire_1"}),
        ("ui_e2", "C_tanker_fire.jpg", 4000.0, {"before_run": "ui_base", "target": "man_1"}),
        ("ui_lost", "B_pool.jpg", 5000.0, {"before_run": "ui_gone", "target": "pool_1"}),
    ])
    (tmp_path / "ui_base" / "interventions.json").write_text(_json.dumps(
        {"tries": [{"after_run": "ui_e1", "status": "measured", "target": "fire_1"},
                   {"after_run": "ui_e2", "status": "failed"}]}))
    opts = ui.saved_records()
    text = [ui.option_text(o) for o in opts]
    # newest run first; an edited run whose own run is gone is a line of its own
    assert text[0].startswith("edit of B_pool · removed pool_1 · ") and text[0].endswith("ui_lost")
    assert text[1].startswith("C_tanker_fire · ") and " · 1 try · ui_base" in text[1]
    # its edited runs under it, newest first, set in with the corner
    assert text[2].startswith(ui.EDIT_MARK + "removed man_1 · ") and "not measured" in text[2]
    assert text[3].startswith(ui.EDIT_MARK + "removed fire_1 · ") and "not measured" not in text[3]
    assert text[4].startswith("A_fire · ") and "tries" not in text[4] and "try" not in text[4]
    assert [o["value"].split("/")[-2] for o in opts] == [
        "ui_lost", "ui_base", "ui_e2", "ui_e1", "ui_old"]
    # an edited line is found by typing its scene or its id
    assert "C_tanker_fire" in opts[2]["search"] and "ui_e2" in opts[2]["search"]
    assert "edit of" not in text[2]
    # the list does not look "changed" every time it is rebuilt
    import json as _j
    as_browser = _j.loads(_j.dumps(opts, default=lambda c: c.to_plotly_json()))
    monkeypatch.setattr(ui, "ctx", type("C", (), {"triggered_id": "run-id"})())
    assert ui.refresh_replay_options("x", 0, as_browser) is ui.dash.no_update
    assert ui.refresh_replay_options("x", 0, as_browser[:-1]) is not ui.dash.no_update


def test_a_torn_or_odd_origin_file_never_breaks_the_list(tmp_path, monkeypatch):
    opts = _listed(tmp_path, monkeypatch, [
        ("ui_base", "C_tanker_fire.jpg", 2000.0, None),
        ("ui_torn", "C_tanker_fire.jpg", 3000.0, "{ torn"),
        ("ui_list", "C_tanker_fire.jpg", 3100.0, "[1, 2]"),
        ("ui_self", "C_tanker_fire.jpg", 3200.0, {"before_run": "ui_self"}),
        ("ui_bare", "C_tanker_fire.jpg", 3300.0, {"before_run": "ui_base"}),
        ("ui_num", "C_tanker_fire.jpg", 3400.0, {"before_run": 7, "target": 9}),
    ])
    (tmp_path / "ui_base" / "interventions.json").write_text("{ torn")
    opts = ui.saved_records()
    text = [ui.option_text(o) for o in opts]
    assert len(opts) == 6
    assert any(t.startswith(ui.EDIT_MARK + "removed something · ") for t in text)   # no target
    assert sum(1 for t in text if t.startswith("edit of C_tanker_fire")) == 4
    assert ui.option_text(None) == "" and ui.option_text({"label": 3}) == ""
    (tmp_path / "ui_empty").mkdir()
    assert len(ui.saved_records()) == 6                                # a folder with no events
