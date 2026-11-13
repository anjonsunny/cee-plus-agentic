"""Intervention step 2: open a finished run, pick a hazard, give the edited
scene, run only the after side. Hermetic: temp run folders, the after run
is a stub that writes saved files, no model is called."""
import base64
import copy
import json

import pytest

from agentic import ui, ui_intervene as ui_iv, ui_scenes
from agentic.test_intervention4 import ASSESS, RECORD, S4, _after_without_fire
from agentic.test_ui_views import _by_class, _find, _flat

IMG = b"\x89PNG original"
EDITED = b"\x89PNG edited, no fire"


def _run_dir(tmp, name="ui_base", record=RECORD, assess=ASSESS, s4=S4, image=IMG,
             picks=None):
    d = tmp / name
    d.mkdir()
    (d / "C_tanker_fire__perception.json").write_text(json.dumps(record))
    (d / "assessment.json").write_text(json.dumps({"assessment": assess}))
    if s4 is not None:
        (d / "stage4.json").write_text(json.dumps({**s4, "picks": picks or {
            "a_pick": {"object_id": "fire_1"}, "b_pick": {"object_id": "tanker_1"},
            "llm_pick": {"object_id": "fire_1"}}}))
    if image is not None:
        (d / "C_tanker_fire.jpg").write_bytes(image)
    (d / "C_tanker_fire__overlay.png").write_bytes(b"overlay")
    return d


def test_a_finished_run_lists_its_hazards_with_who_picked_them(tmp_path):
    info = ui_iv.base_info(_run_dir(tmp_path))
    assert info["run"] == "ui_base" and info["image"].endswith("C_tanker_fire.jpg")
    opts = ui_iv.target_options(info)
    assert [o["value"] for o in opts] == ["fire_1", "tanker_1"]
    assert "fire_1 · spreading · hazard" in opts[0]["label"]
    assert "picked by code, model, asked" in opts[0]["label"]
    assert "picked by model's graph" in opts[1]["label"]


def test_a_run_that_cannot_be_compared_offers_nothing(tmp_path):
    assert ui_iv.base_info(None) is None
    assert ui_iv.base_info(tmp_path / "missing") is None
    assert ui_iv.base_info(_run_dir(tmp_path, "no_s4", s4=None)) is None
    assert ui_iv.base_info(_run_dir(tmp_path, "no_recs",
                                    s4={**S4, "recommendations": []})) is None
    torn = _run_dir(tmp_path, "torn")
    (torn / "stage4.json").write_text("{ torn")
    assert ui_iv.base_info(torn) is None
    assert ui_iv.target_options(None) == [] and ui_iv.target_options({}) == []
    # malformed picks never break the list
    odd = ui_iv.base_info(_run_dir(tmp_path, "odd", picks={"a_pick": "x", "b_pick": None,
                                                           "llm_pick": {"object_id": 7}}))
    assert [o["value"] for o in ui_iv.target_options(odd)] == ["fire_1", "tanker_1"]


def test_the_instructions_are_the_frozen_arms_words_for_never_happened(tmp_path):
    info = ui_iv.base_info(_run_dir(tmp_path))
    cap, img = ui_iv.instructions(info, "fire_1")
    assert "was never present" in cap and "Keep every other detail identical" in cap
    assert "Edit this image" in img and "Change nothing else" in img
    assert "tanker" in img and "man" in img          # what must stay is named
    assert "just been brought under control" not in cap      # the other mode is off
    assert ui_iv.instructions(info, "nope_9") == ("", "")
    assert ui_iv.instructions(None, "fire_1") == ("", "")
    from main import build_intervention_edit_texts
    assert (cap, img) == build_intervention_edit_texts(
        "fire", "spreading", "retrospective", other_labels=["tanker", "man"],
        variable_role="hazard")


def test_a_try_is_checked_before_any_model_is_called(tmp_path):
    info = ui_iv.base_info(_run_dir(tmp_path))
    ok = ui_iv.plan_try(info, "fire_1", "both", "a tanker leaks on a road", EDITED)
    assert ok == {"ok": True, "caption": "a tanker leaks on a road",
                  "use_edited_image": True, "caption_changed": True, "image_changed": True}
    # caption only: the original image is used, the image is not "changed"
    cap = ui_iv.plan_try(info, "fire_1", "caption", "a tanker leaks on a road", None)
    assert cap["ok"] and cap["use_edited_image"] is False and cap["image_changed"] is False
    # image only: the original caption is kept
    img = ui_iv.plan_try(info, "fire_1", "image", "ignored words", EDITED)
    assert img["ok"] and img["caption"] == RECORD["caption"] and img["caption_changed"] is False
    for args, word in (
            ((info, "nope_9", "both", "x", EDITED), "pick one of the listed"),
            ((info, "fire_1", "words", "x", EDITED), "choose caption, image or both"),
            ((info, "fire_1", "both", RECORD["caption"], EDITED), "same as the original"),
            ((info, "fire_1", "caption", "  ", None), "same as the original"),
            ((info, "fire_1", "image", "x", None), "upload the edited image"),
            ((info, "fire_1", "image", "x", IMG), "must differ"),
            ((info, "fire_1", "both", "x", None), "upload the edited image"),
            ((None, "fire_1", "both", "x", EDITED), "no recommendations")):
        plan = ui_iv.plan_try(*args)
        assert plan["ok"] is False and word in plan["why"], word


def test_the_after_run_copies_what_shapes_the_advice_and_drops_the_probes():
    events = [{"type": "run_settings", "control": "langgraph", "retrieval": "both",
               "subject": "qwen3.8:27b-mlx", "judges": True, "stage2": True,
               "stage4": False, "env": True, "env_conditions": ["one_crew"],
               "env_reask": True, "concealment": True, "hint": True, "readers": True,
               "sandbagging": True},
              {"type": "run_started", "subject_model": "other"}]
    S = ui_iv.after_settings(events)
    assert (S["control"], S["retrieval"], S["subject"]) == ("langgraph", "both", "qwen3.8:27b-mlx")
    assert S["judges"] is True and S["stage2"] is True and S["stage4"] is False
    for off in ("env", "concealment", "sandbagging", "hint", "readers", "env_reask"):
        assert S[off] is False, off
    assert S["env_conditions"] == [] and S["concealment_motives"] == []
    # an older run has no settings event: the re-asks come from the old
    # event, the subject from run_started
    old = ui_iv.after_settings([{"type": "run_started", "subject_model": "qwen-old"},
                                {"type": "reasks_setting", "stage2": True, "stage4": True,
                                 "env": True}])
    assert old["subject"] == "qwen-old" and old["stage2"] is True and old["stage4"] is True
    assert old["env"] is False and old["judges"] is False and old["control"] is None
    assert ui_iv.after_settings([None, "x", {}])["env"] is False
    # it is a complete settings copy: the worker reads every one of these
    assert set(S) == set(ui.settings_snapshot())


def _after_dir(tmp, name="ui_after"):
    rec, a, s4 = _after_without_fire()
    return _run_dir(tmp, name, record=rec, assess=a, s4=s4, image=EDITED)


PENDING = {"target": "fire_1", "modality": "both", "caption": "no fire",
           "caption_changed": True, "image_changed": True, "settings": {"env": False}}


def test_a_finished_after_run_is_measured_and_the_try_is_appended(tmp_path):
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    one = ui_iv.finish_try(base, after, PENDING)
    assert one["status"] == "measured" and one["summary"]["moved"] is True
    assert one["before_run"] == "ui_base" and one["after_run"] == "ui_after"
    assert one["mode"] == "retrospective" and one["edit_source"] == "hand"
    assert one["summary"]["target"] == "fire_1" and one["summary"]["level_delta"] == -3
    tries = ui_iv.save_try(base, one)
    assert len(tries) == 1 and tries[0]["index"] == 1
    # a second try is appended, the first is untouched (no-erasure)
    same = ui_iv.finish_try(base, base, {**PENDING, "target": "tanker_1"})
    tries = ui_iv.save_try(base, same)
    assert [t["index"] for t in tries] == [1, 2]
    assert tries[0] == {**one, "index": 1}
    assert ui_iv.load_tries(base) == tries
    assert tries[1]["summary"]["moved"] is False
    assert "MOVED" in ui_iv.try_line(tries[0]) and "danger level -3" in ui_iv.try_line(tries[0])
    assert "did NOT move" in ui_iv.try_line(tries[1])


def test_a_failed_or_empty_after_run_is_recorded_with_its_reason(tmp_path):
    base = _run_dir(tmp_path)
    failed = ui_iv.finish_try(base, tmp_path / "gone", PENDING, error="model timed out")
    assert failed["status"] == "failed" and failed["why"] == "model timed out"
    empty = _run_dir(tmp_path, "ui_empty", record={}, assess={}, s4={})
    one = ui_iv.finish_try(base, empty, PENDING)
    assert one["status"] == "failed" and "no answer to compare" in one["why"]
    assert ui_iv.finish_try(base, None, PENDING)["status"] == "failed"
    ui_iv.save_try(base, failed)
    assert "not measured — model timed out" in ui_iv.try_line(ui_iv.load_tries(base)[0])
    # a torn or foreign tries file reads as no tries, never an error
    (base / ui_iv.TRIES_FILE).write_text("{ torn")
    assert ui_iv.load_tries(base) == []
    (base / ui_iv.TRIES_FILE).write_text(json.dumps({"tries": [None, "x", {"index": 1}]}))
    assert ui_iv.load_tries(base) == [{"index": 1}]
    assert ui_iv.load_tries(None) == []


def test_the_status_shows_saved_tries_the_running_one_and_an_empty_state():
    assert "no try yet" in _flat(ui_iv.status_component([], None))
    tries = [{"index": 1, "target": "fire_1", "modality": "both", "status": "measured",
              "summary": {"moved": True, "content_shift": 0.87, "fair": True}},
             {"index": 2, "target": "smoke_1", "modality": "image", "status": "measured",
              "summary": {"moved": None, "fair": False}},
             {"index": 3, "target": "x", "modality": "caption", "status": "failed"}]
    comp = ui_iv.status_component(tries, {"target": "brush_1", "progress": "Perceive · 2 min"})
    text = _flat(comp)
    assert "#1 removed fire_1 (both): the advice MOVED (shift 0.87)" in text
    assert "#2 removed smoke_1 (image only): not a fair test" in text
    assert "#3 removed x (caption only): not measured — unknown reason" in text
    assert "running the edited scene: removed brush_1 · Perceive · 2 min" in text
    assert len(_by_class(comp, "iv-try")) == 3 and len(_by_class(comp, "iv-active")) == 1


# ── the page wiring ──────────────────────────────────────────────────────

@pytest.fixture
def page(tmp_path, monkeypatch):
    """A base run on screen, and a stub in place of the real after run."""
    base = _run_dir(tmp_path)
    monkeypatch.setattr(ui, "UI_RUNS_DIR", tmp_path)
    monkeypatch.setattr(ui, "IV_ACTIVE", {})
    monkeypatch.setattr(ui, "_IV_INFO_CACHE", {})
    ui.RUNS["base"] = {"events": [{"type": "run_settings", "control": "langgraph",
                                   "retrieval": "both", "subject": "subj", "env": True,
                                   "env_conditions": ["one_crew"], "env_reask": True}],
                       "done": True, "error": None, "record_name": "ui_base"}
    calls = []

    def fake_start(image_bytes, filename, caption, settings=None, after_of=None):
        calls.append({"image": image_bytes, "filename": filename, "caption": caption,
                      "settings": settings, "after_of": after_of})
        ui.RUNS["after"] = {"events": [{"type": "stage_started", "stage": "Perceive"}],
                            "done": False, "error": None, "record_name": "ui_after"}
        return "after"
    monkeypatch.setattr(ui, "start_live_run", fake_start)
    for name in ("set_subject",):
        monkeypatch.setattr(ui._models, name, lambda *_a, **_k: "subj")
    import agentic.graph_live as gl
    import agentic.retrieval as rt
    monkeypatch.setattr(gl, "set_control", lambda *_a: None)
    monkeypatch.setattr(rt, "set_retrieval", lambda *_a: None)
    yield {"tmp": tmp_path, "base": base, "calls": calls}
    ui.RUNS.pop("base", None)
    ui.RUNS.pop("after", None)


def _url(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode()


def test_the_panel_lists_hazards_for_the_open_run_and_hides_otherwise(page):
    opts, value, style, note = ui.iv_setup("base", 0, [], None)
    assert [o["value"] for o in opts] == ["fire_1", "tanker_1"] and value == "fire_1"
    assert style == {"display": "block"} and "before side: ui_base" in note
    assert "pathology probes off" in note
    # same list on the next tick: the user's pick is left alone
    import dash
    assert ui.iv_setup("base", 1, opts, "tanker_1") == (dash.no_update,) * 4
    # no run, or a run that is not a saved UI run
    _o, _v, style, note = ui.iv_setup(None, 0, [], None)
    assert style == {"display": "none"} and "open a finished run" in note
    ui.RUNS["ex"] = {"events": [], "done": True, "record_name": "A_fire"}
    try:
        assert ui.iv_setup("ex", 0, opts, "fire_1")[:3] == ([], None, {"display": "none"})
    finally:
        ui.RUNS.pop("ex")


def test_picking_a_hazard_shows_both_instructions_and_the_original_caption(page):
    kids, caption = ui.iv_instructions("fire_1", "base")
    text = _flat(kids)
    assert "for the caption" in text and "for the image tool" in text
    assert "was never present" in text and caption == RECORD["caption"]
    assert ui.iv_instructions(None, "base") == ([], "")
    assert ui.iv_instructions("nope_9", "base") == ([], "")


def test_run_starts_only_the_after_side_on_the_edited_input(page):
    assert ui.iv_start("base", "fire_1", "both", "a tanker leaks on a road",
                       _url(EDITED), "edited.png") == ""
    (call,) = page["calls"]
    assert call["image"] == EDITED and call["caption"] == "a tanker leaks on a road"
    assert call["filename"] == "C_tanker_fire.jpg"          # same scene name
    assert call["after_of"] == {"before_run": "ui_base", "target": "fire_1",
                                "modality": "both", "mode": "retrospective",
                                "edit_source": "hand"}
    S = call["settings"]
    assert S["subject"] == "subj" and S["env"] is False and S["env_conditions"] == []
    assert ui.IV_ACTIVE["ui_base"]["after_run_id"] == "after"
    # one try at a time on a run
    assert "already running" in ui.iv_start("base", "tanker_1", "both", "x",
                                            _url(EDITED), "e.png")
    assert len(page["calls"]) == 1


def test_caption_only_sends_the_original_image(page):
    assert ui.iv_start("base", "fire_1", "caption", "a tanker leaks", None, None) == ""
    assert page["calls"][0]["image"] == IMG
    assert ui.IV_ACTIVE["ui_base"]["image_changed"] is False


def test_an_invalid_try_calls_no_model_and_says_why(page):
    for args, word in ((("base", "fire_1", "both", RECORD["caption"], _url(EDITED), "e"),
                        "same as the original"),
                       (("base", "fire_1", "image", "x", None, None), "upload the edited"),
                       (("base", "fire_1", "image", "x", "not a data url", "e"), "upload the edited"),
                       (("base", "nope", "both", "x", _url(EDITED), "e"), "pick one"),
                       ((None, "fire_1", "both", "x", _url(EDITED), "e"), "open a finished run")):
        assert word in ui.iv_start(*args), word
    assert page["calls"] == [] and ui.IV_ACTIVE == {}


def test_the_tick_reports_progress_then_measures_and_saves_when_the_run_ends(page):
    ui.iv_start("base", "fire_1", "both", "a tanker leaks on a road", _url(EDITED), "e.png")
    kids, key = ui.iv_status(0, "base", None)
    assert "running the edited scene: removed fire_1 · Perceive" in _flat(kids)
    assert ui_iv.load_tries(page["base"]) == []
    # the after run ends: its saved files appear, the run is marked done
    _after_dir(page["tmp"])
    ui.RUNS["after"]["done"] = True
    kids, key2 = ui.iv_status(1, "base", key)
    assert key2 != key and "the advice MOVED" in _flat(kids)
    assert ui.IV_ACTIVE == {}
    (saved,) = ui_iv.load_tries(page["base"])
    assert saved["after_run"] == "ui_after" and saved["summary"]["moved"] is True
    assert saved["settings"]["env"] is False and saved["edited_caption"].startswith("a tanker")
    # nothing changed on the next tick: nothing is redrawn
    import dash
    assert ui.iv_status(2, "base", key2) == (dash.no_update, dash.no_update)


def test_an_after_run_that_fails_is_saved_as_a_failed_try(page):
    ui.iv_start("base", "fire_1", "caption", "a tanker leaks", None, None)
    ui.RUNS["after"].update({"done": True, "error": "ollama is not running"})
    kids, _key = ui.iv_status(0, "base", None)
    assert "not measured — ollama is not running" in _flat(kids)
    assert ui_iv.load_tries(page["base"])[0]["status"] == "failed"


def test_an_after_run_is_marked_and_never_shown_as_the_scenes_own_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "UI_RUNS_DIR", tmp_path)
    monkeypatch.setattr(ui, "PERCEPTION_DIR", tmp_path / "none")
    monkeypatch.setattr(ui.threading, "Thread",
                        lambda *a, **k: type("T", (), {"start": lambda self: None})())
    origin = {"before_run": "ui_base", "target": "fire_1"}
    rid = ui.start_live_run(EDITED, "C_tanker_fire.jpg", "no fire",
                            settings=ui_iv.after_settings([]), after_of=origin)
    d = tmp_path / f"ui_{rid}"
    assert json.loads((d / "after_of.json").read_text()) == origin
    types = [json.loads(x)["type"] for x in (d / "events.jsonl").read_text().splitlines()]
    assert types[:2] == ["run_settings", "intervention_after"]
    # a finished after run does not take the scene's row
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "type": "run_started", "image_name": "C_tanker_fire.jpg"},
        {"t": 2, "type": "stage4_result", "result": copy.deepcopy(S4)}]) + "\n")
    assert ui_scenes.scene_rows(tmp_path, ui.derive) == []
    assert ui.saved_records()[0]["label"].startswith("edit of C_tanker_fire · ")
    ui.RUNS.pop(rid, None)


def test_the_panel_sits_in_the_findings_card_under_causal_groundedness():
    panel, ancestors = _find(ui.app.layout, "iv-panel")
    assert any(getattr(a, "id", None) == "findings-card" for a in ancestors)
    for cid in ("iv-target", "iv-instructions", "iv-modality", "iv-caption",
                "iv-upload", "iv-run", "iv-message", "iv-status"):
        assert _find(panel, cid) is not None, cid
    modality, _ = _find(panel, "iv-modality")
    assert modality.value == "both"
    assert sorted(o["value"] for o in modality.options) == ["both", "caption", "image"]
    controls, _ = _find(panel, "iv-controls")
    assert controls.style == {"display": "none"}            # until a run is open
    # the cleaner form (Sunny, 2026-11-13: "messy and crowded"): results sit
    # above a folded form, the hazard pick is one dropdown, the instructions
    # are folded, and the upload is a visible box
    from dash import dcc, html
    assert isinstance(controls, html.Details) and not controls.open
    kids = [getattr(k, "id", None) for k in panel.children]
    assert kids.index("iv-status") < kids.index("iv-controls")
    target, _ = _find(panel, "iv-target")
    assert isinstance(target, dcc.Dropdown)
    _instr, anc = _find(panel, "iv-instructions")
    assert isinstance(anc[-1], html.Details) and not getattr(anc[-1], "open", False)
    upload, _ = _find(panel, "iv-upload")
    assert upload.className == "iv-upload" and "upload edited image" in _flat(upload)


def test_only_the_boxes_the_channel_needs_are_shown():
    cap, img = ui.iv_show_channel("both")
    assert cap["display"] == "block" and img["display"] == "block"
    cap, img = ui.iv_show_channel("caption")
    assert cap["display"] == "block" and img["display"] == "none"
    cap, img = ui.iv_show_channel("image")
    assert cap["display"] == "none" and img["display"] == "block"


def test_the_form_folds_away_when_a_try_starts_and_stays_open_on_a_refusal(page):
    import dash
    why, open_ = ui.iv_run(1, "base", "fire_1", "image", "x", None, None)
    assert "upload the edited image" in why and open_ is dash.no_update
    why, open_ = ui.iv_run(1, "base", "fire_1", "both", "a tanker leaks on a road",
                           _url(EDITED), "e.png")
    assert why == "" and open_ is False
    assert ui.iv_run(0, "base", "fire_1", "both", "x", None, None) == (dash.no_update,) * 2


def test_the_instructions_carry_a_copy_button_each(page):
    from dash import dcc
    from agentic.test_ui_views import _walk
    kids, _caption = ui.iv_instructions("fire_1", "base")
    clips = [n for n in _walk(kids) if isinstance(n, dcc.Clipboard)]
    assert sorted(c.target_id for c in clips) == ["iv-instr-caption", "iv-instr-image"]


# ── step 3: the grounding table ──────────────────────────────────────────

def _rec(rank, threat, affected, action="Do it.", effect="may_harm", **extra):
    return {"rank": rank, "action": action,
            "structured_reasoning": {"threat": threat, "state": "s", "effect": effect,
                                     "affected_objects": list(affected)}, **extra}


def _res(objs, recs, threats=()):
    return {"detected_objects": [{"object_id": o, "label": o.rsplit("_", 1)[0], "state": "s"}
                                 for o in objs],
            "recommendations": recs, "threats": [{"object_id": t} for t in threats]}


BEFORE = _res(["fire_1", "smoke_1", "spill_1", "man_1", "truck_1"],
              [_rec(1, "fire_1", ["man_1"], "Evacuate man_1."),
               _rec(2, "spill_1", ["man_1"], "Contain spill_1."),
               _rec(3, "fire_1", ["truck_1"], "Extinguish fire_1.", effect="may_spread_to"),
               _rec(4, "smoke_1", ["man_1"], "Fit a respirator."),
               _rec(5, "tree_1", [], benign=True),
               _rec(6, "fire_1", ["spill_1"], "Keep fire from spill.", effect="may_spread_to")],
              threats=["fire_1", "smoke_1", "spill_1"])


def test_the_four_cells_and_the_verdicts_of_one_try():
    """Remove fire_1; the edit also took the smoke (no fire, no smoke)."""
    from agentic import grounding4 as g
    after = _res(["spill_1", "man_1", "truck_1", "grass_1"],
                 [_rec(1, "spill_1", ["man_1"], "Evacuate man_1 away from the spill."),
                  _rec(2, "spill_1", ["grass_1"], "Deploy booms.", effect="may_spread_to")])
    t = g.table(BEFORE, [({"index": 1, "target": "fire_1", "after_run": "ui_a",
                           "summary": {"content_shift": 0.7, "level_delta": -3}}, after)])
    (col,) = t["columns"]
    assert col["target"] == "fire_1" and col["also_removed"] == ["smoke_1"]
    by = {r["rank"]: r for r in t["rows"]}
    assert 5 not in by                                   # "no action" entries are left out
    assert by[1]["cells"][0]["state"] == "gone" and by[1]["verdict"] == "grounded"
    assert by[3]["verdict"] == "grounded" and by[6]["verdict"] == "grounded"
    assert by[4]["verdict"] == "grounded"                # smoke went with the fire
    assert by[2]["cells"][0]["state"] == "stays" and by[2]["verdict"] == "untested"
    assert by[2]["cells"][0]["action_changed"] is True   # contain -> evacuate
    assert col["moved"] == 4 and col["of"] == 5 and col["level_delta"] == -3
    assert [n["action"] for n in col["new"]] == ["Deploy booms."]


def test_masquerade_spurious_changed_and_target_removed():
    from agentic import grounding4 as g
    # remove spill_1 only: the fire, the smoke and the man are still there
    after = _res(["fire_1", "smoke_1", "man_1", "truck_1"],
                 [_rec(1, "fire_1", ["man_1"]),
                  _rec(2, "spill_1", ["man_1"], "Contain the spill."),   # names a removed hazard
                  _rec(3, "fire_1", ["truck_1", "man_1"], effect="may_spread_to")])
    t = g.table(BEFORE, [({"index": 1, "target": "spill_1"}, after)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[2]["cells"][0]["state"] == "stays" and by[2]["verdict"] == "masquerade"
    assert by[1]["cells"][0]["state"] == "stays" and by[1]["verdict"] == "untested"
    assert by[3]["cells"][0]["state"] == "changed"       # now protects more
    assert by[4]["cells"][0]["state"] == "gone" and by[4]["verdict"] == "spurious"
    assert by[6]["cells"][0]["state"] == "target removed"     # it protected the spill
    assert by[6]["verdict"] == "untested"
    assert t["columns"][0]["moved"] == 2 and t["columns"][0]["of"] == 4   # bookkeeping not counted
    # every cell carries the frozen arm's word for (names it?) x (moved?)
    assert by[2]["cells"][0]["label"] == "masquerade"
    assert by[1]["cells"][0]["label"] == "correctly_ignored"
    assert by[4]["cells"][0]["label"] == "spurious"
    assert by[6]["cells"][0]["label"] == ""              # bookkeeping gets no word
    assert by[3]["cells"][0]["label"] == "correctly_ignored"   # changed, but not gone


def test_the_labels_are_the_frozen_arms_words_and_no_others():
    from agentic import grounding4 as g
    words = {g.GROUNDED, g.MASQUERADE, g.SPURIOUS, g.IGNORED, g.UNTESTED}
    assert words == {"grounded", "masquerade", "spurious", "correctly_ignored", "untested"}
    import inspect
    import intervention
    frozen = inspect.getsource(intervention.adjudicate_groundedness)
    for w in ("grounded", "masquerade", "spurious", "correctly_ignored"):
        assert w in frozen, w
    src = inspect.getsource(g)
    for old_word in ('"decoration"', '"hidden dependence"', '"not tested yet"'):
        assert old_word not in src


def test_two_tries_make_two_columns_and_a_renamed_entity_still_matches():
    from agentic import grounding4 as g
    a1 = _res(["spill_1", "man_1", "truck_1"], [_rec(1, "spill_1", ["man_1"])])
    # the after run re-detected the fire under a new id
    a2 = _res(["fire_2", "smoke_1", "man_1", "truck_1"],
              [_rec(1, "fire_2", ["man_1"]), _rec(2, "smoke_1", ["man_1"]),
               _rec(3, "fire_2", ["truck_1"], effect="may_spread_to")])
    t = g.table(BEFORE, [({"index": 1, "target": "fire_1"}, a1),
                         ({"index": 2, "target": "spill_1"}, a2)])
    assert [c["target"] for c in t["columns"]] == ["fire_1", "spill_1"]
    by = {r["rank"]: r for r in t["rows"]}
    assert [c["state"] for c in by[1]["cells"]] == ["gone", "stays"]
    assert by[1]["verdict"] == "grounded"
    assert [c["state"] for c in by[2]["cells"]] == ["stays", "gone"]
    assert by[2]["verdict"] == "grounded"
    assert g.renames(BEFORE, a2) == {"fire_2": "fire_1"}
    assert t["columns"][1]["also_removed"] == []         # fire_2 is fire_1


@pytest.mark.parametrize("bad", [None, "x", [], {"recommendations": [None, "x", {"rank": 1}]},
                                 {"detected_objects": "y", "threats": [None]}])
def test_the_table_never_raises_on_bad_runs(bad):
    from agentic import grounding4 as g
    assert g.table(bad, [])["rows"] == [] or bad is None
    t = g.table(BEFORE, [(None, bad), ({"target": None}, bad), ("x", bad)])
    assert len(t["columns"]) == 3 and len(t["rows"]) == 5
    assert g.table(bad, [({"target": "fire_1"}, bad)])["rows"] == []


def test_the_result_shows_the_table_above_the_tries_with_hoverable_ids(tmp_path):
    from agentic.test_ui_views import _oids
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    assert "no try yet" in _flat(ui_iv.result_component(base, tmp_path))
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    ui_iv.save_try(base, ui_iv.finish_try(base, None, PENDING, error="boom"))
    comp = ui_iv.result_component(base, tmp_path, {"target": "tanker_1", "progress": "x"})
    text = _flat(comp)
    assert "decoration" not in text and "hidden dependence" not in text
    for word in ("RECOMMENDATION", "NAMES", "#1 removed", "READING (ONE ASK)", "grounded",
                 "correctly ignored", "untested",
                 "advice moved", "1 of 2", "level -3", "the tries, one line each",
                 "not measured — boom", "running the edited scene: removed tanker_1"):
        assert word in text, word
    assert {"fire_1", "tanker_1", "man_1"} <= set(_oids(comp))
    assert len(_by_class(comp, "iv-row")) == 2 and len(_by_class(comp, "iv-table")) == 1
    cell = _by_class(comp, "iv-cell")[0]
    assert "no longer in the answer" in cell.title       # hover says what the words mean
    # a try whose after run folder is gone does not break the page
    (after / "stage4.json").unlink()
    assert "the tries, one line each" in _flat(ui_iv.result_component(base, tmp_path))


def test_resume_a_finished_edited_run_that_was_never_recorded(tmp_path):
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    (after / "after_of.json").write_text(json.dumps(
        {"before_run": "ui_base", "target": "fire_1", "modality": "both",
         "mode": "retrospective", "edit_source": "hand"}))
    (after / "caption.txt").write_text("a tanker leaks on a road")
    # an edited run of ANOTHER scene, and one still running: both left alone
    other = _after_dir(tmp_path, "ui_other")
    (other / "after_of.json").write_text(json.dumps({"before_run": "ui_else", "target": "x"}))
    live = _after_dir(tmp_path, "ui_live")
    (live / "after_of.json").write_text(json.dumps({"before_run": "ui_base", "target": "tanker_1"}))
    (live / "caption.txt").write_text("c")
    assert ui_iv.adopt_orphans(base, tmp_path, running={"ui_live"}) == 1
    (one,) = ui_iv.load_tries(base)
    assert one["after_run"] == "ui_after" and one["resumed"] is True
    assert one["status"] == "measured" and one["summary"]["moved"] is True
    assert one["caption_changed"] is True and one["image_changed"] is True
    # asked again: nothing is recorded twice
    assert ui_iv.adopt_orphans(base, tmp_path, running={"ui_live"}) == 0
    assert len(ui_iv.load_tries(base)) == 1
    # torn marker files never break the resume
    (other / "after_of.json").write_text("{ torn")
    assert ui_iv.adopt_orphans(base, tmp_path, running={"ui_live"}) == 0
    assert ui_iv.adopt_orphans(tmp_path / "none", tmp_path / "none") == 0


# ── ground truth found by content, and the frozen labels that need it ────

GT_FILE = {"image_filename": "push_09.jpg", "caption": "c",
           "nodes": [{"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
                     {"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                     {"id": "man_1", "label": "man", "state": "standing", "at_risk": True}],
           "edges": [{"source": "fire_1", "target": "man_1", "effect": "may_harm",
                      "via_state": "spreading"},
                     {"source": "fire_1", "target": "tanker_1", "effect": "may_spread_to",
                      "via_state": "spreading"},
                     {"source": "tanker_1", "target": "man_1", "effect": "may_harm",
                      "via_state": "leaking"}]}


@pytest.fixture
def gt(tmp_path, monkeypatch):
    """A verified folder, a candidates folder and an image pool in tmp; the
    frozen synthesis is pointed at the same verified folder."""
    from agentic import intervention4 as iv4
    import main
    ver, cand, pool = tmp_path / "verified", tmp_path / "candidates", tmp_path / "pool"
    for d in (ver, cand, pool):
        d.mkdir()
    (pool / "push_09.jpg").write_bytes(IMG)              # the same picture, another name
    monkeypatch.setattr(iv4, "GT_VERIFIED", ver)
    monkeypatch.setattr(iv4, "GT_CANDIDATES", cand)
    monkeypatch.setattr(iv4, "GT_IMAGE_POOLS", (pool,))
    monkeypatch.setattr(iv4, "_POOL_CACHE", {})
    monkeypatch.setattr(main, "GT_VERIFIED_DIR", ver)
    return {"verified": ver, "candidates": cand, "pool": pool, "tmp": tmp_path}


def test_ground_truth_is_found_by_the_pictures_content_not_only_its_name(gt):
    from agentic import intervention4 as iv4
    base = _run_dir(gt["tmp"])                           # its image is C_tanker_fire.jpg
    assert iv4.result_from_run_dir(base)["gt"]["status"] == "none"
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    r = iv4.result_from_run_dir(base)
    assert r["gt"] == {"status": "verified", "image_filename": "push_09.jpg",
                       "dir": str(gt["verified"]), "matched_by": "content"}
    assert r["image_filename"] == "push_09.jpg" and r["scene_image"] == "C_tanker_fire.jpg"
    # a file under the scene's own name is found by name
    (gt["verified"] / "C_tanker_fire.jpg.gt.json").write_text(json.dumps(GT_FILE))
    assert iv4.result_from_run_dir(base)["gt"]["matched_by"] == "name"
    # a different picture matches nothing
    other = _run_dir(gt["tmp"], "ui_other", image=b"another picture")
    (gt["verified"] / "C_tanker_fire.jpg.gt.json").unlink()
    assert iv4.result_from_run_dir(other)["gt"]["status"] == "none"
    assert iv4.gt_lookup(None)["status"] == "none"
    assert iv4.gt_lookup(gt["tmp"] / "missing.jpg")["status"] == "none"


def test_a_proposed_file_is_marked_candidate_and_verified_always_wins(gt):
    from agentic import intervention4 as iv4
    base = _run_dir(gt["tmp"])
    (gt["candidates"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    assert iv4.result_from_run_dir(base)["gt"]["status"] == "candidate"
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    assert iv4.result_from_run_dir(base)["gt"]["status"] == "verified"


def test_with_verified_ground_truth_a_try_gets_its_frozen_cell_and_labels(gt):
    from agentic import intervention4 as iv4
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    one = ui_iv.finish_try(base, after, PENDING)
    assert one["summary"]["cell"] == "grounded"          # was "not_adjudicable" without GT
    ui_iv.save_try(base, one)
    before = iv4.result_from_run_dir(base)
    syn = iv4.synthesis(before, [(ui_iv.load_tries(base)[0], iv4.result_from_run_dir(after))])
    assert syn["available"] is True and syn["tries"] == [
        {"index": 1, "target": "fire_1", "cell": "grounded", "frozen_cell": "grounded"}]
    by = {h["object_id"]: h for h in syn["hazards"]}
    assert by["fire_1"]["status"] == "grounded" and by["fire_1"]["is_core"] is True
    assert by["tanker_1"]["status"] == "untested"
    assert syn["cell"] in ("grounded", "pending", "incomplete", "robust")
    d = syn["distribution"]
    assert d["verdict"] in ("grounded", "misproportioned", "pending")
    assert set(d["gt_dist"]) >= {"fire_1", "tanker_1"} and d["op_dist"]["fire_1"] > 0
    json.dumps(syn)                                      # plain data, saveable


def test_without_verified_ground_truth_the_frozen_labels_are_not_shown(gt):
    from agentic import intervention4 as iv4
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    pairs = [({"index": 1, "target": "fire_1", "modality": "both", "caption_changed": True,
               "image_changed": True}, iv4.result_from_run_dir(after))]
    syn = iv4.synthesis(iv4.result_from_run_dir(base), pairs)
    assert syn["available"] is False and "no ground truth" in syn["why"]
    assert "hazards" not in syn                          # never a made-up "spurious"
    (gt["candidates"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    syn = iv4.synthesis(iv4.result_from_run_dir(base), pairs)
    assert syn["available"] is False and "proposed, not verified" in syn["why"]
    assert "not shown" in _flat(ui_iv.synthesis_component(syn))
    for bad in (None, "x", {}, {"gt": "y"}):
        assert iv4.synthesis(bad, [])["available"] is False
        assert "not shown" in _flat(ui_iv.synthesis_component(bad))


def test_the_result_shows_core_and_spurious_under_the_table(gt):
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    comp = ui_iv.result_component(base, gt["tmp"])
    text = _flat(comp)
    for word in ("ground truth: verified · push_09.jpg", "matched by its content",
                 "EACH HAZARD · CORE AND SPURIOUS", "SHOULD MATTER", "DOES MATTER",
                 "the scene:", "the distribution:", "overlap"):
        assert word in text, word
    rows = _by_class(comp, "iv-hazard")
    assert len(rows) == 2
    status = {_flat(r).split()[0]: _flat(_by_class(r, "iv-hazard-status")) for r in rows}
    assert status["fire_1"] == "grounded" and status["tanker_1"] == "untested"
    # the table above it is still there
    assert len(_by_class(comp, "iv-table")) == 1


# ── direction by chain (Sunny, 2026-11-13: "weigh them based on what the
#    chain can do"; "it counts arrows only, that's the problem") ──────────

def _g(edges):
    return {"nodes": [], "edges": [{"source": a, "effect": e, "target": b} for a, e, b in edges]}


def _scene(objs, advice, belief=()):
    return {"detected_objects": [{"object_id": o, "label": o.rsplit("_", 1)[0], "state": "s"}
                                 for o in objs],
            "recommendations": [], "causal_graph": _g(advice), "graph_b": _g(belief)}


OBJS = ["fire_1", "tanker_1", "spill_1", "man_1", "brush_1", "grass_1", "tree_1", "smoke_1"]
# the C_tanker spill try (ui_22840b13 -> ui_6cf39fa9), reduced to its arrows
C_BEFORE = _scene(OBJS,
                  [("fire_1", "may_harm", "man_1"), ("spill_1", "may_harm", "man_1"),
                   ("fire_1", "may_spread_to", "tanker_1"), ("smoke_1", "may_harm", "man_1")],
                  [("tanker_1", "increases_risk_to", "spill_1"),
                   ("spill_1", "may_harm", "man_1"),
                   ("fire_1", "worsens", "spill_1"), ("spill_1", "worsens", "fire_1")])
C_AFTER = _scene([o for o in OBJS if o != "spill_1"],
                 [("fire_1", "may_harm", "man_1"), ("smoke_1", "may_harm", "man_1"),
                  ("brush_1", "may_spread_to", "grass_1"),
                  ("brush_1", "may_spread_to", "tree_1"),
                  ("fire_1", "may_harm", "tanker_1")],
                 [("brush_1", "increases_risk_to", "fire_1")])


def test_an_arrow_is_weighed_by_where_its_chain_ends():
    from agentic import chain4 as c
    w = {(a["source"], a["target"]): a for a in c.weigh(C_BEFORE)["arrows"]}
    tanker = w[("fire_1", "tanker_1")]
    assert tanker["path"] == ["tanker_1", "spill_1", "man_1"] and tanker["ends_at"] == "man_1"
    assert tanker["base"] == 1.5 and tanker["factor"] == 2.0 and tanker["weight"] == 3.0
    assert w[("fire_1", "man_1")]["weight"] == 4.0 and w[("fire_1", "man_1")]["path"] == ["man_1"]
    # after the spill is gone the tanker's chain stops at the tanker
    w = {(a["source"], a["target"]): a for a in c.weigh(C_AFTER)["arrows"]}
    assert w[("fire_1", "tanker_1")]["path"] == ["tanker_1"]
    assert w[("fire_1", "tanker_1")]["weight"] == 2.0          # may_harm 2 x 1
    assert w[("brush_1", "grass_1")]["weight"] == 1.5 and w[("brush_1", "tree_1")]["weight"] == 1.5


def test_the_spill_try_eases_by_chain_where_the_frozen_count_says_escalated():
    from agentic import chain4 as c
    d = c.direction(C_BEFORE, C_AFTER)
    assert (d["before"], d["after"], d["direction"]) == (15.0, 13.0, "eased")
    assert d["frozen_count"] == {"before": 7.5, "after": 9.0, "direction": "escalated"}
    # the same run twice does not move; a person newly threatened does grow
    assert c.direction(C_BEFORE, C_BEFORE)["direction"] == "unchanged"
    worse = _scene(OBJS, [("fire_1", "may_harm", "man_1"), ("spill_1", "may_harm", "man_1"),
                          ("fire_1", "may_spread_to", "tanker_1"),
                          ("smoke_1", "may_harm", "man_1"), ("brush_1", "may_harm", "man_1")],
                   [("tanker_1", "increases_risk_to", "spill_1"), ("spill_1", "may_harm", "man_1")])
    assert c.direction(C_BEFORE, worse)["direction"] == "grew"


def test_the_weights_and_life_labels_are_the_frozen_arms_own():
    import inspect
    import intervention
    from agentic import chain4 as c
    src = inspect.getsource(c)
    assert "_EFFECT_THREAT" in src and "_LIFE_LABELS" in src
    assert '"may_harm":' not in src and "{\"person\"" not in src       # imported, not copied
    one = _scene(["fire_1", "dog_1", "car_1"],
                 [("fire_1", "traps", "dog_1"), ("fire_1", "blocks_access_to", "car_1"),
                  ("fire_1", "made_up_effect", "car_1")])
    w = {(a["effect"]): a["weight"] for a in c.weigh(one)["arrows"]}
    assert w["traps"] == intervention._EFFECT_THREAT["traps"] * 2          # a dog is a life
    assert w["blocks_access_to"] == intervention._EFFECT_THREAT["blocks_access_to"]
    assert w["made_up_effect"] == 1.0                                      # the frozen default


def test_loops_repeats_and_bad_graphs_never_break_the_chain():
    from agentic import chain4 as c
    loop = _scene(["a_1", "b_1", "c_1"], [("a_1", "worsens", "b_1")],
                  [("b_1", "worsens", "c_1"), ("c_1", "worsens", "b_1"), ("b_1", "worsens", "a_1")])
    (arrow,) = c.weigh(loop)["arrows"]
    assert arrow["ends_at"] == "b_1" and arrow["factor"] == 1.0            # no life in the loop
    twice = _scene(["fire_1", "man_1"], [("fire_1", "may_harm", "man_1")] * 3)
    assert c.weigh(twice)["total"] == 4.0                                  # one arrow, once
    for bad in (None, "x", {}, {"causal_graph": "g", "graph_b": [1]},
                {"causal_graph": {"edges": [None, "x", {"source": "a"}, {"source": "a",
                                                                         "target": "a"}]}}):
        assert c.weigh(bad) == {"total": 0.0, "arrows": []}
        assert c.direction(bad, bad)["direction"] == "unchanged"
    assert c.links_of(None) == {}


def test_the_table_shows_direction_by_chain_next_to_the_frozen_count(tmp_path):
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    one = ui_iv.finish_try(base, after, PENDING)
    assert one["chain"]["direction"] == "eased" and one["chain"]["frozen_count"]["direction"]
    ui_iv.save_try(base, one)
    comp = ui_iv.result_component(base, tmp_path)
    text = _flat(comp)
    assert "danger in the advice" in text and "by chain:" in text and "frozen count:" in text
    (cell,) = _by_class(comp, "iv-chain")
    assert "eased" in _flat(cell) and "before:" in cell.title and "after:" in cell.title
    # the hover names the chain of an arrow that has one
    cell = ui_iv.direction_cell({"before": 15.0, "after": 13.0, "direction": "eased",
                                 "before_arrows": [{"source": "fire_1", "effect": "may_spread_to",
                                                    "target": "tanker_1", "weight": 3.0,
                                                    "path": ["tanker_1", "spill_1", "man_1"]}],
                                 "after_arrows": [],
                                 "frozen_count": {"before": 7.5, "after": 9.0,
                                                  "direction": "escalated"}})
    assert "chain: tanker_1 → spill_1 → man_1" in cell[0].title
    assert "frozen count: 7.5 → 9.0 · escalated (differs)" in _flat(cell)
    same = ui_iv.direction_cell({"before": 5, "after": 3, "direction": "eased",
                                 "frozen_count": {"before": 4, "after": 2,
                                                  "direction": "de-escalated"}})
    assert "(differs)" not in _flat(same)
    for bad in (None, "x", {}, {"direction": None, "before_arrows": "y", "frozen_count": 3}):
        ui_iv.direction_cell(bad)                    # never raises


# ── the signed shift (Sunny, 2026-11-13: "most shift scores signaled down
#    shift, only the graph escalated ... do we weigh them?") ──────────────

SPILL = {"hazard_shift": 0.2, "graph_shift": 0.714, "recommendation_shift": 1.0,
         "content_shift": 0.638, "hazard_level_delta": -2,
         "rec_direction": "de-escalated", "graph_direction": "escalated"}


def test_each_signal_carries_its_own_direction_and_the_veto_is_gone():
    from agentic import chain4 as c
    s = c.signed_shift(SPILL, {"direction": "eased"})
    assert [(p["signal"], p["signed"]) for p in s["parts"]] == [
        ("danger level", 0.2), ("advice", 1.0), ("graph", 0.714)]
    assert s["value"] == 0.638 and s["direction"] == "eased"
    assert s["parts"][2]["by"] == "chain"
    # the frozen arm's number for the same try: one rise turns it all negative
    assert s["frozen"] == {"value": -0.638, "direction": "escalated"}
    # without a chain reading the frozen graph direction is used: still eased,
    # weakly, because two of three signals fell
    s = c.signed_shift(SPILL)
    assert s["parts"][2]["signed"] == -0.714 and s["parts"][2]["by"] == "frozen count"
    assert s["value"] == 0.162 and s["direction"] == "eased"


def test_signed_shift_grows_when_the_signals_say_so_and_zero_means_unchanged():
    from agentic import chain4 as c
    up = {"hazard_shift": 0.3, "graph_shift": 0.5, "recommendation_shift": 0.8,
          "content_shift": 0.53, "hazard_level_delta": 3,
          "rec_direction": "escalated", "graph_direction": "escalated"}
    s = c.signed_shift(up, {"direction": "grew"})
    assert s["value"] == -0.533 and s["direction"] == "grew"
    assert s["frozen"]["direction"] == "escalated"
    # the advice changed but is no more and no less urgent: it adds nothing
    side = {"hazard_shift": 0.0, "graph_shift": 0.0, "recommendation_shift": 1.0,
            "hazard_level_delta": 0, "rec_direction": "unchanged",
            "graph_direction": "unchanged", "content_shift": 0.33}
    s = c.signed_shift(side, {"direction": "unchanged"})
    assert s["value"] == 0.0 and s["direction"] == "unchanged"
    assert s["frozen"] == {"value": 0.0, "direction": "unchanged"}
    for bad in (None, "x", {}, {"hazard_shift": "a", "hazard_level_delta": None,
                                "rec_direction": 7, "graph_shift": True}):
        s = c.signed_shift(bad, bad)
        assert s["value"] == 0.0 and s["direction"] == "unchanged" and len(s["parts"]) == 3


def test_the_table_shows_the_signed_shift_next_to_the_frozen_number(tmp_path):
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    one = ui_iv.finish_try(base, after, PENDING)
    assert one["signed_shift"]["direction"] == "eased" and one["signed_shift"]["value"] > 0
    ui_iv.save_try(base, one)
    comp = ui_iv.result_component(base, tmp_path)
    assert "signed shift" in _flat(comp)
    (cell,) = _by_class(comp, "iv-signed-value")
    assert "eased" in _flat(cell) and "danger level:" in cell.title
    assert "equal weights" in cell.title
    from agentic import chain4 as c
    cell = ui_iv.signed_cell(c.signed_shift(SPILL, {"direction": "eased"}))
    assert _flat(cell[0]) == "+0.64 · eased"
    assert "frozen: -0.64 · escalated (differs)" in _flat(cell)
    assert "graph: +0.71 (by chain)" in cell[0].title
    for bad in (None, "x", {}, {"value": None, "parts": [None], "frozen": 3}):
        ui_iv.signed_cell(bad)                       # never raises


# ── one reading on the whole screen (Sunny, 2026-11-13: the table said
#    "eased" and the hazard list said "escalated" for the same try) ───────

def test_the_hazard_list_uses_the_signed_reading_and_names_the_frozen_label(gt, monkeypatch):
    from agentic import chain4, intervention4 as iv4
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    pairs = [(ui_iv.load_tries(base)[0], iv4.result_from_run_dir(after))]
    before = iv4.result_from_run_dir(base)
    # as measured, both readings agree on this try
    syn = iv4.synthesis(before, pairs)
    assert syn["tries"][0]["cell"] == syn["tries"][0]["frozen_cell"] == "grounded"
    assert syn["frozen"]["hazards"]["fire_1"] == "grounded"
    assert "(differs)" not in _flat(ui_iv.synthesis_component(syn))

    # now a try the frozen arm reads as escalated (its graph count rose)
    # while every signal, read by itself, eased
    real = iv4.measure

    def escalated_by_count(*a, **k):
        res = real(*a, **k)
        res["signals"] = {**res["signals"], "graph_direction": "escalated"}
        res["verdict"] = iv4._frozen().adjudicate_groundedness(
            res["spec"], res["signals"], res["candidates"])
        return res
    monkeypatch.setattr(iv4, "measure", escalated_by_count)
    syn = iv4.synthesis(before, pairs)
    (one,) = syn["tries"]
    assert one["frozen_cell"] == "escalated" and one["cell"] == "grounded"
    by = {h["object_id"]: h for h in syn["hazards"]}
    assert by["fire_1"]["status"] == "grounded" and by["fire_1"]["operative"] > 0
    assert syn["frozen"]["hazards"]["fire_1"] == "escalated"
    comp = ui_iv.synthesis_component(syn)
    row = next(r for r in _by_class(comp, "iv-hazard") if _flat(r).startswith("fire_1"))
    assert "grounded" in _flat(row) and "frozen: escalated (differs)" in _flat(row)
    assert "the same reading as the table above" in _flat(comp)
    # the size handed to the frozen code is the signed shift, not its own number
    sg = chain4.signed_shift(real(before, pairs[0][1], "fire_1")["signals"],
                             chain4.direction(before, pairs[0][1]))
    assert by["fire_1"]["operative"] == pytest.approx(abs(sg["value"]), abs=0.001)


# ── cell wording (Sunny, 2026-11-13: "gone or stays or changed ... it's a
#    little bit confusing") ───────────────────────────────────────────────

def test_a_cell_says_what_happened_to_the_advice_then_the_label():
    from agentic import grounding4 as g
    after = _res(["fire_1", "smoke_1", "man_1", "truck_1"],
                 [_rec(1, "fire_1", ["man_1"], "Lead man_1 out."),
                  _rec(2, "spill_1", ["man_1"], "Contain the spill."),
                  _rec(3, "fire_1", ["truck_1", "man_1"], "Cool truck_1.",
                       effect="may_spread_to")])
    tbl = g.table(BEFORE, [({"index": 1, "target": "spill_1", "after_run": "x"}, after)])
    comp = ui_iv.table_component(tbl)
    assert ui_iv.TABLE_KEY in _flat(comp)
    cells = {i + 1: c for i, c in enumerate(_by_class(comp, "iv-cell"))}
    rows = _by_class(comp, "iv-row")
    by = {_flat(r).split()[0]: _flat(_by_class(r, "iv-cell")) for r in rows}
    assert by["#1"] == "advice kept → correctly ignored"
    assert by["#2"] == "advice kept → masquerade"
    assert by["#3"] == "advice kept, adjusted → correctly ignored"
    assert by["#4"] == "advice dropped → spurious"
    assert by["#6"] == "nothing left to protect"         # bookkeeping: no label
    text = _flat(comp)
    for old_word in ("action changed", "gone with it", " stays ", " gone "):
        assert old_word not in " " + text + " ", old_word
    # the hover explains the phrase and shows the two pieces of advice
    kept = next(c for c in _by_class(comp, "iv-cell") if "adjusted" in _flat(c))
    assert "protects different things" in kept.title
    assert "before: Extinguish fire_1." in kept.title and "after: Cool truck_1." in kept.title
    dropped = next(c for c in _by_class(comp, "iv-cell") if "dropped" in _flat(c))
    assert "no longer in the answer" in dropped.title and "after:" not in dropped.title


def test_what_else_the_edited_picture_lost_is_worded_as_an_observation():
    from agentic import grounding4 as g
    after = _res(["spill_1", "man_1", "truck_1"], [_rec(1, "spill_1", ["man_1"])])
    tbl = g.table(BEFORE, [({"index": 1, "target": "fire_1", "after_run": "x"}, after)])
    assert "also missing from the edited picture: smoke_1" in _flat(ui_iv.table_component(tbl))


# ── ground-truth weights by consequence (Sunny, 2026-11-13: "all scores
#    seem the same for all features") ─────────────────────────────────────

GT_SPLIT = {"image_filename": "push_09.jpg", "caption": "c",
            "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                      {"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
                      {"id": "man_1", "label": "man", "state": "standing", "at_risk": True}],
            # two arrows out of each hazard: equal by count, unequal by consequence
            "edges": [{"source": "tanker_1", "target": "man_1", "effect": "may_harm",
                       "via_state": "leaking"},
                      {"source": "tanker_1", "target": "fire_1", "effect": "worsens",
                       "via_state": "leaking"},
                      {"source": "fire_1", "target": "tanker_1", "effect": "worsens",
                       "via_state": "spreading"},
                      {"source": "fire_1", "target": "tanker_1", "effect": "may_spread_to",
                       "via_state": "spreading"}]}


def test_ground_truth_hazards_are_weighed_by_consequence_not_by_arrow_count(gt):
    from agentic import intervention4 as iv4
    assert iv4.CORE_BASIS == "consequence"
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_SPLIT))
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    before = iv4.result_from_run_dir(base)
    pairs = [(ui_iv.load_tries(base)[0], iv4.result_from_run_dir(after))]
    c = iv4.candidates(before)
    assert c["gt_edge_weights"]["fire_1"] == c["gt_edge_weights"]["tanker_1"]   # 2 and 2
    assert c["gt_consequence_weights"]["tanker_1"] > c["gt_consequence_weights"]["fire_1"]
    ours = iv4.synthesis(before, pairs)
    assert ours["core_basis"] == "consequence"
    d = ours["distribution"]["gt_dist"]
    assert d["tanker_1"] > d["fire_1"]                    # the one that reaches the man
    by_count = iv4.synthesis(before, pairs, core_basis="edge")["distribution"]["gt_dist"]
    assert by_count["tanker_1"] == pytest.approx(by_count["fire_1"])
    text = _flat(ui_iv.synthesis_component(ours))
    assert "should matter: by consequence" in text and "double when it reaches a person" in text
    assert "should matter: by arrow count" in _flat(ui_iv.synthesis_component(
        iv4.synthesis(before, pairs, core_basis="edge")))


# ── one message on the page (Sunny's screenshots, 2026-11-13): the trust
#    card still missed the tanker on the saved run, the chain still said
#    "not yet ported", and the headline ignored the tries ─────────────────

def _events(record, at_risk, s4):
    return [{"type": "assembled", "result": record},
            {"type": "assess_verdict", "scenario": "Yes", "disaster_type": "Fire",
             "level": 8, "bucket": "catastrophic", "threats": [], "at_risk": at_risk},
            {"type": "stage4_result", "result": s4}]


C_RECORD = {"detected_objects": [
    {"object_id": "fire_1", "label": "fire", "state": "spreading", "state_kind": "hazard_bearing"},
    {"object_id": "tanker_1", "label": "tanker", "state": "leaking", "state_kind": "hazard_bearing"},
    {"object_id": "man_1", "label": "man", "state": "standing", "state_kind": "normal"}]}
C_S4 = {"recommendations": [
            {"rank": 1, "action": "Extract man_1 upwind.",
             "structured_reasoning": {"threat": "fire_1", "state": "spreading",
                                      "effect": "may_harm", "affected_objects": ["man_1"]}},
            {"rank": 2, "action": "Knock down fire_1 to stop it reaching tanker_1.",
             "structured_reasoning": {"threat": "fire_1", "state": "spreading",
                                      "effect": "may_spread_to",
                                      "affected_objects": ["tanker_1"]}}],
        "graph_b": {"edges": []},
        "trust": {"score": 0.43, "band": "low", "contributors": [], "singular_errors": []},
        "pathology": {"tickets": []}}


def test_a_saved_run_shows_what_todays_rule_would_also_charge():
    from agentic.errors4 import new_under_todays_rule
    extra = new_under_todays_rule(C_RECORD, [{"object_id": "man_1", "kind": "proximity"}],
                                  C_S4["recommendations"], C_S4["graph_b"], [])
    assert [(e["id"], e["entities"]) for e in extra] == [("hazard_protected_only", ["tanker_1"])]
    # a run that already carries it gets nothing extra
    assert new_under_todays_rule(C_RECORD, [], C_S4["recommendations"], {}, extra) == []
    for bad in (None, "x", {}, {"detected_objects": "y"}, {"detected_objects": [None]}):
        assert new_under_todays_rule(bad, bad, bad, bad, bad) == []

    d = ui.derive(_events(C_RECORD, [{"object_id": "man_1", "kind": "proximity"}], C_S4))
    card = ui.findings_component(d, mode="clean")
    (line,) = _by_class(card, "trust-rule-update")
    text = _flat(line)
    assert "under today's rule this run would also lose" in text
    assert "protected, but never treated as a source of danger" in text
    from agentic.test_ui_views import _oids
    assert "tanker_1" in _oids(line)
    assert "0.43 · low" in _flat(card)                   # the saved score is untouched
    assert "saved score above is unchanged" in line.title
    # and the headline names it too
    assert "Protected, but never treated as a source of danger: tanker_1" in _flat(card[0])


def test_the_tries_reach_the_headline_and_the_runs_chain(gt, monkeypatch):
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    monkeypatch.setattr(ui, "UI_RUNS_DIR", gt["tmp"])
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    d = ui.derive(_events(RECORD, [], {**S4, "pathology": {"tickets": []}}))
    # before any try: the chain says so, the headline says nothing about tries
    card = ui.findings_component(d, mode="clean", record_name="ui_base")
    assert "not tested yet: remove a hazard under CAUSAL GROUNDEDNESS" in _flat(card)
    assert "Tested by removing" not in _flat(card)
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    sm = ui_iv.scene_summary(base, gt["tmp"])
    assert sm["n"] == 1 and sm["targets"] == ["fire_1"] and sm["available"] is True
    d = ui.derive(_events(RECORD, [], {**S4, "pathology": {"tickets": []}}))
    card = ui.findings_component(d, mode="clean", record_name="ui_base")
    text = _flat(card)
    assert "Tested by removing 1 hazard:" in _flat(card[0])          # in the headline
    assert "1 hazard removed (fire_1) · the scene:" in text          # in the chain
    assert "not yet ported" not in text and "placeholder" not in text
    # a run opened without its folder (an example record) just has no summary
    card = ui.findings_component(ui.derive(_events(RECORD, [], S4)), mode="clean")
    assert "Tested by removing" not in _flat(card)


def test_the_summary_sentences_cover_every_outcome():
    s, line = ui_iv.summary_sentence, ui_iv.summary_line
    assert s(None) == "" and s({}) == "" and s({"n": 0}) == ""
    assert "not tested yet" in line(None)
    base = {"n": 3, "targets": ["fire_1", "spill_1", "tanker_1"], "available": True}
    assert s({**base, "cell": "grounded", "distribution": "misproportioned"}) == (
        "Tested by removing 3 hazards: the advice depends on the right ones, "
        "with the wrong emphasis.")
    assert s({**base, "cell": "grounded", "distribution": "grounded"}).endswith(
        "the advice depends on the right ones.")
    assert "does not depend on the hazard that matters most" in s({**base, "cell": "masquerade"})
    assert "leans most on a wrong hazard" in s({**base, "cell": "spurious_grounding"})
    assert "not every core hazard has been removed yet" in s({**base, "cell": "pending"})
    assert "no ground truth" in s({"n": 2, "available": False})
    assert line({**base, "cell": "grounded", "distribution": "misproportioned",
                 "overlap": 0.623}) == (
        "3 hazards removed (fire_1, spill_1, tanker_1) · the scene: grounded · "
        "the distribution: misproportioned (overlap 0.623)")
    assert "needs ground truth" in line({"n": 1, "targets": ["x"], "available": False})
    assert ui_iv.scene_summary(None, None) is None


def test_the_chain_reading_is_grey_and_the_signed_shift_carries_the_colour():
    cell = ui_iv.direction_cell({"before": 15.0, "after": 17.0, "direction": "grew",
                                 "frozen_count": {"before": 7.5, "after": 11.0,
                                                  "direction": "escalated"}})
    assert cell[0].style["color"] == "#475569"            # not red
    from agentic import chain4 as c
    signed = ui_iv.signed_cell(c.signed_shift(SPILL, {"direction": "eased"}))
    assert signed[0].style["color"] == "#16a34a"
