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
    assert "no matching advice" in cell.title            # hover says what it was matched to
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
