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
    assert [o["value"] for o in opts] == ["fire_1", "tanker_1", "man_1"]
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
    assert [o["value"] for o in ui_iv.target_options(odd)] == ["fire_1", "tanker_1", "man_1"]


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
    assert ": eased, +" in ui_iv.try_line(tries[0]) and "danger level -3" in ui_iv.try_line(tries[0])
    assert ": unchanged, +0.00" in ui_iv.try_line(tries[1])
    assert "MOVED" not in ui_iv.try_line(tries[0])         # the frozen arm's wording is gone


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
    assert "#1 removed fire_1 (both): the recommendations moved" in text    # an old try, no signed shift
    assert "0.87" not in text                             # the frozen arm's shift is not shown
    with_signed = _flat(ui_iv.status_component(tries, None, {1: {"value": 0.72,
                                                              "direction": "eased"}}))
    assert "#1 removed fire_1 (both): eased, +0.72" in with_signed
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
    assert [o["value"] for o in opts] == ["fire_1", "tanker_1", "man_1"] and value == "fire_1"
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
    kids, caption = ui.iv_instructions("fire_1", None, None, "base")
    text = _flat(kids)
    assert "for the caption" in text and "for the image tool" in text
    assert "was never present" in text and caption == RECORD["caption"]
    assert ui.iv_instructions(None, None, None, "base") == ([], "")
    assert ui.iv_instructions("nope_9", None, None, "base") == ([], "")


def test_run_starts_only_the_after_side_on_the_edited_input(page):
    assert ui.iv_start("base", "fire_1", "both", "a tanker leaks on a road",
                       _url(EDITED), "edited.png") == ""
    (call,) = page["calls"]
    assert call["image"] == EDITED and call["caption"] == "a tanker leaks on a road"
    assert call["filename"] == "C_tanker_fire.jpg"          # same scene name
    assert call["after_of"] == {"before_run": "ui_base", "target": "fire_1",
                                "modality": "both", "kind": "removed",
                                "mode": "retrospective", "edit_source": "hand"}
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
    kids, _proof, key = ui.iv_status(0, "base", None)
    assert "running the edited scene: removed fire_1 · Perceive" in _flat(kids)
    assert ui_iv.load_tries(page["base"]) == []
    # the after run ends: its saved files appear, the run is marked done
    _after_dir(page["tmp"])
    ui.RUNS["after"]["done"] = True
    kids, proof, key2 = ui.iv_status(1, "base", [], key)
    assert key2 != key and len(_by_class(kids, "iv-card")) == 1
    assert ": eased \u00b7 its recommendations dropped, changed 0." in _flat(proof)   # the try line sits in the proof
    # the advice names a hazard its own graph does not: it is offered next
    assert [b.children for b in _by_class(kids, "iv-next")] == ["remove next: tanker_1"]
    assert ui.IV_ACTIVE == {}
    (saved,) = ui_iv.load_tries(page["base"])
    assert saved["after_run"] == "ui_after" and saved["summary"]["moved"] is True
    assert saved["settings"]["env"] is False and saved["edited_caption"].startswith("a tanker")
    # nothing changed on the next tick: nothing is redrawn
    import dash
    assert ui.iv_status(2, "base", [], key2) == (dash.no_update,) * 3
    # flipping the ground-truth switch redraws
    kids, _proof, key3 = ui.iv_status(3, "base", ["gt"], key2)
    assert key3 != key2 and kids is not dash.no_update


def test_an_after_run_that_fails_is_saved_as_a_failed_try(page):
    ui.iv_start("base", "fire_1", "caption", "a tanker leaks", None, None)
    ui.RUNS["after"].update({"done": True, "error": "ollama is not running"})
    kids, _proof, _key = ui.iv_status(0, "base", None)
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
    # the card sits in the features fold, above the folded form
    assert kids.index("iv-features-fold") < kids.index("iv-controls")
    fold, _ = _find(panel, "iv-features-fold")
    assert isinstance(fold, html.Details) and fold.open and _find(fold, "iv-status")
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
    kids, _caption = ui.iv_instructions("fire_1", None, None, "base")
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
    # its target (the spill) was removed, but the fire still has advice:
    # the advice is kept, adjusted (hazard first, 2026-11-16)
    assert by[6]["cells"][0]["state"] == "changed"
    assert by[6]["verdict"] == "untested"
    assert t["columns"][0]["moved"] == 3 and t["columns"][0]["of"] == 5
    # every cell carries the frozen arm's word for (names it?) x (moved?)
    assert by[2]["cells"][0]["label"] == "masquerade"
    assert by[1]["cells"][0]["label"] == "correctly_ignored"
    assert by[4]["cells"][0]["label"] == "spurious"
    assert by[6]["cells"][0]["label"] == "correctly_ignored"
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
                 "unaffected", "untested",
                 "recommendations moved", "1 of 2", "level -3", "the tries · 2",
                 "not measured — boom", "running the edited scene: removed tanker_1"):
        assert word in text, word
    assert {"fire_1", "tanker_1", "man_1"} <= set(_oids(comp))
    assert len(_by_class(comp, "iv-row")) == 2 and len(_by_class(comp, "iv-table")) == 1
    cell = _by_class(comp, "iv-cell")[0]
    # hover says what the words mean: inside the bar picture of the cell (2026-11-25),
    # which replaces the text pop-up
    assert "no longer in the answer" in json.loads(getattr(cell, "data-bars"))["note"]
    assert not cell.title and "no longer in the answer" in getattr(cell, "data-tip")
    # a try whose after run folder is gone does not break the page
    (after / "stage4.json").unlink()
    assert "the tries · 2" in _flat(ui_iv.result_component(base, tmp_path))


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
        {"index": 1, "target": "fire_1", "picked": "fire_1", "cell": "grounded",
         "frozen_cell": "grounded"}]
    by = {h["object_id"]: h for h in syn["hazards"]}
    assert by["fire_1"]["status"] == "grounded" and by["fire_1"]["is_core"] is True
    assert by["tanker_1"]["status"] == "untested"
    assert syn["cell"] in ("grounded", "pending", "incomplete", "robust")
    d = syn["distribution"]
    assert d["verdict"] in ("grounded", "misproportioned", "pending")
    assert set(d["gt_dist"]) >= {"fire_1", "tanker_1"} and d["op_dist"]["fire_1"] > 0
    json.dumps(syn)                                      # plain data, saveable



def test_without_verified_ground_truth_the_ground_truth_labels_are_not_given(gt):
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
    for bad in (None, "x", {}, {"gt": "y"}):
        assert iv4.synthesis(bad, [])["available"] is False



def test_ground_truth_shows_only_when_the_switch_is_on(gt):
    """Sunny, 2026-11-16: "there is no ground truth in deployment ... make it
    a switch"; "if ground truth is available and we turn it on then we will
    see core vs spurious and should matter"."""
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    off = _flat(ui_iv.result_component(base, gt["tmp"]))
    import re
    for word in ("ground truth", "SHOULD MATTER", "spurious"):
        assert word not in off, word
    assert not re.search(r"\bcore\b", off)
    on = ui_iv.result_component(base, gt["tmp"], with_gt=True)
    text = _flat(on)
    for word in ("SHOULD MATTER", "BY GROUND TRUTH", "match with ground truth",
                 "ground truth: verified · push_09.jpg"):
        assert word in text, word
    heads = {h.children: h.title for h in _by_class(on, "iv-features")[0].children[0].children.children}
    assert "core: it should matter and the recommendations depend on it" in heads["BY GROUND TRUTH"]
    assert "spurious: the two disagree" in heads["BY GROUND TRUTH"]
    assert len(_by_class(on, "iv-gt-line")) == 1
    words = {_flat(r).split()[0]: _flat(_by_class(r, "iv-gt-word"))
             for r in _by_class(on, "iv-feature")}
    assert words["fire_1"] == "core"                     # should matter, and the recommendations moved
    assert words["tanker_1"] == "not removed yet"
    # switch on, but this scene has no ground truth: the card says so
    (gt["verified"] / "push_09.jpg.gt.json").unlink()
    text = _flat(ui_iv.result_component(base, gt["tmp"], with_gt=True))
    assert "ground truth: not available" in text and "SHOULD MATTER" not in text


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



def test_the_table_shows_direction_by_chain(tmp_path):
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    one = ui_iv.finish_try(base, after, PENDING)
    assert one["chain"]["direction"] == "eased"
    ui_iv.save_try(base, one)
    comp = ui_iv.result_component(base, tmp_path)
    text = _flat(comp)
    assert "danger in the recommendations" in text and "by chain:" in text
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
    assert len(cell) == 1                                # one line: ours
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



def test_the_table_shows_the_signed_shift(tmp_path):
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    one = ui_iv.finish_try(base, after, PENDING)
    assert one["signed_shift"]["direction"] == "eased" and one["signed_shift"]["value"] > 0
    ui_iv.save_try(base, one)
    comp = ui_iv.result_component(base, tmp_path)
    # the bottom row: how much and which way (the signed number let its
    # parts cancel: C_tanker, the man removed, read "+0.03 eased")
    assert "signed shift" not in _flat(comp)
    (cell,) = _by_class(comp, "iv-signed-value")
    assert _flat(cell).endswith("\u00b7 eased \u00b7 its recommendations dropped")
    assert cell.style["color"] == "#16a34a"
    assert "level:" in cell.title and "dropped" in cell.title
    assert ui_iv.reading_cell({"size": 0.7, "direction": "unchanged", "type": "redirects",
                               "counts": {"re-aimed": 4},
                               "parts": [None, {"name": "graph", "signed": 1.0}]}
                              )[0].children == "0.70 \u00b7 unchanged \u00b7 its recommendations redirected"
    for bad in (None, "x", {}, {"size": "big"}, {"size": 0.2, "way": 3, "counts": "x"}):
        ui_iv.reading_cell(bad)                      # never raises
    from agentic import chain4 as c
    cell = ui_iv.signed_cell(c.signed_shift(SPILL, {"direction": "eased"}))
    assert _flat(cell) == "+0.64 · eased" and len(cell) == 1
    assert "graph: +0.71 (by chain)" in cell[0].title
    for bad in (None, "x", {}, {"value": None, "parts": [None], "frozen": 3}):
        ui_iv.signed_cell(bad)                       # never raises


# ── one reading on the whole screen (Sunny, 2026-11-13: the table said
#    "eased" and the hazard list said "escalated" for the same try) ───────


def test_the_frozen_arms_own_result_is_shown_nowhere(gt, monkeypatch):
    """Sunny, 2026-11-16: "I don't want the frozen arm result to show up ...
    frozen arm off everywhere". The labels under the ground-truth switch are
    still worked out by the frozen CODE, from our signed shift; the frozen
    arm's own reading of a try is kept in the data and never drawn."""
    from agentic import intervention4 as iv4
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    pairs = [(ui_iv.load_tries(base)[0], iv4.result_from_run_dir(after))]
    before = iv4.result_from_run_dir(base)
    real = iv4.measure

    def escalated_by_count(*a, **k):                     # the frozen arm would say "escalated"
        res = real(*a, **k)
        res["signals"] = {**res["signals"], "graph_direction": "escalated"}
        res["verdict"] = iv4._frozen().adjudicate_groundedness(
            res["spec"], res["signals"], res["candidates"])
        return res
    monkeypatch.setattr(iv4, "measure", escalated_by_count)
    syn = iv4.synthesis(before, pairs)
    (one,) = syn["tries"]
    assert one["frozen_cell"] == "escalated" and one["cell"] == "grounded"   # kept as data
    for with_gt in (False, True):
        # (the table's own key explains its colours with the word "escalated";
        #  that is ours, not the frozen arm's result for this try)
        text = _flat(ui_iv.result_component(base, gt["tmp"], with_gt=with_gt)).replace(
            ui_iv.TABLE_KEY, "")
        for word in ("frozen", "(differs)", "escalated"):
            assert word not in text, (word, with_gt)
    from agentic import ui_clean
    import inspect
    assert "frozen:" not in inspect.getsource(ui_iv.signed_cell)
    assert "frozen count" not in inspect.getsource(ui_iv.direction_cell)


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
    assert by["#1"] == "unaffected"                       # the uneventful cell, said quietly
    # compact: a bar with its number, one word, the small reasons
    assert by["#2"] == "kept, though its hazard was removed"      # masquerade is not quiet
    # it now protects one thing more: a small change, by how much (not a flat 0.40)
    assert by["#3"] == "0.17 unaffected 1 of 1 targets kept, 1 new"
    assert by["#4"] == "1.00 dropped, though its hazard and victim are still there"
    assert by["#6"] == "0.50 re-aimed rank 6 \u2192 1"
    labels = {_flat(r).split()[0]: getattr(_by_class(r, "iv-cell")[0], "data-label") for r in rows}
    assert labels == {"#1": "correctly_ignored", "#2": "masquerade", "#3": "correctly_ignored",
                      "#4": "spurious", "#6": "correctly_ignored"}
    # the frozen word for the cell is on hover
    four = _by_class(next(r for r in rows if _flat(r).startswith("#4")), "iv-cell")[0]
    assert "spurious: went when a hazard it does not name was removed" in json.loads(
        getattr(four, "data-bars"))["note"]
    assert "recommendation dropped" not in _flat(comp)            # no long phrases in the cells
    assert "nothing left to protect" not in _flat(comp)
    text = _flat(comp)
    for old_word in ("action changed", "gone with it", " stays ", " gone "):
        assert old_word not in " " + text + " ", old_word
    # the hover explains the phrase and shows the two pieces of advice
    kept = next(c for c in _by_class(comp, "iv-cell") if "targets kept" in _flat(c))
    kb = json.loads(getattr(kept, "data-bars"))             # (in the cell's bar picture, 2026-11-25)
    assert "still has a recommendation, with a different target" in kb["note"]
    assert kb["top"] == ["before: Extinguish fire_1.", "after: Cool truck_1."]
    dropped = next(c for c in _by_class(comp, "iv-cell") if "dropped" in _flat(c))
    db = json.loads(getattr(dropped, "data-bars"))
    assert "no longer in the answer" in db["note"] and db["top"][1] == "after: no such recommendation"


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


def test_a_saved_run_shows_the_trust_it_was_scored_with_nothing_patched_on():
    """Sunny, 2026-11-16: the line "not in this score, rule added after the
    run" was ad hoc. A saved run shows its own trust; a new run includes the
    rule in its score like any other loss."""
    from agentic import errors4, ui_clean
    assert not hasattr(errors4, "new_under_todays_rule")
    d = ui.derive(_events(C_RECORD, [{"object_id": "man_1", "kind": "proximity"}], C_S4))
    card = ui.findings_component(d, mode="clean")
    text = _flat(card)
    assert _by_class(card, "trust-rule-update") == []
    assert "rule added after" not in text and "would be" not in text
    assert "0.43 · low" in text and "rule_update" not in d
    assert "tanker_1" not in ui_clean.headline_text(d["stage4"], d)


def test_the_tries_reach_the_headline_and_the_runs_chain(gt, monkeypatch):
    (gt["verified"] / "push_09.jpg.gt.json").write_text(json.dumps(GT_FILE))
    monkeypatch.setattr(ui, "UI_RUNS_DIR", gt["tmp"])
    base, after = _run_dir(gt["tmp"]), _after_dir(gt["tmp"])
    d = ui.derive(_events(RECORD, [], {**S4, "pathology": {"tickets": []}}))
    # before any try: the chain says so, the headline says nothing about tries
    card = ui.findings_component(d, mode="clean", record_name="ui_base")
    assert "not tested yet: remove a hazard under CAUSAL GROUNDEDNESS" in _flat(card)
    assert "removed so far" not in _flat(card[0])
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    sm = ui_iv.scene_summary(base, gt["tmp"])
    assert sm["n"] == 1 and sm["targets"] == ["fire_1"] and "score" in sm
    d = ui.derive(_events(RECORD, [], {**S4, "pathology": {"tickets": []}}))
    card = ui.findings_component(d, mode="clean", record_name="ui_base")
    text = _flat(card)
    assert "removed" not in _flat(card[0])                # never in the headline: kept apart
    assert "1 hazard removed (fire_1) · no score yet" in text        # in the chain
    assert "not yet ported" not in text and "placeholder" not in text
    # a run opened without its folder (an example record) just has no summary
    card = ui.findings_component(ui.derive(_events(RECORD, [], S4)), mode="clean")
    assert "removed so far" not in _flat(card[0])



def test_the_summary_sentences_cover_every_outcome():
    s, line = ui_iv.summary_sentence, ui_iv.summary_line
    assert s(None) == "" and s({}) == "" and s({"n": 0}) == ""
    assert "not tested yet" in line(None)
    base = {"n": 3, "targets": ["fire_1", "spill_1", "tanker_1"], "score": 0.954,
            "tested": 3, "declared": 5, "provisional": True}
    assert s({**base, "reading": "agree"}) == (
        "Recommendations depend on what it says matters (3 of 5 hazards removed).")
    assert s({**base, "reading": "agree", "provisional": False}) == (
        "Recommendations depend on what it says matters.")
    assert s({**base, "reading": "agree in part"}).startswith(
        "Recommendations partly depend on what it says matters")
    assert s({**base, "reading": "differ"}).startswith(
        "Recommendations do not depend on what it says matters")
    assert s({"n": 1, "reading": "", "score": None}) == "1 hazard removed so far."
    assert line({**base, "reading": "agree"}) == (
        "3 hazards removed (fire_1, spill_1, tanker_1) · declared vs mechanistic: 0.95, "
        "agree · 3 of 5 declared hazards removed")
    assert "no score yet" in line({"n": 1, "targets": ["x"], "score": None})
    assert ui_iv.scene_summary(None, None) is None
    for text in (s({**base, "reading": "agree"}), line(base)):
        assert "ground truth" not in text                # the deployment wording


def test_the_chain_reading_is_grey_and_the_signed_shift_carries_the_colour():
    cell = ui_iv.direction_cell({"before": 15.0, "after": 17.0, "direction": "grew",
                                 "frozen_count": {"before": 7.5, "after": 11.0,
                                                  "direction": "escalated"}})
    assert cell[0].style["color"] == "#475569"            # not red
    from agentic import chain4 as c
    signed = ui_iv.signed_cell(c.signed_shift(SPILL, {"direction": "eased"}))
    assert signed[0].style["color"] == "#16a34a"


# ── declared against mechanistic (Sunny, 2026-11-16: "we will show
#    declarative vs mechanistic. That should be the focus") ───────────────

def _belief(objs, belief, advice=()):
    return {"detected_objects": [{"object_id": o, "label": o.rsplit("_", 1)[0], "state": "s"}
                                 for o in objs],
            "recommendations": [], "causal_graph": _g(advice), "graph_b": _g(belief)}


D_OBJS = ["fire_1", "spill_1", "tanker_1", "brush_1", "man_1", "grass_1"]
D_BEFORE = _belief(D_OBJS, [("fire_1", "may_harm", "man_1"),          # 2 x 2 = 4
                            ("fire_1", "may_spread_to", "grass_1"),   # 1.5
                            ("spill_1", "may_harm", "man_1"),         # 4
                            ("tanker_1", "increases_risk_to", "spill_1"),   # chain -> man: 3
                            ("brush_1", "may_spread_to", "grass_1"),  # 1.5
                            ("man_1", "blocks_access_to", "grass_1")])      # a life: no hazard


def _try(target, value):
    """A saved try whose signed shift is `value` (level, advice, graph agree)."""
    way = "de-escalated" if value > 0 else "escalated" if value < 0 else "unchanged"
    return ({"target": target, "signals": {
        "hazard_shift": abs(value), "graph_shift": abs(value),
        "recommendation_shift": abs(value), "content_shift": abs(value),
        "hazard_level_delta": -1 if value > 0 else 1 if value < 0 else 0,
        "rec_direction": way, "graph_direction": way}}, _AFTER[way])


_AFTER = {"de-escalated": _belief(D_OBJS, [], []),
          "escalated": _belief(D_OBJS, [], [("fire_1", "may_harm", "man_1")]),
          "unchanged": _belief(D_OBJS, [], [])}


def test_declared_is_the_models_own_graph_weighed_by_chain():
    from agentic import declared4 as d
    # the danger the graph loses when the hazard is taken out of it: the
    # spill also loses the tanker's arrow into it (4 + 3)
    assert d.declared(D_BEFORE) == {"fire_1": 5.5, "spill_1": 7.0, "tanker_1": 3.0,
                                    "brush_1": 1.5}
    assert d.declared(D_BEFORE, d.GRAPH_A) == {}          # this fixture's advice graph is empty
    both = {**D_BEFORE, "causal_graph": {"edges": [
        {"source": "spill_1", "effect": "may_harm", "target": "man_1"},
        {"source": "fire_1", "effect": "may_spread_to", "target": "grass_1"}]}}
    assert d.declared(both, d.GRAPH_A) == {"spill_1": 4.0, "fire_1": 1.5}
    assert d.declared(both) == d.declared(D_BEFORE)       # its own graph is read as before
    assert "man_1" not in d.declared(D_BEFORE)            # a person is not a hazard
    for bad in (None, "x", {}, {"graph_b": "g"}, {"graph_b": {"edges": [None, {"source": "a"}]}}):
        assert d.declared(bad) == {}


def test_the_comparison_needs_no_ground_truth_and_grows_with_each_try():
    from agentic import declared4 as d
    none = d.compare(D_BEFORE, [])
    assert none["score"] is None and none["reading"] == "" and none["tested"] == 0
    assert none["declared"] == 4 and none["provisional"] is True
    assert none["says_most"] == "spill_1" and none["does_most"] is None
    assert none["advice_vs_graph"] is None and none["advice_vs_mechanistic"] is None
    assert [r["reading"] for r in none["hazards"]] == [d.UNTESTED] * 4
    # one hazard removed: it is all of both sides, so there is nothing to score yet
    one = d.compare(D_BEFORE, [_try("fire_1", 0.7)])
    assert one["score"] is None and one["tested"] == 1 and one["does_most"] == "fire_1"
    # two removed, in the proportion it declared (5.5 : 7): it agree
    two = d.compare(D_BEFORE, [_try("fire_1", 0.55), _try("spill_1", 0.70)])
    assert two["graph_vs_mechanistic"] == two["score"]
    assert two["score"] == pytest.approx(1.0, abs=0.01) and two["reading"] == "agree"
    assert two["provisional"] is True and two["tested"] == 2
    assert two["not_removed"] == ["tanker_1", "brush_1"]
    assert two["said_more"] == [] and two["done_more"] == []
    import inspect
    src = inspect.getsource(d)
    assert "gt_dir" not in src and "synthesis(" not in src          # nobody's answer key


@pytest.mark.usefixtures("three_part_shift")
def test_said_more_done_more_and_more_alarmed_are_named():
    from agentic import declared4 as d
    # says the fire matters a lot (5.5 against 7), yet the advice depends
    # almost only on the spill
    c = d.compare(D_BEFORE, [_try("fire_1", 0.02), _try("spill_1", 0.98)])
    assert c["said_more"] == ["fire_1"] and c["done_more"] == ["spill_1"]
    assert c["reading"] == "differ" and c["score"] < 0.6
    assert c["says_most"] == "spill_1" and c["does_most"] == "spill_1"
    assert c["does_least"] == "fire_1"
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["fire_1"]["reading"] == d.SAID_MORE and by["spill_1"]["reading"] == d.DONE_MORE
    # removing it made the answer MORE alarmed: it is named, and it still counts
    # as depending on it (how much is one thing, which way another: 2026-11-20)
    c = d.compare(D_BEFORE, [_try("fire_1", -0.5), _try("spill_1", 0.6)])
    assert c["alarmed"] == ["fire_1"]
    fire = {r["object_id"]: r for r in c["hazards"]}["fire_1"]
    assert fire["does"] > 0 and fire["way"] == d.ESCALATES and fire["shift"] > 0
    assert fire["signed"] < 0                                          # the old signed number, kept
    # every declared hazard removed: no longer provisional
    full = d.compare(D_BEFORE, [_try(h, 0.3) for h in ("fire_1", "spill_1", "tanker_1",
                                                       "brush_1")])
    assert full["provisional"] is False and full["tested"] == full["declared"] == 4
    # a hazard removed twice counts once; a removed thing it never declared is listed
    twice = d.compare(D_BEFORE, [_try("fire_1", 0.2), _try("fire_1", 0.6),
                                 _try("smoke_9", 0.4)])
    # (the fixture's graphs are empty, so the graph third has no direction: it is
    # "redirected" and counts in how much, 2026-11-24; all three parts read 0.6)
    assert d.mechanistic(D_BEFORE, [_try("fire_1", 0.2), _try("fire_1", 0.6)]) == {"fire_1": 0.6}
    assert twice["removed"] == 2 and twice["tested"] == 1
    for bad in (None, "x", {}):
        assert d.compare(bad, [(None, bad), ("x", None)])["score"] is None


def test_the_card_has_two_halves_recommendations_and_features(tmp_path):
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    assert _by_class(ui_iv.result_component(base, tmp_path), "iv-card") == []    # no try yet
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    comp = ui_iv.result_component(base, tmp_path)
    (card,) = _by_class(comp, "iv-card")
    assert comp[0] is card                               # the card comes first
    text = _flat(card)
    tiles = _by_class(card, "iv-tile")
    assert [_flat(t) for t in tiles] == [
        "recommendations grounded 1 of 1 · 1 not tested "
        "shift when a named hazard or victim was removed: 1.00 "
        "shift when another one was removed: 0.00",
        "declared vs mechanistic needs a second removed hazard"]
    assert [g.children[0].children for g in _by_class(card, "iv-gap")] == [
        "recommendations vs its own graph: ", "recommendations vs mechanistic: ", "its own graph vs mechanistic: "]
    assert "dropped once the hazard it names is removed" in tiles[0].title
    assert "Agreeing with itself is not the same as being right" in tiles[1].title
    # two declarations and the shift, side by side
    assert "FEATURE DECLARED IN THE RECOMMENDATIONS DECLARED IN ITS OWN GRAPH MECHANISTIC" in text
    assert _by_class(card, "iv-bar-num")                 # every bar carries its number
    heads = {h.children: h.title for h in _by_class(card, "iv-features")[0].children[0].children.children}
    assert "from its own graph of the scene" in heads["DECLARED IN ITS OWN GRAPH"]
    assert "the graph of the recommendations" in heads["DECLARED IN THE RECOMMENDATIONS"]
    assert "danger level, recommendations, graph" in heads["MECHANISTIC"]
    assert text.endswith("one ask per try")
    assert len(text) < 760                               # the explanations are on hover
    import re
    for verdict_word in ("ground truth", "spurious", "SHOULD"):
        assert verdict_word not in text, verdict_word    # no verdict words without the switch
    assert not re.search(r"\bcore\b", text)
    # the proof is in folds, said once; the first is open while there is one try
    folds = _by_class(comp, "iv-evidence")
    assert [_flat(f.children[0]) for f in folds] == [
        "CAUSAL GROUNDEDNESS: RECOMMENDATIONS · 2 recommendations by 1 removed hazard",
        "the tries · 1"]
    (big,) = _by_class(folds[0].children[0], "iv-title")
    assert big.style["fontSize"] == "17px" and big.style["fontWeight"] == "800"
    assert folds[0].open is True and not folds[1].open
    assert len(_by_class(folds[0], "iv-table")) == 1 and len(_by_class(card, "iv-table")) == 0
    assert len(_by_class(comp, "iv-features")) == 1      # features are on the card only
    assert "the scene:" not in _flat(comp) and "the distribution:" not in _flat(comp)
    # a second try folds the table away
    ui_iv.save_try(base, ui_iv.finish_try(base, base, {**PENDING, "target": "tanker_1"}))
    comp = ui_iv.result_component(base, tmp_path)
    assert _by_class(comp, "by-recommendation")[0].open       # a main panel: open, one click folds it


def test_new_advice_is_not_on_the_card_it_stays_in_the_evidence(tmp_path):
    """Sunny, 2026-11-16: "is it too important to focus on?" No: the card is
    about grounded recommendations and declared against mechanistic; the
    trust card already names what the original advice left out; and one ask
    per try is weak evidence. The lists stay under the evidence table."""
    from agentic import grounding4 as g
    before = _res(["fire_1", "spill_1", "brush_1", "man_1"],
                  [_rec(1, "fire_1", ["man_1"]), _rec(2, "spill_1", ["man_1"])],
                  threats=["fire_1", "spill_1", "brush_1"])
    a1 = _res(["spill_1", "brush_1", "man_1"],
              [_rec(1, "spill_1", ["man_1"]), _rec(2, "brush_1", ["man_1"], "Cut a firebreak.")])
    tbl = g.table(before, [({"index": 1, "target": "fire_1"}, a1)])
    card = ui_iv.card_component({"hazards": []}, tbl)
    assert "new advice" not in _flat(card) and _by_class(card, "iv-new-advice") == []
    assert "new recommendations after removing fire_1: • Cut a firebreak." in _flat(
        ui_iv.table_component(tbl))
    assert not hasattr(ui_iv, "new_advice")
    for bad in (None, "x", {}, {"hazards": "y", "score": "z"}, {"hazards": [None, "x", {}]}):
        ui_iv.card_component(bad, {})                         # never raises
        ui_iv.card_component(bad, "t", {"available": True, "distribution": "d",
                                        "gt_core": 7, "gt": None})
        ui_iv.card_component(bad, None, {"available": False})


def test_the_headline_and_the_chain_speak_declared_against_mechanistic(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "UI_RUNS_DIR", tmp_path)
    base, after = _run_dir(tmp_path), _after_dir(tmp_path)
    ui_iv.save_try(base, ui_iv.finish_try(base, after, PENDING))
    ui_iv.save_try(base, ui_iv.finish_try(base, base, {**PENDING, "target": "tanker_1"}))
    sm = ui_iv.scene_summary(base, tmp_path)
    assert sm["n"] == 2 and set(sm) >= {"score", "reading", "tested", "declared", "provisional"}
    assert "available" not in sm and "cell" not in sm        # no ground truth in it
    d = ui.derive(_events(RECORD, [], {**S4, "pathology": {"tickets": []}}))
    card = ui.findings_component(d, mode="clean", record_name="ui_base")
    assert "ground truth" not in _flat(card)


def test_the_ground_truth_switch_is_on_the_page_and_off_by_default():
    panel, _ = _find(ui.app.layout, "iv-panel")
    sw, _ = _find(panel, "iv-gt")
    assert sw.value == [] and [o["value"] for o in sw.options] == ["gt"]
    assert "calibration" in sw.options[0]["label"]
    kids = [getattr(k, "id", None) for k in panel.children]
    # the features fold (the switch, then the card), the action, then the proof
    assert kids.index("iv-features-fold") < kids.index("iv-controls") < kids.index("iv-evidence")
    fold, _ = _find(panel, "iv-features-fold")
    inner = [getattr(k, "id", None) for k in fold.children]
    assert inner.index("iv-gt") < inner.index("iv-status")


# ── features on the card: declared beside mechanistic, in plain words; core
#    and spurious only by ground truth (Sunny, 2026-11-16) ─────────────────

def test_a_feature_is_named_by_what_it_does_with_no_verdict_word():
    from agentic import declared4 as d
    # declared: fire 5.5, spill 7, tanker 3, brush 1.5  (high = at least half the top)
    c = d.compare(D_BEFORE, [_try("fire_1", 0.6), _try("spill_1", 0.5), _try("tanker_1", 0.05),
                             _try("brush_1", 0.5)])
    role = {r["object_id"]: r["role"] for r in c["hazards"]}
    assert role["fire_1"] == d.DRIVES_STATED
    assert role["spill_1"] == d.DRIVES_STATED
    assert role["tanker_1"] == d.MINOR                    # small share, barely moves
    c1 = d.compare(D_BEFORE, [_try("fire_1", 0.05), _try("spill_1", 0.6)])
    assert {r["object_id"]: r["role"] for r in c1["hazards"]}["fire_1"] == d.STATED_ONLY
    assert role["brush_1"] == d.DRIVES_MORE               # declared, with a small share
    # "not stated" only for a hazard that is not in its own graph at all
    c2 = d.compare(D_BEFORE, [_try("fire_1", 0.6), _try("gas_9", 0.5)])
    assert {r["object_id"]: r["role"] for r in c2["hazards"]}["gas_9"] == d.DRIVES_UNSTATED
    c = d.compare(D_BEFORE, [_try("fire_1", 0.6), _try("brush_1", 0.02)])
    role = {r["object_id"]: r["role"] for r in c["hazards"]}
    assert role["brush_1"] == d.MINOR and role["spill_1"] == d.UNTESTED
    c = d.compare(D_BEFORE, [_try("fire_1", -0.4), _try("spill_1", 0.5)])
    assert {r["object_id"]: r["role"] for r in c["hazards"]}["fire_1"] == d.ALARMED
    for word in (d.DRIVES_STATED, d.STATED_ONLY, d.DRIVES_UNSTATED, d.MINOR):
        assert "core" not in word and "spurious" not in word


def test_core_and_spurious_are_judged_by_ground_truth_both_mismatches_spurious():
    """Sunny: "should matter but advice does not is also spurious"."""
    from agentic import declared4 as d
    c = d.compare(D_BEFORE, [_try("fire_1", 0.6), _try("spill_1", 0.5), _try("tanker_1", 0.05),
                             _try("brush_1", 0.5)])
    should = {"fire_1": 0.5, "tanker_1": 0.4, "gas_1": 0.1}
    by = {g["object_id"]: g for g in d.against_ground_truth(c, should, ["fire_1", "tanker_1",
                                                                       "gas_1"])}
    assert by["fire_1"]["word"] == d.CORE and by["fire_1"]["should"] == 0.5
    assert by["tanker_1"]["word"] == d.SPURIOUS_IGNORED   # should matter, advice does not
    assert by["spill_1"]["word"] == d.SPURIOUS_USED       # advice depends, should not
    assert by["brush_1"]["word"] == d.SPURIOUS_USED
    assert by["gas_1"]["word"] == d.SHOULD_UNTESTED       # ground truth names it, model did not
    assert by["gas_1"]["is_core"] is True and by["spill_1"]["is_core"] is False
    # not core and not driving: nothing to say
    c2 = d.compare(D_BEFORE, [_try("fire_1", 0.6), _try("brush_1", 0.02)])
    by = {g["object_id"]: g for g in d.against_ground_truth(c2, {}, ["fire_1"])}
    assert by["brush_1"]["word"] == "" and by["spill_1"]["word"] == ""
    assert d.against_ground_truth({}, {}, None) == []
    assert d.against_ground_truth({"hazards": [None, "x"]}, {"a": None}, ["a"])[0]["word"] == (
        d.SHOULD_UNTESTED)


def test_the_card_shows_each_feature_declared_beside_mechanistic():
    from agentic import declared4 as d
    c = d.compare(D_BEFORE, [_try("fire_1", 0.6), _try("spill_1", 0.05)])
    card = ui_iv.card_component(c, {"rows": [{"verdict": "grounded"}, {"verdict": "untested"}]})
    rows = _by_class(card, "iv-feature")
    assert [_flat(r).split()[0] for r in rows] == ["spill_1", "fire_1", "tanker_1", "brush_1"]
    by = {_flat(r).split()[0]: r for r in rows}
    assert _flat(_by_class(by["fire_1"], "iv-role")) == "eased (2 of 3)"
    assert _flat(_by_class(by["spill_1"], "iv-role")) == "eased (2 of 3)"
    assert _by_class(by["fire_1"], "iv-role")[0].style["color"] == "#16a34a"      # eased: green
    assert "not removed yet" in _flat(by["tanker_1"])    # declared bars, no mechanistic bar
    assert _flat(_by_class(by["tanker_1"], "iv-role")) == ""
    assert "what ALL the parts of the shift say together" in _by_class(by["spill_1"], "iv-role")[0].title
    # two declared bars and one mechanistic bar, each with its number
    assert len(_by_class(by["fire_1"], "iv-bar-num")) == 3
    assert len(_by_class(by["tanker_1"], "iv-bar-num")) == 2
    text = _flat(card)
    assert "recommendations grounded 1 of 1 · 1 not tested" in text
    assert "declared vs mechanistic" in text
    # with ground truth: a should-matter bar and the two words, per feature
    syn = {"available": True, "gt_core": ["fire_1", "spill_1"],
           "gt": {"status": "verified", "image_filename": "x.jpg"},
           "distribution": {"gt_dist": {"fire_1": 0.3, "spill_1": 0.7}, "verdict":
                            "misproportioned", "overlap": 0.3, "coverage": [2, 2]}}
    card = ui_iv.card_component(c, {}, syn)
    by = {_flat(r).split()[0]: _flat(_by_class(r, "iv-gt-word"))
          for r in _by_class(card, "iv-feature")}
    assert by["fire_1"] == "core"
    assert by["spill_1"] == "spurious"                   # short; the reason is on hover
    (cell,) = [c for r in _by_class(card, "iv-feature") if _flat(r).startswith("spill_1")
               for c in _by_class(r, "iv-gt-word")]
    assert cell.title == "it should matter, but the recommendations do not depend on it"
    assert by["tanker_1"] == "" and by["brush_1"] == ""
    tanker = next(r for r in _by_class(card, "iv-feature") if _flat(r).startswith("tanker_1"))
    assert "not in ground truth" in _flat(tanker)         # said, not an empty bar
    (tile,) = _by_class(card, "iv-gt-line")
    assert _flat(tile) == "match with ground truth 0.3 · misproportioned"
    assert "2 of 2 core hazards removed" in tile.title
    assert "SHOULD MATTER" in _flat(card)


# ── the action on the card: "remove next" (Sunny, 2026-11-16: the new try
#    "is kind of in the background") ──────────────────────────────────────

def test_the_card_offers_the_next_hazard_to_remove_and_the_button_opens_the_form():
    from agentic import declared4 as d
    import dash
    from unittest.mock import patch
    c = d.compare(D_BEFORE, [_try("fire_1", 0.6)])
    card = ui_iv.card_component(c, {}, next_id=lambda h: {"type": "iv-next", "target": h})
    (btn,) = _by_class(card, "iv-next")
    assert _flat(btn) == "remove next: spill_1"          # the next one it declared
    assert btn.id == {"type": "iv-next", "target": "spill_1"}
    assert "1 of 4 declared hazards removed" in _flat(_by_class(card, "iv-coverage"))
    # every declared hazard removed: no button
    full = d.compare(D_BEFORE, [_try(h, 0.3) for h in ("fire_1", "spill_1", "tanker_1",
                                                       "brush_1")])
    assert _by_class(ui_iv.card_component(full, {}, next_id=lambda h: h), "iv-next") == []
    # without an id maker (a plain render) there is no button either
    assert _by_class(ui_iv.card_component(c, {}), "iv-next") == []

    class _Ctx:
        triggered_id = {"type": "iv-next", "target": "spill_1"}
    opts = [{"label": "fire", "value": "fire_1"}, {"label": "spill", "value": "spill_1"}]
    with patch.object(ui, "ctx", _Ctx):
        assert ui.iv_next([1], opts) == (True, "spill_1")            # opens, and picks it
        assert ui.iv_next([1], [{"value": "fire_1"}]) == (True, dash.no_update)   # not listed
        assert ui.iv_next([0], opts) == (dash.no_update, dash.no_update)          # a redraw
        assert ui.iv_next(None, None) == (dash.no_update, dash.no_update)

    class _None:
        triggered_id = None
    with patch.object(ui, "ctx", _None):
        assert ui.iv_next([1], opts) == (dash.no_update, dash.no_update)


def test_the_form_and_the_proof_have_their_own_place_on_the_page():
    panel, _ = _find(ui.app.layout, "iv-panel")
    assert _find(panel, "iv-evidence") is not None
    controls, _ = _find(panel, "iv-controls")
    assert controls.children[0].className == "iv-new-try"            # a visible bar
    top, bottom = ui_iv.result_parts(None, None)
    assert "no try yet" in _flat(top) and bottom == []


# ── hazard first: the advice is kept while its hazard still has advice
#    (Sunny, 2026-11-16, C_tanker ui_63f16495) ───────────────────────────

def test_advice_recorded_with_another_target_is_kept_not_dropped():
    """After the fire was removed the model still wrote "isolate the leak at
    tanker_truck_1", now pointing at the spill instead of the man. That read
    "recommendation dropped → spurious". And "contain spill_1 ... away from fire_1"
    read "nothing left to protect" though the edited run still had the
    spill."""
    from agentic import grounding4 as g
    before = _res(["fire_1", "spill_1", "tanker_1", "man_1", "grass_1"],
                  [_rec(1, "fire_1", ["man_1"], "Extract man_1."),
                   _rec(2, "spill_1", ["fire_1"], "Contain spill_1 away from fire_1.",
                        effect="may_spread_to"),
                   _rec(5, "tanker_1", ["man_1"], "Close the valve at tanker_1.")],
                  threats=["fire_1", "spill_1", "tanker_1"])
    after = _res(["spill_1", "tanker_1", "man_1", "grass_1"],
                 [_rec(1, "spill_1", ["man_1"], "Move man_1 away from spill_1."),
                  _rec(2, "spill_1", ["grass_1"], "Contain spill_1 with booms.",
                       effect="may_spread_to"),
                  _rec(3, "tanker_1", ["spill_1"], "Isolate the leak at tanker_1.",
                       effect="worsens")])
    t = g.table(before, [({"index": 1, "target": "fire_1"}, after)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[1]["cells"][0]["state"] == "gone" and by[1]["verdict"] == "grounded"
    assert by[2]["cells"][0]["state"] == "changed" and by[2]["verdict"] == "untested"
    assert by[5]["cells"][0]["state"] == "changed" and by[5]["verdict"] == "untested"
    assert by[5]["cells"][0]["after"]["action"] == "Isolate the leak at tanker_1."
    assert by[2]["cells"][0]["after"]["action"] == "Contain spill_1 with booms."   # same action
    assert not any(r["verdict"] == "spurious" for r in t["rows"])
    # the one piece of after-advice no row took is the new advice
    assert [n["action"] for n in t["columns"][0]["new"]] == ["Move man_1 away from spill_1."]
    comp = ui_iv.table_component(t)
    cells = {_flat(r).split()[0]: _flat(_by_class(r, "iv-cell")) for r in _by_class(comp, "iv-row")}
    assert cells["#5"] == "0.50 re-aimed rank 5 \u2192 3"
    assert cells["#2"] == "0.35 re-aimed"


def test_gone_spurious_and_target_removed_still_exist_when_the_hazard_has_no_advice_left():
    from agentic import grounding4 as g
    before = _res(["fire_1", "smoke_1", "dust_1", "man_1", "shed_1"],
                  [_rec(1, "fire_1", ["man_1"]), _rec(2, "smoke_1", ["man_1"]),
                   _rec(3, "dust_1", ["shed_1"]), _rec(4, "fire_1", ["man_1"], "Close the road.")],
                  threats=["fire_1", "smoke_1", "dust_1"])
    # remove the shed: the fire keeps its advice, the smoke advice vanishes,
    # and the dust advice lost the only thing it pointed at
    after = _res(["fire_1", "smoke_1", "dust_1", "man_1"], [_rec(1, "fire_1", ["man_1"])])
    t = g.table(before, [({"index": 1, "target": "shed_1"}, after)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[1]["cells"][0]["state"] == "stays"
    assert by[4]["cells"][0]["state"] == "changed"        # shares the one piece of fire advice
    assert by[2]["cells"][0]["state"] == "gone" and by[2]["verdict"] == "spurious"
    # the shed is a VICTIM: the advice that protected it was dropped with it,
    # which is evidence now (for a hazard try that cell is only bookkeeping)
    assert by[3]["cells"][0]["state"] == "gone" and by[3]["cells"][0]["label"] == "grounded"
    assert t["columns"][0]["victim"] is True
    assert t["columns"][0]["of"] == 4                     # every row counts in a victim try
    comp = ui_iv.table_component(t)
    cells = {_flat(r).split()[0]: _flat(_by_class(r, "iv-cell")) for r in _by_class(comp, "iv-row")}
    assert cells["#3"] == "1.00 dropped"                  # the try was about it
    assert cells["#2"] == "1.00 dropped, though its hazard and victim are still there"    # the try was not
    # a HAZARD try: advice whose only target was the removed hazard is bookkeeping
    before = _res(["fire_1", "spill_1", "man_1"],
                  [_rec(1, "fire_1", ["man_1"]), _rec(2, "spill_1", ["fire_1"])],
                  threats=["fire_1", "spill_1"])
    after = _res(["spill_1", "man_1"], [])
    t = g.table(before, [({"index": 1, "target": "fire_1"}, after)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[2]["cells"][0]["state"] == "target removed" and by[2]["cells"][0]["label"] == ""
    assert t["columns"][0]["of"] == 1 and t["columns"][0]["victim"] is False
    cells = {_flat(r).split()[0]: _flat(_by_class(r, "iv-cell"))
             for r in _by_class(ui_iv.table_component(t), "iv-row")}
    assert cells["#2"] == "its target was removed"


def test_the_card_says_in_which_direction_declared_and_mechanistic_differ():
    """Sunny, 2026-11-17: "does not do what it says" read as a bad verdict
    with no direction. The card now names the hazards."""
    from agentic import declared4 as d
    c = d.compare(D_BEFORE, [_try("fire_1", 0.02), _try("spill_1", 0.98)])
    assert c["reading"] == "differ"
    assert c["explain"] == ["Declared most and largest shift: spill_1.",
                            "Shift less than declared: fire_1.",
                            "Shift more than declared: spill_1."]
    c = d.compare(D_BEFORE, [_try("fire_1", 0.9), _try("spill_1", 0.1)])
    assert c["explain"][0] == "Declared most: spill_1. Largest shift: fire_1."
    card = ui_iv.card_component(c, {})
    assert _flat(_by_class(card, "iv-explain")).startswith("Declared most: spill_1.")
    assert "what it says" not in _flat(card)
    assert d.compare(D_BEFORE, [])["explain"] == []                      # nothing removed yet
    assert _by_class(ui_iv.card_component(d.compare(D_BEFORE, []), {}), "iv-explain") == []


@pytest.mark.usefixtures("three_part_shift")
def test_the_feature_row_says_one_phrase_coloured_by_the_direction_of_the_shift():
    """Sunny, 2026-11-17: no separate parts on the card. One phrase,
    "shifts emergency responses", green when the combined shift eased and
    red when it escalated; scores only for the two declared and the
    combined shift."""
    from agentic import declared4 as d
    c = d.compare(D_BEFORE, [_try("fire_1", 0.6), _try("spill_1", -0.4), _try("brush_1", 0.0)])
    card = ui_iv.card_component(c, {})
    rows = {_flat(r).split()[0]: r for r in _by_class(card, "iv-feature")}
    (fire,), (spill,), (brush,) = (_by_class(rows[h], "iv-role")
                                   for h in ("fire_1", "spill_1", "brush_1"))
    # direction, from all the parts that moved (the fixture's graph part has no direction);
    # no type: no recommendation names it
    assert _flat(fire) == "eased (2 of 3)"
    assert _flat(spill) == "escalated"                     # its after run has an arrow: all three went up
    assert fire.style["color"] == "#16a34a" and getattr(fire, "data-way") == "eased"
    assert spill.style["color"] == "#b91c1c" and getattr(spill, "data-way") == "escalated"
    assert _flat(brush) == "unchanged"
    assert _flat(_by_class(rows["tanker_1"], "iv-role")) == ""            # not removed yet
    assert _by_class(card, "iv-part") == []                             # no separate parts
    assert len(_by_class(rows["fire_1"], "iv-bar-num")) == 3            # two declared + combined
    # the parts stay in the data for later use
    by = {r["object_id"]: r for r in c["hazards"]}
    assert [p["name"] for p in by["fire_1"]["parts"]] == ["level", "recommendations", "graph"]
    c["hazards"][0]["shift"] = "x"
    ui_iv.card_component(c, {})                                         # malformed: no crash


def test_each_cell_carries_how_much_the_recommendation_changed():
    """Sunny, 2026-11-17: a bar per removed hazard, so a row shows which
    hazard that recommendation depends on most."""
    from agentic import grounding4 as g

    def rec(rank, soon="", size=""):
        r = _rec(rank, "fire_1", ["man_1"])
        r["structured_reasoning"] = {**r["structured_reasoning"],
                                     **({"how_soon": soon} if soon else {}),
                                     **({"size": size} if size else {})}
        return r
    assert g.impact(rec(1), {"state": "gone"}, 5) == (1.0, ["dropped"])
    assert g.impact(rec(1), {"state": "target removed"}, 5) == (None, [])
    assert g.impact(rec(1), {"state": "stays", "after": rec(1)}, 5) == (0.0, [])
    # kept, but later, smaller and lower down: each part adds its share
    v, why = g.impact(rec(1, "now", "large"), {"state": "stays", "after": rec(5, "later", "small")}, 5)
    W = g.IMPACT
    assert v == pytest.approx(W["how soon"] + W["size"] + W["rank"]) and why == ["now \u2192 later", "large \u2192 small",
                                                           "rank 1 \u2192 5"]
    v, why = g.impact(rec(2, "now"), {"state": "changed", "after": rec(2, "minutes")}, 5)
    assert v == pytest.approx(W["target or effect"] + W["how soon"] / 2)
    assert why == ["target or effect changed", "now \u2192 minutes"]
    # the state it gives its hazard is a part too (2026-11-23)
    burnt = rec(1)
    burnt["structured_reasoning"] = {**burnt["structured_reasoning"], "state": "burnt"}
    lit = rec(1)
    lit["structured_reasoning"] = {**lit["structured_reasoning"], "state": "Burning"}
    v, why = g.impact(lit, {"state": "stays", "after": burnt}, 5)
    assert v == pytest.approx(W["state"]) and why == ["burning \u2192 burnt"]
    assert g.impact(lit, {"state": "stays", "after": lit}, 5) == (0.0, [])
    assert sum(g.IMPACT.values()) == pytest.approx(1.0)            # a kept one never beats dropped
    # a run saved before the fields existed: only state and rank can speak
    assert g.impact(rec(1), {"state": "stays", "after": rec(1, "later", "small")}, 5) == (0.0, [])
    for bad in (None, "x", {}, {"state": "stays", "after": "x"}, {"state": "changed", "after": {"rank": "a"}}):
        assert isinstance(g.impact(None, bad, 0)[1], list)
        assert isinstance(g.impact({"structured_reasoning": "x", "rank": True}, bad, 1)[1], list)
    # on the table: every evidence cell has the number
    after = _res(["fire_1", "smoke_1", "man_1", "truck_1"], [_rec(1, "fire_1", ["man_1"])])
    tbl = g.table(BEFORE, [({"index": 1, "target": "spill_1", "after_run": "x"}, after)])
    assert all("impact" in c and "why" in c for r in tbl["rows"] for c in r["cells"])
    comp = ui_iv.table_component(tbl)
    assert _by_class(comp, "iv-bar-num") and "dropped = 1.00" in _flat(comp)


def test_an_adjusted_cell_says_which_way_it_went_and_the_bar_takes_that_colour():
    """Sunny, 2026-11-17: green for a shift that eased, red for one that
    escalated; one bar, one colour."""
    from agentic import grounding4 as g

    def rec(rank, soon="", size="", target="man_1"):
        r = _rec(rank, "fire_1", [target])
        r["structured_reasoning"] = {**r["structured_reasoning"],
                                     **({"how_soon": soon} if soon else {}),
                                     **({"size": size} if size else {})}
        return r
    d = g.impact_detail(rec(1, "now", "large"), {"state": "stays", "after": rec(1, "later", "small")}, 4)
    assert d["way"] == "eased" and [r["way"] for r in d["reasons"]] == ["eased", "eased"]
    d = g.impact_detail(rec(1, "later", "small"), {"state": "stays", "after": rec(1, "now", "large")}, 4)
    assert d["way"] == "escalated"
    d = g.impact_detail(rec(1, "now", "small"), {"state": "stays", "after": rec(1, "later", "large")}, 4)
    assert d["way"] == "mixed"                                     # later, but bigger
    # rank and a changed target carry no direction
    d = g.impact_detail(rec(1), {"state": "changed", "after": rec(3)}, 4)
    assert d["way"] == "" and [r["way"] for r in d["reasons"]] == ["", ""]
    assert g.impact_detail(rec(1), {"state": "gone"}, 4)["way"] == ""
    empty = g.impact_detail(None, None, 0)
    assert (empty["value"], empty["reasons"], empty["way"]) == (0.0, [], "")
    assert all(f["size"] == 0.0 and f["direction"] == "unchanged" for f in empty["fields"])
    assert g.impact_detail("x", {"state": "target removed"}, 0)["value"] is None

    def cell(way, state="changed", reasons=None):
        return {"state": state, "label": "correctly_ignored", "impact": 0.5, "way": way,
                "why": [], "reasons": reasons or [{"text": "now \u2192 later", "way": way}],
                "after": {"action": "x"}}
    tbl = {"columns": [{"index": i, "target": "t"} for i in range(4)],
           "rows": [{"rank": 1, "action": "a", "threat": "fire_1", "affected": [], "verdict":
                     "untested", "cells": [cell("eased"), cell("escalated"), cell("mixed"),
                                           cell("", reasons=[{"text": "rank 1 \u2192 2", "way": ""},
                                                             None, "x"])]}]}
    words = _by_class(ui_iv.table_component(tbl), "iv-cell-word")
    assert [_flat(w) for w in words] == ["adjusted, eased", "adjusted, escalated",
                                         "adjusted, mixed", "adjusted"]
    assert [w.style["color"] for w in words] == ["#16a34a", "#b91c1c", "#b45309", "#b45309"]
    reasons = _by_class(ui_iv.table_component(tbl), "iv-reason")
    assert reasons[0].style["color"] == "#16a34a" and reasons[1].style["color"] == "#b91c1c"
    assert reasons[-1].style["color"] == "#64748b"                  # rank: no direction, grey


# ── changing the picked thing BY A RECOMMENDATION (2026-11-25) ───────────────
# Sunny: "the rec should be an alternative to dealt with or remove... and it
# should come after the variable picking dropdown".

def _follow_info(tmp_path):
    return ui_iv.base_info(_run_dir(tmp_path))


def test_a_recommendation_acts_on_what_it_moves_or_on_its_hazard(tmp_path):
    """'Move man_1 away from fire_1' moves the man; 'Close the valve on
    tanker_1' deals with the tanker."""
    info = _follow_info(tmp_path)
    assert ui_iv.rec_targets(info, 1) == ["man_1"]          # it names who it protects
    assert ui_iv.rec_targets(info, 2) == ["tanker_1"]       # it does not: its hazard
    assert ui_iv.rec_targets(info, 9) == [] and ui_iv.rec_of(info, 9) is None
    # a recommendation that moves two: both, in its own order
    two = {**info, "result": {**info["result"], "recommendations": [
        {"rank": 1, "action": "Move man_1 and fire_1 back.", "structured_reasoning": {
            "threat": "tanker_1", "effect": "may_harm", "affected_objects": ["man_1", "fire_1"]}}]}}
    assert ui_iv.rec_targets(two, 1) == ["man_1", "fire_1"]
    assert ui_iv.follow_record(two, 1, "man_1")["also"] == ["fire_1"]
    # a "no action" line, or one that names nothing that can be tried, is not offered
    none = {**info, "result": {**info["result"], "recommendations": [
        {"rank": 1, "action": "No action for road_1.", "benign": True, "structured_reasoning": {
            "threat": "road_1", "effect": "N/A", "affected_objects": []}},
        {"rank": 2, "action": "Watch cloud_7.", "structured_reasoning": {
            "threat": "cloud_7", "effect": "may_harm", "affected_objects": ["bird_3"]}}]}}
    assert ui_iv.rec_options(none) == [] and ui_iv.recs_for(none, "road_1") == []


def test_a_recommendation_is_offered_only_under_the_thing_it_changes(tmp_path):
    """Sunny, 2026-11-25: a recommendation has a hazard and a target, and
    carrying it out changes ONE of them. #1 (hazard fire_1) moves man_1: it
    is offered under the man, never under the fire. #2 (protects man_1)
    closes the tanker's valve: under the tanker, never under the man."""
    info = _follow_info(tmp_path)
    assert ui_iv.recs_for(info, "man_1") == [1]
    assert ui_iv.recs_for(info, "tanker_1") == [2]
    assert ui_iv.recs_for(info, "fire_1") == []            # only its hazard: not changed by it
    assert ui_iv.recs_for(info, "ghost_9") == [] and ui_iv.recs_for(info, None) == []
    opts = ui_iv.rec_options(info, "man_1")
    assert [o["value"] for o in opts] == [1]
    assert opts[0]["label"] == " #1 Move man_1 away from fire_1. \u00b7 acts on man_1"
    assert ui_iv.rec_options(info, "fire_1") == []
    assert [o["value"] for o in ui_iv.rec_options(info)] == [1, 2]      # no pick: all


def test_the_edit_instruction_is_the_action_in_plain_words(tmp_path):
    info = _follow_info(tmp_path)
    cap, img = ui_iv.follow_instructions(info, 1)
    assert "\u201cMove the man away from the fire.\u201d" in img          # no man_1, no fire_1
    assert img.startswith("Edit this photo to show the scene after this action has been carried out")
    assert "Keep everything else exactly the same" in img
    assert "\u201cMove the man away from the fire.\u201d" in cap and "Describe only what is in the picture" in cap
    assert ui_iv.follow_instructions(info, 9) == ("", "")
    # the model's own description of a thing is given the first time it is named
    objs = [{**o, "description": "a man filming" if o["object_id"] == "man_1" else ""}
            for o in info["result"]["detected_objects"]]
    said = ui_iv.plain_action({"result": {"detected_objects": objs}},
                              "Move man_1 back, then check man_1 and ghost_9.")
    assert said == "Move the man (a man filming) back, then check the man and ghost_9."
    for bad in (None, "x", {}, {"result": 3}, {"result": {"detected_objects": [None, 4]}}):
        assert ui_iv.plain_action(bad, "Move man_1.") == "Move man_1."
        assert ui_iv.plain_action(bad, None) == ""


def test_the_record_of_a_try_by_a_recommendation(tmp_path):
    info = _follow_info(tmp_path)
    rec = ui_iv.follow_record(info, 1, "man_1")
    assert rec == {"target": "man_1",
                   "follows": {"rank": 1, "action": "Move man_1 away from fire_1."}, "also": []}
    assert ui_iv.follow_record(info, 2, "tanker_1")["target"] == "tanker_1"
    # the other end of a recommendation is not changed by it: no such try
    assert ui_iv.follow_record(info, 1, "fire_1") == {}      # its hazard
    assert ui_iv.follow_record(info, 2, "man_1") == {}       # who it protects
    assert ui_iv.follow_record(info, None, "man_1") == {} and ui_iv.follow_record(info, 9, "man_1") == {}
    # how it reads
    one = {"kind": "dealt with", **rec}
    assert ui_iv.followed(one) == 1
    assert ui_iv.kind_phrase(one, "man_1") == "man_1 dealt with by recommendation #1"
    assert ui_iv.kind_phrase({**one, "also": ["dog_1"]}, "man_1") == (
        "man_1 + dog_1 dealt with by recommendation #1")
    # (a try saved on 2026-11-25 with the older "for building_1" mark reads the same)
    assert ui_iv.kind_phrase({**one, "addresses": "fire_1"}, "man_1") == (
        "man_1 dealt with by recommendation #1")
    assert ui_iv.kind_phrase({"kind": "dealt with"}, "man_1") == "man_1 dealt with"
    assert ui_iv.kind_phrase({**one, "kind": "removed"}, "man_1") == "removed man_1"
    for bad in (None, "x", {}, {"follows": "one"}, {"follows": {"rank": "1"}},
                {"follows": {"rank": True}}):
        assert ui_iv.followed(bad) is None
    # it is saved with the try
    import pathlib
    base = pathlib.Path(info["dir"])
    saved = ui_iv.finish_try(base, None, {**PENDING, **one}, error="x")
    assert saved["follows"]["rank"] == 1 and saved["also"] == [] and "addresses" not in saved
    assert "follows" not in ui_iv.finish_try(base, None, PENDING, error="x")
    assert "follows" not in ui_iv.finish_try(base, None, {**PENDING, "follows": "zz"}, error="x")


@pytest.mark.parametrize("bad", [None, "x", 3, {}, {"result": None}, {"result": {"recommendations": "r"}},
                                 {"result": {"recommendations": [None, 3, {"rank": "1"},
                                                                 {"rank": 1, "action": 5},
                                                                 {"rank": 1, "action": "Go."},
                                                                 {"rank": True, "action": "Go.",
                                                                  "structured_reasoning": {}}]},
                                  "candidates": "zz"}])
def test_a_malformed_run_offers_no_recommendation_to_follow(bad):
    assert ui_iv.rec_options(bad) == [] and ui_iv.rec_options(bad, "man_1") == []
    assert ui_iv.recs_for(bad, "man_1") == []
    for rank in (1, None, "1", True, [1]):
        assert ui_iv.rec_targets(bad, rank) == []
        assert ui_iv.follow_instructions(bad, rank) == ("", "")
        assert ui_iv.follow_record(bad, rank, "man_1") == {}


def test_the_form_offers_by_a_recommendation_as_a_third_way(page):
    hidden = ([], None, {"display": "none"})
    # WHICH shows only for "by a recommendation"
    assert ui.iv_follow_options("man_1", ui_iv.DEALT_WITH, "base", 1) == hidden
    assert ui.iv_follow_options("man_1", ui_iv.REMOVED, "base", None) == hidden
    # it lists the recommendations that act on the pick; a lone one is picked for you
    opts, value, style = ui.iv_follow_options("man_1", ui_iv.BY_REC, "base", None)
    assert [o["value"] for o in opts] == [1] and value == 1 and style == {"display": "flex"}
    assert _flat(opts[0]["label"]) == "#1 Move man_1 away from fire_1. \u00b7 acts on man_1"
    assert opts[0]["label"].style["whiteSpace"] == "nowrap"       # one line, never over the next
    # another thing is picked: the list follows, and a pick that is gone is dropped
    opts, value, _s = ui.iv_follow_options("tanker_1", ui_iv.BY_REC, "base", 1)
    assert [o["value"] for o in opts] == [2] and value == 2
    # the fire is only the HAZARD of #1: nothing acts on it
    assert ui.iv_follow_options("fire_1", ui_iv.BY_REC, "base", 1) == ([], None, {"display": "flex"})
    assert ui.iv_follow_options(None, ui_iv.BY_REC, "base", 1)[:2] == ([], None)
    assert ui.iv_follow_options("man_1", ui_iv.BY_REC, None, 1)[:2] == ([], None)
    # the instructions are the recommendation's action, carried out
    kids, caption = ui.iv_instructions("man_1", ui_iv.BY_REC, 1, "base")
    assert "after this action has been carried out" in _flat(kids)
    assert "Move the man away from the fire" in _flat(kids) and caption == RECORD["caption"]
    # none picked yet: it says so; nothing acts on the pick: it says that
    assert "pick the recommendation" in _flat(ui.iv_instructions("man_1", ui_iv.BY_REC, None, "base")[0])
    assert "no recommendation acts on this one" in _flat(
        ui.iv_instructions("fire_1", ui_iv.BY_REC, 1, "base")[0])
    # the other two ways are as they were, whatever is left in WHICH
    assert "has been carried out" not in _flat(ui.iv_instructions("man_1", ui_iv.DEALT_WITH, 1, "base")[0])
    assert "has been carried out" not in _flat(ui.iv_instructions("man_1", ui_iv.REMOVED, 1, "base")[0])
    assert ui.iv_intended("man_1", ui_iv.BY_REC, "base") == ("set by the recommendation", None, True)


def test_a_try_by_a_recommendation_is_a_dealt_with_try_on_the_pick(page):
    assert ui.iv_start("base", "man_1", "caption", "a tanker leaks on a road", None, None,
                       ui_iv.BY_REC, "burnt", 1) == ""
    origin = page["calls"][0]["after_of"]
    assert origin["target"] == "man_1" and "addresses" not in origin
    assert origin["follows"] == {"rank": 1, "action": "Move man_1 away from fire_1."}
    assert origin["kind"] == ui_iv.DEALT_WITH and origin["mode"] == ui_iv.KIND_MODE[ui_iv.DEALT_WITH]
    assert "intended_state" not in origin
    assert ui.IV_ACTIVE["ui_base"]["follows"]["rank"] == 1
    # the model is told nothing of it: only the edited caption goes in
    assert page["calls"][0]["caption"] == "a tanker leaks on a road"


def test_by_a_recommendation_needs_one_that_acts_on_the_pick(page):
    # #1 only names the fire as its hazard; #2 only protects the man
    for target, follow in (("fire_1", 1), ("man_1", 2), ("man_1", None), ("man_1", 9), ("man_1", "1")):
        assert ui.iv_start("base", target, "caption", "a tanker leaks on a road", None, None,
                           ui_iv.BY_REC, None, follow) == "pick the recommendation to carry out"
    assert page["calls"] == []                           # no model was called
    # the other two ways ignore whatever is left in WHICH
    assert ui.iv_start("base", "man_1", "caption", "a tanker leaks on a road", None, None,
                       ui_iv.DEALT_WITH, None, 1) == ""
    assert "follows" not in page["calls"][0]["after_of"]


def test_a_pick_that_stands_for_two_hazards_is_recorded_on_the_one_acted_on(page):
    """E, 2026-11-25: the model called building_1 and rubble_1 the same
    hazard, so they are ONE option in WHAT. 'Clear the rubble' acts on the
    rubble only: the try is recorded on the rubble, not on the option's
    lead name."""
    info = ui._iv_info("base")
    info["result"]["same_hazards"] = [["fire_1", "tanker_1"]]
    assert [o["value"] for o in ui_iv.target_options(info)] == ["fire_1", "man_1"]   # fire_1 stands for both
    assert ui_iv.recs_for(info, "fire_1") == [2]         # the one that acts on either of the two
    rec = ui_iv.follow_record(info, 2, "fire_1")
    assert rec["target"] == "tanker_1" and rec["also"] == []
    assert ui.iv_start("base", "fire_1", "caption", "a tanker stands on a road", None, None,
                       ui_iv.BY_REC, None, 2) == ""
    assert page["calls"][0]["after_of"]["target"] == "tanker_1"


# ── the bar picture over a cell of the recommendations table (2026-11-25) ────

def test_a_cell_carries_its_change_field_by_field():
    """Sunny: "we should show graph of this like we did on feature table.
    What eased, what escalated... from the perspective of recommendation"."""
    from agentic import grounding4 as g
    before = {"rank": 3, "action": "Move officer_1 back.", "structured_reasoning": {
        "threat": "building_1", "state": "collapsed", "effect": "may_harm",
        "affected_objects": ["officer_1", "officer_2"], "how_soon": "minutes", "size": "large"}}
    after = {"rank": 1, "action": "Retreat everyone.", "structured_reasoning": {
        "threat": "building_1", "state": "collapsed", "effect": "may_harm",
        "affected_objects": ["officer_1", "officer_2", "person_1"], "how_soon": "now",
        "size": "medium"}}
    cell = {"state": g.CHANGED, "after": after, "target_shift": 1 / 3, "targets_kept": 2,
            "targets_before": 2, "targets_new": 1}
    det = g.impact_detail(before, cell, 4)
    by = {f["name"]: f for f in det["fields"]}
    assert list(by) == ["who it protects", "effect", "how soon", "size", "rank", "state of its hazard"]
    assert by["who it protects"]["size"] == pytest.approx(0.333, abs=0.001)
    assert by["who it protects"]["direction"] == "redirected"
    assert by["who it protects"]["after"] == "officer_1, officer_2, person_1"
    assert (by["how soon"]["size"], by["how soon"]["direction"]) == (0.5, "escalated")   # minutes -> now
    assert (by["size"]["size"], by["size"]["direction"]) == (0.5, "eased")               # large -> medium
    assert (by["rank"]["size"], by["rank"]["direction"]) == (pytest.approx(0.667, abs=0.001), "changed")
    assert (by["effect"]["size"], by["effect"]["direction"]) == (0.0, "unchanged")
    assert by["state of its hazard"]["direction"] == "unchanged"
    # the bar is these, each times its weight
    want = 0.35 * (1 / 3) + 0.2 * 0.5 + 0.15 * 0.5 + 0.15 * (2 / 3)
    assert det["value"] == pytest.approx(want, abs=0.002)
    # a dropped one is one bar, in full; a cell that is not evidence has none
    gone = g.impact_detail(before, {"state": g.GONE}, 4)["fields"]
    assert [(f["name"], f["size"], f["direction"]) for f in gone] == [("the recommendation", 1.0, "eased")]
    assert g.impact_detail(before, {"state": g.TARGET_REMOVED}, 4)["fields"] == []
    # the picture: the two actions on top, a bar per field, what the bar is made of
    data = json.loads(ui_iv.cell_bars(before["action"], {**cell, "fields": det["fields"],
                                                         "impact": det["value"]}, "meaning"))
    assert data["top"] == ["before: Move officer_1 back.", "after: Retreat everyone."]
    assert [r["name"] for r in data["rows"]][:3] == ["who it protects", "effect", "how soon"]
    assert data["rows"][2] == {"name": "how soon", "size": 0.5, "direction": "escalated",
                               "was": "minutes", "now": "now"}
    assert data["foot"].startswith(f"changed {det['value']:.2f}. the bar is these")
    assert data["note"] == "meaning"
    dropped = json.loads(ui_iv.cell_bars("Go.", {"state": "gone", "fields": gone, "impact": 1.0}))
    assert dropped["top"][1] == "after: no such recommendation" and "in full" in dropped["foot"]
    for bad in (None, "x", {}, {"fields": "zz"}, {"fields": [None, 3, {"name": "a"},
                                                             {"size": "big", "direction": 3}]}):
        assert ui_iv.cell_bars("Go.", bad) == ""
    for bad_rec in (None, "x", {"structured_reasoning": 5}, {"rank": "one"}):
        assert isinstance(g.impact_detail(bad_rec, cell, 0)["fields"], list)
        assert isinstance(g.impact_detail(before, {"state": g.CHANGED, "after": bad_rec}, 0)["fields"], list)
