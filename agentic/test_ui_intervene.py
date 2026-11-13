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
    assert [o["value"] for o in modality.options] == ["caption", "image", "both"]
    controls, _ = _find(panel, "iv-controls")
    assert controls.style == {"display": "none"}            # until a run is open
