"""The two loop switches on the run strip (Sunny, 2026-11-20): perception
repair and assessment reflection, tick = on, unticked by default like the
other boxes. Hermetic."""
import pytest
from PIL import Image

from agentic import perception, ui, ui_clean, ui_intervene as ui_iv
from agentic.test_ui_intervene import _flat
from agentic.test_ui_views import _find


@pytest.fixture()
def fake_geometry(monkeypatch):
    def fake_detect(image, entities):
        cands = {}
        for i, e in enumerate(entities):
            p = perception._detector_phrase(e["label"], e.get("description", ""))
            e["_phrase"] = p
            cands.setdefault(p, []).append(
                {"score": 0.8, "bbox": [10 + i * 30, 10, 100 + i * 30, 120]})
        return cands
    monkeypatch.setattr(perception, "detect_candidates", fake_detect)
    monkeypatch.setattr(perception, "mask_for_box",
                        lambda image, bbox: Image.new("L", image.size, 255))


def _perceive(tmp_path, monkeypatch, answer, **kw):
    img = tmp_path / "scene.jpg"
    Image.new("RGB", (300, 200), "gray").save(img)
    import copy
    asked = []
    # a fresh copy each time: the pipeline edits the answer it is handed
    monkeypatch.setattr(perception, "query_vlm_entities",
                        lambda url, cap, notes: copy.deepcopy(answer))
    monkeypatch.setattr(perception, "_query_vlm_raw",
                        lambda *a, **k: asked.append(1) or copy.deepcopy(answer))
    events = []
    result = perception.run_perception(img, with_masks=False, out_dir=tmp_path / "out",
                                       on_event=events.append, **kw)
    return result, events, asked


BAD = [{"label": "vehicle", "state": "intact", "description": "tanker",
        "bbox": [12, 12, 98, 118]}]                          # a family name: a P-check fails
GOOD = [{"label": "tanker_truck", "state": "intact", "description": "tanker",
         "bbox": [12, 12, 98, 118]}]


def test_with_the_repair_loop_off_the_checks_still_run_and_the_model_is_not_asked_again(
        tmp_path, monkeypatch, fake_geometry):
    result, events, asked = _perceive(tmp_path, monkeypatch, BAD, repair=False)
    assert asked == []                                       # no second call: the time saved
    assert result.repair_trace is None
    found = [e for e in events if e["type"] == "violation_found"]
    assert found, "the problem is still reported"
    (stop,) = [e for e in events if e["type"] == "repair_stopped"]
    assert stop["reason"] == "loop_off" and stop["rounds"] == 0 and stop["remaining"]
    assert [e.get("stopped") for e in events
            if e["type"] == "stage_done" and e.get("stage") == "Repair"] == ["loop_off"]
    # it stays visibly unrepaired in the record
    assert result.detected_objects[0].family_name_as_label
    # nothing wrong: it simply reads clean
    _result, events, asked = _perceive(tmp_path, monkeypatch, GOOD, repair=False)
    (stop,) = [e for e in events if e["type"] == "repair_stopped"]
    assert stop["reason"] == "clean" and asked == []
    # with the loop on, the model IS asked again
    _result, _events, asked = _perceive(tmp_path, monkeypatch, BAD, repair=True)
    assert asked


def test_a_broken_first_answer_never_breaks_the_run_with_the_loop_off(
        tmp_path, monkeypatch, fake_geometry):
    for bad in ([{"label": None, "state": 3, "bbox": "x"}], [{}], []):
        result, events, _ = _perceive(tmp_path, monkeypatch, bad, repair=False)
        assert any(e["type"] == "repair_stopped" for e in events)
    monkeypatch.setattr("agentic.repair_loop.detect_violations",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    _result, events, _ = _perceive(tmp_path, monkeypatch, BAD, repair=False)
    (stop,) = [e for e in events if e["type"] == "repair_stopped"]
    assert stop["reason"] == "clean"                         # the check failed: nothing claimed


def test_the_two_boxes_are_unticked_by_default_and_tick_means_on(monkeypatch):
    box, _ = _find(ui.app.layout, "loops-mode")
    assert box.value == [] and [o["value"] for o in box.options] == ["repair", "reflect"]
    assert [o["label"].strip() for o in box.options] == ["perception repair",
                                                         "assessment reflection"]
    monkeypatch.setitem(ui.LOOPS, "repair", False)
    monkeypatch.setitem(ui.LOOPS, "reflect", False)
    S = ui.settings_snapshot()
    assert S["repair"] is False and S["reflect"] is False
    monkeypatch.setitem(ui.LOOPS, "reflect", True)
    assert ui.settings_snapshot()["reflect"] is True
    # the strip says what this run will use
    assert _flat(ui.loops_chip(None)) == "loops off (faster)"
    assert _flat(ui.loops_chip(["repair"])) == "loops: perception repair only"
    assert _flat(ui.loops_chip(["repair", "reflect"])) == "loops: both on"
    assert _flat(ui.loops_chip("zz")) == "loops off (faster)"
    chips = _flat(ui.settings_summary("python", "rulebook", "m", [], "off", [], [], []))
    assert "loops off (faster)" in chips


def test_the_run_line_says_which_loops_were_off_and_an_older_run_says_nothing():
    line = ui_clean.run_settings_line({"run_settings": {"repair": False, "reflect": False}})
    assert "loops off: perception repair, assessment reflection" in _flat(line)
    line = ui_clean.run_settings_line({"run_settings": {"repair": True, "reflect": False}})
    assert _flat(line).endswith("loops off: assessment reflection")
    # a run recorded before the switch existed had both on
    assert "loops off" not in _flat(ui_clean.run_settings_line({"run_settings": {}}))
    assert "loops off" not in _flat(ui_clean.run_settings_line(
        {"run_settings": {"repair": True, "reflect": True}}))


def test_an_edited_run_uses_the_loops_its_first_run_used():
    new = ui_iv.after_settings([{"type": "run_settings", "repair": False, "reflect": False}])
    assert new["repair"] is False and new["reflect"] is False
    old = ui_iv.after_settings([{"type": "run_settings", "subject": "m"}])
    assert old["repair"] is True and old["reflect"] is True       # before the switch: both on
    none = ui_iv.after_settings([])
    assert none["repair"] is True and none["reflect"] is True
    mixed = ui_iv.after_settings([None, "x", {"type": "run_settings", "repair": True,
                                              "reflect": False}])
    assert (mixed["repair"], mixed["reflect"]) == (True, False)


def test_a_live_run_passes_the_two_switches_to_the_pipeline(tmp_path, monkeypatch):
    import threading
    seen = {}
    monkeypatch.setattr(ui, "UI_RUNS_DIR", tmp_path)

    class Now:                                   # run the worker in the test, not a thread
        def __init__(self, target=None, **k):
            self.target = target

        def start(self):
            self.target()
    monkeypatch.setattr(ui.threading, "Thread", Now)
    assert threading is ui.threading

    def fake_perception(image_path, **kw):
        seen["repair"] = kw.get("repair")
        raise RuntimeError("stop here")
    monkeypatch.setattr("agentic.perception.run_perception", fake_perception)
    for settings, want in (({"repair": False, "reflect": False}, False),
                           ({"repair": True, "reflect": True}, True),
                           ({"subject": "m"}, True)):          # an older settings copy
        rid = ui.start_live_run(b"img", "s.jpg", "cap", settings={
            "control": "python", "retrieval": "rulebook", "subject": None, "judges": False,
            "stage2": False, "stage4": False, "concealment": False,
            "concealment_motives": [], "sandbagging": False, "sandbagging_motives": [],
            "hint": False, "readers": False, "env": False, "env_conditions": [],
            "env_reask": False, **settings})
        assert seen["repair"] is want
        ui.RUNS.pop(rid, None)
