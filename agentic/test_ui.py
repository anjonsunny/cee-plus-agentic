"""Hermetic tests for the live UI's pure core. No browser, no server.

Everything the screen shows is a pure function of an event prefix
(derive -> components). These tests drive exactly that path: event
streams in, component trees out; plus the replay builder against a
frozen-record shape, and the event emission of an instrumented
run_perception with mocked models.

Run:  pytest agentic/test_ui.py -q
"""
from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("dash")
from PIL import Image  # noqa: E402

from agentic import perception  # noqa: E402
from agentic.ui import (  # noqa: E402
    build_replay_events,
    derive,
    instruments_component,
    rail_component,
    scene_component,
    tickets_component,
)

# ── A miniature event stream, hand-authored ─────────────────────────────

EVENTS = [
    {"type": "run_started", "image_size": [400, 300], "caption": "a tanker truck",
     "image_name": "s.jpg"},
    {"type": "stage_started", "stage": "Perceive"},
    {"type": "stage_done", "stage": "Perceive", "seconds": 12.0, "n_entities": 1},
    {"type": "stage_started", "stage": "Repair"},
    {"type": "violation_found", "round": 1, "entity_index": 0,
     "raw_label": "vehicle", "kind": "family_name_as_label",
     "instruction": "Entity 0: 'vehicle' is a family name..."},
    {"type": "repair_round_started", "round": 1, "open_violations": 1},
    {"type": "repair_round_done", "round": 1, "changed": True,
     "remaining_violations": 0,
     "entities": [{"label": "tanker_truck", "state": "stationary",
                   "bbox": [10, 10, 200, 150]}]},
    {"type": "repair_stopped", "reason": "clean", "rounds": 1, "remaining": []},
    {"type": "stage_done", "stage": "Repair", "seconds": 30.0, "stopped": "clean"},
    {"type": "anchors_ready", "entities": [
        {"object_id": "tanker_truck_1", "label": "tanker_truck",
         "state": "stationary", "description": "", "anchor_bbox": [10, 10, 200, 150]}]},
    {"type": "entity_bound", "object_id": "tanker_truck_1",
     "box_source": "dino_matched", "bbox": [12, 11, 195, 148], "confidence": 0.9},
]


def test_derive_midstream_open_violation():
    """Scrubbed to just after the violation: ticket open, Repair active."""
    d = derive(EVENTS[:5])
    assert d["stages"]["Perceive"]["status"] == "done"
    assert d["stages"]["Repair"]["status"] == "active"
    v = list(d["violations"].values())[0]
    assert v["status"] == "open" and v["kind"] == "family_name_as_label"


def test_derive_after_stop_violation_fixed():
    d = derive(EVENTS)
    v = list(d["violations"].values())[0]
    assert v["status"] == "fixed"
    assert d["stop_reason"] == "clean" and d["rounds_used"] == 1
    assert d["bound"]["tanker_truck_1"]["box_source"] == "dino_matched"


def test_derive_stood_ground_marks_remaining():
    ev = EVENTS[:6] + [{"type": "repair_stopped", "reason": "no_change",
                        "rounds": 2, "remaining": [
                            {"kind": "family_name_as_label", "raw_label": "vehicle",
                             "entity_index": 0}]}]
    d = derive(ev)
    assert list(d["violations"].values())[0]["status"] == "stood"


def test_components_render_at_every_prefix():
    """The scrubber can stop at ANY event index; no prefix may crash."""
    for k in range(len(EVENTS) + 1):
        d = derive(EVENTS[:k])
        rail_component(d)
        tickets_component(d)
        instruments_component(d)
        scene_component(d, "data:image/png;base64,x")


def test_scene_returns_children_for_the_positioned_container():
    """Regression (Sunny's screenshot, 2026-07-21): boxes anchored to the
    page because the wrapper's .scene class was lost. scene_component now
    returns children destined for the layout's positioned container, and
    every box style must stay inside 0-100% of it."""
    d = derive(EVENTS + [{"type": "assembled", "result": {
        "image_size": [400, 300],
        "detected_objects": [{"object_id": "tanker_truck_1", "label": "tanker_truck",
                              "state": "stationary", "state_kind": "normal",
                              "description": "", "bbox": [12, 11, 195, 148],
                              "box_source": "dino_matched", "box_confidence": 0.9,
                              "anchor_bbox": [10, 10, 200, 150], "mask_path": None,
                              "label_note": "", "vocab_extension": False,
                              "family_name_as_label": False}]}}])
    children = scene_component(d, "data:image/png;base64,x")
    assert isinstance(children, list)
    for child in children:
        style = getattr(child, "style", None) or {}
        for key in ("left", "top", "width", "height"):
            if key in style and str(style[key]).endswith("%"):
                assert 0.0 <= float(str(style[key]).rstrip("%")) <= 100.0


def test_perceive_detail_and_repair_ledger():
    """The rail's Perceive station lists the first answer; Repair carries
    the violation ledger. (Sunny: 'should include more details'.)"""
    ev = EVENTS[:3]
    ev[2] = {**ev[2], "entities": [{"label": "vehicle", "state": "intact"}]}
    d = derive(ev)
    assert d["perceive_entities"] == [{"label": "vehicle", "state": "intact"}]
    rail = rail_component(derive(EVENTS))          # renders without crashing
    assert rail is not None


def test_inspector_close_button_uses_pattern_id():
    """Regression: a plain id on the sometimes-absent close button silently
    disabled the whole click callback."""
    from agentic.ui import inspector_component
    d = derive(EVENTS + [{"type": "assembled", "result": {
        "image_size": [400, 300],
        "detected_objects": [{"object_id": "tanker_truck_1", "label": "tanker_truck",
                              "state": "stationary", "state_kind": "normal",
                              "description": "", "bbox": [1, 1, 9, 9],
                              "box_source": "dino_matched", "box_confidence": 0.9,
                              "anchor_bbox": None, "mask_path": None, "label_note": "",
                              "vocab_extension": False, "family_name_as_label": False}]}}])
    modal = inspector_component(d, "tanker_truck_1")

    def find_button(node):
        if getattr(node, "id", None) and isinstance(node.id, dict):
            if node.id.get("type") == "insp-close":
                return True
        kids = getattr(node, "children", None)
        kids = kids if isinstance(kids, list) else ([kids] if kids else [])
        return any(find_button(k) for k in kids if hasattr(k, "children") or hasattr(k, "id"))

    assert find_button(modal)


def test_shared_heavy_modules_preloaded_in_main_thread():
    """Import-race regression (pandas.NaT AttributeError, 2026-07-21):
    modules Dash's serializer may touch must be fully imported by the time
    the ui module is loaded, so no worker thread can race their import."""
    import importlib.util
    import sys
    for mod in ("pandas", "numpy"):
        if importlib.util.find_spec(mod) is not None:
            assert mod in sys.modules, f"{mod} not preloaded"


def test_malformed_model_boxes_never_crash_the_scene():
    """Live-run regression (2026-07-21): a raw mid-repair entity carried a
    one-element bbox and the chip renderer crashed. Every malformed shape
    must be skipped; valid-but-oversized boxes are clamped into frame."""
    from agentic.ui import _pct_box, _valid_box

    bad = [[5], None, "box", [1, 2, 3], ["a", "b", "c", "d"],
           [10, 10, 10, 10], [50, 50, 10, 10], [1, 2, 3, 4, 5]]
    for b in bad:
        assert not _valid_box(b), b
    assert _valid_box([0, 0, 10, 10]) and _valid_box([0.5, 1, 9.5, 12])

    # Oversized box clamps to the frame instead of overflowing it.
    style = _pct_box([-50, -50, 800, 600], [400, 300])
    assert style["left"] == "0.00%" and style["top"] == "0.00%"
    assert style["width"] == "100.00%" and style["height"] == "100.00%"

    # A violation chip pointing at an entity with a broken bbox: no crash.
    ev = EVENTS[:6] + [{"type": "repair_round_done", "round": 1, "changed": True,
                        "entities": [{"label": "vehicle", "state": "intact",
                                      "bbox": [5]}]}]
    d = derive(ev)
    scene_component(d, "data:image/png;base64,x")   # must not raise


def test_smaller_boxes_paint_on_top_of_larger_ones():
    """Z-order regression (Sunny: road_1 buried spill_1): boxes render
    largest-first, so later (topmost) children are the smaller boxes."""
    result = {"image_size": [1000, 800], "detected_objects": [
        {"object_id": "spill_1", "label": "spill", "state": "leaking",
         "state_kind": "hazard_bearing", "description": "", "bbox": [400, 500, 600, 700],
         "box_source": "dino_matched", "box_confidence": 0.5, "anchor_bbox": None,
         "mask_path": None, "label_note": "", "vocab_extension": False,
         "family_name_as_label": False},
        {"object_id": "road_1", "label": "road", "state": "dry",
         "state_kind": "normal", "description": "", "bbox": [0, 300, 1000, 800],
         "box_source": "dino_matched", "box_confidence": 0.5, "anchor_bbox": None,
         "mask_path": None, "label_note": "", "vocab_extension": False,
         "family_name_as_label": False},
    ]}
    events = [{"type": "run_started", "image_size": [1000, 800]},
              {"type": "anchors_ready", "entities": [
                  {"object_id": "spill_1"}, {"object_id": "road_1"}]},
              {"type": "assembled", "result": result}]
    children = scene_component(derive(events), "data:image/png;base64,x")
    order = [c.id["oid"] for c in children
             if getattr(c, "id", None) and isinstance(c.id, dict)
             and c.id.get("type") == "scene-box"]
    assert order == ["road_1", "spill_1"]       # big first, small on top


def test_upload_pick_shows_thumbnail_and_name():
    """After picking an image, the upload control shows what was picked."""
    from agentic.ui import cache_upload
    data, picked = cache_upload("data:image/png;base64,xyz", "A_fire.png")
    assert data["filename"] == "A_fire.png"
    assert picked.className == "upload-inner"
    img, name = picked.children
    assert img.src.startswith("data:image/") and name.children == "A_fire.png"


def test_event_sink_writes_jsonl_flight_recorder(tmp_path):
    """Every event lands in events.jsonl as it happens: the durable record
    (Sunny: 'are you writing everything in a log file?')."""
    import json as _json
    from agentic import ui as ui_mod
    ui_mod.RUNS["testrun"] = {"events": [], "done": False, "error": None}
    sink = ui_mod.make_event_sink("testrun", tmp_path)
    sink({"type": "stage_started", "stage": "Perceive"})
    sink({"type": "violation_found", "kind": "x", "raw_label": "y"})
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2 == len(ui_mod.RUNS["testrun"]["events"])
    first = _json.loads(lines[0])
    assert first["type"] == "stage_started" and "t" in first
    del ui_mod.RUNS["testrun"]


def test_stations_are_collapsible_details():
    """Stations render as <details> (collapsible); non-pending ones open."""
    from dash import html
    rail = rail_component(derive(EVENTS))
    stations = [c for c in rail.children if isinstance(c, html.Details)]
    assert len(stations) == 6
    opens = {s.className.split()[-1]: s.open for s in stations}
    assert opens["st-perceive"] and not opens["st-assemble"]


def test_rail_renders_timeline_nodes():
    """The station activity feed renders as a mini timeline (nodes on a
    connector), with a warn node for violation lines."""
    rail = str(rail_component(derive(EVENTS)))
    assert "tl-row" in rail and "tl-node" in rail
    assert "warn" in rail                       # the violation line's node


def test_flight_recorder_timestamp_never_leaks_into_station_info():
    """Regression (Sunny's screenshot): the events.jsonl 't' stamp showed
    up as 't=178...' in station headers."""
    d = derive([{"type": "run_started", "image_size": [10, 10]},
                {"type": "stage_started", "stage": "Ground", "t": 1784622358.4},
                {"type": "stage_done", "stage": "Ground", "seconds": 2.0,
                 "n_candidates": 6, "t": 1784622360.4}])
    assert "t=" not in d["stages"]["Ground"]["info"]
    assert "n_candidates=6" in d["stages"]["Ground"]["info"]


def test_station_activity_feeds_accumulate():
    """Cowork-style narration: each stage collects its own activity lines
    (Sunny, 2026-07-21)."""
    d = derive(EVENTS)
    per = d["stage_activities"]["Perceive"]
    rep = d["stage_activities"]["Repair"]
    assert per[0].startswith("asking the VLM")
    assert any("first answer: 1 entities" in a for a in per)
    assert any(a.startswith("found: family name as label") for a in rep)
    assert any("round 1: asking the model" in a for a in rep)
    assert any("model revised its answer" in a for a in rep)
    assert rep[-1] == "all problems resolved — clean"
    assert any("tanker_truck_1: matched" in a for a in d["stage_activities"]["Bind"])


def test_ticket_shows_fixing_during_inflight_round():
    """Mid-round, open violations read FIXING (the 'which problem is being
    fixed right now' focus)."""
    mid = EVENTS[:6]                       # violation found + round started
    d = derive(mid)
    assert d["round_in_progress"]
    tickets = tickets_component(d)
    text = str(tickets)
    assert "FIXING" in text
    d_done = derive(EVENTS)                # after round done + clean
    assert not d_done["round_in_progress"]
    assert "FIXING" not in str(tickets_component(d_done))


def test_activity_ribbon_follows_events():
    """The scene narrates the run: ribbon text and spotlight target derive
    from the latest event."""
    d = derive(EVENTS[:2])
    assert d["activity"]["text"] == "model reading the scene..."
    assert d["activity"]["busy"]                    # sweep shows while thinking
    d = derive(EVENTS[:5])
    assert "violation" in d["activity"]["text"]
    d = derive(EVENTS + [{"type": "masking_entity", "object_id": "tanker_truck_1"}])
    assert d["activity"]["oid"] == "tanker_truck_1"  # spotlight target
    assert "masking" in d["activity"]["text"]


def test_inspector_is_modal_and_optional():
    from agentic.ui import inspector_component
    d = derive(EVENTS)
    assert inspector_component(d, None) is None      # no selection: no modal
    d_final = derive(EVENTS + [{"type": "assembled", "result": {
        "image_size": [400, 300],
        "detected_objects": [{"object_id": "tanker_truck_1", "label": "tanker_truck",
                              "state": "stationary", "state_kind": "normal",
                              "description": "d", "bbox": [1, 1, 9, 9],
                              "box_source": "dino_matched", "box_confidence": 0.9,
                              "anchor_bbox": None, "mask_path": None, "label_note": "",
                              "vocab_extension": False, "family_name_as_label": False}]}}])
    modal = inspector_component(d_final, "tanker_truck_1")
    assert modal is not None and modal.className == "modal-backdrop"


def test_replay_builder_roundtrip():
    record = {
        "image_path": "/x/B_pool.jpg", "image_size": [1679, 941], "caption": "cap",
        "detected_objects": [
            {"object_id": "child_1", "label": "child", "state": "drowning",
             "state_kind": "at_risk", "description": "", "bbox": [5, 5, 50, 50],
             "box_source": "dino_matched", "box_confidence": 0.7,
             "anchor_bbox": [4, 4, 52, 52], "mask_path": "m.png",
             "label_note": "", "vocab_extension": False,
             "family_name_as_label": False}],
        "repair_trace": {"rounds": [], "clean_on_arrival": True,
                         "stopped_reason": "clean"},
    }
    d = derive(build_replay_events(record))
    assert d["stop_reason"] == "clean"
    assert d["result"]["detected_objects"][0]["object_id"] == "child_1"
    assert d["stages"]["Assemble"]["status"] == "done"
    assert d["bound"]["child_1"]["box_source"] == "dino_matched"


# ── Event emission from the instrumented pipeline ───────────────────────


def test_run_perception_emits_coherent_stream(tmp_path, monkeypatch):
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

    img = tmp_path / "s.jpg"
    Image.new("RGB", (300, 200), "gray").save(img)
    events: list[dict] = []
    perception.run_perception(
        img, caption="a burning house",
        entities=[{"label": "vehicle", "state": "intact", "bbox": [12, 12, 98, 118]}],
        out_dir=tmp_path / "out",
        repair_query_fn=lambda p: [{"label": "house", "state": "burning",
                                    "bbox": [12, 12, 98, 118]}],
        on_event=events.append,
    )
    types = [e["type"] for e in events]
    # Coherence: starts with run_started, every stage opens and closes,
    # the violation stream sits inside Repair, assembled is last.
    assert types[0] == "run_started"
    for s in ("Perceive", "Repair", "Ground", "Bind", "Mask", "Assemble"):
        assert {"stage_started", "stage_done"} <= {
            e["type"] for e in events if e.get("stage") == s}
    assert "violation_found" in types and "repair_stopped" in types
    assert types[-1] == "assembled"
    # And the stream is UI-consumable end to end.
    d = derive(events)
    assert d["result"] is not None
    assert d["stop_reason"] == "clean"
