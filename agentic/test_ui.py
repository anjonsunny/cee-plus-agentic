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
