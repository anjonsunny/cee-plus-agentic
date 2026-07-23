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
    data, picked, caption = cache_upload("data:image/png;base64,xyz", "A_fire.png")
    assert caption == ""     # stale caption cleared on new image (Sunny's bug)
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


# ── Stage 2 in the UI (PHASE 1 · ASSESS) ───────────────────────────────

ASSESS_EVENTS = [
    {"type": "stage_started", "stage": "assess"},
    {"type": "assess_context", "n_entities": 3,
     "hazard_ids": ["fire_1"], "at_risk_ids": ["child_1"]},
    {"type": "assess_probe", "index": 0, "scenario": "Yes", "level": 7,
     "bucket": "catastrophic"},
    {"type": "assess_verdict", "scenario": "Yes", "disaster_type": "fire",
     "level": 7, "bucket": "catastrophic", "self_confidence": 0.95,
     "n_violations": 0},
    {"type": "assess_uncertainty", "score": 0.2, "scenario_agreement": 1.0,
     "type_agreement": 0.6, "bucket_agreement": 0.8,
     "drivers": ["type_split"], "explanation": "probes split on type",
     "explainer": "llm"},
    {"type": "stage_done", "stage": "assess"},
]


def test_derive_assess_events_fill_the_phase1_state():
    d = derive(EVENTS + ASSESS_EVENTS)
    A = d["assess"]
    assert A["status"] == "done"
    assert A["verdict"]["bucket"] == "catastrophic"
    assert A["verdict"]["self_confidence"] == 0.95
    assert A["uncertainty"]["score"] == 0.2
    assert A["context"]["hazard_ids"] == ["fire_1"]
    assert any("verdict:" in a for a in A["activities"])
    # assess must NOT leak into the Phase 0 stage cards
    assert "assess" not in d["stages"]


def test_derive_assess_midstream_is_active_with_ribbon():
    d = derive(EVENTS + ASSESS_EVENTS[:2])
    assert d["assess"]["status"] == "active"
    assert "assess" in d["activity"]["text"]
    from agentic.ui import scene_component
    children = scene_component(d, "data:image/png;base64,x")
    txt = str(children)
    assert "assessing scene from declared state" in txt


def test_assess_component_three_views():
    from agentic.ui import assess_component
    d = derive(EVENTS + ASSESS_EVENTS)
    both = str(assess_component(d, "both"))
    assert "model says" in both and "we measured" in both
    assert "calibration gap" in both
    assert "DISASTER · fire" in both
    self_only = str(assess_component(d, "self"))
    assert "model says" in self_only and "we measured" not in self_only
    measured_only = str(assess_component(d, "measured"))
    assert "we measured" in measured_only and "model says" not in measured_only
    assert "probes split on type" in measured_only     # the causal why


def test_assess_component_pending_and_violations():
    from agentic.ui import assess_component
    assert "waiting for perception" in str(assess_component(derive(EVENTS), "both"))
    bad = ASSESS_EVENTS[:2] + [
        {"type": "assess_violation", "kind": "scenario_no_level_gt0",
         "evidence": "No but level=3"},
        {"type": "assess_verdict", "scenario": "No", "disaster_type": "N/A",
         "level": 3, "bucket": "minor", "self_confidence": None,
         "n_violations": 1},
        {"type": "stage_done", "stage": "assess"}]
    out = str(assess_component(derive(EVENTS + bad), "both"))
    assert "scenario no level gt0" in out
    assert "NO DISASTER" in out


def test_scene_wears_verdict_ribbon_tinted_by_bucket():
    d = derive(EVENTS + ASSESS_EVENTS)
    txt = str(scene_component(d, "data:image/png;base64,x"))
    assert "DISASTER · fire" in txt and "CATASTROPHIC" in txt
    assert "#ef4444" in txt                       # catastrophic tint


def test_instruments_gain_bucket_and_u_tiles():
    d = derive(EVENTS + ASSESS_EVENTS)
    txt = str(instruments_component(d))
    assert "CATASTROPHIC" in txt and "measured U" in txt


def test_assess_replay_events_from_frozen_record(tmp_path, monkeypatch):
    """A frozen assessment json replays as a full PHASE 1 event stream."""
    import json as _json

    from agentic import ui as ui_mod
    monkeypatch.setattr(ui_mod, "ASSESSMENT_DIR", tmp_path)
    (tmp_path / "X_scene__assessment.json").write_text(_json.dumps({
        "assessment": {"disaster_scenario": "Yes", "disaster_type": "flood",
                       "disaster_level": 5, "severity_bucket": "serious",
                       "self_confidence": 0.9, "reasoning": ""},
        "parse_notes": ["level_clamped(11)->10"],
        "violations": [],
        "measured_uncertainty": {"n_probes": 5, "score": 0.1,
                                 "scenario_agreement": 1.0,
                                 "type_agreement": 0.8,
                                 "bucket_agreement": 1.0,
                                 "level": {"min": 5, "max": 6, "std": 0.4},
                                 "drivers": [{"kind": "type_split",
                                              "evidence": "e", "action": "a"}],
                                 "explanation": "why-line",
                                 "explainer": "llm"}}))
    record = {"detected_objects": [
        {"object_id": "river_1", "state_kind": "hazard_bearing"}]}
    ev = ui_mod.build_assess_replay_events("X_scene", record)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "stage_started" and kinds[-1] == "stage_done"
    assert "assess_verdict" in kinds and "assess_uncertainty" in kinds
    d = derive(EVENTS + ev)
    assert d["assess"]["verdict"]["bucket"] == "serious"
    assert d["assess"]["uncertainty"]["explanation"] == "why-line"
    # missing file -> no events, replay still works
    assert ui_mod.build_assess_replay_events("nope", record) == []


def test_assess_replay_survives_malformed_frozen_json(tmp_path, monkeypatch):
    from agentic import ui as ui_mod
    monkeypatch.setattr(ui_mod, "ASSESSMENT_DIR", tmp_path)
    (tmp_path / "Y__assessment.json").write_text("{broken json")
    assert ui_mod.build_assess_replay_events("Y", {}) == []


def test_assess_component_shows_entity_lists_with_granular_u():
    from agentic.ui import assess_component
    merged = ASSESS_EVENTS[:2] + [
        {"type": "assess_verdict", "scenario": "Yes",
         "disaster_type": "house fire", "level": 7, "bucket": "catastrophic",
         "self_confidence": 0.9, "n_violations": 0,
         "threats": [{"object_id": "house_1", "reason": "burning"}],
         "at_risk": [{"object_id": "person_1", "kind": "proximity",
                      "reason": "adjacent to house_1"}]},
        {"type": "assess_uncertainty", "score": 0.13, "n_probes": 5,
         "scenario_agreement": 1.0, "type_agreement": 1.0,
         "bucket_agreement": 1.0,
         "granular": {"fields": {},
                      "threats": {"house_1": {"u": 0.0, "votes": "5/5"}},
                      "at_risk": {"person_1": {"u": 0.4, "votes": "3/5"},
                                  "dog_1": {"u": 0.8, "votes": "1/5"}}},
         "drivers": ["at_risk_membership_split"], "explanation": "x",
         "explainer": "llm"},
        {"type": "stage_done", "stage": "assess"}]
    out = str(assess_component(derive(EVENTS + merged), "both"))
    assert "house_1" in out and "U 0.0 (5/5)" in out
    assert "person_1" in out and "U 0.4 (3/5)" in out and "proximity" in out
    # dog_1 flickered in probes but is NOT in the final answer -> ghost row
    assert "dog_1" in out and "NOT" in out and "1/5" in out
    assert "disaster scenario 1.0" in out       # renamed label


def test_derive_and_render_reflection_ledger():
    from agentic.ui import assess_component
    refl = ASSESS_EVENTS[:4] + [
        {"type": "assess_uncertainty", "score": 0.3, "n_probes": 5,
         "scenario_agreement": 1.0, "type_agreement": 1.0,
         "bucket_agreement": 0.8, "granular": {}, "drivers": [],
         "explanation": "x", "explainer": "llm", "phase": "initial"},
        {"type": "reflect_round_started", "round": 1, "n_triggers": 2,
         "triggers": [{"type": "violation",
                       "kind": "threat_state_not_hazardous"},
                      {"type": "membership_split", "list": "at_risk",
                       "object_id": "dog_1", "votes": "1/5"}]},
        {"type": "reflect_round_done", "round": 1, "changed": True,
         "violations_after": 0},
        {"type": "reflect_stopped", "reason": "clean", "rounds": 1,
         "u_before": 0.3, "u_after": 0.1},
        {"type": "stage_done", "stage": "assess"}]
    d = derive(EVENTS + refl)
    R = d["assess"]["reflection"]
    assert R["stopped"] == "clean" and R["u_after"] == 0.1
    assert R["rounds"][0]["changed"] is True
    out = str(assess_component(d, "both"))
    assert "REFLECTION" in out and "revised" in out and "CLEAN" in out
    assert "0.3 → 0.1" in out and "−0.2" in out
    assert "ΔU alone is not proof" in out
    # activity feed narrated the loop
    assert any("reflection round 1" in a for a in d["assess"]["activities"])
    assert any("U 0.3 -> 0.1" in a for a in d["assess"]["activities"])


# ── Increment-2 UI: tickets, round detail, verdict diff, GT overlay ─────

MERGED_REFL = [
    {"type": "stage_started", "stage": "assess"},
    {"type": "assess_context", "n_entities": 2, "hazard_ids": [],
     "at_risk_ids": ["child_1", "child_2"]},
    {"type": "assess_violation", "kind": "threat_state_not_hazardous",
     "evidence": "child_1 listed as threat but state 'drowning' is at_risk"},
    {"type": "assess_verdict", "scenario": "Yes",
     "disaster_type": "drowning", "level": 9, "bucket": "catastrophic",
     "self_confidence": 0.9, "n_violations": 1,
     "threats": [{"object_id": "child_1", "reason": "drowning"}],
     "at_risk": [{"object_id": "child_1", "kind": "distress", "reason": "d"}]},
    {"type": "reflect_round_started", "round": 1, "n_triggers": 1,
     "instruction": "PROBLEM: child_1 as threat...\nRULE S5: A threat must",
     "triggers": [{"type": "violation",
                   "kind": "threat_state_not_hazardous",
                   "evidence": "child_1 as threat"}]},
    {"type": "reflect_round_done", "round": 1, "changed": True,
     "violations_after": 0, "violations_after_kinds": []},
    {"type": "assess_verdict", "scenario": "Yes",
     "disaster_type": "drowning", "level": 9, "bucket": "catastrophic",
     "self_confidence": 0.9, "n_violations": 0, "threats": [],
     "at_risk": [{"object_id": "child_1", "kind": "distress", "reason": "d"},
                 {"object_id": "child_2", "kind": "distress", "reason": "t"}]},
    {"type": "reflect_stopped", "reason": "clean", "rounds": 1,
     "u_before": 0.3, "u_after": 0.1},
    {"type": "stage_done", "stage": "assess"},
]


def test_assess_tickets_lifecycle_open_fixing_fixed():
    # mid-round: ticket is FIXING
    d_mid = derive(EVENTS + MERGED_REFL[:5])
    tks = list(d_mid["assess"]["tickets"].values())
    assert tks[0]["status"] == "fixing"
    # after the round resolved it: FIXED
    d_done = derive(EVENTS + MERGED_REFL)
    tks = list(d_done["assess"]["tickets"].values())
    assert tks[0]["status"] == "fixed"
    from agentic.ui import assess_component
    out = str(assess_component(d_done, "both"))
    assert "FIXED" in out and "RULE S5" in out       # rulebook in the body


def test_assess_tickets_stood_when_cap_reached():
    ev = [e for e in MERGED_REFL]
    ev[5] = {"type": "reflect_round_done", "round": 1, "changed": True,
             "violations_after": 1,
             "violations_after_kinds": ["threat_state_not_hazardous"]}
    ev[7] = {"type": "reflect_stopped", "reason": "cap_reached",
             "rounds": 1, "u_before": 0.3, "u_after": None}
    d = derive(EVENTS + ev)
    assert list(d["assess"]["tickets"].values())[0]["status"] == "stood"


def test_round_detail_shows_instruction_and_verdict_diff():
    from agentic.ui import assess_component
    d = derive(EVENTS + MERGED_REFL)
    assert len(d["assess"]["verdict_history"]) == 2
    out = str(assess_component(d, "both"))
    assert "WHAT THE RULEBOOK SENT" in out
    assert "RULE S5: A threat must" in out           # the instruction text
    assert "BEFORE:" in out and "AFTER:" in out
    assert "threats: child_1" in out and "threats: -" in out


def test_gt_overlay_renders_for_calibration_scene_only():
    from agentic.ui import assess_component, gt_eval_overlay
    d = derive(EVENTS + MERGED_REFL)
    # unknown scene -> overlay absent (the production shape)
    assert gt_eval_overlay(d, "ui_abc123") is None
    assert gt_eval_overlay(d, None) is None
    # calibration scene -> overlay present with quadrant + errors
    ge = gt_eval_overlay(d, "B_pool")
    assert ge is not None
    assert ge["pre"]["error_score"] > ge["post"]["error_score"]  # improved
    assert "REFINEMENT" in ge["quadrant"] or "improvement" in ge["quadrant"]
    out = str(assess_component(d, "both", record_name="B_pool"))
    assert "THE JUDGES" in out and "GROUND TRUTH" in out
    assert "error 5.0 -> " in out or "error " in out   # GT card verdict line
    assert "RUN PAIRWISE JUDGE" in out               # button offered
    out2 = str(assess_component(d, "both", record_name="B_pool",
                                judge_result={"winner": "post",
                                              "raw": "B covers both"}))
    assert "winner: POST" in out2
    running = str(assess_component(d, "both", record_name="B_pool",
                                   judge_result={"status": "running"}))
    assert "deliberating" in running


def test_judges_roster_shows_task_allocation():
    """Every judge card carries the uniform anatomy: TASK / AUTHORITY /
    THIS RUN — the interpretability contract."""
    from agentic.ui import assess_component
    with_mu = MERGED_REFL[:-1] + [
        {"type": "assess_uncertainty", "score": 0.1, "n_probes": 5,
         "scenario_agreement": 1.0, "type_agreement": 1.0,
         "bucket_agreement": 1.0, "granular": {}, "drivers": [],
         "explanation": "x", "explainer": "llm"},
        {"type": "stage_done", "stage": "assess"}]
    d = derive(EVENTS + with_mu)
    out = str(assess_component(d, "both"))
    for judge in ("CODE CHECKS", "RULEBOOK", "PROBE METER", "GEOMETRY",
                  "PAIRWISE (LLM)", "GROUND TRUTH"):
        assert judge in out, judge
    assert out.count("TASK ") >= 5 and out.count("AUTHORITY ") >= 5
    assert "never edits" in out                     # the iron rule, visible
    assert "no GT for this scene (production shape)" in out  # honest absence
    assert "cited: S5" in out                       # rulebook's live verdict


def test_zero_probe_membership_shows_max_uncertainty():
    """house_1 case: in the final at-risk list but in 0/5 probe lists —
    must badge U 1.0 (0/5), never silence."""
    from agentic.ui import assess_component
    ev = MERGED_REFL[:4] + [
        {"type": "assess_uncertainty", "score": 0.2, "n_probes": 5,
         "scenario_agreement": 1.0, "type_agreement": 1.0,
         "bucket_agreement": 1.0,
         "granular": {"fields": {}, "threats": {},
                      "at_risk": {}},              # probes never saw anyone
         "drivers": [], "explanation": "x", "explainer": "llm"},
        {"type": "stage_done", "stage": "assess"}]
    out = str(assess_component(derive(EVENTS + ev), "both"))
    assert "U 1.0 (0/5)" in out


def test_ribbon_calms_down_after_assessment():
    ev = MERGED_REFL[:4] + [
        {"type": "assess_probe", "index": 4, "scenario": "Yes", "level": 9,
         "bucket": "catastrophic"},
        {"type": "assess_uncertainty", "score": 0.1, "n_probes": 5,
         "scenario_agreement": 1.0, "type_agreement": 1.0,
         "bucket_agreement": 1.0, "granular": {}, "drivers": [],
         "explanation": "x", "explainer": "llm"},
        {"type": "stage_done", "stage": "assess"}]
    d = derive(EVENTS + ev)
    assert d["activity"]["busy"] is False
    assert "probing" not in d["activity"]["text"]
    assert "assessment done" in d["activity"]["text"]


def test_judge_cards_and_verdict_change_chips():
    from agentic.ui import assess_component
    d = derive(EVENTS + MERGED_REFL)
    out = str(assess_component(d, "both"))
    # judge cards for the round's contributors
    assert "CODE CHECKS" in out and "RULEBOOK" in out
    assert "cited: S5" in out
    # verdict-change chips: threats lost child_1, at-risk gained child_2
    assert "VERDICT CHANGE:" in out
    assert "−child_1 (threats)" in out
    assert "+child_2 (at_risk)" in out


def test_gt_overlay_matches_live_run_by_image_name():
    from agentic.ui import gt_eval_overlay
    ev = [dict(EVENTS[0])] + EVENTS[1:] + MERGED_REFL
    ev[0] = {"type": "run_started", "image_size": [400, 300],
             "caption": "c", "image_name": "B_pool.png"}
    d = derive(ev)
    ge = gt_eval_overlay(d, "ui_12345")       # live run id, no GT match
    assert ge is not None and ge["matched_by"] == "image_name"


def test_round_summary_labels_runoff_not_none_none():
    ev = MERGED_REFL[:4] + [
        {"type": "reflect_round_started", "round": 1, "n_triggers": 2,
         "instruction": "x",
         "triggers": [{"type": "violation", "kind": "k1", "evidence": "e"},
                      {"type": "candidate_runoff", "winner": "top1",
                       "raw": "r", "top1_votes": "3/5"}]},
        {"type": "stage_done", "stage": "assess"}]
    d = derive(EVENTS + ev)
    summ = d["assess"]["reflection"]["rounds"][0]["summary"]
    assert "None" not in summ and "runoff advice (top1)" in summ


def test_judge_cards_carry_specifics():
    from agentic.ui import assess_component
    ev = MERGED_REFL[:2] + [
        dict(MERGED_REFL[1], n_spatial_hints=3),
    ] + MERGED_REFL[2:4] + [
        {"type": "assess_uncertainty", "score": 0.3, "n_probes": 5,
         "scenario_agreement": 1.0, "type_agreement": 1.0,
         "bucket_agreement": 1.0,
         "granular": {"fields": {},
                      "threats": {"fire_1": {"u": 0.4, "votes": "3/5"}},
                      "at_risk": {}},
         "drivers": ["threat_membership_split"], "explanation": "x",
         "explainer": "llm"},
        {"type": "assess_runoff", "winner": "top1", "raw": "cites burning",
         "top1_votes": "3/5", "top2_votes": "2/5",
         "top1_line": "Yes·L7 threats[fire_1]",
         "top2_line": "Yes·L4 threats[-]"},
        {"type": "stage_done", "stage": "assess"}]
    out = str(assess_component(derive(EVENTS + ev), "both",
                               judge_result={"winner": "pre",
                                             "raw": "prefers minimal",
                                             "pre_line": "Yes·L9 threats[a]",
                                             "post_line": "Yes·L9 threats[-]"}))
    assert "threat state not hazardous → " in out       # problem→status
    assert "S5 → for threat state not hazardous" in out  # rule→problem
    assert "fire_1: 3/5 threats lists" in out            # probe specifics
    assert "judged: Yes·L7 threats[fire_1] (3/5)" in out  # runoff readings
    assert "judged: pre  Yes·L9 threats[a]" in out       # pairwise readings
    assert "winner: PRE — prefers minimal" in out


def test_petition_events_narrate_and_render():
    from agentic.ui import assess_component
    ev = MERGED_REFL[:4] + [
        {"type": "petition_started", "reasons": [
            {"kind": "threat_state_not_hazardous",
             "evidence": "child_2 as threat", "locates": "child_2"}]},
        {"type": "petition_done", "added": ["pool·engulfing"],
         "removed": [], "n_petitioned": 1},
        {"type": "petition_outcome", "resolved": True,
         "violations_before": ["threat_state_not_hazardous"],
         "violations_after": []},
        {"type": "stage_done", "stage": "assess"}]
    d = derive(EVENTS + ev)
    P = d["assess"]["petition"]
    assert P["status"] == "merged" and P["added"] == ["pool·engulfing"]
    assert P["outcome"]["resolved"] is True
    assert any("PETITION to Stage 1" in a for a in d["assess"]["activities"])
    out = str(assess_component(d, "both"))
    assert "PETITION → STAGE 1" in out
    assert "+pool·engulfing" in out
    assert "RESOLVED" in out


def test_failed_petition_renders_honestly():
    from agentic.ui import assess_component
    ev = MERGED_REFL[:4] + [
        {"type": "petition_started", "reasons": [
            {"kind": "false_alarm_incoherence", "evidence": "e"}]},
        {"type": "petition_failed", "error": "detector grounded nothing"},
        {"type": "stage_done", "stage": "assess"}]
    d = derive(EVENTS + ev)
    out = str(assess_component(d, "both"))
    assert "FAILED" in out and "pathology signal" in out


def test_petition_snapshots_epoch0_and_resets_for_rerun():
    """Sunny: keep the first pass visible. petition_started freezes both
    stages as epoch0; the re-run builds fresh state underneath."""
    rerun = [
        {"type": "petition_started", "reasons": [
            {"kind": "threat_state_not_hazardous", "evidence": "e",
             "locates": "child_2"}]},
        # the re-perception re-fires standard stage events
        {"type": "stage_started", "stage": "Perceive"},
        {"type": "stage_done", "stage": "Perceive", "seconds": 9.0,
         "n_entities": 3},
        {"type": "petition_done", "added": ["pool·engulfing"],
         "removed": [], "n_petitioned": 1},
        {"type": "stage_started", "stage": "assess"},
        {"type": "assess_verdict", "scenario": "Yes",
         "disaster_type": "drowning", "level": 9, "bucket": "catastrophic",
         "self_confidence": 0.9, "n_violations": 0,
         "threats": [{"object_id": "pool_1", "reason": "engulfing"}],
         "at_risk": []},
        {"type": "petition_outcome", "resolved": True,
         "violations_before": ["threat_state_not_hazardous"],
         "violations_after": []},
        {"type": "stage_done", "stage": "assess"}]
    d = derive(EVENTS + MERGED_REFL + rerun)
    # epoch0 froze the FIRST pass
    e0 = d["epoch0"]
    assert e0["stages"]["Perceive"]["status"] == "done"
    assert e0["assess"]["verdict"]["threats"] == []       # post-reflection
    assert len(e0["assess"]["verdict_history"]) == 2
    # the live state is the SECOND pass
    assert d["assess"]["verdict"]["threats"][0]["object_id"] == "pool_1"
    assert len(d["assess"]["verdict_history"]) == 1
    assert d["stages"]["Repair"]["status"] == "pending"   # reset


def test_petition_panel_shows_cross_epoch_comparison():
    from agentic.ui import assess_component
    rerun = [
        {"type": "petition_started", "reasons": [
            {"kind": "threat_state_not_hazardous", "evidence": "e"}]},
        {"type": "petition_done", "added": ["pool·engulfing"],
         "removed": [], "n_petitioned": 1},
        {"type": "assess_verdict", "scenario": "Yes",
         "disaster_type": "drowning", "level": 9, "bucket": "catastrophic",
         "self_confidence": 0.9, "n_violations": 0,
         "threats": [{"object_id": "pool_1", "reason": "engulfing"}],
         "at_risk": []},
        {"type": "petition_outcome", "resolved": True,
         "violations_before": ["threat_state_not_hazardous"],
         "violations_after": []}]
    d = derive(EVENTS + MERGED_REFL + rerun)
    out = str(assess_component(d, "both"))
    # The ending is always shown: a grey BEFORE line and a loud FINAL
    # DECISION banner with the full answer (Sunny: no scrolling back).
    assert "BEFORE:" in out and "FINAL DECISION" in out
    assert "threats: pool_1" in out


def test_petition_panel_final_decision_when_nothing_changed():
    """Empty merge, no cascade: the panel must still end with the FINAL
    DECISION line (marked unchanged), not trail off."""
    from agentic.ui import assess_component
    rerun = [
        {"type": "petition_started", "reasons": [
            {"kind": "threat_state_not_hazardous", "evidence": "e"}]},
        {"type": "petition_done", "added": [], "removed": [],
         "disputed": ["person·standing"], "n_petitioned": 0,
         "note": "second look omitted entities; recorded as dispute"},
    ]
    d = derive(EVENTS + MERGED_REFL + rerun)
    out = str(assess_component(d, "both"))
    assert "FINAL DECISION (unchanged)" in out


def test_stage2_petition_epoch_keeps_perception_panels_alive():
    """Live crash regression (Sunny's screenshot, KeyError 'stages'):
    a stage-2 petition snapshots only the assessment; the rail must
    render fine and the assess panel must carry the FRESH RE-ASK story."""
    from agentic.ui import assess_component
    rerun = [
        {"type": "petition_started", "target": "stage2", "reasons": [
            {"kind": "threat_state_not_hazardous", "evidence": "e"}]},
        {"type": "assess_verdict", "scenario": "Yes",
         "disaster_type": "collapse", "level": 9, "bucket": "catastrophic",
         "self_confidence": 0.9, "n_violations": 0,
         "threats": [{"object_id": "building_1", "reason": "collapsed"}],
         "at_risk": []},
        {"type": "petition_done", "added": [], "removed": [],
         "disputed": [], "n_petitioned": 0,
         "note": "fresh answer CHANGED the decision"},
        {"type": "petition_outcome", "resolved": True,
         "violations_before": ["threat_state_not_hazardous"],
         "violations_after": []}]
    d = derive(EVENTS + MERGED_REFL + rerun)
    e0 = d["epoch0"]
    assert "stages" not in e0 and "assess" in e0     # assessment-only snapshot
    assert d["stages"]["Repair"]["status"] == "done"  # perception NOT reset
    rail_component(d)                                 # must not crash
    instruments_component(d)
    out = str(assess_component(d, "both"))
    assert "SAME STAGE" in out                        # target-aware title
    assert "FINAL DECISION" in out


def test_progress_strip_shows_stage2_steps():
    """Sunny: the top bar showed stage 1's steps but not stage 2's.
    The strip must carry Answer / Probes / Reflect chips too."""
    from agentic.ui import progress_strip
    d = derive(EVENTS + MERGED_REFL)
    out = str(progress_strip(d))
    for name in ("Perceive", "Assemble", "Answer", "Probes", "Reflect"):
        assert name in out
    # never crashes at any prefix, including pending stage 2
    for k in range(len(EVENTS) + 1):
        progress_strip(derive(EVENTS[:k]))


def test_phase_status_badges():
    """Sunny: show running-status ON the stage cards, not a top bar."""
    from agentic.ui import phase_status_span, pipeline_steps
    # mid-run: stage 1 active
    d = derive(EVENTS[:4])
    s1, s2 = pipeline_steps(d)
    b = phase_status_span(s1)
    assert "●" in str(b.children) and "step" in str(b.children)
    assert "○ waiting" in str(phase_status_span(s2).children)
    # finished stage 1
    d2 = derive(EVENTS + [{"type": "stage_done", "stage": s}
                          for s in ("Ground", "Bind", "Mask", "Assemble")])
    s1b, _ = pipeline_steps(d2)
    assert "✓ done" in str(phase_status_span(s1b).children)
    # empty steps never crash
    phase_status_span([])


def test_merged_duplicate_leaves_the_picture():
    """Sunny's E_collapse re-run: the record was clean but person_2/3
    boxes stayed on the image — the scene draws from the bind
    accumulators. A duplicate_merged event must remove the dropped id
    from bound boxes and anchors."""
    ev = EVENTS + [
        {"type": "anchors_ready", "entities": [
            {"object_id": "person_2", "label": "person",
             "state": "standing", "description": "",
             "anchor_bbox": [443, 660, 558, 941]}]},
        {"type": "entity_bound", "object_id": "person_2",
         "box_source": "dino_matched", "bbox": [443, 660, 558, 941],
         "confidence": 0.9},
        {"type": "duplicate_merged", "kept": "police_officer_1",
         "dropped": "person_2", "iou": 0.93}]
    d = derive(ev)
    assert "person_2" not in d["bound"]
    assert all(a.get("object_id") != "person_2" for a in d["anchors"])
    assert "tanker_truck_1" in d["bound"]          # others untouched
    scene_component(d, "data:image/png;base64,x")  # renders fine
