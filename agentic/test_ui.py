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
    stage4_component,
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


def test_retrieval_shadow_renders():
    """rag/both mode emits a retrieval_shadow event; the assess panel shows
    the exact-key vs RAG top-1 comparison for the rules the run used."""
    from agentic.ui import assess_component
    ev = EVENTS + [
        {"type": "stage_started", "stage": "assess"},
        {"type": "assess_verdict", "scenario": "Yes", "disaster_type": "fire",
         "level": 7, "bucket": "catastrophic", "self_confidence": 0.9,
         "n_violations": 0, "threats": [], "at_risk": []},
        {"type": "retrieval_shadow", "rows": [
            {"kind": "hazard_not_in_threats", "exact_rule_id": "S6",
             "rag_kind": "threat_reason_victim_shaped", "rag_rule_id": "S8",
             "agree": False, "score": 4.0, "backend": "llamaindex_chroma"},
            {"kind": "state_out_of_vocab", "exact_rule_id": "P3",
             "rag_kind": "state_out_of_vocab", "rag_rule_id": "P3",
             "agree": True, "score": 9.0, "backend": "llamaindex_chroma"}]},
    ]
    d = derive(ev)
    assert d["retrieval_shadow"] and len(d["retrieval_shadow"]) == 2
    out = str(assess_component(d, "both"))
    assert "RAG SHADOW" in out and "1/2 rules agree" in out
    assert "S6" in out and "S8" in out and "differs" in out
    # a fired rule differs but the diff event hasn't landed yet -> "comparing"
    assert "Comparing answers" in out


def test_fallback_reason_shows_in_panel():
    """When the RAG engine is the word-count fallback, the panel must say
    WHY the real embedding didn't run — never a silent keyword result."""
    from agentic.ui import tickets_component
    ev = EVENTS + [
        {"type": "retrieval_shadow", "stage": "perceive",
         "backend_reason": "ProxyError: 403 Forbidden (bge download blocked)",
         "rows": [
             {"kind": "state_out_of_vocab", "exact_rule_id": "P3",
              "rag_kind": "state_out_of_vocab", "rag_rule_id": "P3",
              "agree": True, "score": 5.0, "backend": "keyword_fallback"}]},
    ]
    d = derive(ev)
    assert d["retrieval_backend_reason"]
    out = str(tickets_component(d))
    assert "word-count fallback" in out          # engine named plainly
    assert "real embedding OFF" in out           # and WHY
    assert "403" in out


def test_real_embedding_engine_named_plainly():
    """When the real vector engine answered, the panel names it as the
    semantic engine and shows no fallback warning."""
    from agentic.ui import tickets_component
    ev = EVENTS + [
        {"type": "retrieval_shadow", "stage": "perceive",
         "backend_reason": None,
         "rows": [
             {"kind": "state_out_of_vocab", "exact_rule_id": "P3",
              "rag_kind": "state_out_of_vocab", "rag_rule_id": "P3",
              "agree": True, "score": 0.8, "backend": "llamaindex_chroma"}]},
    ]
    out = str(tickets_component(derive(ev)))
    assert "real embedding (semantic)" in out
    assert "real embedding OFF" not in out


def test_stage4_panel_renders_from_event():
    """A stage4_result event flows through derive → stage4_component and the
    panel shows the three picks + agreement, the recommendations with quads,
    the assumptions advisory, and both graph sizes. (Phase 1a wiring.)"""
    s4 = {"frame": {}, "recommendations": [
        {"rank": 1, "action": "Rescue person_1 from the upper window.",
         "reason": "Because building_1 is collapsed, it may_harm person_1.",
         "related_object_ids": ["building_1", "person_1"],
         "structured_reasoning": {"threat": "building_1", "state": "collapsed",
                                  "effect": "may_harm",
                                  "affected_objects": ["person_1"]},
         "expected_consequence": "person_1 is out.",
         "remaining_risk": "(dust_1, rising)",
         "possible_follow_up_action": "oxygen"}],
        "advisory": [{"suspected": "occupants inside building_1",
                      "anchor_object_id": "building_1", "cue": "residential",
                      "suggested_action": "search"}],
        "graph_a": {"nodes": [
            {"id": "building_1", "label": "building", "state": "collapsed",
             "hazardous": True},
            {"id": "person_1", "label": "person", "state": "trapped",
             "hazardous": False}],
            "edges": [{"source": "building_1", "target": "person_1",
                       "effect": "may_harm", "via_state": "collapsed"}],
            "graph_warnings": []},
        "graph_b": {"nodes": [
            {"id": "building_1", "label": "building", "state": "collapsed",
             "hazardous": True}], "edges": []},
        "picks": {"a_pick": {"threat": "dust_1"},
                  "b_pick": {"threat": "building_1"},
                  "llm_pick": {"threat": "building_1"},
                  "agreement": 0.667, "unanimous": False}, "parse_notes": []}
    d = derive([{"type": "stage4_result", "result": s4}])
    assert d["stage4"] is not None
    out = str(stage4_component(d))
    for needle in ("SUPPRESSION TARGET (FOR THE CAUSAL TEST)", "dust_1", "building_1", "0.667",
                   "they disagree", "Rescue person_1", "may_harm",
                   "ASSUMPTIONS ADVISORY", "GRAPH A", "GRAPH B"):
        assert needle in out, f"missing {needle!r}"


def test_stage4_conformance_and_alignment_panels():
    """Phase 1b: the conformance breakdown (by failure pattern, corrected +
    Arm A raw) and the A-vs-B alignment render in the Stage 4 panel."""
    s4 = {"frame": {}, "recommendations": [], "advisory": [],
          "graph_a": {"nodes": [], "edges": []},
          "graph_b": {"nodes": [], "edges": []},
          "picks": {"a_pick": {"object_id": "h_1"},
                    "b_pick": {"object_id": "h_1"},
                    "llm_pick": {"object_id": "h_1"},
                    "agreement": 1.0, "unanimous": True},
          "conformance": {"validity": 0.54, "n_issues": 3,
                          "raw_a_validity": 0.5, "raw_b_validity": 0.0,
                          "breakdown": [
                              {"category": "role mix-up", "count": 1,
                               "severity": 2, "examples": ["hazard_flag: h_1"]},
                              {"category": "cosmetic", "count": 2,
                               "severity": 0, "examples": ["node_budget: x"]}]},
          "internal_alignment": {"score": 0.7, "n_failures": 1, "breakdown": [
              {"category": "coverage gap", "count": 1, "severity": 2,
               "examples": ["at-risk dog_1 not addressed"]}]},
          "alignment": {"a_fidelity": 0.5, "b_coverage": 0.33,
                        "structural": 0.4, "a_only": ["x"], "b_only": ["y", "z"]}}
    out = str(stage4_component(derive([{"type": "stage4_result", "result": s4}])))
    # F37: conformance no longer has its own panel — the findings render under
    # the graph or card they judge, and the ROLLUP sits at the foot of the
    # graph section rather than above the things it summarises.
    # F46: the score now says which way is good ("clean"), and states that it
    # covers both graphs and the cards together — there is no per-graph number,
    # which is why each graph's CONFORMANCE band shows counts and not a score.
    assert "0.54 clean" in out
    assert "there is no per-graph number" in out
    assert "Arm A raw" in out                      # frozen number kept
    assert "ALIGNMENT ·" in out and "DIVERGE" in out         # self-consistency reframe
    assert "ALIGNMENT (rung 2)" not in out          # NOT mislabeled as an intervention rung
    # F43: named, so the screen connects to "a_fidelity < 0.4 fires sycophancy"
    assert "a_fidelity" in out and "b_coverage" in out
    # ...and the three lines that served none of the six objectives are gone
    assert "overall agreement" not in out
    assert "same whole claim" not in out and "same who-harms-whom" not in out
    # F43: the Pearl's-rung-2 footnote printed on every run and belongs in the
    # docs. The panel header still says what the check IS.             # rung 2 named only to say we don't do it
    # F43: the edge-level dumps are gone — fourteen lines restating what the
    # two entity lines already say. Those entity lines are what remains.            # b_only wording
    assert "y" in out and "z" in out                # actual disagreeing edges shown


def test_stage4_uncertainty_panel_renders():
    """Phase 1b: the measured-uncertainty panel shows the score, the flipped
    top target, and the mechanism that won't commit."""
    s4 = {"frame": {}, "recommendations": [], "advisory": [],
          "graph_a": {"nodes": [], "edges": []},
          "graph_b": {"nodes": [], "edges": []},
          "picks": {"a_pick": {"object_id": "house_1"},
                    "b_pick": {"object_id": "house_1"},
                    "llm_pick": {"object_id": "car_1"},
                    "agreement": 0.667, "unanimous": False},
          "uncertainty": {"n_probes": 5, "score": 0.34,
                          "explanation": "probes split",
                          "granular": {
                              "fields": {
                                  "top_priority_target": {
                                      "u": 0.4, "evidence": "house_1×3, car_1×2"},
                                  "recommendation_count": {
                                      "u": 0.0, "evidence": "2×5"}},
                              "threats": {"car_1": {"u": 0.4, "votes": "3/5"}},
                              "affected": {},
                              "effects": {"house_1": {
                                  "u": 0.4, "votes": "3/5",
                                  "evidence": "may_harm×3, may_spread_to×2"}}},
                          "candidates": [{"votes": 3, "edges": []},
                                         {"votes": 2, "edges": []}],
                          "drivers": []}}
    out = str(stage4_component(derive([{"type": "stage4_result", "result": s4}])))
    assert "MEASURED UNCERTAINTY" in out and "0.34" in out
    assert "house_1" in out and "car_1" in out and "×3" in out  # top-target flip, chipped
    assert "mechanism won't commit" in out          # per-threat effect wobble
    assert "DIFFERENT recommendation sets" in out   # distinct sets surfaced (renamed)


def test_stage4_trust_panel_and_per_rec_uncertainty():
    """The trust panel renders as its own headline (score, band, ranked why),
    and each recommendation card carries its granular uncertainty slice — the
    car_1 'never reappeared' case."""
    s4 = {"frame": {}, "advisory": [],
          "graph_a": {"nodes": [{"id": "car_1", "label": "car"}], "edges": []},
          "graph_b": {"nodes": [], "edges": []},
          "picks": {"a_pick": {"object_id": "house_1"}, "agreement": 0.667},
          "recommendations": [
              {"rank": 3, "action": "tow the car", "reason": "car_1 blocks",
               "structured_reasoning": {"threat": "car_1", "effect":
                                        "blocks_access_to", "affected_objects":
                                        ["house_1"]}}],
          "uncertainty": {"n_probes": 5, "score": 0.2,
                          "granular": {"threats": {"house_1": {"u": 0.0,
                                                               "votes": "5/5"}},
                                       "effects": {}, "fields": {}},
                          "candidates": []},
          "trust": {"score": 0.55, "band": "moderate", "global_penalty": 0.45,
                    "explanation": "Trust is moderate (0.55). Biggest reason: "
                                   "the recommendations diverge...",
                    "contributors": [
                        {"signal": "ab_alignment", "contribution": 0.21,
                         "text": "the recommendations diverge from the model's "
                                 "own independent causal graph",
                         "evidence": "agreement 0.3"},
                        {"signal": "conformance", "contribution": 0.0,
                         "text": "well-formed", "evidence": "validity 1.0"}],
                    "per_rec": [{"rank": 3, "threat": "car_1", "score": 0.0,
                                 "worst_contributor": {"signal": "uncertainty",
                                     "text": "car_1 never reappeared in 5 re-asks"},
                                 "contributors": []}]}}
    out = str(stage4_component(derive([{"type": "stage4_result", "result": s4}])))
    # trust headline panel
    assert "TRUST · can we trust" in out and "moderate" in out
    assert "what pulls trust down" in out
    assert "the recommendations diverge" in out          # ranked contributor
    assert "every recommendation" in out and "trust 0.0" in out  # all recs, best→worst
    assert "what to do:" in out                          # band interpretation line
    # per-recommendation granular uncertainty on the card
    assert "never reappeared in 5 re-asks" in out


def _stage4_pin_fixture():
    ga = {"nodes": [{"id": "house_1", "hazardous": True, "state": "burning",
                     "label": "house"},
                    {"id": "person_1", "hazardous": False, "state": "trapped",
                     "label": "person"}],
          "edges": [{"source": "house_1", "target": "person_1",
                     "effect": "may_harm"}]}
    d = {"image_size": [100, 100], "bound": {}, "anchors": [],
         "result": {"detected_objects": [
             {"object_id": "house_1", "bbox": [10, 10, 40, 40]},
             {"object_id": "person_1", "bbox": [60, 60, 80, 90]}]},
         "stage4": {"graph_a": ga,
                    "recommendations": [{"rank": 1, "action": "evacuate person_1",
                        "structured_reasoning": {"threat": "house_1",
                            "effect": "may_harm", "affected_objects": ["person_1"]}}],
                    "uncertainty": {"n_probes": 5, "granular": {
                        "threats": {"house_1": {"u": 0.0, "votes": "5/5"}},
                        "effects": {"house_1": {"u": 0.4, "votes": "3/5",
                                                "evidence": "may_harm×3"}}}},
                    "trust": {"per_rec": [{"rank": 1, "threat": "house_1",
                                           "score": 0.8, "consequence_band": "high"}]}}}
    return ga, d


def test_stage4_pins_hazard_and_victim():
    """ON THE MAIN IMAGE: a hazard pin (its rec + consequence + uncertainty
    flag) and a victim pin (who's at risk + the protecting rec)."""
    from agentic.ui import _make_chipper, _stage4_pins
    ga, d = _stage4_pin_fixture()
    out = str(_stage4_pins(d, _make_chipper(ga)))
    assert "house_1" in out and "person_1" in out          # both pinned
    assert "evacuate person_1" in out                      # rec in the expand
    assert "may_harm" in out                               # the harm
    assert "mechanism won't commit" in out                 # uncertainty flag
    assert "at risk" in out                                # victim pin


def test_stage4_pins_render_on_the_main_scene_canvas():
    """The pins go onto the primary scene image, not a separate panel image."""
    from agentic.ui import scene_component
    _ga, fx = _stage4_pin_fixture()
    d = derive([])                                    # full scene state...
    d.update({k: fx[k] for k in ("image_size", "result", "bound", "anchors",
                                 "stage4")})           # ...then the Stage-4 bits
    out = str(scene_component(d, "data:image/png;base64,xx"))
    assert "scene-img" in out                              # the big canvas image
    assert "evacuate person_1" in out                     # pin content is on it


def test_stage4_pins_empty_without_boxes_or_stage4():
    from agentic.ui import _make_chipper, _stage4_pins
    d = {"image_size": [100, 100], "result": {"detected_objects": []},
         "bound": {}, "anchors": [], "stage4": {"graph_a": {}}}
    assert _stage4_pins(d, _make_chipper({})) == []        # no boxes
    assert _stage4_pins({"stage4": None}, _make_chipper({})) == []  # no stage4


def test_stage4_live_running_view():
    """While Stage 4 runs (no final result yet) the body shows a live step
    checklist, not a blank — the 'shows nothing' bug."""
    d = derive([{"type": "recommendations_ready", "ranks": [1], "n_recs": 1,
                 "n_advisory": 0},
                {"type": "recommend_probe", "index": 0, "n_recs": 1,
                 "top_threat": "house_1"}])
    out = str(stage4_component(d))
    assert "running" in out and "probe 1/5" in out


def test_stage4_status_badge():
    """The Stage 4 card shows a running/done badge like Stages 1-2:
    waiting → active (with step count) → done."""
    from agentic.ui import stage4_status_span

    def badge(events):
        return str(stage4_status_span(derive(events)))
    assert "waiting" in badge([{"type": "stage_started",
                                "stage": "assess"}]).lower()
    assert "●" in badge([{"type": "stage_started", "stage": "recommend"}])
    mid = badge([{"type": "recommendations_ready", "ranks": [1], "n_recs": 1,
                  "n_advisory": 0},
                 {"type": "graph_a_built", "n_nodes": 3, "n_edges": 2}])
    assert "step 3/9" in mid    # 9 steps: the judges own their header time
    assert "done" in badge([{"type": "stage4_result",
                             "result": {"picks": {}}}])


def test_stage4_panel_empty_and_error():
    assert "after assessment" in str(stage4_component(derive([]))).lower()
    err = derive([{"type": "stage4_error", "message": "boom"}])
    assert "could not run" in str(stage4_component(err)).lower()


def test_stage1_shadow_routes_to_perception_panel():
    """A retrieval_shadow event tagged stage='perceive' must land in the
    Stage 1 (repair) panel via tickets_component — NOT the Stage 2 panel.
    This is the bug Sunny hit: repair quoted rules but the drain before
    Stage 2 swallowed them, so Stage 1 showed nothing."""
    from agentic.ui import assess_component, tickets_component
    ev = EVENTS + [
        {"type": "retrieval_shadow", "stage": "perceive", "rows": [
            {"kind": "state_out_of_vocab", "exact_rule_id": "P3",
             "rag_kind": "state_out_of_vocab", "rag_rule_id": "P3",
             "agree": True, "score": 9.0, "backend": "llamaindex_chroma"}]},
    ]
    d = derive(ev)
    assert d["retrieval_shadow_perceive"] and not d["retrieval_shadow"]
    # shows in Stage 1 (tickets), not Stage 2 (assess)
    assert "RAG SHADOW" in str(tickets_component(d))
    assert "Stage 1" in str(tickets_component(d))
    assert "RAG SHADOW" not in str(assess_component(d, "both"))


def test_retrieval_result_diff_renders_automatically():
    """When a fired rule differs, the worker auto-computes the result diff and
    emits retrieval_result_diff. The shadow panel shows it inline — no button,
    no command line. Both the 'moved' and 'identical' cases render."""
    from agentic.ui import assess_component
    base = EVENTS + [
        {"type": "stage_started", "stage": "assess"},
        {"type": "assess_verdict", "scenario": "Yes", "disaster_type": "fire",
         "level": 7, "bucket": "catastrophic", "self_confidence": 0.9,
         "n_violations": 0, "threats": [], "at_risk": []},
        {"type": "retrieval_shadow", "rows": [
            {"kind": "hazard_not_in_threats", "exact_rule_id": "S6",
             "rag_kind": "threat_reason_victim_shaped", "rag_rule_id": "S8",
             "agree": False, "score": 4.0, "backend": "llamaindex_chroma"}]},
    ]
    # answer moved
    moved = derive(base + [
        {"type": "retrieval_result_diff", "changed": True,
         "rulebook_line": "Yes · fire · L7 · threats[building_1] · at_risk[-]",
         "rag_line": "Yes · fire · L7 · threats[-] · at_risk[-]"}])
    out = str(assess_component(moved, "both"))
    assert "MOVED the answer" in out
    assert "building_1" in out and "threats[-]" in out
    assert "--diff" not in out and "Comparing answers" not in out
    # answer identical
    same = derive(base + [
        {"type": "retrieval_result_diff", "changed": False,
         "rulebook_line": "Yes · fire · L7 · threats[-] · at_risk[-]",
         "rag_line": "Yes · fire · L7 · threats[-] · at_risk[-]"}])
    assert "did NOT change it" in str(assess_component(same, "both"))


# ── The petition never leaves the ticket panel claiming the run hasn't started

def _fold(events):
    """Replay an event list through derive(), the way the UI does."""
    from agentic.ui import derive
    return derive(events)


def _panel_text(d):
    from agentic.ui import tickets_component
    return str(tickets_component(d))


def test_ticket_panel_never_says_waiting_after_a_petition():
    """ui_3049cd31: after `petition_started` the accumulators are cleared so
    the re-perception can build its own panels, which left the empty-state
    branch claiming the model had not answered yet — on a second pass that had
    already finished. Fold the real stream and assert the claim never appears
    once a petition is in flight."""
    import json
    import pathlib
    p = (pathlib.Path(__file__).parent.parent / "exports" / "agentic_runs"
         / "ui_3049cd31" / "events.jsonl")
    if not p.exists():                       # record not present in this tree
        return
    events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    start = next(i for i, e in enumerate(events)
                 if e.get("type") == "petition_started")
    for i in range(start + 1, len(events) + 1):
        text = _panel_text(_fold(events[:i]))
        assert "waiting for the model's first answer" not in text, (
            f"panel claimed the run had not started, {i - start} events "
            f"after petition_started")


def test_ticket_panel_reports_a_clean_re_perception():
    """The third empty state has to say what actually happened."""
    events = [
        {"type": "run_started", "image_size": [10, 10]},
        {"type": "petition_started", "reasons": [{"kind": "x"}]},
        {"type": "run_started", "image_size": [10, 10]},
        {"type": "petition_done", "added": [], "removed": [], "rejected": [],
         "disputed": [], "n_petitioned": 0},
    ]
    text = _panel_text(_fold(events))
    assert "re-perception raised no violations" in text
    assert "waiting for the model's first answer" not in text


def test_ticket_panel_still_says_waiting_before_any_answer():
    """The original empty state must survive for the case it was written for."""
    text = _panel_text(_fold([{"type": "run_started", "image_size": [10, 10]}]))
    assert "waiting for the model's first answer" in text


# ── A settled Stage 2 must stop pulsing, and must say what stands ───────

def _stage2_badge(d):
    from agentic.ui import phase_status_span, pipeline_steps
    _, s2 = pipeline_steps(d)
    p = ((d.get("assess") or {}).get("petition") or {})
    return phase_status_span(s2, "", settled=bool(p) and
                             p.get("status") != "in_flight")


def test_stage2_badge_stops_pulsing_after_a_petition_concludes():
    """ui_3049cd31: the petition added nothing, so Stage 2 never re-ran and no
    petition_outcome ever fires. petition_started had blanked the live
    assessment, so Answer/Probes/Reflect read 'pending', the badge hit the
    partial branch, got className 'active' — the class the CSS pulses — and
    blinked forever on a stage that was finished."""
    import json
    import pathlib
    p = (pathlib.Path(__file__).parent.parent / "exports" / "agentic_runs"
         / "ui_3049cd31" / "events.jsonl")
    if not p.exists():
        return
    events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    end = next(i for i, e in enumerate(events)
               if e.get("type") == "petition_done")
    for i in range(end + 1, len(events) + 1):
        badge = str(_stage2_badge(_fold(events[:i])))
        assert "phase-status active" not in badge, (
            f"Stage 2 still pulsing {i - end} events after petition_done")


def test_the_first_pass_is_not_lost_when_the_petition_adds_nothing():
    """Answer/Probes/Reflect DID run — the record is in epoch0. Reading the
    blanked live dict made a finished stage report pending."""
    import json
    import pathlib
    p = (pathlib.Path(__file__).parent.parent / "exports" / "agentic_runs"
         / "ui_3049cd31" / "events.jsonl")
    if not p.exists():
        return
    from agentic.ui import pipeline_steps
    events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    _, s2 = pipeline_steps(_fold(events))
    by = dict(s2)
    assert by["Answer"] == "done" and by["Probes"] == "done"
    assert by["Reflect"] == "done" and by["Second look"] == "done"


def test_a_no_change_petition_says_the_verdict_stands_and_what_is_open():
    """Silence read as 'still working'. The panel must state the outcome and
    name the pressure that is still unresolved."""
    events = [
        {"type": "run_started", "image_size": [10, 10]},
        {"type": "assess_verdict", "scenario": "Yes", "level": 7,
         "threats": [], "at_risk": [], "n_violations": 0},
        {"type": "reflect_stopped", "reason": "clean", "rounds": 0},
        {"type": "petition_started", "target": "stage1", "reasons": [
            {"kind": "caption_state_contradiction",
             "evidence": "caption describes 'unconscious' but no entity "
                         "carries that state"}]},
        {"type": "petition_done", "added": [], "removed": [], "rejected": [],
         "disputed": [], "n_petitioned": 0},
    ]
    from agentic.ui import assess_component
    text = str(assess_component(_fold(events), None, "r", None))
    assert "verdict STANDS" in text
    assert "still unresolved" in text
    assert "caption state contradiction" in text


# ── Instruction 5: panels split by producer ─────────────────────────────

def _s4_panel(stage4):
    """Render the Stage 4 panel from a stage4_result event."""
    from agentic.ui import derive, stage4_component
    d = derive([{"type": "run_started", "image_size": [10, 10]},
                {"type": "stage4_result", "result": stage4}])
    return str(stage4_component(d))


def test_each_graph_carries_its_own_conformance_score():
    """F37: the score-only CONFORMANCE panel is gone. Each graph's count and
    worst severity ride in that graph's own header, so a number never sits
    above the thing it describes."""
    s4 = {"conformance": {
        "validity": 0.6, "n_issues": 2, "breakdown": [],
        "issues": [
            {"graph": "graph_b", "entity": "child_1", "severity": 2,
             "category": "role mix-up", "rule": "edge_from_non_hazardous",
             "detail": "child_1->pool_1"},
            {"graph": "graph_b", "entity": "child_1", "severity": 1,
             "category": "inconsistency", "rule": "via_state_not_hazard_bearing",
             "detail": "child_1->pool_1: via 'drowning'"}],
        "by_graph": {
            "graph_a": {"count": 0, "max_severity": 0, "breakdown": []},
            "graph_b": {"count": 2, "max_severity": 2, "breakdown": []}}}}
    t = _s4_panel(s4)
    a, b = t.index("GRAPH A ·"), t.index("GRAPH B ·")
    assert "0 issue(s)" in t[a:b]                    # clean graph stays visible
    assert "2 issue(s) · worst severity 2" in t[b:]
    # the findings render under the graph they judge, once
    assert t.count("edge_from_non_hazardous") == 1
    assert "via_state_not_hazard_bearing" in t
    # two rules on one edge is ONE defect counted twice — say so
    assert "the same edge, 2 rules — one defect" in t


def test_the_gate_verdict_renders_under_graph_b():
    """F37: the gate is a verdict about Graph B's FITNESS, so it belongs under
    Graph B. Buried in the trust panel, a failed gate showed up only as
    "signals_measured 4/5" with no way to see why."""
    s4 = {"conformance": {"validity": 0.4, "n_issues": 1, "breakdown": [],
                          "by_graph": {
                              "graph_a": {"count": 0, "max_severity": 0,
                                          "breakdown": []},
                              "graph_b": {"count": 1, "max_severity": 2,
                                          "breakdown": []}}},
          "trust": {"score": 0.5, "band": "moderate", "contributors": [],
                    "graph_b_gate": {
                        "trusted": False,
                        "reasons": ["Graph B contradicts its own declarations "
                                    "(self-consistency 0.2, 3 issue(s))"]}}}
    t = _s4_panel(s4)
    b = t.index("GRAPH B ·")
    assert "FIT TO BE THE YARDSTICK?" in t[b:]
    assert "contradicts its own declarations" in t[b:]
    assert "meaningless number, not a low one" in t[b:]


def test_a_passing_gate_says_nothing():
    """F39. A passing gate was one green line reporting that nothing happened,
    on every clean run. Its whole value is the FAILURE case, where it explains
    why A-vs-B disappeared from trust."""
    t = _s4_panel({"trust": {"graph_b_gate": {"trusted": True, "reasons": []}}})
    assert "FIT TO BE THE YARDSTICK" not in t


def test_graph_b_uncertainty_panel_shows_belief_not_every_edge():
    """F42. It used to print EVERY edge any probe produced — sixteen lines on
    D_aerial, eleven seen once, plus a mechanism line per pair. A wall.

    What a reader needs is three things: what the model consistently BELIEVES
    (links a majority of probes repeat), where it CONTRADICTS itself, and
    whether it invented entities. The once-only tail is noise by definition —
    that is exactly what the instability score summarises."""
    s4 = {"graph_b_uncertainty": {
        "n_probes": 5, "score": 0.6, "edge_set_instability": 0.8,
        "direction_instability": 0.6, "pick_instability": 0.4, "flags": [],
        "direction_evidence": [
            {"source": "child_1", "target": "pool_1", "votes": 4, "of": 5},
            {"source": "pool_1", "target": "child_1", "votes": 3, "of": 5},
            # a minority link — seen once, must be counted and not listed
            {"source": "swing_1", "target": "child_2", "votes": 1, "of": 5}],
        "both_directions_in_one_probe": [
            {"probe": 2, "a": "child_1", "b": "pool_1"},
            {"probe": 4, "a": "child_1", "b": "pool_1"}],
        "effect_evidence": {"pool_1->child_2": {
            "exposes": 1, "increases_risk_to": 1, "may_harm": 1}},
        "pick_evidence": {"pool_1": 3, "child_1": 2}}}
    t = _s4_panel(s4)
    assert "GRAPH B UNCERTAINTY" in t          # not "stability"
    # links a MAJORITY of probes repeat are listed with their counts...
    assert "4/5 probes" in t and "3/5 probes" in t
    # ...and the once-seen tail is counted, never listed. On D_aerial that tail
    # was fifteen lines.
    assert "1 further link(s)" in t and "not shown" in t
    assert "swing_1" not in t
    assert "contradicts itself inside one response" in t
    # the numbers read as agreement, in words — "direction 0.6" left the reader
    # to work out which way is good
    assert "40% agreement" in t and "flips which end is the hazard" in t
    assert "would suppress pool_1 in 3 of 5 probes" in t


def test_the_withheld_notice_lives_under_graph_b():
    """F37 moved the gate verdict under GRAPH B, where the question "is this
    fit to be the yardstick" belongs. It used to be visible only as
    `signals_measured 4/5` in the trust panel."""
    s4 = {"graph_b_uncertainty": {
              "n_probes": 5, "score": 0.6, "edge_set_instability": 0.8,
              "direction_instability": 0.6, "pick_instability": 0.4,
              "flags": [], "direction_evidence": [], "pick_evidence": {}},
          "trust": {"score": 0.5, "band": "moderate", "contributors": [],
                    "graph_b_gate": {"trusted": False, "reasons": [
                        "the model does not reproduce its own arrows "
                        "(direction instability 0.6 over 5 probes)"]}}}
    t = _s4_panel(s4)
    assert "FIT TO BE THE YARDSTICK" in t
    assert "does not reproduce its own arrows" in t


def test_graph_b_internal_panel_renders_beside_graph_a():
    s4 = {"graph_b_internal": {
        "score": 0.4, "n_failures": 2, "measured": True,
        "breakdown": [{"category": "role contradiction", "count": 2,
                       "severity": 2, "examples": ["child_1 hazardous: false"]}]}}
    t = _s4_panel(s4)
    assert "INTERNAL ALIGNMENT (B)" in t
    assert "role contradiction" in t


def test_the_numbers_are_shown_with_a_warning_when_graph_b_fails_the_gate():
    """F44. "NOT COMPUTED" was untrue — the comparison IS computed and stored
    in every run; only TRUST withholds it. Hiding the panel also meant the
    worse graph escaped measurement: on D_aerial, Graph B named the two hazmat
    workers as the victims and failed the gate on reproducibility, while
    Graph A protected three vehicles and ignored the people — and nothing
    measured Graph A at all, because the yardstick had been disqualified.

    Show the numbers, warn at the top, and say plainly that trust ignored
    them."""
    s4 = {"alignment": {"a_fidelity": 0.0, "b_coverage": 0.0,
                        "structural": 0.0, "a_only": [], "b_only": []},
          "trust": {"score": 0.5, "band": "moderate", "contributors": [],
                    "graph_b_gate": {"trusted": False, "reasons": [
                        "conformance flags 1 serious issue(s) against Graph B"]}}}
    t = _s4_panel(s4)
    assert "a_fidelity" in t and "b_coverage" in t     # shown, not hidden
    assert "WITHHELD from trust" in t
    assert "read them as a lead, not a measurement" in t
    assert "conformance flags 1 serious issue" in t    # WHICH check failed


def test_alignment_still_renders_when_graph_b_passes():
    s4 = {"alignment": {"a_fidelity": 0.9, "b_coverage": 0.9,
                        "structural": 0.9, "a_only": [], "b_only": []},
          "trust": {"score": 0.9, "band": "high", "contributors": [],
                    "graph_b_gate": {"trusted": True, "reasons": []}}}
    t = _s4_panel(s4)
    assert "WITHHELD from trust" not in t
    assert "SAME story" in t


def test_ui_toggles_default_to_langgraph_and_both():
    """Sunny, 2026-07-28. The env defaults and the UI defaults must agree —
    a UI that silently re-selects the old path would make every run through
    the app disagree with every run through the CLI."""
    import agentic.ui as U
    from agentic.graph_live import control_flag, set_control
    from agentic.retrieval import retrieval_mode

    def _find(node, wanted):
        if getattr(node, "id", None) == wanted:
            return node
        for kid in (getattr(node, "children", None) or []
                    if isinstance(getattr(node, "children", None), list)
                    else [getattr(node, "children", None)]):
            if kid is None or isinstance(kid, str):
                continue
            hit = _find(kid, wanted)
            if hit is not None:
                return hit
        return None

    assert _find(U.app.layout, "control-mode").value == "langgraph"
    assert _find(U.app.layout, "retrieval-mode").value == "both"
    # and the env defaults agree, so CLI and UI runs match
    set_control(None)
    assert control_flag() == "langgraph"
    assert retrieval_mode() == "both"


# ── The picture must show the record the run reasoned on ────────────────

def test_a_refused_petition_reverts_the_displayed_perception():
    """ui_9d48a00e: the second look returned three `other` entities, all
    vlm_sam_fallback, so the two-witness rule refused them all and the run
    proceeded on the ORIGINAL record. But petition_started had cleared the
    live accumulators and nothing restored them, so the overlay drew the
    rejected pass — boxes for entities that were turned away, and none for the
    hazmat workers and the spill every later stage reasoned about."""
    import json
    import pathlib
    p = (pathlib.Path(__file__).parent.parent / "exports" / "agentic_runs"
         / "ui_9d48a00e" / "events.jsonl")
    if not p.exists():
        return
    from agentic.ui import derive
    d = derive([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
    shown = set(d["bound"])
    assert {"hazmat_worker_1", "hazmat_worker_2", "spill_1"} <= shown
    assert not any(o.startswith("other_") for o in shown)
    # the record the pipeline used and the picture agree
    used = {o["object_id"]
            for o in ((d.get("result") or {}).get("detected_objects") or [])}
    assert used == shown


def test_the_refused_pass_is_not_erased():
    """No-erasure: reverting the DISPLAY must not delete the evidence of what
    was refused and why — that stays in the petition panel."""
    import json
    import pathlib
    p = (pathlib.Path(__file__).parent.parent / "exports" / "agentic_runs"
         / "ui_9d48a00e" / "events.jsonl")
    if not p.exists():
        return
    from agentic.ui import derive
    d = derive([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
    pet = (d.get("assess") or {}).get("petition") or {}
    assert pet.get("status") == "failed"
    assert "did not survive" in pet.get("error", "") or pet.get("error")
    assert "other" in pet.get("error", "")          # what was refused
    assert (d.get("epoch0") or {}).get("bound")     # the first pass kept too


def test_a_merged_petition_that_adds_something_keeps_the_new_view():
    """The revert is for petitions that add NOTHING. A real merge must still
    show the merged record."""
    from agentic.ui import derive
    evs = [{"type": "run_started", "image_size": [10, 10]},
           {"type": "entity_bound", "object_id": "a_1",
            "box_source": "dino_matched", "bbox": [0, 0, 5, 5]},
           {"type": "petition_started", "target": "stage1", "reasons": []},
           {"type": "entity_bound", "object_id": "b_1",
            "box_source": "dino_matched", "bbox": [1, 1, 6, 6]},
           {"type": "petition_done", "added": ["b_1·leaking"], "removed": [],
            "rejected": [], "disputed": [], "n_petitioned": 1}]
    d = derive(evs)
    assert "b_1" in d["bound"]


# ── F24: each card carries its own verdict, under it ───────────────────

def _text(tree) -> str:
    """Flatten a Dash component tree to the text a reader would see."""
    out = []
    def walk(n):
        if isinstance(n, (list, tuple)):
            for x in n:
                walk(x)
        elif isinstance(n, str):
            out.append(n)
        elif hasattr(n, "children"):
            walk(n.children)
    walk(tree)
    return " ".join(out)


def _s4_with_findings():
    return {"stage4": {"recommendations": [
        {"rank": 1, "action": "Isolate the area.",
         "reason": "Because person_1 is standing it may_harm house_1.",
         "structured_reasoning": {"threat": "house_1", "state": "burning",
                                  "effect": "may_harm",
                                  "affected_objects": ["person_1"]}}],
        "explanation_alignment": {
            "modes": [{"rank": 1, "mode": "hazard_directed"}],
            "by_rank": {"1": [
                {"rule": "action_names_no_object_id", "signal": "conformance",
                 "level": "card", "severity": 2, "rank": 1,
                 "detail": "rec 1: the action names no object_id"},
                {"rule": "subject_mismatch", "signal": "internal_alignment",
                 "level": "card", "severity": 2, "rank": 1,
                 "detail": "rec 1: the reason blames person_1, the quad blames house_1"},
                {"rule": "victim_named_with_no_hazard_declared",
                 "signal": "conformance", "level": "card", "severity": 0,
                 "rank": 1, "detail": "rec 1: no hazard was declared"},
                {"rule": "rank_not_a_triage", "signal": "conformance",
                 "level": "set", "severity": 1, "rank": None,
                 "detail": "rank 1 used more than once"}]}},
        "card_judge": {"verdicts": [
            {"rank": 1, "advisory": True,
             "prose": {"verdict": "not_causally_aligned", "votes": 4, "n": 5,
                       "unanimous": False},
             "structure": {"verdict": "causally_aligned", "votes": 5, "n": 5,
                           "unanimous": True}}]},
        "graph_a": {}, "graph_b": {}}}


def test_a_cards_verdict_renders_under_that_card():
    """The whole point of the footer: before this, the checks fired in one
    panel and the evidence sat in another, so '5 alignment failures' pointed at
    no card in particular."""
    txt = _text(stage4_component(_s4_with_findings()))
    assert "the action names no object_id" in txt
    assert "the reason blames person_1, the quad blames house_1" in txt


def test_the_two_signals_stay_visibly_apart():
    """'the surface broke a rule' and 'two surfaces disagree' are different
    failures and map to different pathologies."""
    txt = _text(stage4_component(_s4_with_findings())).upper()
    assert "CONFORMANCE" in txt and "ALIGNMENT" in txt


def test_a_severity_zero_finding_says_it_is_not_charged():
    """These are the cases where OUR constraint had no legal answer. Rendering
    them like a real defect would undo the point of separating them."""
    txt = _text(stage4_component(_s4_with_findings()))
    assert "recorded, not charged" in txt


def test_the_judge_is_labelled_advisory_on_the_card():
    """Judges advise, never overwrite — the panel has to say so, or a reader
    will treat a judge verdict as a measurement."""
    txt = _text(stage4_component(_s4_with_findings()))
    assert "ADVISORY" in txt.upper()
    assert "NOT causally aligned" in txt
    # the vote split rides beside the verdict — 4/5 and 5/5 are different
    # findings, and showing only the winner throws away the more useful half
    assert "(4/5)" in txt and "(5/5)" in txt


def test_action_mode_is_shown_and_says_what_it_is_for():
    txt = _text(stage4_component(_s4_with_findings()))
    assert "hazard-directed" in txt and "suppression" in txt


def test_set_level_findings_do_not_land_on_a_card():
    """rank and cross-card duplication are about the SET; they have no card to
    sit under, so they must never appear in a card footer."""
    txt = _text(stage4_component(_s4_with_findings()))
    assert "rank 1 used more than once" not in txt


# ── F29: the ACROSS ALL RECOMMENDATIONS panel ──────────────────────────

def _with_set_report(**over):
    d = _s4_with_findings()
    rep = {"coverage": [], "pairwise": [], "n_findings": 0, "n_cards": 2,
           "modes": {"hazard_directed": 0, "victim_directed": 0, "mixed": 0,
                     "unattributed": 2},
           "mode_verdict": "nothing in this set is testable by hazard "
                           "suppression",
           "suppression_testable": 0}
    rep.update(over)
    d["stage4"]["set_report"] = rep
    return d


def test_the_panel_renders_even_when_clean():
    """The opposite of a card footer. A missing footer means "this card is
    clean"; a missing PANEL would be ambiguous between "no cross-card problems"
    and "not rendered" — and absence of duplication is a real positive
    signal."""
    txt = _text(stage4_component(_with_set_report()))
    assert "ACROSS ALL RECOMMENDATIONS" in txt
    assert "no duplicated quads, actions or remaining risks" in txt
    assert "every at-risk entity is acted on" in txt


def test_coverage_and_pairwise_stay_in_separate_blocks():
    """They mean different things and, at S5, map to different pathologies: a
    coverage gap is under-response, duplication is padding."""
    d = _with_set_report(
        coverage=[{"severity": 2,
                   "detail": "at-risk dog_1 is not addressed by any "
                             "recommendation"}],
        pairwise=[{"severity": 1, "detail": "rec 2: same quad as rec 1"}])
    txt = _text(stage4_component(d))
    assert "COVERAGE" in txt and "PAIRWISE" in txt
    assert "at-risk dog_1 is not addressed" in txt
    assert "rec 2: same quad as rec 1" in txt
    assert "every at-risk entity is acted on" not in txt


def test_the_mode_rollup_says_whether_anything_is_testable():
    """D_aerial round 4 produced two unattributed recommendations, so nothing
    in that run was testable by hazard suppression — the most important
    sentence about it, computed nowhere before this."""
    txt = _text(stage4_component(_with_set_report()))
    assert "WHAT THE SET ACTS ON" in txt
    assert "hazard-directed 0" in txt and "unattributed 2" in txt
    assert "nothing in this set is testable by hazard suppression" in txt


def test_a_severity_zero_set_finding_is_not_shown_as_a_defect():
    d = _with_set_report(pairwise=[
        {"severity": 0, "detail": "rec 2: remaining_risk duplicates rec 1"}])
    assert "○ rec 2: remaining_risk duplicates rec 1" in _text(
        stage4_component(d))


def test_no_panel_before_the_report_exists():
    """Older runs have no set_report; they must not grow an empty panel."""
    d = _s4_with_findings()
    d["stage4"].pop("set_report", None)
    assert "ACROSS ALL RECOMMENDATIONS" not in _text(stage4_component(d))


def test_a_card_with_no_findings_grows_no_footer():
    """Scoped to the CARD footer. Since F36 the word CONFORMANCE also appears
    under each graph, which is correct and not a card footer."""
    d = _s4_with_findings()
    d["stage4"]["explanation_alignment"] = {}
    d["stage4"]["card_judge"] = {}
    txt = _text(stage4_component(d))
    # the card's own bands, not the graph sections' identically-named ones
    assert "rec 1:" not in txt
    assert "ADVISORY, NOT SCORED" not in txt
    assert "◆ hazard-directed" not in txt


def test_the_alignment_band_shows_id_level_failures_too():
    """D_aerial rec 1: the band rendered EMPTY while two internal-alignment
    findings sat in the panel above. F24's cross-checks cannot fire when the
    reason does not parse — but the older id-level checks still can, and they
    carry a rank, so they belong under their card."""
    from agentic.ui import stage4_component
    d = _s4_with_findings()
    d["stage4"]["internal_alignment"] = {"failures": [
        {"category": "coverage gap", "severity": 1, "rank": 1,
         "detail": "rec 1: quad ids not in reason: ['fire_truck_1']"},
        {"category": "coverage gap", "severity": 2, "rank": None,
         "detail": "at-risk hazmat_worker_1 is not addressed"}]}
    txt = _text(stage4_component(d))
    assert "quad ids not in reason" in txt
    # rank-less findings are about the SET and have no card to sit under —
    # they render in ACROSS ALL RECOMMENDATIONS, never in a card footer
    card_end = txt.index("ACROSS ALL RECOMMENDATIONS")
    assert "hazmat_worker_1" not in txt[:card_end]
    assert "hazmat_worker_1" in txt[card_end:]


def test_the_semantic_block_completes_the_panel():
    """F30. Conformance and alignment each have a card view and a set view;
    the judge only had a card view."""
    d = _with_set_report()
    d["stage4"]["card_judge"] = {"rollup": {
        "headline": "2 finding(s) across 2 card(s) · 1 card(s) clean",
        "findings": [
            {"kind": "not_aligned", "text": "rec 1: the reason is not causally"
             " aligned with its action", "votes": 3, "n": 5, "thin": True},
            {"kind": "undecided", "text": "rec 1: the judge could not decide "
             "about the quad", "votes": 2, "n": 5, "thin": True}],
        "clean_ranks": [2], "n_judged": 2, "unreachable": 0,
        "advisory": True}}
    txt = _text(stage4_component(d))
    assert "SEMANTIC" in txt and "ADVISORY, NOT SCORED" in txt
    assert "rec 1: the reason is not causally aligned" in txt
    # a scraped majority is marked, never hidden
    assert "thin majority" in txt and "(3/5)" in txt


def test_an_undecided_verdict_is_visually_apart_from_a_finding():
    """◇ the judge could not decide · ◈ the judge found something. An
    undecided verdict is a finding about the INSTRUMENT."""
    d = _with_set_report()
    d["stage4"]["card_judge"] = {"rollup": {
        "headline": "h", "findings": [
            {"kind": "undecided", "text": "rec 1: the judge could not decide "
             "about the quad", "votes": 2, "n": 5, "thin": True}],
        "clean_ranks": [], "n_judged": 1, "unreachable": 0}}
    txt = _text(stage4_component(d))
    assert "◇ rec 1: the judge could not decide" in txt
    assert "◈" not in txt


def test_no_semantic_block_when_the_judge_did_not_run():
    d = _with_set_report()
    d["stage4"]["card_judge"] = {}
    assert "SEMANTIC" not in _text(stage4_component(d))


# ── F31: a run that died mid-stage must not look like it is still working ──

_DEAD = [
    {"type": "run_started", "image_size": [400, 300], "caption": "c",
     "image_name": "s.jpg"},
    {"type": "stage_started", "stage": "Perceive"},
    {"type": "run_error", "message": "HTTPConnectionPool(host='localhost', "
     "port=11434): Connection refused"},
]


def test_a_stage_in_flight_when_the_run_died_is_marked_failed():
    """A_fire errored 10ms in — Ollama was not listening — and the stage sat
    at 'active' forever, so the screen showed an eternal spinner on Perceive.
    The failure was in the event stream and rendered nowhere."""
    d = derive(_DEAD)
    assert d["stages"]["Perceive"]["status"] == "failed"
    assert "Connection refused" in d["stages"]["Perceive"]["info"]


def test_the_failure_message_reaches_the_screen():
    from agentic.ui import progress_strip
    txt = _text(progress_strip(derive(_DEAD)))
    assert "failed at perceive" in txt.lower()
    assert "Connection refused" in txt


def test_a_dead_run_never_renders_as_busy():
    """The CSS reads 'active' as still working; that is what made it pulse."""
    d = derive(_DEAD)
    assert d["activity"]["busy"] is False
    assert "run failed" in d["activity"]["text"]


def test_stages_that_never_started_stay_pending():
    """Only the stage that was in flight failed — the rest were never
    reached, and marking them failed would overstate what went wrong."""
    d = derive(_DEAD)
    others = [s for n, s in d["stages"].items() if n != "Perceive"]
    assert all(s["status"] == "pending" for s in others)


def test_a_healthy_run_is_untouched():
    d = derive(EVENTS)
    assert not any(s["status"] == "failed" for s in d["stages"].values())


# ── F36: findings render under the graph they judge ────────────────────

def _s4_graphs(**over):
    d = {"recommendations": [],
         "graph_a": {"nodes": [{"id": "spill_1", "label": "spill",
                                "state": "chemical_spill", "hazardous": True}],
                     "edges": [{"source": "spill_1", "effect": "may_harm",
                                "target": "worker_1"}]},
         "graph_b": {"nodes": [], "edges": []},
         "conformance": {"validity": 0.6, "n_issues": 2, "breakdown": [],
                         "issues": [
                             {"graph": "graph_a", "entity": "spill_1",
                              "severity": 2, "category": "role mix-up",
                              "rule": "hazard_flag_state_mismatch",
                              "detail": "spill_1: state vs hazardous"},
                             {"graph": "graph_b", "entity": "dog_1",
                              "severity": 1, "category": "inconsistency",
                              "rule": "via_state_not_hazard_bearing",
                              "detail": "dog_1->person_1: via 'running'"}],
                         "by_graph": {
                             "graph_a": {"count": 1, "max_severity": 2,
                                         "breakdown": []},
                             "graph_b": {"count": 1, "max_severity": 1,
                                         "breakdown": []}}}}
    d.update(over)
    return {"stage4": d}


def test_each_graph_carries_its_own_conformance_findings():
    """Before F36 the graphs rendered as bare edge lists at the bottom while
    five separate panels above judged them — the same disconnect F24 fixed for
    the recommendation cards."""
    t = _text(stage4_component(_s4_graphs()))
    a = t.index("GRAPH A ·")
    b = t.index("GRAPH B ·")
    assert a < t.index("hazard_flag_state_mismatch") < b
    assert b < t.index("via_state_not_hazard_bearing")


def test_a_finding_is_not_printed_twice_on_one_screen():
    """The score panel keeps the count; the detail lives under the graph."""
    t = _text(stage4_component(_s4_graphs()))
    assert t.count("hazard_flag_state_mismatch") == 1
    assert "GRAPH A ·" in t


def test_a_clean_graph_says_so_rather_than_vanishing():
    d = _s4_graphs()
    d["stage4"]["conformance"]["issues"] = []
    t = _text(stage4_component(d))
    assert t.count("no conformance issues") == 2      # one per graph


def test_graph_a_states_why_it_has_no_self_consistency_band():
    """Graph A is BUILT BY CODE from the quads, so it cannot contradict itself
    the way a model's answer can. The asymmetry with Graph B is intentional
    and the panel says so rather than looking like an oversight."""
    t = _text(stage4_component(_s4_graphs()))
    assert "built by code from the recommendation quads" in t
    assert "no self-consistency band" in t


def test_a_vs_b_is_its_own_section_after_the_graphs():
    """Sunny (C run on the rewritten prompt): A-vs-B is the earned comparison
    the trust factors read — too important to bury inside the graphs section.
    Section 5, open by default, after THE CAUSAL GRAPHS."""
    d = _s4_graphs(alignment={"a_fidelity": 0.0, "b_coverage": 0.0,
                              "structural": 0.0, "a_only": [], "b_only": [],
                              "decomposition": {}})
    t = _text(stage4_component(d))
    assert "5 · ALIGNMENT" in t
    assert t.index("4 · THE CAUSAL GRAPHS") < t.index("5 · ALIGNMENT")
    assert t.index("5 · ALIGNMENT") < t.index("6 · THE JUDGES' BENCH")


def test_no_finding_is_printed_twice_anywhere_on_the_screen():
    """F37. The whole point of the reorganisation: a finding renders once,
    under the thing it judges. Before it, `hazard_flag_state_mismatch` appeared
    four times, the card findings appeared in the trust panel AND their own
    footers, and the internal-alignment panel reprinted all five of its
    findings a second time."""
    import json
    from agentic.ui import stage4_component
    d = json.load(open('exports/agentic_runs/ui_21f1cdad/stage4.json'))
    t = _text(stage4_component({"stage4": d}))
    # one per graph — two graphs, not four printings
    assert t.count("hazard_flag_state_mismatch") == 2
    # one per card
    assert t.count("quad ids not in reason") == 2
    # one per unaddressed entity
    assert t.count("is not addressed by any recommendation") == 2


def test_a_set_level_finding_survives_on_runs_that_predate_set_report():
    """Older runs have no set_report, and their coverage findings used to live
    in the panel F37 removed. They are the evidence the bias work needs."""
    d = _s4_with_findings()
    d["stage4"].pop("set_report", None)
    d["stage4"]["internal_alignment"] = {"failures": [
        {"category": "coverage gap", "severity": 2, "rank": None,
         "detail": "at-risk dog_1 is not addressed by any recommendation"}]}
    t = _text(stage4_component(d))
    assert "at-risk dog_1 is not addressed" in t


def test_the_trust_row_names_what_dented_a_card_not_the_finding():
    """The finding belongs in that card's footer. Printing it in the trust
    panel too put the same sentence above the cards and under them."""
    d = _s4_with_findings()
    d["stage4"]["trust"] = {"score": 0.7, "band": "moderate", "contributors": [],
                            "per_rec": [{"rank": 1, "score": 0.7,
                                         "worst_contributor": {
                                             "signal": "internal_alignment",
                                             "penalty": 0.5,
                                             "text": "rec 1: quad ids not in reason"}}]}
    t = _text(stage4_component(d))
    assert "its parts don't line up" in t
    assert t.count("rec 1: quad ids not in reason") == 0


# ── F38: the graph judge renders in the A-vs-B panel ───────────────────

def _s4_graph_judge(**over):
    gj = {"advisory": True, "n_probes": 5,
          "victims": {"verdict": "graph_b", "votes": 4, "n": 5,
                      "sets": {"graph_a": ["fire_truck_1"],
                               "graph_b": ["hazmat_worker_1"]}},
          "mechanisms": [
              {"source": "tanker_truck_1", "target": "spill_1",
               "effect_a": "exposes", "effect_b": "may_spread_to",
               "verdict": "same_response", "votes": 5, "n": 5}]}
    gj.update(over)
    return {"stage4": {"recommendations": [],
                       "graph_a": {"nodes": [], "edges": []},
                       "graph_b": {"nodes": [], "edges": []},
                       "alignment": {"a_fidelity": 0.0, "b_coverage": 0.0,
                                     "structural": 0.0, "a_only": [],
                                     "b_only": [], "decomposition": {}},
                       "graph_judge": gj}}


def test_the_victim_verdict_is_stated_without_diagnosing():
    """"Graph B's victims are more exposed" IS "Graph A is minimizing the
    victims" — established by a closed choice with a right answer. The naming
    stays in the pathology layer; the judge only supplies the evidence."""
    t = _text(stage4_component(_s4_graph_judge()))
    # F43: NAME both sets. "Graph B's harmed entities are more exposed" made
    # the reader hold in their head that Graph B is the model's belief, Graph A
    # is the advice, and that the point is the advice protects the wrong people.
    assert "hazmat_worker_1" in t and "fire_truck_1" in t
    assert "MORE danger" in t
    assert "(4/5)" in t
    assert "minimiz" not in t.lower()        # no diagnosis in the judge's words


def test_the_mechanism_verdict_names_the_pair_and_both_effects():
    t = _text(stage4_component(_s4_graph_judge()))
    assert "tanker_truck_1 → spill_1" in t
    assert "'exposes' vs 'may_spread_to'" in t
    assert "same response either way" in t


def test_the_graph_judge_is_labelled_advisory():
    """Judges advise, never overwrite — the panel has to say so, or a reader
    treats a judge verdict as a measurement."""
    t = _text(stage4_component(_s4_graph_judge()))
    assert "ADVISORY, NOT SCORED" in t


def test_an_undecided_graph_verdict_is_visually_apart():
    d = _s4_graph_judge(victims={"verdict": "unclear", "votes": 2, "n": 5})
    t = _text(stage4_component(d))
    assert "◇ the judge could not decide which set is more exposed" in t


def test_nothing_renders_when_the_graph_judge_did_not_run():
    d = _s4_graph_judge()
    d["stage4"]["graph_judge"] = {}
    assert "ADVISORY, NOT SCORED" not in _text(stage4_component(d))


def test_the_graph_judge_still_shows_when_the_gate_withheld_the_numbers():
    """F38. The judge reads the two graphs directly, so its verdict does not
    depend on the gate. When the arithmetic is withheld as meaningless is
    exactly when the semantic reading is the only thing left saying anything —
    hiding it there was backwards. D_aerial is the live case: gate failed,
    a_fidelity withheld, and the judge still says Graph B's victims are the
    exposed ones."""
    d = _s4_graph_judge()
    d["stage4"]["trust"] = {"graph_b_gate": {
        "trusted": False, "reasons": ["conformance flags 1 serious issue"]}}
    t = _text(stage4_component(d))
    assert "WITHHELD from trust" in t
    assert "MORE danger" in t


def test_invented_entities_are_named_as_a_finding_not_buried_in_edges():
    """F42. One D_aerial run's probes invented four entity names. Three of them
    map to nothing in the scene — the model put entities in its own causal
    graph that do not exist — and that was invisible inside a sixteen-line
    edge dump."""
    s4 = {"graph_b_uncertainty": {
        "n_probes": 5, "score": 0.5, "direction_evidence": [],
        "invented_ids": ["chemical_worker_1", "spill_consequence_1"]}}
    t = _s4_panel(s4)
    assert "named 2 entit" in t
    assert "chemical_worker_1" in t and "spill_consequence_1" in t


# ── F45: the effect toggle, and the wiring warning ──────────────────────

def _al(**kw):
    base = {"a_fidelity": 0.625, "b_coverage": 0.666,
            "a_fidelity_strict": 0.0, "b_coverage_strict": 0.0,
            "decomposition": {"hazards": 1.0, "victims": 0.25, "pairs": 0.25,
                              "reading": "agrees on the hazards, disagrees on "
                                         "who they threaten"}}
    base.update(kw)
    return base


def test_the_toggle_says_the_effect_is_ignored_and_carries_the_old_numbers():
    """Sunny: "Make it a toggle in that panel. by default effect is ignored."
    Default state is stated on the summary line; the pre-F45 whole-edge pair is
    one click away, so no earlier run becomes unquotable."""
    from agentic.ui import _effect_toggle_rows
    s = str(_effect_toggle_rows(_al()))
    assert "effect word: IGNORED" in s
    assert "0.00 / 0.00" in s                 # the strict pair, inside
    assert "same pairs" in s                  # the wiring check, inside


def test_the_wiring_warning_fires_only_when_the_wires_are_crossed():
    """Same hazards and same victims but connected differently reads 1.00 on
    the default — a mean of two sets cannot see wiring. That case, and only
    that case, gets a warning."""
    from agentic.ui import _effect_toggle_rows
    crossed = _al(a_fidelity=1.0,
                  decomposition={"hazards": 1.0, "victims": 1.0, "pairs": 0.0,
                                 "reading": "x"})
    assert "wired to each other differently" in str(_effect_toggle_rows(crossed))
    # the ordinary D_aerial case must NOT be warned about — its victims differ,
    # which the split above already says plainly.
    assert "wired to each other differently" not in str(_effect_toggle_rows(_al()))


def test_the_split_reads_as_the_two_halves_of_a_fidelity():
    """F45 made `hazards` and `victims` the parts a_fidelity is the mean OF,
    not a commentary beside it. The `+` is what makes that legible."""
    from agentic.ui import _ab_decomposition_rows
    s = str(_ab_decomposition_rows(_al()["decomposition"]))
    assert "same hazards" in s and "same victims" in s
    assert "+" in s


def test_a_loose_id_merge_says_it_is_loose():
    """`head noun, by number` is a weaker claim than `verbatim`, and a reader
    who cannot tell them apart cannot audit either."""
    from agentic.ui import _resolved_id_rows
    s = str(_resolved_id_rows({"chemical_worker_1": "hazmat_worker_1"},
                              {"chemical_worker_1": "head noun, by number"}))
    assert "chemical_worker_1 → hazmat_worker_1 (head noun, by number)" in s


# ── F46: every organized score is stated, and says which way is good ────

def _s4_scores():
    """A stage-4 result carrying one of every score family."""
    return {
        "recommendations": [{"rank": 1, "action": "cool fire_1",
                             "reason": "fire_1 is burning and may_harm person_1",
                             "structured_reasoning": {"threat": "fire_1",
                                                      "state": "burning",
                                                      "effect": "may_harm",
                                                      "affected_objects": ["person_1"]}}],
        "conformance": {"validity": 0.765, "n_issues": 11,
                        "raw_a_validity": 0.7, "raw_b_validity": 0.9,
                        "by_graph": {"card": {"count": 7, "max_severity": 2}}},
        "internal_alignment": {"score": 0.667, "n_failures": 3},
        "explanation_alignment": {"score": 0.533, "n_failures": 7,
                                  "by_rank": {}, "modes": {}},
        "set_report": {"n_cards": 2, "n_findings": 0, "coverage": [],
                       "pairwise": [], "modes": {}, "suppression_testable": 1},
        "uncertainty": {"score": 0.446, "n_probes": 5},
        "graph_b_internal": {"measured": True, "score": 0.778,
                             "n_failures": 1, "breakdown": []},
        "graph_b_uncertainty": {"score": 0.217, "n_probes": 5,
                                "edge_set_instability": 0.45,
                                "direction_instability": 0.2,
                                "pick_instability": 0.0},
        "trust": {"score": 0.447, "band": "moderate", "contributors": {}},
    }


def test_the_card_rule_score_is_stated_not_just_tallied():
    """It was computed and never shown; the header carried "7 rule breaks",
    which cannot be compared between runs or scenes because there is no
    denominator on screen."""
    out = str(stage4_component(derive([{"type": "stage4_result",
                                        "result": _s4_scores()}])))
    assert "7 rule breaks" in out
    assert "0.53 clean" in out


def test_graph_b_uncertainty_states_its_score():
    """0.217 is the number the yardstick gate and the pathology layer read.
    The panel showed the probe count and three per-axis agreements, and not
    the score itself."""
    out = str(stage4_component(derive([{"type": "stage4_result",
                                        "result": _s4_scores()}])))
    assert "0.22 unsure" in out


def test_higher_is_worse_numbers_never_borrow_the_higher_is_better_word():
    """Two families run in OPPOSITE directions and both used to read "score".
    `clean` rises as things get better; `unsure` rises as they get worse. A
    number must never be labelled with the wrong family's word."""
    out = str(stage4_component(derive([{"type": "stage4_result",
                                        "result": _s4_scores()}])))
    assert "0.45 unsure" in out                 # measured uncertainty
    assert "0.45 clean" not in out
    assert "0.78 clean" in out                  # Graph B self-consistency
    assert "0.78 unsure" not in out
    # and "score N" as a bare, directionless label is gone from the headers
    assert "score 0.446" not in out and "score 0.778" not in out


def test_no_score_is_presented_as_a_pass_fraction():
    """None of these are "N of M checks passed" — they are 1 - x/(x + size)
    shapes. Printing a denominator we do not have would be a fabrication."""
    out = str(stage4_component(derive([{"type": "stage4_result",
                                        "result": _s4_scores()}])))
    for bad in ("of 15 checks", "checks passed", "out of 15", "/15"):
        assert bad not in out


# ── the sectioned Stage 4 layout + the judges' bench (2026-08-08) ───────

def _s4_sectioned():
    ro = {"application": "recommendations", "advisory": True,
          "prompt_version": "runoff-v1", "n_probes": 3,
          "candidate_a": "A-text", "candidate_b": "B-text",
          "text": {"verdict": "answer_a", "votes": 3, "n": 3,
                   "counts": {"answer_a": 3}, "tie": False, "reasoning": "r"},
          "image": {"verdict": "answer_b", "votes": 2, "n": 3,
                    "counts": {"answer_b": 2, "answer_a": 1}, "tie": False,
                    "reasoning": "r2"},
          "twins_agree": False,
          "code_facts": {"invented_ids_a": [], "invented_ids_b": ["ghost_1"]},
          "verified_by_intervention": None}
    return {"recommendations": [], "trust": {"score": 0.6, "band": "moderate",
                                             "contributors": []},
            "uncertainty": {"n_probes": 5, "score": 0.3, "granular": {},
                            "candidates": []},
            "runoff_judge": {"recommendations": ro}}


def test_stage4_renders_five_numbered_sections():
    """Sunny: "stage 4 is getting crowded. We need to divide it into
    subsections." Five collapsible sections, evidence-ordered."""
    out = str(stage4_component(derive([{"type": "stage4_result",
                                        "result": _s4_sectioned()}])))
    for head in ("1 · THE VERDICT", "2 · THE RECOMMENDATIONS",
                 "3 · STABILITY", "6 · THE JUDGES' BENCH"):
        assert head in out, head


def test_a_judge_verdict_lives_on_the_bench_and_only_there():
    """Sunny: "I had a hard time finding runoff judges. It's almost they are
    hidden. Judges should be visible and in separate cards." One verdict, one
    home: the bench card carries the detail; the judged panel keeps a pointer
    chip. Nothing is printed twice (F37, applied to judges)."""
    out = str(stage4_component(derive([{"type": "stage4_result",
                                        "result": _s4_sectioned()}])))
    assert "⚖ RUNOFF · RECOMMENDATIONS" in out
    assert "advises — never enters a score" in out
    assert "→ see THE JUDGES' BENCH" in out          # the pointer chip
    assert out.count("the two candidates") == 1       # detail on bench only


def test_twin_disagreement_is_named_not_just_flagged():
    """The chip names both verdicts so a reader sees the split itself."""
    out = str(stage4_component(derive([{"type": "stage4_result",
                                        "result": _s4_sectioned()}])))
    assert "TWINS DISAGREE" in out
    assert "text: candidate A" in out and "image: candidate B" in out


def test_judge_votes_default_is_three():
    """5 -> 3 (Sunny): the first full run spent 55 of 59 minutes judging.
    Env-overridable; the discrimination sets pass explicit counts."""
    from agentic import models
    from agentic.judge_card import DEFAULT_JUDGE_PROBES
    assert models.JUDGE_VOTES == 3
    assert DEFAULT_JUDGE_PROBES == 3
