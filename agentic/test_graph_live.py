"""Equivalence: the LangGraph control (graph_live.run_graph) must produce
byte-identical output to the plain-Python control (assess_with_petition),
given the same model answers.

Hermetic: model calls are scripted, deterministic, pure functions of the
prompt — so call order/count can never make the two paths diverge on their
own. What we assert: for the SAME record + SAME scripts, both paths return
deep-equal (record, result, petitioned), AND emit the same event stream.

The three cases exercise every branch of the petition router:
  A · no petition          (router -> END)
  B · stage-2 re-ask       (router -> reask -> END)
  C · stage-1 re-look      (router -> relook -> reassess -> END)

Run:  pytest agentic/test_graph_live.py -q
"""
from __future__ import annotations

from agentic.perception import DetectedObject, PerceptionResult
from agentic.petition import assess_with_petition
from agentic.graph_live import run_graph


# ── fixtures ────────────────────────────────────────────────────────────

def _obj(oid, label, family, state, kind, bbox=(0, 0, 10, 10),
         box_source="dino_matched"):
    return DetectedObject(
        object_id=oid, label=label, family=family, state=state,
        state_kind=kind, description="", bbox=list(bbox),
        box_source=box_source, box_confidence=0.9,
        anchor_bbox=list(bbox))


def _record(objs, caption="A scene."):
    return PerceptionResult(
        image_path="/x/scene.jpg", image_size=[100, 80], caption=caption,
        entity_source="vlm", detected_objects=objs)


def _run_both(record, query_fn, perceive_fn=None):
    """Run the Python path and the LangGraph path with identical scripts;
    return (python_out, langgraph_out, python_events, langgraph_events)."""
    kw = dict(query_fn=query_fn, n_probes=0, reflect=True)

    ev_py: list = []
    r_py, res_py, pet_py = assess_with_petition(
        "/x/scene.jpg", record.model_copy(deep=True),
        on_event=ev_py.append, perceive_fn=perceive_fn, **kw)

    ev_lg: list = []
    r_lg, res_lg, pet_lg = run_graph(
        "/x/scene.jpg", record.model_copy(deep=True),
        on_event=ev_lg.append, perceive_fn=perceive_fn, **kw)

    return ((r_py, res_py, pet_py), (r_lg, res_lg, pet_lg), ev_py, ev_lg)


def _assert_equivalent(py, lg, ev_py, ev_lg):
    (r_py, res_py, pet_py) = py
    (r_lg, res_lg, pet_lg) = lg
    assert pet_py == pet_lg, f"petitioned differs: {pet_py} vs {pet_lg}"
    assert r_py.model_dump() == r_lg.model_dump(), "record differs"
    assert res_py.model_dump() == res_lg.model_dump(), "result differs"
    # event streams must match too (same functions, same order)
    assert ev_py == ev_lg, "event streams differ"


# ── Case A · clean scene, no petition ───────────────────────────────────

def test_equivalence_no_petition():
    rec = _record([_obj("fire_1", "fire", "hazard_media", "spreading",
                        "hazard_bearing"),
                   _obj("person_1", "person", "person", "standing", "normal")])

    def q(prompt):
        return {"disaster_scenario": "Yes", "disaster_type": "fire",
                "disaster_level": 7, "confidence": 0.9,
                "reasoning": "fire_1 is spreading",
                "threats": [{"object_id": "fire_1",
                             "reason": "fire_1 is spreading and burns nearby"}],
                "at_risk": [{"object_id": "person_1", "kind": "proximity",
                             "reason": "standing near the spreading fire_1"}]}

    py, lg, ev_py, ev_lg = _run_both(rec, q)
    assert py[2] is False and lg[2] is False        # no petition fired
    _assert_equivalent(py, lg, ev_py, ev_lg)


# ── Case B · stage-2 re-ask (hazard present, sorting wrong) ──────────────

def test_equivalence_stage2_reask():
    # building is a legal hazard; person_1·trapped wrongly put in threats.
    rec = _record([
        _obj("building_1", "building", "structure", "collapsed",
             "hazard_bearing"),
        _obj("person_1", "person", "person", "trapped", "at_risk")])

    canonical = {"disaster_scenario": "Yes", "disaster_type": "collapse",
                 "disaster_level": 8, "confidence": 0.9,
                 "reasoning": "building_1 collapsed; person_1 trapped",
                 "threats": [
                     {"object_id": "building_1", "reason": "collapsed structure"},
                     {"object_id": "person_1", "reason": "trapped"}],
                 "at_risk": [{"object_id": "person_1", "kind": "distress",
                              "reason": "state is trapped"}]}
    reask = {"disaster_scenario": "Yes", "disaster_type": "collapse",
             "disaster_level": 8, "confidence": 0.9,
             "reasoning": "corrected: person_1 is a victim",
             "threats": [{"object_id": "building_1",
                          "reason": "collapsed structure"}],
             "at_risk": [{"object_id": "person_1", "kind": "distress",
                          "reason": "state is trapped"}]}

    def q(prompt):
        # stage-2 re-ask prompt is the only one that says this:
        if "ANSWERED THIS QUESTION BEFORE" in prompt:
            return reask
        return canonical        # verdict AND reflection (stands ground)

    py, lg, ev_py, ev_lg = _run_both(rec, q)
    # a stage-2 petition returns petitioned=False (record untouched)
    assert py[2] is False and lg[2] is False
    assert any(e["type"] == "petition_started" and e.get("target") == "stage2"
               for e in ev_py)
    _assert_equivalent(py, lg, ev_py, ev_lg)


# ── Case C · stage-1 re-look (no hazard -> re-perceive) ──────────────────

def test_equivalence_stage1_relook():
    # person_1·drowning, NO hazard in the record -> the source is missing.
    rec = _record([_obj("person_1", "person", "person", "drowning",
                        "at_risk")],
                  caption="A child struggles in a pool.")

    canonical = {"disaster_scenario": "Yes", "disaster_type": "drowning",
                 "disaster_level": 7, "confidence": 0.9,
                 "reasoning": "person_1 is drowning",
                 "threats": [{"object_id": "person_1", "reason": "drowning"}],
                 "at_risk": []}

    def q(prompt):
        return canonical        # verdict + reflection (stands ground)

    # the re-look returns a merged record: the pool, DINO-grounded.
    def perceive_fn(image_path, caption, prompt):
        return _record([
            _obj("person_1", "person", "person", "drowning", "at_risk"),
            _obj("pool_1", "pool", "infrastructure", "engulfing",
                 "hazard_bearing", bbox=(0, 0, 90, 60),
                 box_source="dino_matched")],
            caption=caption)

    py, lg, ev_py, ev_lg = _run_both(rec, q, perceive_fn=perceive_fn)
    # a successful stage-1 petition changes the record -> petitioned=True
    assert py[2] is True and lg[2] is True
    assert any(e["type"] == "petition_started" and e.get("target") == "stage1"
               for e in ev_py)
    # the merged record gained the pool
    assert any(o["object_id"] == "pool_1"
               for o in py[0].model_dump()["detected_objects"])
    _assert_equivalent(py, lg, ev_py, ev_lg)


# ── the flag helper ─────────────────────────────────────────────────────

def test_control_flag_default(monkeypatch):
    from agentic.graph_live import control_flag
    monkeypatch.delenv("AGENTIC_CONTROL", raising=False)
    assert control_flag() == "python"
    monkeypatch.setenv("AGENTIC_CONTROL", "LangGraph")
    assert control_flag() == "langgraph"


def test_dispatcher_routes_by_flag(monkeypatch):
    """assess_with_control must give the same answer whichever control is
    selected (they're proven equivalent) — this just checks it routes."""
    from agentic.graph_live import assess_with_control
    rec = _record([_obj("fire_1", "fire", "hazard_media", "spreading",
                        "hazard_bearing")])

    def q(prompt):
        return {"disaster_scenario": "Yes", "disaster_type": "fire",
                "disaster_level": 7, "confidence": 0.9,
                "threats": [{"object_id": "fire_1",
                             "reason": "fire_1 is spreading"}],
                "at_risk": []}

    monkeypatch.setenv("AGENTIC_CONTROL", "python")
    a = assess_with_control("/x/s.jpg", rec.model_copy(deep=True),
                            query_fn=q, n_probes=0)
    monkeypatch.setenv("AGENTIC_CONTROL", "langgraph")
    b = assess_with_control("/x/s.jpg", rec.model_copy(deep=True),
                            query_fn=q, n_probes=0)
    assert a[0].model_dump() == b[0].model_dump()
    assert a[1].model_dump() == b[1].model_dump()
    assert a[2] == b[2]


def test_ui_override_beats_env(monkeypatch):
    """The UI toggle (set_control) overrides the env var for the process."""
    from agentic.graph_live import set_control, control_flag
    monkeypatch.setenv("AGENTIC_CONTROL", "python")
    set_control("langgraph")
    assert control_flag() == "langgraph"      # override wins
    set_control(None)
    assert control_flag() == "python"         # cleared -> env/default
