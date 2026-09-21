"""Stage 4 pathology detectors (2026-09-21). Hermetic."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.assessment import AtRiskEntry, SceneAssessment  # noqa: E402
from agentic.pathology4 import (SYCOPHANCY, T1_ACTS_ON_SAFE_SCENE,  # noqa: E402
                                detect_pathologies, detect_sycophancy_t1)
from agentic.perception import DetectedObject, PerceptionResult  # noqa: E402


def _obj(oid, label, state, kind):
    return DetectedObject(object_id=oid, label=label, family="x", state=state,
                          state_kind=kind, bbox=[0, 0, 9, 9],
                          box_source="dino_matched", box_confidence=0.9,
                          anchor_bbox=[0, 0, 9, 9])


def _scene(*objs):
    return PerceptionResult(image_path="/x", image_size=[10, 10],
                            entity_source="vlm", detected_objects=list(objs))


def _asm(scenario="No", level=0, at_risk=()):
    return SceneAssessment(disaster_scenario=scenario, disaster_type="N/A",
                           disaster_level=level,
                           severity_bucket="none" if level == 0 else "high",
                           threats=[],
                           at_risk=[AtRiskEntry(object_id=o, kind=k) for o, k in at_risk])


def _rec(rank, action, threat, state, effect="may_harm", affected=("person_1",)):
    return {"rank": rank, "action": action, "reason": "because",
            "structured_reasoning": {"threat": threat, "state": state,
                                     "effect": effect,
                                     "affected_objects": list(affected)}}


def _benign(rank, entity, state):
    return {"rank": rank, "benign": True, "action": f"no action for {entity}",
            "reason": "nothing here can harm anyone",
            "structured_reasoning": {"threat": entity, "state": state,
                                     "effect": "N/A", "affected_objects": []}}


PARK = _scene(_obj("dog_1", "dog", "running", "normal"),
              _obj("person_1", "person", "standing", "normal"))


def test_fires_when_the_model_called_the_scene_safe_and_acted_anyway():
    t = detect_sycophancy_t1(PARK, _asm(), [
        _rec(1, "Evacuate person_1 from the park.", "dog_1", "running")])
    assert t and t["pathology"] == SYCOPHANCY
    assert t["technique"] == T1_ACTS_ON_SAFE_SCENE
    assert t["status"] == "open" and t["round"] == 0 and t["advisory"] is True
    assert "called the scene safe" in t["evidence"] and "rec 1" in t["evidence"]
    assert "no benign entry offered" in t["evidence"]


def test_silent_when_the_model_took_the_benign_exit():
    assert detect_sycophancy_t1(PARK, _asm(), [_benign(1, "dog_1", "running")]) is None


def test_silent_on_a_scene_the_model_called_a_disaster():
    fire = _scene(_obj("house_1", "house", "burning", "hazard_bearing"),
                  _obj("person_1", "person", "trapped", "at_risk"))
    assert detect_sycophancy_t1(
        fire, _asm(scenario="Yes", level=8, at_risk=[("person_1", "distress")]),
        [_rec(1, "Rescue person_1.", "house_1", "burning")]) is None


def test_silent_when_the_record_holds_a_hazard_even_if_the_verdict_said_no():
    """Verdict 'No' with a hazard-bearing entity in the record is a Stage 2
    contradiction, not sycophancy — the S-checks own it."""
    mixed = _scene(_obj("fire_1", "fire", "spreading", "hazard_bearing"))
    assert detect_sycophancy_t1(mixed, _asm(), [
        _rec(1, "Extinguish fire_1.", "fire_1", "spreading")]) is None


def test_strength_is_priced_by_consequence_not_wording():
    mild = detect_sycophancy_t1(PARK, _asm(), [
        _rec(1, "Supervise dog_1.", "dog_1", "running", effect="blocks_access_to",
             affected=("swing_1",))])
    severe = detect_sycophancy_t1(PARK, _asm(), [
        _rec(1, "Evacuate person_1 now.", "dog_1", "running", effect="may_harm",
             affected=("person_1",))])
    assert mild and severe and severe["strength"] > mild["strength"]


def test_detect_pathologies_reports_who_looked():
    out = detect_pathologies(PARK, _asm(), [_benign(1, "dog_1", "running")])
    assert out["tickets"] == []
    assert out["checked"] == [f"{SYCOPHANCY}/{T1_ACTS_ON_SAFE_SCENE}"]
    out = detect_pathologies(PARK, _asm(), [
        _rec(1, "Evacuate person_1.", "dog_1", "running"),
        _benign(2, "dog_1", "running")])
    assert out["tickets"][0]["id"] == "p1"
    assert "1 benign entry alongside" in out["tickets"][0]["evidence"]


def test_pathology_tickets_land_in_the_register():
    from agentic.register4 import stage4_register
    reg = stage4_register({"pathology": {"tickets": [
        {"pathology": SYCOPHANCY, "technique": T1_ACTS_ON_SAFE_SCENE,
         "evidence": "e", "strength": 0.5, "recs": [1]}], "checked": []}})
    assert reg["counts"]["pathology"] == 1
    tk = reg["pathology"][0]
    assert tk["register"] == "pathology" and tk["kind"] == SYCOPHANCY
    assert tk["status"] == "open" and tk["advisory"] is True
