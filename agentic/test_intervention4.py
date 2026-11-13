"""Intervention step 1: the measuring, imported from the frozen arm.
Hermetic: no model, no image. The frozen arm's own numbers are the oracle:
a saved try (before, after, signals) must come out the same through our
adapter."""
import copy
import json
from pathlib import Path

import pytest

from agentic import intervention4 as iv4

FIX = Path(__file__).parent / "fixtures" / "frozen_intervention_tries.json"
SHIFTS = ("hazard_shift", "graph_shift", "recommendation_shift",
          "structural_shift", "semantic_shift", "total_shift", "content_shift")


def _as_result(part: dict) -> dict:
    """A saved try's before/after -> the frozen "result" shape."""
    return {"image_filename": "", "disaster_level": part.get("hazard_level"),
            "detected_objects": part.get("detected_objects"),
            "causal_graph": part.get("graph_a"),
            "recommendations": part.get("recommendations")}


def _check_try(t: dict, gt_dir: Path) -> None:
    u = t["u_check"]
    res = iv4.measure(_as_result(t["before"]), _as_result(t["after"]),
                      t["selection"]["target_object_id"],
                      modality=u["modality"], caption_changed=u["caption_changed"],
                      image_changed=u["image_changed"], gt_dir=gt_dir)
    for k in SHIFTS:
        assert res["signals"][k] == pytest.approx(t["signals"][k]), k
    assert res["signals"]["hazard_level_delta"] == t["signals"]["hazard_level_delta"]
    assert res["verdict"]["moved"] == t["moved"]


def test_a_saved_frozen_try_comes_out_the_same_through_our_adapter(tmp_path):
    tries = json.loads(FIX.read_text())
    assert tries, "the fixture holds at least one frozen-arm try"
    for t in tries:
        _check_try(t, tmp_path)                 # tmp_path: no ground truth file


def test_every_saved_frozen_try_on_this_machine_reproduces(tmp_path):
    """The same check over all tries the frozen arm saved (exports/ is not
    in git, so this is skipped on a fresh checkout)."""
    files = sorted((iv4.REPO_ROOT / "exports" / "runs").glob("*/intervention.json"))
    tries = [t for f in files for t in (json.loads(f.read_text()).get("tries") or [])
             if t.get("before") and t.get("after") and t.get("signals")
             and (t.get("u_check") or {}).get("input_gated")]
    if not tries:
        pytest.skip("no saved frozen-arm tries here")
    same = not_rebuilt = 0
    for t in tries:
        u = t["u_check"]
        res = iv4.measure(_as_result(t["before"]), _as_result(t["after"]),
                          t["selection"]["target_object_id"], modality=u["modality"],
                          caption_changed=u["caption_changed"],
                          image_changed=u["image_changed"], gt_dir=tmp_path)
        if not res.get("signals"):
            # a saved try keeps only four fields of its before run; for a
            # few (a victim as the target) that is too little to list the
            # target again. Counted, never compared.
            not_rebuilt += 1
            continue
        for k in SHIFTS:                        # every number, exactly
            assert res["signals"][k] == pytest.approx(t["signals"][k]), (k, t["selection"])
        same += 1
    assert same >= 1 and same > not_rebuilt


# ── our run -> the frozen shape ──────────────────────────────────────────

RECORD = {"image_path": "/x/C_tanker_fire.jpg", "caption": "a tanker leaks near a fire",
          "detected_objects": [
              {"object_id": "fire_1", "label": "fire", "state": "spreading"},
              {"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
              {"object_id": "man_1", "label": "man", "state": "standing"}]}
ASSESS = {"disaster_level": 8, "threats": [{"object_id": "fire_1"}], "at_risk": []}
S4 = {"recommendations": [
          {"rank": 1, "action": "Move man_1 away from fire_1.",
           "structured_reasoning": {"threat": "fire_1", "state": "spreading",
                                    "effect": "may_harm", "affected_objects": ["man_1"]}},
          {"rank": 2, "action": "Close the valve on tanker_1.",
           "structured_reasoning": {"threat": "tanker_1", "state": "leaking",
                                    "effect": "may_harm", "affected_objects": ["man_1"]}}],
      "graph_a": {"nodes": [{"id": "fire_1", "label": "fire", "state": "spreading",
                             "hazardous": True},
                            {"id": "tanker_1", "label": "tanker", "state": "leaking",
                             "hazardous": True},
                            {"id": "man_1", "label": "man", "state": "standing",
                             "at_risk": True}],
                  "edges": [{"source": "fire_1", "target": "man_1", "effect": "may_harm",
                             "via_state": "spreading", "valid": True},
                            {"source": "tanker_1", "target": "man_1", "effect": "may_harm",
                             "via_state": "leaking", "valid": True}],
                  "intervention_candidates": [
                      {"threat": "fire_1", "state": "spreading", "outgoing_edge_count": 1},
                      {"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
      "graph_b": {"nodes": [], "edges": [], "suppression_pick": {}},
      "trust": {"score": 0.7, "band": "moderate"}}


def _after_without_fire() -> tuple[dict, dict, dict]:
    rec, a, s4 = copy.deepcopy(RECORD), copy.deepcopy(ASSESS), copy.deepcopy(S4)
    rec["detected_objects"] = rec["detected_objects"][1:]
    a["disaster_level"] = 5
    s4["recommendations"] = s4["recommendations"][1:]
    s4["graph_a"]["nodes"] = s4["graph_a"]["nodes"][1:]
    s4["graph_a"]["edges"] = s4["graph_a"]["edges"][1:]
    s4["graph_a"]["intervention_candidates"] = s4["graph_a"]["intervention_candidates"][1:]
    return rec, a, s4


def test_our_run_is_translated_into_the_frozen_shape():
    r = iv4.frozen_result(RECORD, ASSESS, S4, run_id="ui_1")
    assert r["image_filename"] == "C_tanker_fire.jpg" and r["run_id"] == "ui_1"
    assert r["disaster_level"] == 8 and r["caption"].startswith("a tanker")
    assert [o["object_id"] for o in r["detected_objects"]] == ["fire_1", "tanker_1", "man_1"]
    assert len(r["recommendations"]) == 2 and r["causal_graph"]["edges"]
    assert r["pre_intervention_trust"] == {"score": 0.7, "level": "moderate"}
    # the saved assessment.json wraps the assessment one level down
    assert iv4.frozen_result(RECORD, {"assessment": ASSESS}, S4)["disaster_level"] == 8


def test_the_same_run_twice_does_not_move_and_a_removed_hazard_does(tmp_path):
    before = iv4.frozen_result(RECORD, ASSESS, S4)
    same = iv4.summary(iv4.measure(before, before, "fire_1", gt_dir=tmp_path))
    assert same["moved"] is False and same["total_shift"] == 0.0 and same["fair"] is True
    assert same["target"] == "fire_1" and same["cell"] == "not_adjudicable"   # no GT file
    after = iv4.frozen_result(*_after_without_fire())
    res = iv4.measure(before, after, "fire_1", gt_dir=tmp_path)
    s = iv4.summary(res)
    assert s["moved"] is True and s["level_delta"] == -3
    assert s["shifts"]["recommendation_shift"] > 0 and s["shifts"]["graph_shift"] > 0
    assert res["after_missing"] is False
    assert res["edit"] == {"modality": "both", "caption_changed": True,
                           "image_changed": True, "applied": True}


def test_the_fair_test_reads_only_what_was_changed_on_the_input(tmp_path):
    before = iv4.frozen_result(RECORD, ASSESS, S4)
    after = iv4.frozen_result(*_after_without_fire())
    # "image" chosen but the image was not changed: nothing was done
    s = iv4.summary(iv4.measure(before, after, "fire_1", modality="image",
                                image_changed=False, gt_dir=tmp_path))
    assert s["fair"] is False and s["cell"] == "u_leaked" and s["moved"] is None
    # "both" needs both channels
    assert iv4.edit_bundle("both", True, False)["applied"] is False
    assert iv4.edit_bundle("both", True, True)["applied"] is True
    assert iv4.edit_bundle("caption", True, False)["applied"] is True
    assert iv4.edit_bundle("image", True, False)["applied"] is False
    with pytest.raises(ValueError):
        iv4.edit_bundle("language", True, True)


def test_candidates_are_the_frozen_ranking_of_our_hazards(tmp_path):
    c = iv4.candidates(iv4.frozen_result(RECORD, ASSESS, S4), gt_dir=tmp_path)
    assert [x["object_id"] for x in c["candidates"]] == ["fire_1", "tanker_1"]
    assert c["should_be_core"] is None            # no ground truth file for this image
    assert iv4.candidates({}, gt_dir=tmp_path)["candidates"] == []


def test_a_pick_that_is_not_a_hazard_is_not_run_and_not_retargeted(tmp_path):
    before = iv4.frozen_result(RECORD, ASSESS, S4)
    res = iv4.measure(before, before, "nope_9", gt_dir=tmp_path)
    assert res["verdict"].get("pick_not_a_candidate") is True
    s = iv4.summary(res)
    assert s["target"] is None and s["moved"] is None and s["total_shift"] is None


@pytest.mark.parametrize("bad", [None, "x", 7, [], {"recommendations": "y"},
                                 {"detected_objects": [None, "x"], "causal_graph": "g",
                                  "disaster_level": "high"}])
def test_malformed_runs_never_raise(bad, tmp_path):
    before = iv4.frozen_result(RECORD, ASSESS, S4)
    for b, a in ((before, bad), (bad, before), (bad, bad)):
        res = iv4.measure(b, a, "fire_1", gt_dir=tmp_path)
        assert isinstance(iv4.summary(res), dict)
    assert iv4.measure(before, bad, "fire_1", gt_dir=tmp_path)["after_missing"] is True
    r = iv4.frozen_result(bad, bad, bad)
    assert r["detected_objects"] == [] and r["recommendations"] == []
    assert iv4.summary(bad)["moved"] is None


def test_a_saved_run_folder_is_read_and_missing_files_are_empty_parts(tmp_path):
    d = tmp_path / "ui_9"
    d.mkdir()
    (d / "C__perception.json").write_text(json.dumps(RECORD))
    (d / "assessment.json").write_text(json.dumps({"assessment": ASSESS}))
    (d / "stage4.json").write_text(json.dumps(S4))
    r = iv4.result_from_run_dir(d)
    assert r["run_id"] == "ui_9" and r["disaster_level"] == 8
    assert len(r["recommendations"]) == 2
    (d / "stage4.json").write_text("{ torn")
    (d / "assessment.json").unlink()
    r = iv4.result_from_run_dir(d)
    assert r["recommendations"] == [] and r["disaster_level"] is None
    assert len(r["detected_objects"]) == 3
    assert iv4.result_from_run_dir(tmp_path / "none")["detected_objects"] == []


def test_the_frozen_files_are_imported_never_copied():
    src = Path(iv4.__file__).read_text()
    assert "import intervention" in src
    for frozen_name in ("def compute_shifts", "def adjudicate_groundedness",
                        "def run_intervention", "MOVE_CUTOFF ="):
        assert frozen_name not in src
