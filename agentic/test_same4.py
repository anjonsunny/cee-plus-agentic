"""Same hazard, entered twice: the model is asked which pairs of hazards
are one and the same, and the groundedness tables follow its judgment.
Hermetic; covers malformed model output at every boundary."""
import re

import pytest

from agentic import declared4, grounding4, recommend, same4, ui_intervene as ui_iv
from agentic.test_ui_intervene import (D_BEFORE, D_OBJS, _belief, _by_class, _flat, _rec,
                                       _res, _try)


# ── the prompt line and the parser ──

def test_the_prompt_asks_for_the_pairs_with_no_scene_in_it():
    p = recommend.RECOMMEND_PROMPT
    line = p[p.index("- same_hazards:"):p.index("- recommendations:")]
    assert "one and the same physical hazard" in line
    assert "Leave the array empty if there are none." in line
    assert not re.search(r"\b[a-z]+_\d+\b", line)                 # no id-shaped token
    for word in ("fire", "brush", "house", "spill", "smoke"):
        assert word not in line                                    # no scene, no example
    assert p.index("- same_hazards:") < p.index("- recommendations:")


def test_well_formed_pairs_are_kept_each_once():
    notes: list = []
    assert same4.parse_pairs([["fire_1", "brush_1"], ["brush_1", "fire_1"],
                              ("spill_1·leaking", "tank_1")], notes) == [
        ["brush_1", "fire_1"], ["spill_1", "tank_1"]]
    assert notes == []


@pytest.mark.parametrize("bad", ["fire_1 and brush_1", {"a": "b"}, 7, True])
def test_a_value_that_is_not_a_list_is_empty_with_a_note(bad):
    notes: list = []
    assert same4.parse_pairs(bad, notes) == [] and len(notes) == 1


def test_a_bad_entry_is_dropped_with_a_note_and_the_rest_kept():
    notes: list = []
    got = same4.parse_pairs([["fire_1"], ["a", "a"], "x", None, ["a", "b", "c"], [1, 2],
                             ["fire_1", "brush_1"]], notes)
    assert got == [["brush_1", "fire_1"]] and len(notes) == 6
    for empty in (None, "", []):
        assert same4.parse_pairs(empty, []) == []


def test_a_pair_must_be_two_hazards_of_this_scene():
    notes: list = []
    got = same4.valid_pairs([["brush_1", "fire_1"], ["fire_1", "man_1"], ["x"], None],
                            ["fire_1", "brush_1", "spill_1"], notes)
    assert got == [["brush_1", "fire_1"]] and len(notes) == 3
    assert same4.valid_pairs(None, None, []) == []


def test_parse_recommend_carries_the_pairs_in_the_frame():
    frame, _r, _a, notes = recommend.parse_recommend(
        {"same_hazards": [["fire_1", "brush_1"], "junk"], "recommendations": []})
    assert frame["same_hazards"] == [["brush_1", "fire_1"]]
    assert any(n.startswith("same_hazards_entry_dropped") for n in notes)
    frame, *_ = recommend.parse_recommend({"recommendations": []})    # an older answer
    assert frame["same_hazards"] == []
    frame, *_ = recommend.parse_recommend(None)
    assert frame["same_hazards"] == []


def test_pairs_chain_into_groups_and_a_group_goes_by_one_name():
    grp = same4.groups([["a", "b"], ["b", "c"], ["x", "y"]])
    assert grp["a"] == grp["c"] == ("a", "b", "c") and grp["y"] == ("x", "y")
    assert same4.key("b", grp) == "a + b + c" and same4.key("z", grp) == "z"
    assert same4.members("a + b + c") == ["a", "b", "c"] and same4.members(None) == []
    for bad in (None, "x", [None, ["a"], [1, 2], "ab"]):
        assert same4.groups(bad) == {}
    assert same4.pairs_of(None) == [] and same4.pairs_of({"same_hazards": "x"}) == []
    assert same4.pairs_of({"same_hazards": [["b", "a"], ["a", "a"], [1, 2], ["a"]]}) == [["a", "b"]]


def test_a_saved_run_carries_its_pairs():
    from agentic import intervention4 as iv4
    r = iv4.frozen_result({}, {}, {"frame": {"same_hazards": [["brush_1", "fire_1"], "x"]}})
    assert r["same_hazards"] == [["brush_1", "fire_1"]]
    assert iv4.frozen_result({}, {}, {})["same_hazards"] == []        # a run from before
    assert iv4.frozen_result(None, None, {"frame": "x"})["same_hazards"] == []


# ── the recommendation table follows the model's judgment ──

def _scene(pairs):
    b = _res(["fire_1", "brush_1", "spill_1", "man_1"],
             [_rec(1, "fire_1", ["man_1"], "Move man_1 away from fire_1."),
              _rec(2, "brush_1", ["fire_1"], "Knock down the burning brush_1.",
                   effect="may_spread_to"),
              _rec(3, "spill_1", ["man_1"], "Contain spill_1.")],
             threats=["fire_1", "brush_1", "spill_1"])
    return {**b, "same_hazards": pairs}


BRUSH_GONE = _res(["fire_1", "spill_1", "man_1"],
                  [_rec(1, "fire_1", ["man_1"], "Move man_1 away from fire_1."),
                   _rec(2, "fire_1", ["man_1"], "Attack fire_1 with a crew."),
                   _rec(3, "spill_1", ["man_1"], "Contain spill_1.")])
BOTH_GONE = _res(["spill_1", "man_1"], [_rec(1, "spill_1", ["man_1"], "Contain spill_1.")])


def test_without_a_declared_pair_the_brush_advice_reads_dropped_as_before():
    t = grounding4.table(_scene([]), [({"index": 1, "target": "brush_1"}, BRUSH_GONE)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[2]["cells"][0]["state"] == "gone" and by[2]["verdict"] == "grounded"
    assert t["same_hazards"] == [] and t["columns"][0]["left_behind"] == []


def test_with_the_pair_declared_advice_on_either_is_advice_on_the_pair():
    """C_tanker ui_3e525100: the brush was removed, the model still wrote
    "Attack fire_1", and the brush advice read dropped -> grounded."""
    b = _scene([["brush_1", "fire_1"]])
    t = grounding4.table(b, [({"index": 1, "target": "brush_1"}, BRUSH_GONE)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[2]["cells"][0]["state"] != "gone"                    # the advice is still there
    assert by[2]["cells"][0]["after"]["action"] == "Attack fire_1 with a crew."
    assert by[2]["verdict"] == "untested"                          # the pair is not removed yet
    assert by[1]["cells"][0]["state"] == "stays"
    assert by[2]["same_as"] == ["fire_1"] and by[1]["same_as"] == ["brush_1"]
    assert by[3]["same_as"] == []
    # the edit took one and left the other: said, not hidden
    assert t["columns"][0]["left_behind"] == ["fire_1"]
    assert t["same_hazards"] == [("brush_1", "fire_1")]
    comp = ui_iv.table_component(t)
    assert "the model calls it the same hazard as fire_1; still in the edited picture" in _flat(
        _by_class(comp, "iv-left-behind"))
    assert [_flat(x) for x in _by_class(comp, "iv-same")] == ["same hazard as brush_1",
                                                               "same hazard as fire_1"]


def test_the_pair_is_removed_when_both_are_gone_and_both_rows_are_affected():
    b = _scene([["brush_1", "fire_1"]])
    t = grounding4.table(b, [({"index": 1, "target": "fire_1"}, BOTH_GONE)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[1]["cells"][0]["state"] == "gone" and by[1]["verdict"] == "grounded"
    assert by[2]["cells"][0]["state"] == "gone" and by[2]["verdict"] == "grounded"
    assert by[3]["cells"][0]["state"] == "stays"
    assert t["columns"][0]["left_behind"] == []


# ── the feature table: a pair is one row ──

def _features(pairs):
    return {**D_BEFORE, "same_hazards": pairs}


def test_a_declared_pair_is_one_feature_row_weighed_together():
    alone = declared4.declared(D_BEFORE)
    paired = declared4.declared(_features([["brush_1", "fire_1"]]))
    assert "fire_1" not in paired and "brush_1" not in paired
    assert paired["brush_1 + fire_1"] == alone["fire_1"] + alone["brush_1"]
    assert paired["spill_1"] == alone["spill_1"]
    assert declared4.declared({**D_BEFORE, "same_hazards": "junk"}) == alone


def test_the_shift_goes_to_the_pair_only_when_the_edit_removed_both():
    b = _features([["brush_1", "fire_1"]])
    one, after = _try("fire_1", 0.6)
    both_gone = {**after, "detected_objects": [o for o in after["detected_objects"]
                                               if o["object_id"] not in ("fire_1", "brush_1")]}
    assert declared4.removed_name(b, one, both_gone) == "brush_1 + fire_1"
    assert declared4.removed_name(b, one, after) == "fire_1"          # the brush is still there
    assert declared4.removed_name(D_BEFORE, one, both_gone) == "fire_1"   # no pair declared
    assert declared4.removed_name(None, None, None) == ""
    c = declared4.compare(b, [(one, both_gone), _try("spill_1", 0.3)])
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["brush_1 + fire_1"]["shift"] is not None and "fire_1" not in by
    assert c["same_hazards"] == [("brush_1", "fire_1")]
    card = ui_iv.card_component(c, {})
    assert "the model says: brush_1 and fire_1 are the same hazard" in _flat(card)
    # removed alone: its own row, with the reason said
    c = declared4.compare(b, [(one, after), _try("spill_1", 0.3)])
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["fire_1"]["alone_of"] == ["brush_1"] and by["brush_1 + fire_1"]["shift"] is None
    assert "removed alone; the model calls it the same hazard as brush_1" in _flat(
        ui_iv.card_component(c, {}))


def test_the_remove_next_button_picks_one_hazard_of_a_pair():
    c = declared4.compare(_features([["brush_1", "fire_1"]]), [_try("spill_1", 0.3)])
    assert "brush_1 + fire_1" in c["not_removed"]
    card = ui_iv.card_component(c, {}, next_id=lambda h: {"type": "iv-next", "target": h})
    (btn,) = _by_class(card, "iv-next")
    assert btn.id["target"] in D_OBJS and " + " not in btn.id["target"]


def test_ground_truth_reads_a_pair_as_core_when_either_is():
    c = declared4.compare(_features([["brush_1", "fire_1"]]), [_try("spill_1", 0.3)])
    rows = {g["object_id"]: g for g in declared4.against_ground_truth(
        c, {"fire_1": 0.4, "brush_1": 0.1}, ["fire_1"])}
    assert rows["brush_1 + fire_1"]["is_core"] is True
    assert rows["brush_1 + fire_1"]["should"] == 0.5
    assert "fire_1" not in rows                                       # not listed a second time


# ── shown as soon as the run is done, before any try ──

def test_the_pair_is_named_in_the_try_form_not_under_the_recommendations():
    from agentic import ui_clean
    s4 = {"frame": {"same_hazards": [["brush_1", "fire_1"]]},
          "recommendations": [_rec(1, "fire_1", ["man_1"])]}
    line = ui_clean.same_hazards_line(s4)
    assert _flat(line) == "the model says: brush_1 and fire_1 are the same hazard"
    # not under the recommendations: it is shown where the intervention is
    assert "same hazard" not in _flat(ui_clean.clean_recs(s4))
    for none in ({}, None, {"frame": "x"}, {"frame": {"same_hazards": []}},
                 {"frame": {"same_hazards": "junk"}}):
        assert ui_clean.same_hazards_line(none) is None
    assert "same hazard" not in _flat(ui_clean.clean_recs({"recommendations": s4["recommendations"]}))
    note = ui_iv.same_note({"same_hazards": [["brush_1", "fire_1"]]})
    assert "brush_1 and fire_1 are the same hazard" in note and "remove them together" in note
    assert ui_iv.same_note({}) == "" and ui_iv.same_note(None) == ""


def test_the_try_form_offers_a_declared_pair_as_one_option_with_joint_instructions():
    def info(pairs):
        return {"candidates": [{"object_id": "brush_1", "label": "brush", "state": "burning"},
                               {"object_id": "fire_1", "label": "fire", "state": "spreading"},
                               {"object_id": "spill_1", "label": "spill", "state": "leaking"}],
                "picked_by": {"fire_1": ["code"], "brush_1": ["model, asked"]},
                "result": {"same_hazards": pairs, "detected_objects": [
                    {"object_id": o, "label": o.rsplit("_", 1)[0]}
                    for o in ("brush_1", "fire_1", "spill_1", "man_1")]}}
    plain = ui_iv.target_options(info([]))
    assert [o["value"] for o in plain] == ["brush_1", "fire_1", "spill_1"]
    opts = ui_iv.target_options(info([["brush_1", "fire_1"]]))
    assert [o["value"] for o in opts] == ["brush_1", "spill_1"]         # the pair is one option
    assert "brush_1 + fire_1 · the model says: same hazard · remove both" in opts[0]["label"]
    assert "model, asked" in opts[0]["label"] and "code" in opts[0]["label"]
    # the instructions cover both, and neither is told to stay as it is
    cap, img = ui_iv.instructions(info([["brush_1", "fire_1"]]), "brush_1")
    one_cap, one_img = ui_iv.instructions(info([]), "brush_1")
    assert one_cap in cap and "fire" in cap and "fire" not in one_cap
    assert len(img) > len(one_img)
    assert ui_iv.instructions(info([["brush_1", "fire_1"]]), "nope") == ("", "")
    # a pair naming something that is not a candidate changes nothing
    assert ui_iv.target_options(info([["brush_1", "ghost_9"]])) == plain
    for bad in (None, {}, {"candidates": [None, "x", {}]}):
        assert ui_iv.target_options(bad) == []
