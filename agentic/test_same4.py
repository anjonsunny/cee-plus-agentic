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
                   _rec(3, "spill_1", ["man_1"], "Contain spill_1.")],
                  threats=["fire_1", "spill_1"])
# the fire is out of the picture; the brush is still there, as plain brush
FIRE_OUT_BRUSH_PLAIN = _res(["brush_1", "spill_1", "man_1"],
                            [_rec(1, "spill_1", ["man_1"], "Contain spill_1.")],
                            threats=["spill_1"])
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


def test_a_member_left_in_the_picture_as_a_plain_object_counts_as_removed():
    """A burning house is edited to an intact house, not deleted. Same for
    burning brush: it stays as plain brush, and the pair is still removed."""
    b = _scene([["brush_1", "fire_1"]])
    t = grounding4.table(b, [({"index": 1, "target": "fire_1"}, FIRE_OUT_BRUSH_PLAIN)])
    by = {r["rank"]: r for r in t["rows"]}
    assert by[1]["verdict"] == "grounded" and by[2]["verdict"] == "grounded"
    assert t["columns"][0]["left_behind"] == []
    assert "brush_1" in t["columns"][0]["also_removed"]
    one = {"target": "fire_1"}
    assert declared4.removed_name(b, one, FIRE_OUT_BRUSH_PLAIN) == "brush_1 + fire_1"
    # picked the other way round, same reading
    assert declared4.removed_name(b, {"target": "brush_1"}, FIRE_OUT_BRUSH_PLAIN) == "brush_1 + fire_1"
    # the brush gone but the fire still a hazard: the edit separated them
    assert declared4.removed_name(b, {"target": "brush_1"}, BRUSH_GONE) == "brush_1"


# ── the feature table: a pair is one row ──

def _features(pairs):
    return {**D_BEFORE, "same_hazards": pairs}


def test_a_declared_pair_stays_two_feature_rows_each_with_its_own_weight():
    """Sunny, 2026-11-19: "they are separate features". The pair shares a
    try, not a row."""
    alone = declared4.declared(D_BEFORE)
    assert declared4.declared(_features([["brush_1", "fire_1"]])) == alone
    assert declared4.declared({**D_BEFORE, "same_hazards": "junk"}) == alone


def _gone(after, ids):
    return {**after, "detected_objects": [o for o in after["detected_objects"]
                                          if o["object_id"] not in ids]}


def test_one_try_on_a_pair_fills_the_shift_on_both_rows():
    b = _features([["brush_1", "fire_1"]])
    one, after = _try("fire_1", 0.6)
    both_gone = _gone(after, ("fire_1", "brush_1"))
    assert declared4.removed_name(b, one, both_gone) == "brush_1 + fire_1"
    still = {**after, "threats": [{"object_id": "brush_1"}]}
    assert declared4.removed_name(b, one, still) == "fire_1"          # the brush still burns
    assert declared4.removed_name(D_BEFORE, one, both_gone) == "fire_1"   # no pair declared
    assert declared4.removed_name(None, None, None) == ""
    c = declared4.compare(b, [(one, both_gone), _try("spill_1", 0.3)])
    by = {r["object_id"]: r for r in c["hazards"]}
    assert "brush_1 + fire_1" not in by                                # two rows, not one
    assert by["fire_1"]["shift"] == by["brush_1"]["shift"] is not None
    assert by["fire_1"]["parts"] == by["brush_1"]["parts"] != []
    assert by["fire_1"]["same_as"] == ["brush_1"] and by["brush_1"]["same_as"] == ["fire_1"]
    assert by["fire_1"]["alone_of"] == [] and by["spill_1"]["same_as"] == []
    assert by["fire_1"]["says"] != by["brush_1"]["says"]               # each its own weight
    assert c["same_hazards"] == [("brush_1", "fire_1")]
    card = ui_iv.card_component(c, {})
    assert "the model says: brush_1 and fire_1 are the same hazard" in _flat(card)
    rows = {_flat(r).split()[0]: _flat(r) for r in _by_class(card, "iv-feature")}
    assert "same hazard as brush_1" in rows["fire_1"] and "same hazard as fire_1" in rows["brush_1"]
    # removed alone: the other row stays untested, and the reason is said
    c = declared4.compare(b, [(one, still), _try("spill_1", 0.3)])
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["fire_1"]["alone_of"] == ["brush_1"] and by["brush_1"]["shift"] is None
    assert "removed alone; the model calls it the same hazard as brush_1" in _flat(
        ui_iv.card_component(c, {}))


def test_the_score_counts_a_pair_once():
    one, after = _try("fire_1", 0.6)
    both_gone = _gone(after, ("fire_1", "brush_1"))
    tries = [(one, both_gone), _try("spill_1", 0.6)]
    paired = declared4.compare(_features([["brush_1", "fire_1"]]), tries)
    # declared: fire 5.5 + brush 1.5 = 7 against spill 7; shifts 0.4 and 0.4
    assert paired["score"] == pytest.approx(1.0, abs=0.01)
    assert paired["graph_vs_mechanistic"] == paired["score"]
    # one try on the pair and nothing else: one unit, so no score yet
    assert declared4.compare(_features([["brush_1", "fire_1"]]), [(one, both_gone)])["score"] is None


def test_each_feature_row_shows_the_state_of_its_hazard():
    b = {**D_BEFORE, "detected_objects": [
        {**o, "state": {"fire_1": "spreading", "brush_1": "burning"}.get(o["object_id"], "")}
        for o in D_BEFORE["detected_objects"]]}
    c = declared4.compare(b, [_try("spill_1", 0.3)])
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["fire_1"]["state"] == "spreading" and by["brush_1"]["state"] == "burning"
    rows = {_flat(r).split()[0]: r for r in _by_class(ui_iv.card_component(c, {}), "iv-feature")}
    assert _flat(_by_class(rows["fire_1"], "iv-state")) == "\u00b7 spreading"
    assert _by_class(rows["spill_1"], "iv-state") == []                # no state, nothing shown
    assert declared4.compare({**D_BEFORE, "detected_objects": [None, "x"]}, [])["hazards"] is not None


def test_ground_truth_judges_each_row_of_a_pair_by_itself():
    c = declared4.compare(_features([["brush_1", "fire_1"]]), [_try("spill_1", 0.3)])
    rows = {g["object_id"]: g for g in declared4.against_ground_truth(
        c, {"fire_1": 0.4, "brush_1": 0.1}, ["fire_1"])}
    assert rows["fire_1"]["is_core"] is True and rows["brush_1"]["is_core"] is False
    assert rows["fire_1"]["should"] == 0.4 and rows["brush_1"]["should"] == 0.1


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


# ── a hazard that went with another (no fire, so no smoke) ──

def _smoke_scene():
    objs = ["fire_1", "smoke_1", "spill_1", "man_1"]
    b = _belief(objs, [("fire_1", "may_harm", "man_1"), ("smoke_1", "may_harm", "man_1"),
                       ("spill_1", "may_harm", "man_1")])
    return {**b, "threats": [{"object_id": h} for h in ("fire_1", "smoke_1", "spill_1")]}, objs


def _after(objs, gone, threats):
    return {**_belief([o for o in objs if o not in gone], []),
            "threats": [{"object_id": h} for h in threats]}


def test_a_hazard_that_went_with_the_removed_one_carries_the_same_shift():
    """Sunny, 2026-11-19: "smoke_1 also should get affected in feature
    table". The recommendation table already counted it as removed."""
    b, objs = _smoke_scene()
    one, _ = _try("fire_1", 0.6)
    after = _after(objs, ("fire_1", "smoke_1"), ["spill_1"])
    c = declared4.compare(b, [(one, after)])
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["smoke_1"]["shift"] == by["fire_1"]["shift"] is not None
    assert by["smoke_1"]["does"] == by["fire_1"]["does"] == 1.0        # the same number
    assert by["smoke_1"]["gone_with"] == ["fire_1"] and by["fire_1"]["gone_with"] == []
    assert by["spill_1"]["shift"] is None
    assert c["score"] is None                                          # one try is one unit
    rows = {_flat(r).split()[0]: _flat(r) for r in
            _by_class(ui_iv.card_component(c, {}), "iv-feature")}
    assert "gone with fire_1" in rows["smoke_1"] and "gone with" not in rows["fire_1"]
    assert "shifts emergency responses" in rows["smoke_1"]


def test_removing_it_by_itself_later_is_its_own_test_and_wins():
    b, objs = _smoke_scene()
    with_fire = (_try("fire_1", 0.9)[0], _after(objs, ("fire_1", "smoke_1"), ["spill_1"]))
    alone = (_try("smoke_1", 0.15)[0], _after(objs, ("smoke_1",), ["fire_1", "spill_1"]))
    for tries in ([with_fire, alone], [alone, with_fire]):
        by = {r["object_id"]: r for r in declared4.compare(b, tries)["hazards"]}
        assert by["smoke_1"]["gone_with"] == []                        # its own try now
        assert by["smoke_1"]["shift"] < by["fire_1"]["shift"]          # the smaller, direct one
    # a try that took out three counts once in the score, not three times
    c = declared4.compare(b, [with_fire, (_try("spill_1", 0.9)[0],
                                          _after(objs, ("spill_1",), ["fire_1", "smoke_1"]))])
    assert c["score"] == pytest.approx(1 - abs(2 / 3 - 0.5), abs=0.02)
    for bad in (None, "x", [None, (1,), ("a", "b", "c"), ({"target": "a"}, None)]):
        assert isinstance(declared4.mechanistic(b, bad), dict)
        assert isinstance(declared4.mechanistic_parts(b, bad), dict)


# ── ground truth words for a pair and for a hazard that only went along ──

def _pair_smoke_scene():
    objs = ["fire_1", "brush_1", "smoke_1", "spill_1", "man_1"]
    b = _belief(objs, [(h, "may_harm", "man_1") for h in objs[:4]])
    return {**b, "threats": [{"object_id": h} for h in objs[:4]],
            "same_hazards": [["brush_1", "fire_1"]]}, objs


def test_a_pair_member_is_judged_as_the_hazard_ground_truth_names():
    """C_tanker ui_2345d3fc: fire_1 read core while brush_1, which the
    model calls the same fire, read spurious."""
    b, objs = _pair_smoke_scene()
    after = _after(objs, ("fire_1", "brush_1", "smoke_1"), ["spill_1"])
    c = declared4.compare(b, [(_try("brush_1", 0.6)[0], after)])
    gt = {g["object_id"]: g for g in declared4.against_ground_truth(
        c, {"fire_1": 0.4, "spill_1": 0.6}, ["fire_1", "spill_1"])}
    assert gt["fire_1"]["word"] == gt["brush_1"]["word"] == declared4.CORE
    assert gt["brush_1"]["via"] == ["fire_1"] and gt["fire_1"]["via"] == []
    assert gt["brush_1"]["should"] == 0.0                              # no weight invented
    # smoke only went along: neither core nor spurious
    assert gt["smoke_1"]["word"] == declared4.NOT_ALONE
    assert gt["spill_1"]["word"] == declared4.SHOULD_UNTESTED
    syn = {"available": True, "distribution": {"gt_dist": {"fire_1": 0.4, "spill_1": 0.6}},
           "gt_core": ["fire_1", "spill_1"], "gt": {}}
    rows = {_flat(r).split()[0]: r for r in
            _by_class(ui_iv.card_component(c, {}, syn), "iv-feature")}
    assert _flat(_by_class(rows["brush_1"], "iv-gt-word")) == "core, as fire_1"
    assert _flat(_by_class(rows["brush_1"], "iv-gt-via")) == "with fire_1"
    assert _flat(_by_class(rows["smoke_1"], "iv-gt-word")) == "not tested alone"
    assert _flat(_by_class(rows["fire_1"], "iv-gt-word")) == "core"


def test_a_pair_member_removed_alone_or_tested_by_itself_gets_its_own_word():
    b, objs = _pair_smoke_scene()
    # brush removed alone (the edit kept the fire): judged by itself
    alone = (_try("brush_1", 0.6)[0], _after(objs, ("brush_1",), ["fire_1", "smoke_1", "spill_1"]))
    c = declared4.compare(b, [alone])
    gt = {g["object_id"]: g for g in declared4.against_ground_truth(c, {}, ["fire_1"])}
    assert gt["brush_1"]["via"] == [] and gt["brush_1"]["word"] != declared4.CORE
    # smoke removed by itself: no longer "not tested alone"
    own = (_try("smoke_1", 0.6)[0], _after(objs, ("smoke_1",), ["fire_1", "brush_1", "spill_1"]))
    c = declared4.compare(b, [own])
    gt = {g["object_id"]: g for g in declared4.against_ground_truth(c, {}, ["fire_1"])}
    assert gt["smoke_1"]["word"] != declared4.NOT_ALONE
    # malformed rows never raise
    for bad in ({"hazards": [{"object_id": "a", "same_as": None, "gone_with": "x", "does": 1}]},
                {"hazards": [{"object_id": "a", "same_as": [None, 3], "alone_of": 5}]}):
        assert isinstance(declared4.against_ground_truth(bad, {"a": "x"}, ["a", None]), list)


def test_one_try_that_took_out_several_is_said_once_and_later_bars_are_faded():
    b, objs = _pair_smoke_scene()
    after = _after(objs, ("fire_1", "brush_1", "smoke_1"), ["spill_1"])
    c = declared4.compare(b, [(_try("brush_1", 0.6)[0], after),
                              (_try("spill_1", 0.6)[0],
                               _after(objs, ("spill_1",), ["fire_1", "brush_1", "smoke_1"]))])
    assert [sorted(u) for u in c["one_try"]] == [["brush_1", "fire_1", "smoke_1"]]
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["spill_1"]["try_with"] == [] and by["smoke_1"]["try_with"] == ["brush_1", "fire_1"]
    card = ui_iv.card_component(c, {})
    assert _flat(_by_class(card, "iv-one-try")) == (
        "removed in one try, counted once: brush_1, fire_1, smoke_1")
    feats = _by_class(card, "iv-feature")
    faded = [_flat(r).split()[0] for r in feats if _by_class(r, "iv-again")]
    full = [_flat(r).split()[0] for r in feats if _by_class(r, "iv-own")]
    assert len(faded) == 2 and "spill_1" in full and len(full) == 2    # one full bar per try
    # nothing shared, nothing said
    c1 = declared4.compare(b, [(_try("spill_1", 0.6)[0],
                                _after(objs, ("spill_1",), ["fire_1", "brush_1", "smoke_1"]))])
    assert c1["one_try"] == [] and _by_class(ui_iv.card_component(c1, {}), "iv-one-try") == []
    assert ui_iv.card_component({"one_try": [None, "x", ["a"]], "hazards": [
        {"object_id": "a", "try_with": "zz", "does": 0.5}]}, {}) is not None


# ── ground truth lists every thing, and says which are the same hazard ──

def _gt_before(tmp_path, body):
    import json as _json
    (tmp_path / "pic.jpg.gt.json").write_text(body if isinstance(body, str) else _json.dumps(body))
    objs = [{"object_id": "fire_1", "label": "fire"}, {"object_id": "brush_1", "label": "brush"},
            {"object_id": "tanker_truck_1", "label": "tanker"}, {"object_id": "man_1", "label": "man"},
            {"object_id": "man_2", "label": "man"}, None, "x"]
    return {"gt": {"dir": str(tmp_path), "image_filename": "pic.jpg", "status": "verified"},
            "detected_objects": objs, "same_hazards": [["man_1", "man_2"]]}


def test_ground_truth_lists_every_thing_and_its_own_same_hazards(tmp_path):
    from agentic import intervention4 as iv4
    b = _gt_before(tmp_path, {
        "nodes": [{"id": "fire_1", "label": "fire"}, {"id": "brush_1", "label": "brush"},
                  {"id": "tanker_1", "label": "tanker"},      # another id, one label: matched
                  {"id": "driver_1", "label": "man"},         # two men in the run: no match
                  {"id": "ghost_1", "label": "ghost"}, None, {"id": 7}],
        "same_hazards": [["fire_1", "brush_1"], ["fire_1", "ghost_1"], ["fire_1"], None, "x",
                         ["fire_1", "fire_1"]]})
    got = iv4.gt_objects(b)
    assert got["listed"] == ["brush_1", "fire_1", "tanker_truck_1"]
    assert got["same_hazards"] == [["brush_1", "fire_1"]]
    assert iv4.same_pairs_for_gt(b) == [["brush_1", "fire_1"]]         # the human's, not the model's
    # an older file does not answer the question: the model's own pairs stand
    old = _gt_before(tmp_path, {"nodes": [{"id": "fire_1", "label": "fire"}]})
    assert iv4.gt_objects(old)["same_hazards"] is None
    assert iv4.same_pairs_for_gt(old) == [["man_1", "man_2"]]
    # no file, a broken file, nonsense: nothing listed, never raises
    for bad in (None, {}, {"gt": None}, {"gt": {"dir": str(tmp_path), "image_filename": "no"}},
                _gt_before(tmp_path, "{not json"), _gt_before(tmp_path, "[1, 2]"),
                _gt_before(tmp_path, {"nodes": "x", "same_hazards": "y"})):
        got = iv4.gt_objects(bad)
        assert got["listed"] == [] and got["same_hazards"] is None and got["roles"] == {}
        assert not (got["graph"] or {}).get("nodes")


def test_a_try_counts_as_the_hazard_ground_truth_has():
    """C_tanker ui_2345d3fc: the fire was removed, the try was saved under
    brush_1, ground truth has only fire_1, and the match read 0.786."""
    from agentic import intervention4 as iv4
    pairs, gt = [["brush_1", "fire_1"]], {"fire_1", "spill_1"}
    gone = {"threats": [{"object_id": "spill_1"}]}
    assert iv4.counted_as("brush_1", gone, gt, pairs) == "fire_1"
    assert iv4.counted_as("fire_1", gone, gt, pairs) == "fire_1"       # already a GT hazard
    assert iv4.counted_as("smoke_1", gone, gt, pairs) == "smoke_1"     # no pair: itself
    # the fire is still a hazard after the edit: the brush was removed alone
    kept = {"threats": [{"object_id": "fire_1"}, None, "x"]}
    assert iv4.counted_as("brush_1", kept, gt, pairs) == "brush_1"
    for bad in (None, "x", [], {"threats": "zz"}):
        assert iv4.counted_as("brush_1", bad, gt, pairs) == "fire_1"
    assert iv4.counted_as("brush_1", gone, None, None) == "brush_1"
    assert iv4.counted_as(None, gone, gt, pairs) == ""


def test_ground_truth_pairs_decide_the_word_and_a_listed_thing_gets_a_bar():
    b, objs = _pair_smoke_scene()
    after = _after(objs, ("fire_1", "brush_1", "smoke_1"), ["spill_1"])
    c = declared4.compare(b, [(_try("brush_1", 0.6)[0], after)])
    should, core = {"fire_1": 0.4, "spill_1": 0.6}, ["fire_1", "spill_1"]
    # ground truth answers "none are the same": the model's pair is not used
    gt = {g["object_id"]: g for g in declared4.against_ground_truth(c, should, core, [], [])}
    assert gt["brush_1"]["via"] == [] and gt["brush_1"]["word"] != declared4.CORE
    # ground truth says the pair: core with the fire
    gt = {g["object_id"]: g for g in declared4.against_ground_truth(
        c, should, core, ["brush_1", "smoke_1"], [["brush_1", "fire_1"]])}
    assert gt["brush_1"]["via"] == ["fire_1"] and gt["brush_1"]["word"] == declared4.CORE
    assert gt["smoke_1"]["listed"] is True
    syn = {"available": True, "distribution": {"gt_dist": should}, "gt_core": core, "gt": {},
           "gt_listed": ["brush_1", "smoke_1"], "gt_same_hazards": [["brush_1", "fire_1"]]}
    card = ui_iv.card_component(c, {}, syn)
    rows = {_flat(r).split()[0]: _flat(r) for r in _by_class(card, "iv-feature")}
    assert "not in ground truth" not in rows["smoke_1"]                # listed: a 0.00 bar
    assert _flat(_by_class(card, "iv-same-gt")) == (
        "ground truth says: brush_1 and fire_1 are the same hazard · the model says the same")
    # the two disagree, each way
    card = ui_iv.card_component(c, {}, {**syn, "gt_same_hazards": []})
    assert _flat(_by_class(card, "iv-same-gt")) == (
        "ground truth does not call brush_1 and fire_1 the same hazard")
    card = ui_iv.card_component({**c, "same_hazards": []}, {}, syn)
    assert _flat(_by_class(card, "iv-same-gt")).endswith("the model does not say so")
    # an older file, or nonsense: no line, no crash
    for bad in (None, "x", [None, 3, ["a"]]):
        card = ui_iv.card_component(c, {}, {**syn, "gt_same_hazards": bad, "gt_listed": bad})
        assert card is not None
    assert _by_class(ui_iv.card_component(c, {}, {**syn, "gt_same_hazards": None}),
                     "iv-same-gt") == []


# ── the declared weight of a group: its hazards taken out together ──

def _linked_scene():
    """fire -> man, brush -> man, and an arrow BETWEEN the two: fire -> brush."""
    objs = ["fire_1", "brush_1", "spill_1", "man_1"]
    arrows = [("fire_1", "may_harm", "man_1"), ("brush_1", "may_harm", "man_1"),
              ("fire_1", "may_spread_to", "brush_1"), ("spill_1", "may_harm", "man_1")]
    b = _belief(objs, arrows, arrows)
    return {**b, "threats": [{"object_id": h} for h in objs[:3]],
            "same_hazards": [["brush_1", "fire_1"]]}, objs


def test_an_arrow_between_two_hazards_of_a_group_is_counted_once():
    """Sunny, 2026-11-20: "fire -> brush twice? how?" The arrow goes when
    either end goes, so it is in both hazards' own weights."""
    b, objs = _linked_scene()
    alone = declared4.declared(b)
    together = declared4.declared_together(b, ["fire_1", "brush_1"])
    between = alone["fire_1"] + alone["brush_1"] - together
    assert between > 0                                                 # added: counted twice
    assert together == pytest.approx(declared4.declared_together(b, ("brush_1", "fire_1")))
    # one hazard taken out "together" is just its own weight
    assert declared4.declared_together(b, ["spill_1"]) == pytest.approx(alone["spill_1"])
    # a person in the group is not a hazard and stays
    assert declared4.declared_together(b, ["fire_1", "man_1"]) == pytest.approx(alone["fire_1"])
    assert declared4.declared_together(b, ["man_1"]) == 0.0
    for bad in (None, "fire_1", [], [None, 3, ""], {"a": 1}):
        assert declared4.declared_together(b, bad) == 0.0
    for bad in (None, "x", {"graph_b": "zz"}, {"detected_objects": [None]}):
        assert declared4.declared_together(bad, ["fire_1"]) == 0.0


def test_the_score_uses_the_group_weight_not_the_added_one():
    b, objs = _linked_scene()
    pair = (_try("brush_1", 0.6)[0], _after(objs, ("fire_1", "brush_1"), ["spill_1"]))
    spill = (_try("spill_1", 0.6)[0], _after(objs, ("spill_1",), ["fire_1", "brush_1"]))
    c = declared4.compare(b, [pair, spill])
    g = {tuple(x["members"]): x for x in c["groups"]}
    two = g[("brush_1", "fire_1")]
    assert two["declared_graph"] < two["added_graph"]                  # the arrow between, once
    assert two["declared_graph"] == declared4.declared_together(b, ["fire_1", "brush_1"])
    one = g[("spill_1",)]
    assert one["declared_graph"] == one["added_graph"]                 # alone: nothing changes
    # the score is the shared part of the group shares and the shift shares
    total = two["declared_graph"] + one["declared_graph"]
    assert c["score"] == pytest.approx(
        min(two["declared_graph"] / total, 0.5) + min(one["declared_graph"] / total, 0.5),
        abs=0.01)
    # both rows of the pair show the group's share
    by = {r["object_id"]: r for r in c["hazards"]}
    assert by["fire_1"]["says_tested"] == by["brush_1"]["says_tested"] == pytest.approx(
        two["declared_graph"] / total, abs=0.01)
    assert declared4.compare(b, [])["groups"] == []


# ── ground truth measured like the declared side; roles written by the human ──

GT_NODES = [{"id": "fire_1", "label": "fire", "role": "core"},
            {"id": "brush_1", "label": "brush", "role": "core"},
            {"id": "spill_1", "label": "spill"},                        # no word: arrow out = core
            {"id": "shed_1", "label": "shed"},                          # no word, arrow in: spurious
            {"id": "smoke_1", "label": "smoke", "role": "spurious"},    # the human's word wins
            {"id": "driver_1", "label": "man"},                         # a person: core
            {"id": "odd_1", "label": "odd", "role": "banana"}]          # not a role: the fallback
GT_EDGES = [{"source": "fire_1", "target": "driver_1", "effect": "may_harm"},
            {"source": "fire_1", "target": "shed_1", "effect": "may_spread_to"},
            {"source": "spill_1", "target": "driver_1", "effect": "may_harm"},
            {"source": "smoke_1", "target": "driver_1", "effect": "may_harm"},
            {"source": "ghost_9", "target": "driver_1", "effect": "may_harm"}, None, "x"]


def _gt_scene(tmp_path, same=None):
    import json as _json
    body = {"nodes": GT_NODES + [None, {"id": 3}], "edges": GT_EDGES}
    if same is not None:
        body["same_hazards"] = same
    (tmp_path / "pic.jpg.gt.json").write_text(_json.dumps(body))
    objs = ["fire_1", "brush_1", "spill_1", "shed_1", "smoke_1", "man_1", "odd_1"]
    b = _belief(objs, [(h, "may_harm", "man_1") for h in objs[:3]])
    return {**b, "threats": [{"object_id": h} for h in ("fire_1", "brush_1", "spill_1", "smoke_1")],
            "same_hazards": [["brush_1", "fire_1"]],
            "gt": {"dir": str(tmp_path), "image_filename": "pic.jpg", "status": "verified"}}, objs


def test_every_thing_has_a_role_and_a_spurious_thing_may_have_an_arrow(tmp_path):
    """Sunny, 2026-11-20: "spurious objects might have arrow" (the shed the
    fire may reach) and "human is also core"."""
    from agentic import intervention4 as iv4
    b, _ = _gt_scene(tmp_path)
    extra = iv4.gt_objects(b)
    assert extra["roles"] == {"fire_1": "core", "brush_1": "core", "spill_1": "core",
                              "shed_1": "spurious", "smoke_1": "spurious",
                              "man_1": "core", "odd_1": "spurious"}
    w = declared4.should_matter(extra)
    assert set(w) == {"fire_1", "spill_1"}                             # smoke has an arrow, and 0
    assert w["fire_1"] > w["spill_1"]                                  # the arrow to the shed too
    # the brush is core and adds nothing: taken out with the fire, the fire's weight
    assert declared4.should_matter_together(extra, ["fire_1", "brush_1", "smoke_1"]) == (
        pytest.approx(w["fire_1"]))
    assert declared4.should_matter_together(extra, ["smoke_1"]) == 0.0
    for bad in (None, "x", {}, {"graph": "zz", "roles": 3}, {"graph": {"nodes": [None]}}):
        assert declared4.should_matter(bad) == {}
        assert declared4.should_matter_together(bad, ["fire_1"]) == 0.0
        assert declared4.match_ground_truth({}, bad)["available"] is False
    assert declared4.should_matter_together(extra, None) == 0.0


def test_the_match_with_ground_truth_is_measured_like_declared_vs_mechanistic(tmp_path):
    from agentic import intervention4 as iv4
    b, objs = _gt_scene(tmp_path, same=[["fire_1", "brush_1"]])
    extra = iv4.gt_objects(b)
    fire = (_try("brush_1", 0.6)[0], _after(objs, ("fire_1", "brush_1", "smoke_1"), ["spill_1"]))
    c1 = declared4.compare(b, [fire])
    m1 = declared4.match_ground_truth(c1, extra)
    assert m1["available"] and m1["score"] is None                     # one group: nothing to compare
    assert (m1["removed"], m1["of"]) == (1, 2)
    assert sum(m1["should"].values()) == pytest.approx(1.0, abs=0.01)
    spill = (_try("spill_1", 0.6)[0], _after(objs, ("spill_1",), ["fire_1", "brush_1", "smoke_1"]))
    c2 = declared4.compare(b, [fire, spill])
    m2 = declared4.match_ground_truth(c2, extra)
    w = declared4.should_matter(extra)
    share = w["fire_1"] / (w["fire_1"] + w["spill_1"])
    assert m2["score"] == pytest.approx(min(share, 0.5) + min(1 - share, 0.5), abs=0.01)
    assert m2["reading"] in ("agree", "agree in part", "differ") and m2["removed"] == 2
    # on the card: the new tile and bars; the brush reads core with the fire
    syn = {"available": True, "distribution": {"verdict": "misproportioned", "overlap": 0.1},
           "gt": {}, "gt_extra": extra, "gt_listed": extra["listed"],
           "gt_same_hazards": extra["same_hazards"]}
    card = ui_iv.card_component(c2, {}, syn)
    tile = _flat(_by_class(card, "iv-gt-line"))
    assert f"{m2['score']:.2f}" in tile and m2["reading"] in tile and "misproportioned" not in tile
    rows = {_flat(r).split()[0]: r for r in _by_class(card, "iv-feature")}
    assert _flat(_by_class(rows["brush_1"], "iv-gt-word")) == "core, as fire_1"
    assert _flat(_by_class(rows["smoke_1"], "iv-gt-word")) == "not tested alone"
    # the file cannot be read: the older numbers stand, nothing breaks
    old = ui_iv.card_component(c2, {}, {**syn, "gt_extra": None})
    assert "misproportioned" in _flat(_by_class(old, "iv-gt-line"))
    for bad in ("x", {"graph": None}, {"graph": {"nodes": "n", "edges": 3}, "roles": []}):
        assert ui_iv.card_component(c2, {}, {**syn, "gt_extra": bad}) is not None
    assert declared4.match_ground_truth(None, extra)["score"] is None
    assert declared4.match_ground_truth({"groups": [None, {"members": "x"}, {"members": ["fire_1"],
                                         "shift": "big"}]}, extra)["available"] is True


def test_ground_truth_names_are_matched_one_to_one(tmp_path):
    from agentic import intervention4 as iv4
    objs = [{"object_id": "tanker_truck_1", "label": "tanker_truck"},
            {"object_id": "man_1", "label": "man"}, {"object_id": "tree_1", "label": "tree"},
            {"object_id": "tree_2", "label": "tree"}]
    nodes = [{"id": "tanker_1", "label": "tanker"}, {"id": "driver_1", "label": "driver"},
             {"id": "oak_1", "label": "tree"}]
    got = iv4._gt_to_model(nodes, objs)
    assert got["tanker_1"] == "tanker_truck_1"                         # one label inside the other
    assert got["driver_1"] == "man_1"                                  # a person for the person
    assert "oak_1" not in got                                          # two trees: no guess
    assert iv4._gt_to_model([], objs) == {} and iv4._gt_to_model(nodes, []) == {}
