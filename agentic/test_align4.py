"""The same object under the same name in the before run and the edited
run (agentic/align4.py), and a recommendation kept under another hazard
word. Hermetic. From B_pool ui_b3ada51c, 2026-11-20."""
import pytest

from agentic import align4, grounding4 as g
from agentic.test_ui_intervene import _by_class, _flat, _rec


def _o(oid, box, label=None, state="s"):
    return {"object_id": oid, "label": label or oid.rsplit("_", 1)[0], "state": state, "bbox": box}


POOL = [0, 250, 1670, 940]
BEFORE = {"image_size": [1672, 941], "detected_objects": [
    _o("child_1", [380, 490, 760, 760], state="struggling"),
    _o("child_2", [1355, 344, 1569, 372], state="unconscious"),
    _o("pool_1", POOL)],
    "threats": [],
    "recommendations": [_rec(1, "pool_1", ["child_1"], "Rescue child_1 from pool_1 with a pole."),
                        _rec(2, "pool_1", ["child_2"], "Swim to child_2 in pool_1."),
                        _rec(3, "pool_1", ["child_2"], "Begin CPR on child_2.")]}
# the edit moved the struggling child to the deck; the model swapped the names
AFTER = {"image_size": [1672, 941], "detected_objects": [
    _o("child_1", [1354, 343, 1568, 372], state="unconscious"),      # the floating child
    _o("child_2", [132, 342, 413, 410], state="unconscious"),        # the child on the deck
    _o("pool_1", POOL)],
    "threats": [{"object_id": "child_1"}],
    "recommendations": [_rec(1, "child_1", ["child_1"], "Water rescue of child_1 from pool_1."),
                        _rec(2, "child_2", ["child_2"], "Medical care for child_2 on the deck.")],
    "causal_graph": {"nodes": [{"id": "child_1"}, {"id": "pool_1"}],
                     "edges": [{"source": "pool_1", "target": "child_1 · unconscious"}],
                     "intervention_candidates": [{"object_id": "child_1"}]},
    "graph_b": None, "same_hazards": [["child_1", "child_2"]]}


def test_objects_in_the_same_place_are_the_same_object_whatever_they_are_called():
    m = align4.same_place(BEFORE, AFTER)
    # the floating child is where child_2 was; the moved one is the one left over
    assert m == {"child_1": "child_2", "child_2": "child_1"}
    out = align4.aligned(BEFORE, AFTER)
    assert out is not AFTER and out["renamed"] == m and out["renamed_new"] == {}
    assert [o["object_id"] for o in out["detected_objects"]] == ["child_2", "child_1", "pool_1"]
    assert out["threats"] == [{"object_id": "child_2"}]
    q = out["recommendations"][0]["structured_reasoning"]
    assert q["threat"] == "child_2" and q["affected_objects"] == ["child_2"]
    # the ids inside the action follow, and the model's own words are kept
    assert out["recommendations"][0]["action"] == "Water rescue of child_2 from pool_1."
    assert out["recommendations"][0]["action_as_written"] == "Water rescue of child_1 from pool_1."
    assert out["causal_graph"]["edges"][0]["target"] == "child_2 · unconscious"
    assert out["causal_graph"]["nodes"][0]["id"] == "child_2"
    assert out["causal_graph"]["intervention_candidates"] == [{"object_id": "child_2"}]
    assert out["same_hazards"] == [["child_2", "child_1"]]
    assert AFTER["recommendations"][0]["action"].startswith("Water rescue of child_1")   # untouched
    assert align4.note(out["renamed"]) == (
        'matched by its place in the picture: in the edited run the model calls '
        'child_1 "child_2" and child_2 "child_1"')


def test_names_that_already_agree_and_things_that_cannot_be_matched_are_left_alone():
    assert align4.same_place(BEFORE, BEFORE) == {}
    same = align4.aligned(BEFORE, BEFORE)
    assert same is not BEFORE and "renamed" not in same and same["unsure"] == []   # a copy
    assert same["detected_objects"] == BEFORE["detected_objects"]
    assert align4.summary(same["tracks"]) == {"same place": 3, "by looks": 0, "not sure": 0,
                                              "new": 0, "gone": 0}
    # a REMOVED child: one child left, the floating one, named child_1 by the edited run
    removed = {**AFTER, "detected_objects": [AFTER["detected_objects"][0],
                                             AFTER["detected_objects"][2]]}
    assert align4.same_place(BEFORE, removed) == {"child_1": "child_2"}
    # nothing sits where anything was: no guessing, also not by elimination
    moved = {"detected_objects": [_o("child_1", [10, 10, 60, 60]), _o("child_2", [900, 10, 960, 60])]}
    assert align4.same_place(BEFORE, moved) == {}
    # a newcomer that carries a name now given to another object gets a free one
    extra = {"detected_objects": [_o("tree_2", [100, 100, 200, 300]),
                                  _o("tree_1", [700, 100, 800, 300])]}
    one_tree = {"detected_objects": [_o("tree_1", [100, 100, 200, 300])]}
    m = align4.same_place(one_tree, extra)
    assert m == {"tree_2": "tree_1", "tree_1": "tree_3"}
    out = align4.aligned(one_tree, extra)
    assert out["renamed_new"] == {"tree_1": "tree_3"}
    assert align4.note(out["renamed"], out["renamed_new"]) == (
        'matched by its place in the picture: in the edited run the model calls '
        'tree_1 "tree_2"; its "tree_1" is another object, shown as tree_3')
    # pictures of different sizes are compared as fractions of the picture
    half = {"image_size": [836, 470], "detected_objects": [
        _o("child_1", [677, 171, 784, 186]), _o("child_2", [66, 171, 206, 205])]}
    assert align4.same_place(BEFORE, half)["child_1"] == "child_2"
    # one size unknown: raw pixels, so the half-size picture matches nothing
    assert align4.same_place(BEFORE, {**half, "image_size": None}) == {}


def test_malformed_runs_never_break_the_matching():
    for bad in (None, "x", {}, {"detected_objects": "zz"},
                {"detected_objects": [None, 3, {"object_id": 7}, {"object_id": "a_1"},
                                      {"object_id": "b_1", "bbox": [1, 2]},
                                      {"object_id": "c_1", "bbox": [5, 5, 1, 1]},
                                      {"object_id": "d_1", "bbox": ["a", 1, 2, 3]}]}):
        assert align4.same_place(bad, AFTER) == {} and align4.same_place(BEFORE, bad) == {}
        assert "renamed" not in (align4.aligned(BEFORE, bad) or {})
        assert "renamed" not in (align4.aligned(bad, AFTER) or {})
    odd = {"detected_objects": AFTER["detected_objects"], "threats": [None, "x", {"k": 1}],
           "recommendations": [None, {"action": 5, "structured_reasoning": "q"},
                               {"structured_reasoning": {"affected_objects": "child_1"}}],
           "causal_graph": {"nodes": "n", "edges": [None, {"source": 4}]},
           "graph_b": "zz", "same_hazards": [None, ["child_1"]]}
    assert align4.aligned(BEFORE, odd)["renamed"] == {"child_1": "child_2", "child_2": "child_1"}
    assert align4.rename(AFTER, None) is AFTER and align4.rename(AFTER, {3: 4}) is AFTER
    assert align4.rename("x", {"a": "b"}) == "x"
    assert align4.note(None) == "" and align4.note({3: 4}) == "" and align4.note({}) == ""
    assert align4.iou(None, [0, 0, 1, 1]) == 0.0 and align4.iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert align4.iou([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)
    assert align4._swap("man_1 and manual_1 and man_10", {"man_1": "man_2"}) == (
        "man_2 and manual_1 and man_10")                       # whole names only


def test_with_the_names_matched_the_table_reads_the_pool_try_the_way_it_happened():
    """It read "every recommendation dropped": #1 was dropped, #2 and #3
    were kept (the rescue of the floating child, under another name and
    with the child itself written as the hazard)."""
    one = {"index": 1, "target": "child_1", "kind": "dealt with"}
    raw = g.table(BEFORE, [(one, AFTER)])
    # without the matching, #2 ("swim to child_2") is paired with the WRONG child
    assert raw["rows"][1]["cells"][0]["after"]["action"] == "Medical care for child_2 on the deck."
    t = g.table(BEFORE, [(one, align4.aligned(BEFORE, AFTER))])
    by = {r["rank"]: r["cells"][0] for r in t["rows"]}
    assert by[1]["state"] == "gone" and by[1]["label"] == "grounded"
    assert by[2]["state"] == "stays" and by[2]["hazard_renamed"] is True
    assert by[3]["state"] == "stays"                                   # shares the one rescue
    assert "same victim, the hazard is no longer named" in by[2]["why"]
    assert by[2]["after"]["action"] == "Water rescue of child_2 from pool_1."
    assert t["shift"]["named"] == 1.0 and t["shift"]["other"] < 0.3
    # the medical care for the child on the deck is new, not a kept #1
    assert [n["action"] for n in t["columns"][0]["new"]] == ["Medical care for child_1 on the deck."]
    col = t["columns"][0]
    assert col["renamed"] == {"child_1": "child_2", "child_2": "child_1"}
    from agentic import ui_intervene as ui_iv
    comp = ui_iv.table_component(t)
    assert "matched by its place in the picture" in _flat(_by_class(comp, "iv-renamed"))
    assert _by_class(ui_iv.table_component(raw), "iv-renamed") == []


def test_another_hazard_for_the_same_victim_is_not_the_same_recommendation():
    """smoke -> man is not "kept" because fire -> man is still there: only a
    recommendation that names the VICTIM as its hazard counts as the same one."""
    before = {"detected_objects": [_o("fire_1", [0, 0, 9, 9]), _o("smoke_1", [0, 0, 9, 9]),
                                   _o("man_1", [0, 0, 9, 9]), _o("shed_1", [0, 0, 9, 9])],
              "threats": [{"object_id": "fire_1"}, {"object_id": "smoke_1"}],
              "recommendations": [_rec(1, "fire_1", ["man_1"]), _rec(2, "smoke_1", ["man_1"])]}
    after = {**before, "recommendations": [_rec(1, "fire_1", ["man_1"])]}
    t = g.table(before, [({"index": 1, "target": "shed_1"}, after)])
    assert t["rows"][1]["cells"][0]["state"] == "gone"
    # the victim written as the hazard: kept
    after = {**before, "recommendations": [_rec(1, "fire_1", ["man_1"]),
                                           _rec(2, "man_1", ["man_1"], "Give man_1 a mask.")]}
    t = g.table(before, [({"index": 1, "target": "shed_1"}, after)])
    assert t["rows"][1]["cells"][0]["state"] == "stays"
    # but not when what it protected was the thing removed
    t = g.table(before, [({"index": 1, "target": "man_1"}, after)])
    assert t["rows"][1]["cells"][0].get("hazard_renamed") is None


def test_a_try_measured_before_its_names_were_matched_is_measured_again(tmp_path, monkeypatch):
    from agentic import intervention4 as iv4, ui_intervene as ui_iv
    calls = []
    monkeypatch.setattr(iv4, "result_from_run_dir",
                        lambda d, **k: BEFORE if Path(d).name == "ui_base" else AFTER)
    monkeypatch.setattr(iv4, "measure", lambda b, a, target, **k: calls.append(
        (target, a.get("renamed"))) or {"signals": {"hazard_shift": 0.5}, "verdict": {"cell": "x"}})
    old = {"index": 1, "target": "child_1", "after_run": "ui_after", "signals": {"stale": 1}}
    before, pairs = ui_iv.try_pairs(tmp_path / "ui_base", tmp_path, [old, None, {"index": 2}])
    assert before is BEFORE and len(pairs) == 1
    t, after = pairs[0]
    assert after["renamed"] == {"child_1": "child_2", "child_2": "child_1"}
    assert t["measured_again"] is True and t["signals"] == {"hazard_shift": 0.5}
    assert calls == [("child_1", after["renamed"])] and old["signals"] == {"stale": 1}
    # a try that already carries these names is not measured twice
    calls.clear()
    _, pairs = ui_iv.try_pairs(tmp_path / "ui_base", tmp_path, [{**old, "renamed": after["renamed"]}])
    assert calls == [] and pairs[0][0].get("measured_again") is None
    # the measuring fails: the saved try is used as it is
    monkeypatch.setattr(iv4, "measure", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    _, pairs = ui_iv.try_pairs(tmp_path / "ui_base", tmp_path, [old])
    assert pairs[0][0] is old


from pathlib import Path  # noqa: E402  (used by the last test)


# ── tracking: place, then looks, then a marked guess (Sunny, 2026-11-20:
#    "it's tracking... object tracking"; the model never sees any of it) ──

def _fam(oid, box, family, label=None):
    return {**_o(oid, box, label), "family": family}


def test_a_person_called_a_man_in_the_other_run_is_the_same_person_in_the_same_place():
    """H ui_4a253340: the porch person was person_1, then man_2."""
    b = {"detected_objects": [_fam("person_1", [267, 615, 315, 664], "person"),
                              _fam("man_1", [956, 660, 1054, 962], "person")]}
    a = {"detected_objects": [_fam("man_1", [955, 660, 1054, 962], "person"),
                              _fam("man_2", [265, 615, 315, 665], "person")]}
    t = align4.tracks(b, a)
    how = {p["before"]: (p["after"], p["how"], p["sure"]) for p in t["pairs"]}
    assert how == {"man_1": ("man_1", "same place", True),
                   "person_1": ("man_2", "same place", True)}
    assert align4.same_place(b, a) == {"man_2": "person_1"}
    out = align4.aligned(b, a)
    assert out["unsure"] == [] and out["tracks"]["new"] == [] and out["tracks"]["gone"] == []
    # the same label wins over another label of the family that overlaps more loosely
    b = {"detected_objects": [_fam("brush_1", [100, 100, 400, 300], "vegetation")]}
    a = {"detected_objects": [_fam("tree_1", [130, 100, 400, 300], "vegetation"),     # 0.9
                              _fam("brush_1", [100, 100, 330, 300], "vegetation")]}    # 0.77
    t = align4.tracks(b, a)
    assert [(p["before"], p["after"]) for p in t["pairs"]] == [("brush_1", "brush_1")]
    assert t["new"] == ["tree_1"]
    # another label is taken only when the boxes are nearly the same
    a = {"detected_objects": [_fam("tree_1", [100, 100, 330, 300], "vegetation")]}     # 0.77
    assert align4.tracks(b, a)["pairs"] == []
    # another family is never the same object, wherever it is
    a = {"detected_objects": [_fam("rock_1", [100, 100, 400, 300], "terrain")]}
    assert align4.tracks(b, a)["pairs"] == []


def _haz(oid, box, family, label=None):
    return {**_fam(oid, box, family, label), "state_kind": "hazard_bearing"}


def test_a_smoke_is_the_same_smoke_only_when_it_comes_from_the_same_thing():
    """H ui_4a253340 (Sunny, 2026-11-20: "Smoke got shifted. Car's was not
    detected before... now it does"): the house's smoke and, after the
    house fire was put out, the car's smoke were both "smoke_1"."""
    house, car = [0, 200, 540, 800], [500, 680, 850, 890]
    b = {"detected_objects": [_haz("house_1", house, "structure"), _haz("car_1", car, "vehicle"),
                              _haz("smoke_1", [2, 1, 481, 467], "hazard_media")]}
    assert align4.source_of(b["detected_objects"][2], b["detected_objects"]) == "house_1"
    # the edited run: the house no longer burns, its smoke is gone; the car's is found
    a = {"detected_objects": [_fam("house_1", house, "structure"), _haz("car_1", car, "vehicle"),
                              _haz("smoke_1", [447, 1, 920, 756], "hazard_media")]}
    assert align4.source_of(a["detected_objects"][2], a["detected_objects"]) == "car_1"
    t = align4.tracks(b, a)
    assert t["gone"] == ["smoke_1"] and t["new"] == ["smoke_1"]         # two different smokes
    out = align4.aligned(b, a)
    assert out["renamed"] == {"smoke_1": "smoke_2"} and out["renamed_new"] == {"smoke_1": "smoke_2"}
    assert out["tracks"]["new"] == ["smoke_2"] and out["unsure"] == []
    # the same source, another shape: the same smoke, and sure
    a2 = {"detected_objects": [_haz("house_1", house, "structure"), _haz("car_1", car, "vehicle"),
                               _haz("smoke_7", [100, 1, 400, 300], "hazard_media", "smoke")]}
    (p,) = [x for x in align4.tracks(b, a2)["pairs"] if x["before"] == "smoke_1"]
    assert (p["after"], p["how"], p["sure"], p["source"]) == ("smoke_7", "the same source",
                                                               True, "house_1")
    # nothing near to come from: taken as the same by its name, as a guess
    lone_b = {"detected_objects": [_fam("smoke_1", [0, 0, 500, 200], "hazard_media")]}
    lone_a = {"detected_objects": [_fam("smoke_1", [500, 300, 900, 500], "hazard_media")]}
    (p,) = align4.tracks(lone_b, lone_a)["pairs"]
    assert p["how"] == "the same name" and p["sure"] is False
    assert align4.aligned(lone_b, lone_a)["unsure"] == ["smoke_1"]
    # in the same place it needs no source at all
    (p,) = align4.tracks(lone_b, lone_b)["pairs"]
    assert p["how"] == "same place" and p["sure"] is True
    assert align4.source_of({}, []) is None
    assert align4.source_of(lone_b["detected_objects"][0], [None] if False else []) is None


def test_a_moved_object_is_matched_by_looks_and_a_changed_one_is_a_marked_guess(tmp_path):
    from PIL import Image
    def scene(name, boxes):
        d = tmp_path / name
        d.mkdir()
        img = Image.new("RGB", (400, 200), (40, 90, 200))                # water
        for box, colour in boxes:
            img.paste(colour, tuple(box))
        img.save(d / "s.png")
        return str(d)
    red, green, white = (220, 30, 30), (30, 200, 60), (240, 240, 240)
    b = {"run_dir": scene("b", [([20, 20, 80, 80], red), ([300, 20, 360, 80], green)]),
         "scene_image": "s.png",
         "detected_objects": [_fam("child_1", [20, 20, 80, 80], "person", "child"),
                              _fam("child_2", [300, 20, 360, 80], "person", "child")]}
    # the red child moved; it still looks the same
    a = {"run_dir": scene("a1", [([150, 120, 210, 180], red), ([300, 20, 360, 80], green)]),
         "scene_image": "s.png",
         "detected_objects": [_fam("child_1", [300, 20, 360, 80], "person", "child"),
                              _fam("child_2", [150, 120, 210, 180], "person", "child")]}
    how = {p["before"]: (p["after"], p["how"], p["sure"]) for p in align4.tracks(b, a)["pairs"]}
    assert how["child_2"] == ("child_1", "same place", True)
    assert how["child_1"] == ("child_2", "by looks", True)
    # the moved child is wrapped in a white towel: it no longer looks the same
    a2 = {**a, "run_dir": scene("a2", [([150, 120, 210, 180], white),
                                       ([300, 20, 360, 80], green)])}
    how = {p["before"]: (p["after"], p["how"], p["sure"]) for p in align4.tracks(b, a2)["pairs"]}
    assert how["child_1"] == ("child_2", "the only one left", False)
    assert align4.aligned(b, a2)["unsure"] == ["child_1"]
    assert align4.summary(align4.tracks(b, a2)) == {"same place": 1, "by looks": 0,
                                                    "not sure": 1, "new": 0, "gone": 0}
    # a picture that cannot be read: looks say nothing, nothing breaks
    assert align4._looks({"run_dir": str(tmp_path / "nope"), "scene_image": "s.png"},
                         b["detected_objects"][0]) is None
    assert align4._looks({}, {}) is None and align4._alike(None, None) == 0.0
    for bad in (None, "x", {"pairs": "zz"}, {"pairs": [None, 3], "new": 5}):
        assert align4.summary(bad)["not sure"] == 0


# ── a hazard that was dealt with, and when a recommendation may borrow another ──

def _house():
    objs = [_o("house_1", [0, 0, 9, 9]), _o("car_1", [20, 0, 29, 9]), _o("man_1", [40, 0, 49, 9]),
            _o("shed_1", [60, 0, 69, 9])]
    return {"detected_objects": objs,
            "threats": [{"object_id": "house_1"}, {"object_id": "car_1"}],
            "recommendations": [
                _rec(1, "house_1", ["man_1"], "Move man_1 away from house_1."),
                _rec(2, "house_1", ["shed_1"], "Attack house_1 with a fire engine."),
                _rec(3, "car_1", ["man_1"], "Keep man_1 back from car_1.")]}


def _soon(rec, soon, size):
    return {**rec, "structured_reasoning": {**rec["structured_reasoning"],
                                            "how_soon": soon, "size": size}}


def test_a_dealt_with_hazard_whose_recommendation_stays_but_eases_is_grounded():
    """H ui_4a253340: the house fire was put out; "get the person off the
    porch" stayed, now later and smaller. It read masquerade."""
    b = _house()
    b["recommendations"] = [_soon(r, "now", "large") for r in b["recommendations"]]
    eased = {**b, "recommendations": [
        _soon(_rec(1, "house_1", ["man_1"], "Move man_1 away from house_1."), "minutes", "small"),
        _soon(_rec(2, "car_1", ["man_1"], "Keep man_1 back from car_1."), "now", "large")]}
    one = {"index": 1, "target": "house_1", "kind": "dealt with"}
    t = g.table(b, [(one, eased)])
    by = {r["rank"]: r["cells"][0] for r in t["rows"]}
    assert by[1]["state"] == "stays" and by[1]["way"] == "eased" and by[1]["label"] == "grounded"
    # the same answer after the house was REMOVED is a masquerade: nothing should be left
    t = g.table(b, [({"index": 1, "target": "house_1"}, eased)])
    assert t["rows"][0]["cells"][0]["label"] == "masquerade"
    # dealt with, and the recommendation stays exactly as it was: a masquerade
    same = {**b, "recommendations": [b["recommendations"][0], b["recommendations"][2]]}
    t = g.table(b, [(one, same)])
    assert t["rows"][0]["cells"][0]["label"] == "masquerade"
    # dealt with, and it got worse: a masquerade
    worse = {**b, "recommendations": [_soon(b["recommendations"][0], "now", "large")]}
    b2 = {**b, "recommendations": [_soon(b["recommendations"][0], "later", "small")]}
    t = g.table(b2, [(one, worse)])
    assert t["rows"][0]["cells"][0]["way"] == "escalated"
    assert t["rows"][0]["cells"][0]["label"] == "masquerade"


def test_a_recommendation_borrows_another_only_when_it_protects_the_same_thing():
    """H: "attack the house with a fire engine" read kept because "move the
    person off the porch" was still there. The two protect different things."""
    b = _house()
    after = {**b, "recommendations": [_rec(1, "house_1", ["man_1"], "Move man_1 away."),
                                      _rec(2, "car_1", ["man_1"], "Keep man_1 back.")]}
    t = g.table(b, [({"index": 1, "target": "house_1", "kind": "dealt with"}, after)])
    by = {r["rank"]: r["cells"][0] for r in t["rows"]}
    assert by[2]["state"] == "gone" and by[2]["label"] == "grounded"     # the fire engine: dropped
    assert by[1]["state"] == "stays"
    # two recommendations that protect the same man from the same hazard: one
    # piece left, the second borrows it and reads kept
    b2 = {**b, "recommendations": [_rec(1, "house_1", ["man_1"], "Move man_1 away."),
                                   _rec(2, "house_1", ["man_1"], "Shield man_1.")]}
    after2 = {**b, "recommendations": [_rec(1, "house_1", ["man_1"], "Move man_1 away.")]}
    t = g.table(b2, [({"index": 1, "target": "shed_1"}, after2)])
    assert [r["cells"][0]["state"] for r in t["rows"]] == ["stays", "changed"]
    for bad in (None, {}, {"recommendations": [None, {"structured_reasoning": 3}]}):
        assert g.table(b, [({"index": 1, "target": "house_1", "kind": "dealt with"}, bad)])["rows"]


def test_the_older_renaming_does_not_undo_the_tracking():
    """H: with the house fire out, its smoke is gone and the car's smoke is
    another one. The frozen rule pairs by label and glued them together."""
    house, car = [0, 200, 540, 800], [500, 680, 850, 890]
    b = {"detected_objects": [_haz("house_1", house, "structure"), _haz("car_1", car, "vehicle"),
                              _haz("smoke_1", [2, 1, 481, 467], "hazard_media")],
         "threats": [{"object_id": x} for x in ("house_1", "car_1", "smoke_1")],
         "recommendations": [_rec(1, "house_1", ["car_1"]), _rec(2, "smoke_1", ["car_1"])]}
    a = {"detected_objects": [_fam("house_1", house, "structure"), _haz("car_1", car, "vehicle"),
                              _haz("smoke_1", [447, 1, 920, 756], "hazard_media")],
         "threats": [{"object_id": "car_1"}, {"object_id": "smoke_1"}],
         "recommendations": [_rec(1, "smoke_1", ["car_1"])]}
    al = align4.aligned(b, a)
    assert g.renames(b, al) == {}
    t = g.table(b, [({"index": 1, "target": "house_1", "kind": "dealt with"}, al)])
    assert t["columns"][0]["also_removed"] == ["smoke_1"]              # went with the house fire
    # the recommendation about the house's smoke is gone; the one about the car's is new
    assert t["rows"][1]["cells"][0]["state"] == "gone"
    assert [n["structured_reasoning"]["threat"] for n in t["columns"][0]["new"]] == ["smoke_2"]


# ── the table check on H, 2026-11-23: four mistakes and five plain words ──

def _two_fires():
    objs = [_o("house_1", [0, 0, 9, 9]), _o("car_1", [20, 0, 29, 9]),
            {**_o("man_1", [40, 0, 49, 9]), "family": "person"},
            {**_o("person_1", [60, 0, 69, 9]), "family": "person"},
            _o("house_2", [80, 0, 89, 9]), _o("house_3", [100, 0, 109, 9]),
            _o("tree_1", [120, 0, 129, 9]), _o("bush_1", [140, 0, 149, 9])]
    recs = [_soon(_rec(1, "house_1", ["person_1"], "Get person_1 out of house_1."), "now", "large"),
            _soon(_rec(2, "car_1", ["man_1"], "Move man_1 back from car_1."), "minutes", "medium"),
            _soon(_rec(3, "house_1", ["house_2", "house_3", "tree_1", "bush_1"],
                       "Attack house_1 with a fire engine."), "minutes", "large")]
    return {"detected_objects": objs,
            "threats": [{"object_id": "house_1"}, {"object_id": "car_1"}],
            "recommendations": recs}


def test_a_recommendation_that_protected_only_the_victim_is_dropped_not_adjusted():
    """A. H: with person_1 moved to safety, "get the person off the porch"
    was compared with "deploy a hoseline on the house" and read adjusted 0.63."""
    b = _two_fires()
    after = {**b, "recommendations": [
        _soon(_rec(1, "car_1", ["man_1"], "Move man_1 back from car_1."), "minutes", "medium"),
        _soon(_rec(2, "house_1", ["house_2"], "Deploy a hoseline on house_1."), "minutes", "large")]}
    one = {"index": 1, "target": "person_1", "kind": "dealt with"}
    t = g.table(b, [(one, after)])
    cell = t["rows"][0]["cells"][0]
    assert cell["state"] == "gone" and cell["impact"] == 1.0 and cell["label"] == "grounded"
    assert cell["after"] is None and cell["spill"] == ""               # the try was about it
    # but a recommendation about the same hazard that still protects him is this one, kept
    still = {**b, "recommendations": [
        _soon(_rec(1, "house_1", ["person_1", "house_2"], "Shield person_1."), "now", "large")]}
    cell = g.table(b, [(one, still)])["rows"][0]["cells"][0]
    assert cell["state"] == "stays" and cell["still_names"] is True and cell["label"] == "masquerade"


def test_a_changed_target_counts_by_how_much_and_also_when_the_hazard_was_dealt_with():
    """B and C. Losing one target of four read the same 0.40 as losing three;
    and a kept recommendation that named the dealt-with hazard was never
    charged for it."""
    b = _two_fires()
    def after(targets):
        return {**b, "recommendations": [
            b["recommendations"][0], b["recommendations"][1],
            _soon(_rec(3, "house_1", targets, "Attack house_1."), "minutes", "large")]}
    car = {"index": 1, "target": "car_1", "kind": "dealt with"}
    one = g.table(b, [(car, after(["house_2", "tree_1", "bush_1"]))])["rows"][2]["cells"][0]
    three = g.table(b, [(car, after(["house_2"]))])["rows"][2]["cells"][0]
    T = g.IMPACT["target or effect"]
    assert one["impact"] == pytest.approx(T * 0.25, abs=0.001)
    assert three["impact"] == pytest.approx(T * 0.75, abs=0.001)
    assert one["why"] == ["3 of 4 targets kept"] and one["spill"] == "unaffected"
    assert three["spill"] == "re-aimed"
    # the same change in the column of the hazard that WAS dealt with is charged too
    house = {"index": 1, "target": "house_1", "kind": "dealt with"}
    named = g.table(b, [(house, after(["house_2"]))])["rows"][2]["cells"][0]
    assert named["named_removed"] is True and named["state"] == "changed"
    assert named["impact"] == pytest.approx(T * 0.75, abs=0.001)
    # only the effect word changed: half
    eff = {**b, "recommendations": [b["recommendations"][0], b["recommendations"][1],
                                    {**b["recommendations"][2], "structured_reasoning": {
                                        **b["recommendations"][2]["structured_reasoning"],
                                        "effect": "threatens"}}]}
    cell = g.table(b, [(car, eff)])["rows"][2]["cells"][0]
    assert cell["impact"] == pytest.approx(T / 2) and cell["why"] == ["effect changed"]
    # an older cell that does not say how far counts in full, as before
    assert g.impact_detail({}, {"state": "changed"}, 3)["value"] == T


def test_what_a_try_did_to_a_recommendation_it_was_not_about_has_a_plain_word():
    """E. "Demoting rec 1 to 2 when we put out the car fire" read "correctly
    ignored" (Sunny, 2026-11-23)."""
    b = _two_fires()
    car = {"index": 1, "target": "car_1", "kind": "dealt with"}
    def run(first):
        return g.table(b, [(car, {**b, "recommendations": [first, b["recommendations"][2]]})])
    r1 = b["recommendations"][0]
    assert run(r1)["rows"][0]["cells"][0]["spill"] == "unaffected"
    assert run(_soon(r1, "minutes", "large"))["rows"][0]["cells"][0]["spill"] == "demoted"
    low = {**b, "recommendations": [_soon(r1, "later", "small")] + b["recommendations"][1:]}
    t = g.table(low, [(car, {**b, "recommendations": [r1, b["recommendations"][2]]})])
    assert t["rows"][0]["cells"][0]["spill"] == "promoted"
    other = _soon(_rec(1, "house_1", ["man_1"], "Get man_1 out."), "minutes", "large")
    assert run(other)["rows"][0]["cells"][0]["spill"] == "re-aimed, demoted"
    # gone, though the try was about the car
    t = g.table(b, [(car, {**b, "recommendations": [b["recommendations"][2]]})])
    assert t["rows"][0]["cells"][0]["spill"] == "dropped, though its hazard and victim are still there"
    # a change of rank alone is unaffected
    moved = {**r1, "rank": 2}
    assert run(moved)["rows"][0]["cells"][0]["spill"] == "unaffected"
    # the cells the try WAS about carry no such word
    assert t["rows"][1]["cells"][0]["spill"] == ""
    for bad in (None, "x", {}, {"state": 3}):
        assert g.spill(bad) in ("", "unaffected")


def test_the_reading_no_longer_hides_what_another_try_did():
    """D. H: #5 read "grounded" though it was dropped without cause in another try."""
    from agentic import ui_intervene as ui_iv
    b = _two_fires()
    house = {"index": 1, "target": "house_1", "kind": "dealt with"}
    car = {"index": 2, "target": "car_1", "kind": "dealt with"}
    gone1 = {**b, "recommendations": b["recommendations"][1:2]}      # the house ones are gone
    gone2 = {**b, "recommendations": b["recommendations"][2:]}       # #1 and #2 are gone
    t = g.table(b, [(house, gone1), (car, gone2)])
    row = t["rows"][0]
    assert row["verdict"] == "grounded" and row["spill"] == {"dropped, though its hazard and victim are still there": 1}
    comp = ui_iv.table_component(t)
    reading = _by_class(_by_class(comp, "iv-row")[0], "iv-verdict")[0]
    assert _flat(reading) == "grounded 1 dropped, though its hazard and victim are still there"
    assert _flat(_by_class(_by_class(comp, "iv-row")[2], "iv-verdict")[0]) == "grounded"
    # the cell itself says it, in red, and never "correctly ignored" or "spurious"
    text = _flat(comp)
    assert "dropped, though its hazard and victim are still there" in text
    assert "correctly ignored" not in text and "spurious" not in text


# ── two person labels only; a dealt-with victim follows the hazard rule ──

def test_adults_are_person_and_children_are_child_and_older_words_still_work():
    """Sunny, 2026-11-23: "why the model is labeling same class as person
    and man? That's not good." The same human was person_1, then man_2."""
    from agentic import vocabulary as v
    assert v.LABEL_FAMILIES["person"] == ["person", "child"]
    block = v.vocabulary_prompt_block()
    assert "- person: person, child\n" in block
    assert "man" not in block.split("- responder")[0].replace("person", "")
    for word, want in (("man", "person"), ("woman", "person"), ("guy", "person"),
                       ("Lady", "person"), ("men", "person"), ("boy", "child"),
                       ("girl", "child"), ("kid", "child"), ("children", "child")):
        canon, note, in_vocab, fam = v.canonicalize_label(word)
        assert canon == want and in_vocab and not fam, word
        assert note == f"synonym:{word.lower()}->{want}"            # said, not silent
    assert v.canonicalize_label("person") == ("person", "", True, False)
    assert v.canonicalize_label("child") == ("child", "", True, False)
    # an older run's "man" is still a person, and still a life
    for old in ("man", "woman", "boy", "girl"):
        assert v.family_of(old) == "person" and v.is_life(old)
    assert v.family_of("rock") == "unknown" and not v.is_life("")
    for bad in (None, "", 3):
        assert v.canonicalize_label(bad)[0] == v.OTHER_LABEL


def test_a_dealt_with_victim_still_protected_but_eased_is_grounded_and_not_listed_twice():
    """H ui_4a253340, man_1 moved far down the street: "keep him out of the
    smoke" stayed, later and smaller. It read masquerade, and the same
    recommendation was also listed as new."""
    b = _two_fires()                         # rec 2: car_1 -> man_1, minutes, medium
    eased = _soon(_rec(1, "car_1", ["man_1"], "Account for man_1."), "later", "small")
    after = {**b, "recommendations": [b["recommendations"][0], eased, b["recommendations"][2]]}
    one = {"index": 1, "target": "man_1", "kind": "dealt with"}
    t = g.table(b, [(one, after)])
    cell = t["rows"][1]["cells"][0]
    assert cell["still_names"] is True and cell["way"] == "eased" and cell["state"] == "stays"
    assert cell["label"] == "grounded"
    assert t["columns"][0]["new"] == []                                # not "new" as well
    # still protected exactly as before: the masquerade stands
    t = g.table(b, [(one, b)])
    assert t["rows"][1]["cells"][0]["label"] == "masquerade"
    # the same eased answer after he was REMOVED is a masquerade (nothing should name him)
    t = g.table(b, [({"index": 1, "target": "man_1"}, after)])
    assert t["rows"][1]["cells"][0]["label"] == "masquerade"
    # two recommendations about him, one left that still protects him: it is
    # the counterpart of both, and it is still not a new one
    b2 = _two_fires()
    b2["recommendations"].append(_soon(_rec(4, "car_1", ["man_1"], "Shield man_1."),
                                       "now", "large"))
    t = g.table(b2, [(one, after)])
    states = sorted(r["cells"][0]["state"] for r in t["rows"] if r["threat"] == "car_1")
    assert states == ["stays", "stays"] and t["columns"][0]["new"] == []


def test_a_dropped_recommendation_says_what_is_still_in_the_picture_not_why():
    """Sunny, 2026-11-23: "how do you determine if there is a cause or not".
    We cannot. H: "keep the man out of the smoke" (the house's smoke, on
    paper) was dropped when the car fire was put out; the house still smokes."""
    b = _two_fires()
    car = {"index": 1, "target": "car_1", "kind": "dealt with"}
    # #1 (house_1 -> person_1) is gone; the house and the person are both still there
    after = {**b, "recommendations": b["recommendations"][2:]}
    cell = g.table(b, [(car, after)])["rows"][0]["cells"][0]
    assert cell["spill"] == "dropped, though its hazard and victim are still there"
    assert cell["hazard_there"] is True and cell["victim_there"] is True
    # what it protected is no longer listed in the edited run: plain "dropped"
    fewer = {**after, "detected_objects": [o for o in b["detected_objects"]
                                           if o["object_id"] != "person_1"]}
    assert g.table(b, [(car, fewer)])["rows"][0]["cells"][0]["spill"] == "dropped"
    # it protected nothing by name: only the hazard can be said
    b2 = {**b, "recommendations": [_rec(1, "house_1", [], "Vent house_1.")] + b["recommendations"][1:]}
    cell = g.table(b2, [(car, {**b2, "recommendations": b2["recommendations"][2:]})])["rows"][0]["cells"][0]
    assert cell["spill"] == "dropped, though its hazard is still there"
    # on the screen: the long form in the cell, and never "without cause"
    from agentic import ui_intervene as ui_iv
    t = g.table(b, [(car, after)])
    text = _flat(ui_iv.table_component(t))
    assert "dropped, though its hazard and victim are still there" in text
    assert "without cause" not in text and "spurious" not in text
    assert t["rows"][0]["spill"] == {"dropped, though its hazard and victim are still there": 1}
    for bad in ({"state": "gone"}, {"state": "gone", "named_removed": True}, None):
        assert isinstance(g.spill(bad), str)


def test_nothing_but_the_try_is_under_a_column_picture_and_the_notes_are_folded_below():
    """Sunny, 2026-11-23: "Remove the huge texts you have underneath the
    images for a column", and the column's chip was black, so its box could
    not be seen on the thumbnail."""
    from agentic import ui_intervene as ui_iv
    one = {"index": 1, "target": "child_1", "kind": "dealt with"}
    t = g.table(BEFORE, [(one, align4.aligned(BEFORE, AFTER))])
    comp = ui_iv.table_component(t)
    ths = _by_class(comp, "iv-table")[0].children[0].children.children
    # the try, and whether the model saw the edit: nothing else
    assert _flat(ths[2]) == "#1 child_1 dealt with \u00b7 victim edit seen"
    (notes,) = _by_class(comp, "iv-notes")
    assert notes.open is False and "matched by its place in the picture" in _flat(notes)
    assert _flat(_by_class(notes, "iv-renamed")).startswith("#1 child_1: matched by")
    # the chip carries the colour of its kind: orange for a victim, red for a hazard
    assert ui_iv.chip_col({"victim": True}) == "#f97316" and ui_iv.chip_col({}) == "#ef4444"
    assert ui_iv.chip_col(None) == "#ef4444"
    # a try with nothing to say: no fold at all
    assert ui_iv.column_notes([{"index": 1, "target": "x"}, None, "y"]) == []
    assert ui_iv.column_notes(None) == []


# ── did the model see the edit that was meant? ──

def _seen_pair(after_state="burnt", box=None, gone=False, how="same place"):
    b = {"detected_objects": [{**_o("house_1", [0, 0, 9, 9]), "state": "burning"},
                              {**_o("man_1", [40, 0, 49, 9]), "state": "standing"}]}
    objs = [] if gone else [{**_o("house_1", [0, 0, 9, 9]), "state": after_state}]
    a = {"detected_objects": objs + [{**_o("man_1", box or [40, 0, 49, 9]), "state": "standing"}],
         "tracks": {"pairs": [{"before": "house_1", "after": "house_1", "how": "same place"},
                              {"before": "man_1", "after": "man_1", "how": how}]}}
    return b, a


def test_the_intended_state_is_compared_with_the_state_the_model_saw():
    """Sunny, 2026-11-23: "we want to check whether the meant edit is the
    same as what the model has seen after edit... I also meant the state"."""
    one = {"target": "house_1", "kind": "dealt with", "intended_state": "Burnt"}
    b, a = _seen_pair("burnt")
    e = g.edit_seen(b, a, one)
    assert (e["seen"], e["as_intended"], e["head"]) == (True, True, "edit seen")
    assert e["text"] == "intended: burnt \u00b7 model saw: burnt \u2713"
    # another new state: seen, but not as intended
    e = g.edit_seen(*_seen_pair("intact"), one)
    assert (e["seen"], e["as_intended"], e["head"]) == (True, False, "edit seen, not as intended")
    assert e["text"].endswith("model saw: intact \u2717 changed, but not as intended")
    # the old state: the model did not see the edit, so the try is not a fair test
    e = g.edit_seen(*_seen_pair("burning"), one)
    assert (e["seen"], e["head"]) == (False, "edit not seen")
    assert e["text"] == "intended: burnt \u00b7 model saw: still burning \u2717 not a fair test"
    # nothing was intended: only whether the state changed
    plain = {"target": "house_1", "kind": "dealt with"}
    e = g.edit_seen(*_seen_pair("burnt"), plain)
    assert (e["seen"], e["as_intended"]) == (True, None)
    assert e["text"] == "model saw: burning \u2192 burnt \u2713"
    assert g.edit_seen(*_seen_pair("burning"), plain)["seen"] is False


def test_a_moved_victim_keeps_its_state_and_a_removed_thing_must_be_gone():
    man = {"target": "man_1", "kind": "dealt with", "intended_state": "standing"}
    # moved (its box is somewhere else): seen, as intended, though the state is
    # the same; however the tracking matched it (H: "the only one of its kind")
    e = g.edit_seen(*_seen_pair(how="the only one of its kind", box=[90, 0, 99, 9]), man)
    assert (e["seen"], e["as_intended"], e["saw"]) == (True, True, "moved")
    # in the same place with the same state: not seen
    assert g.edit_seen(*_seen_pair(), man)["seen"] is False
    # removed: gone is seen; still there in the same state is not
    gone = {"target": "house_1"}
    e = g.edit_seen(*_seen_pair(gone=True), gone)
    assert (e["seen"], e["text"]) == (
        True, "intended: not in the picture \u00b7 model saw: not in the picture \u2713")
    assert g.edit_seen(*_seen_pair("burning"), gone)["head"] == "edit not seen"
    assert g.edit_seen(*_seen_pair("intact"), gone)["seen"] is True    # there, no longer burning
    # dealt with, and it vanished: seen, not as intended
    e = g.edit_seen(*_seen_pair(gone=True), {"target": "house_1", "kind": "dealt with",
                                             "intended_state": "burnt"})
    assert (e["seen"], e["as_intended"]) == (True, False)
    # a target the before run does not have, and nonsense: says nothing, never raises
    for bad_one in (None, {}, {"target": "ghost_9"}, {"target": 3, "intended_state": 7}):
        assert g.edit_seen(*_seen_pair(), bad_one)["text"] == ""
    for bad in (None, "x", {}, {"detected_objects": "zz", "tracks": 3}):
        assert isinstance(g.edit_seen(bad, bad, {"target": "house_1"}), dict)


def test_the_form_asks_for_the_intended_state_and_the_result_is_shown_either_way(tmp_path):
    from agentic import ui, ui_intervene as ui_iv, ui_who
    from agentic.test_ui_views import _find
    box, anc = _find(ui.app.layout, "iv-intended")
    assert box.value is None and len(box.options) > 20                  # the model's own state words
    assert {"burning", "burnt", "intact"} <= {o["value"] for o in box.options}
    left = [x for x in anc if getattr(x, "className", "") == "iv-form-rows"][0]
    order = [str(k) for k in left.children]
    assert order.index(next(x for x in order if "iv-kind" in x)) < order.index(
        next(x for x in order if "iv-intended" in x)) < order.index(
        next(x for x in order if "iv-modality" in x))
    # the row follows the pick; a "removed" try has nothing to choose
    from unittest.mock import patch
    info = {"candidates": [{"object_id": "house_1", "state": "burning"}, None]}
    with patch.object(ui, "_iv_info", lambda r: info):
        assert ui.iv_intended("house_1", "dealt with", "base") == ("burning \u2192 ", None, False)
        assert ui.iv_intended("house_1", "removed", "base") == ("not in the picture", None, True)
        assert ui.iv_intended("ghost_9", "dealt with", "base") == ("", None, False)
    # saved with the try
    one = ui_iv.finish_try(tmp_path, None, {"target": "house_1", "kind": "dealt with",
                                            "intended_state": " Burnt "}, error="x")
    assert one["intended_state"] == "burnt"
    assert ui_iv.finish_try(tmp_path, None, {"target": "house_1"}, error="x")["intended_state"] == ""
    # said in the column head whether it was seen or not
    b, a = _seen_pair("burnt")
    b["recommendations"] = [_rec(1, "house_1", ["man_1"])]
    t = g.table(b, [({"index": 1, "target": "house_1", "kind": "dealt with",
                      "intended_state": "burnt"}, {**a, "recommendations": []})])
    (mark,) = _by_class(ui_iv.table_component(t), "iv-edit-seen")
    assert _flat(mark) == "edit seen" and mark.style["color"] == "#16a34a"
    assert "intended: burnt" in mark.title
    b2, a2 = _seen_pair("burning")
    b2["recommendations"] = b["recommendations"]
    t = g.table(b2, [({"index": 1, "target": "house_1", "kind": "dealt with",
                       "intended_state": "burnt"}, {**a2, "recommendations": []})])
    (mark,) = _by_class(ui_iv.table_component(t), "iv-edit-seen")
    assert _flat(mark) == "edit not seen" and mark.style["color"] == "#b91c1c"
    assert ui_iv.state_of(None, "x") == "" and ui_iv.state_options()


# ── the feature shift: the graph without the state, and the state by itself ──

def _g3(house_state):
    return {"nodes": [], "edges": [
        {"source": f"house_1 · {house_state}", "via_state": house_state,
         "effect": "may_harm", "target": "person_1"},
        {"source": "house_1", "via_state": house_state, "effect": "may_spread_to",
         "target": "house_2 · intact"},
        {"source": "car_1", "via_state": "burning", "effect": "may_harm", "target": "man_1"}]}


def _run3(house_state, car_state="burning"):
    return {"causal_graph": _g3(house_state),
            "threats": [{"object_id": "house_1"}, {"object_id": "car_1"}],
            "at_risk": [{"object_id": "person_1"}],
            "detected_objects": [
                {"object_id": "house_1", "label": "house", "state": house_state},
                {"object_id": "car_1", "label": "car", "state": car_state},
                {"object_id": "person_1", "label": "person", "state": "standing"},
                {"object_id": "man_1", "label": "person", "state": "standing"},
                {"object_id": "house_2", "label": "house", "state": "intact"}]}


def test_a_change_of_state_alone_is_not_a_change_of_who_threatens_whom():
    """Sunny, 2026-11-23, the three-arrow example: the house goes from
    burning to burnt and nothing else changes. With the state inside the
    arrow that read 0.80 of the graph changed; nobody threatens anybody
    differently."""
    from agentic import declared4
    b, a = _run3("burning"), _run3("burnt")
    assert declared4.graph_part(b, a) == 0.0
    # the state is its own part: one of the three threats-and-at-risk changed
    assert declared4.state_part(b, a) == pytest.approx(1 / 3, abs=0.001)
    assert declared4.state_part(b, b) == 0.0 and declared4.graph_part(b, b) == 0.0
    # a real change of structure does count: one arrow of three is another one
    a2 = _run3("burning")
    a2["causal_graph"]["edges"][2] = {"source": "car_1", "effect": "may_harm", "target": "person_1"}
    assert declared4.graph_part(b, a2) == pytest.approx(1 - 2 / 4)       # A B C against A B D
    # a thing that is gone counts as changed
    gone = _run3("burning")
    gone["detected_objects"] = [o for o in gone["detected_objects"] if o["object_id"] != "car_1"]
    assert declared4.state_part(b, gone) == pytest.approx(1 / 3, abs=0.001)
    # no arrows at all, or nothing that is a threat or at risk: no such part
    assert declared4.graph_part({}, {"causal_graph": {"edges": []}}) is None
    assert declared4.state_part({"detected_objects": [{"object_id": "x_1", "state": "s"}]}, a) is None
    for bad in (None, "x", {"causal_graph": "zz", "detected_objects": 3},
                {"causal_graph": {"edges": [None, {"source": 3}, {"target": "a"}]}}):
        assert declared4.graph_part(bad, bad) in (None, 0.0)
        assert declared4.state_part(bad, bad) in (None, 0.0)


def test_the_feature_shift_is_four_parts_and_the_state_is_one_of_them():
    from agentic import declared4
    b, a = {**_run3("burning"), "disaster_level": 8}, {**_run3("burnt"), "disaster_level": 7}
    b["recommendations"] = a["recommendations"] = []
    one = {"target": "house_1", "kind": "dealt with", "signals": {
        "hazard_shift": 0.1, "graph_shift": 0.8, "recommendation_shift": 0.6,
        "hazard_level_delta": -1, "rec_direction": "de-escalated",
        "graph_direction": "de-escalated"}}
    rd = declared4.try_reading(b, (one, a))
    parts = {p["name"]: p["signed"] for p in rd["parts"]}
    assert set(parts) == {"level", "recommendations", "graph", "state"}
    assert parts["graph"] == 0.0                                        # not the frozen 0.8
    assert parts["state"] == pytest.approx(1 / 3, abs=0.001)
    assert rd["size"] == pytest.approx((0.1 + 0.6 + 0.0 + 1 / 3) / 4, abs=0.002)
    # nothing that is a threat or at risk: three parts, as before
    bare = {"detected_objects": [], "causal_graph": None, "recommendations": []}
    rd = declared4.try_reading(bare, (one, bare))
    assert [p["name"] for p in rd["parts"]] == ["level", "recommendations", "graph"]
    # (the older graph number has no direction here, so it adds nothing, as before)
    assert rd["size"] == pytest.approx((0.1 + 0.6 + 0.0) / 3, abs=0.002)
