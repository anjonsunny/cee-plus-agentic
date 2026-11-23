"""A flag: a recommendation's words put someone where the picture does not
have them (agentic/place4.py). Hermetic. From H_car_house_fire, 2026-11-20."""
from agentic import grounding4 as g, place4, ui_intervene as ui_iv
from agentic.test_ui_intervene import _by_class, _flat, _rec


def _o(oid, box, label, family):
    return {"object_id": oid, "label": label, "family": family, "state": "s", "bbox": box}


HOUSE, ROAD = [0, 200, 540, 800], [0, 700, 1250, 1250]
OBJS = [_o("house_1", HOUSE, "house", "structure"), _o("road_1", ROAD, "road", "infrastructure"),
        _o("person_1", [956, 660, 1054, 962], "person", "person"),     # in the road
        _o("person_2", [265, 615, 315, 665], "person", "person"),      # on the porch
        _o("car_1", [500, 680, 850, 890], "car", "vehicle")]


def _run(*actions):
    return {"image_size": [1254, 1254], "detected_objects": [dict(o) for o in OBJS],
            "recommendations": [_rec(i + 1, "house_1", ["person_1"], a)
                                for i, a in enumerate(actions)]}


def test_the_one_in_the_road_is_told_to_leave_the_porch_and_that_is_flagged():
    """Two tries in a row: "Evacuate person_1 from the porch of house_1",
    and person_1 was the one in the road."""
    (f,) = place4.mismatches(_run("Evacuate person_1 from the porch of house_1 to safety."))
    assert (f["rank"], f["who"], f["at"], f["instead"]) == (1, "person_1", "house_1", ["person_2"])
    assert f["text"] == ('its words put person_1 at house_1 ("from the porch of house_1"); '
                         'in the picture person_1 is not there, person_2 is')
    (f,) = place4.mismatches(_run("Direct person_1 to step off the porch of house_1."))
    assert f["phrase"] == "off the porch of house_1"
    assert place4.mismatches(_run("Tell person_1 to leave house_1 by the rear exit."))[0]["at"] == (
        "house_1")


def test_it_says_nothing_when_the_words_fit_the_picture_or_it_cannot_tell():
    quiet = [
        "Evacuate person_2 from the porch of house_1.",              # the right person
        "Direct person_1 off road_1 to a safe upwind location.",     # person_1 IS in the road
        "Move person_1 to a safe distance from house_1.",            # "from" alone: no claim
        "Keep person_1 away from car_1.",
        "Attack house_1 with a fire engine.",                        # nobody named
        "Evacuate everyone from the porch of house_1.",              # no id before the phrase
        "Evacuate person_1 from the porch of shed_9.",               # not an object of the run
    ]
    for action in quiet:
        assert place4.mismatches(_run(action)) == [], action
    # nobody else is at the house either: not a mix-up we can show
    alone = _run("Evacuate person_1 from the porch of house_1.")
    alone["detected_objects"] = [o for o in alone["detected_objects"]
                                 if o["object_id"] != "person_2"]
    assert place4.mismatches(alone) == []
    # a "no action needed" entry is never flagged
    benign = _run("Evacuate person_1 from the porch of house_1.")
    benign["recommendations"][0]["benign"] = True
    assert place4.mismatches(benign) == []


def test_malformed_runs_never_raise():
    for bad in (None, "x", {}, {"detected_objects": "zz", "recommendations": 3},
                {"detected_objects": [None, {"object_id": "a_1"}],
                 "recommendations": [None, {"action": 7}, {"action": ""}, {"action": "leave a_1"}]},
                {"detected_objects": [{"object_id": "person_1", "label": "person", "bbox": "q"},
                                      {"object_id": "house_1", "label": "house", "bbox": [1, 2]}],
                 "recommendations": [{"rank": 1, "action": "person_1 must leave house_1"}]}):
        assert place4.mismatches(bad) == []


def test_the_table_carries_the_flag_for_the_run_and_for_each_edited_run():
    before = _run("Evacuate person_2 from the porch of house_1.")
    after = _run("Evacuate person_1 from the porch of house_1.")
    t = g.table(before, [({"index": 1, "target": "car_1", "kind": "dealt with"}, after)])
    assert t["place_flags"] == [] and len(t["columns"][0]["place_flags"]) == 1
    comp = ui_iv.table_component(t)
    (flag,) = _by_class(comp, "iv-place-flag")
    assert _flat(flag).startswith("flag · #1: its words put person_1 at house_1")
    assert "a flag only" in flag.title
    # the run itself can carry one too, said above the table
    t = g.table(after, [({"index": 1, "target": "car_1"}, before)])
    assert len(t["place_flags"]) == 1 and t["columns"][0]["place_flags"] == []
    assert _flat(_by_class(ui_iv.table_component(t), "iv-place-flag")[0]).startswith(
        "flag · this run, #1:")
    # nothing is changed by it: the recommendation reads as it did
    assert after["recommendations"][0]["action"] == "Evacuate person_1 from the porch of house_1."


def test_a_recommendation_that_changed_hands_in_a_flagged_run_is_marked_not_removed():
    """Sunny, 2026-11-23: "mark it". H: "move man_1 back from the car" became
    "direct person_1 to leave the street": the same recommendation with the
    wrong name on it, counted as re-aimed."""
    before = {"image_size": [1254, 1254], "detected_objects": [dict(o) for o in OBJS],
              "threats": [{"object_id": "house_1"}, {"object_id": "car_1"}],
              "recommendations": [
                  _rec(1, "house_1", ["person_2"], "Get person_2 out of house_1."),
                  _rec(2, "car_1", ["person_1"], "Move person_1 back from car_1."),
                  _rec(3, "car_1", ["road_1"], "Put out car_1.")]}
    # the edited run swaps the two people in its words
    after = {**before, "recommendations": [
        _rec(1, "house_1", ["person_1"], "Evacuate person_1 from the porch of house_1."),
        _rec(2, "car_1", ["person_2"], "Move person_2 back from car_1."),
        _rec(3, "car_1", ["road_1"], "Put out car_1.")]}
    t = g.table(before, [({"index": 1, "target": "road_1", "kind": "dealt with"}, after)])
    assert len(t["columns"][0]["place_flags"]) == 1
    by = {r["rank"]: r["cells"][0] for r in t["rows"]}
    assert by[1]["mixup"] is True and by[2]["mixup"] is True
    assert by[2]["spill"] == "re-aimed" and by[2]["impact"] == 0.4     # still counted in full
    assert by[3]["mixup"] is False                                     # it names neither person
    comp = ui_iv.table_component(t)
    marks = _by_class(comp, "iv-mixup")
    assert len(marks) == 2 and _flat(marks[0]) == "possible name mix-up"
    assert "still counts in the numbers" in marks[0].title
    # no flag in the run: no mark, whatever changed
    clean = {**before, "recommendations": [
        _rec(1, "house_1", ["person_2"], "Get person_2 out of house_1."),
        _rec(2, "car_1", ["person_2"], "Move person_2 back from car_1."),
        _rec(3, "car_1", ["road_1"], "Put out car_1.")]}
    t = g.table(before, [({"index": 1, "target": "road_1", "kind": "dealt with"}, clean)])
    assert not any(r["cells"][0]["mixup"] for r in t["rows"])
    assert _by_class(ui_iv.table_component(t), "iv-mixup") == []
