"""Every part of a try's shift says how much and which direction; the
fifth part is the tried object itself (agentic/parts4.py, 2026-11-24)."""
import copy

import pytest

from agentic import declared4, parts4


def _rec(rank, action, threat, affected, effect="may_harm", state="burning"):
    return {"rank": rank, "action": action, "structured_reasoning": {
        "threat": threat, "state": state, "effect": effect, "affected_objects": affected}}


def _scene(man_at_risk=True, man_arrows=True, man_rec=True, car_state="burning",
           man_state="standing", man_kind="proximity", man_listed=True, car_threat=True):
    objs = [{"object_id": "car_1", "label": "car", "state": car_state},
            {"object_id": "house_1", "label": "house", "state": "intact"}]
    if man_listed:
        objs.append({"object_id": "person_1", "label": "person", "state": man_state})
    edges = [{"source": "car_1", "effect": "may_spread_to", "target": "house_1"}]
    if man_arrows:
        edges.append({"source": "car_1", "effect": "may_harm", "target": "person_1"})
    recs = [_rec(1, "Suppress the fire at car_1", "car_1", ["house_1"], "may_spread_to")]
    if man_rec:
        recs.append(_rec(2, "Evacuate person_1 away from car_1", "car_1", ["person_1"]))
    return {"detected_objects": objs,
            "threats": [{"object_id": "car_1"}] if car_threat else [],
            "at_risk": [{"object_id": "person_1", "kind": man_kind}] if man_at_risk else [],
            "graph_b": {"nodes": [], "edges": copy.deepcopy(edges)},
            "causal_graph": {"nodes": [], "edges": copy.deepcopy(edges)},
            "recommendations": recs, "disaster_level": 7}


SIGNALS = {"hazard_shift": 0.1, "graph_shift": 0.5, "recommendation_shift": 0.5,
           "hazard_level_delta": -1, "rec_direction": "de-escalated",
           "graph_direction": "de-escalated"}


def _one(target, kind="dealt with"):
    return {"target": target, "kind": kind, "signals": dict(SIGNALS)}


def _lines(p):
    return {x["name"]: x for x in p["lines"]}


def test_every_state_word_perception_may_use_has_a_severity():
    from agentic import perception as p
    words = (set(p.NORMAL_STATES) | set(p.AT_RISK_STATES) | set(p.HAZARD_BEARING_STATES)
             | set(p.EXTRA_NORMAL_STATES) | set(p.EXTRA_AT_RISK_STATES)
             | set(p.EXTRA_HAZARD_BEARING_STATES))
    assert not [w for w in words if parts4.severity(w) is None]
    # the normal states are the calm end
    assert all(parts4.severity(w) == 0 for w in set(p.NORMAL_STATES) | set(p.EXTRA_NORMAL_STATES))
    assert parts4.severity("Burning") == 3 and parts4.severity("chemical spill") == 2
    for bad in (None, 3, "", "wobbling", ["burning"]):
        assert parts4.severity(bad) is None


def test_the_direction_of_one_state_change():
    f = parts4.state_direction
    assert f("burning", "burnt") == "eased" and f("burnt", "burning") == "escalated"
    assert f("standing", "unconscious") == "escalated" and f("standing", "standing") == "unchanged"
    assert f("burning", None) == "eased"                 # not listed after
    assert f(None, "burning") == "unchanged"             # nothing to compare with
    # a word not on the list, or two words of the same severity: changed, no direction
    assert f("burning", "wobbling") == "changed" and f("standing", "walking") == "changed"


def test_a_part_carries_how_much_and_which_direction():
    p = parts4.part("graph", 0.4, "escalated")
    assert p == {"name": "graph", "size": 0.4, "direction": "escalated", "signed": -0.4}
    assert parts4.part("graph", 0.4, "eased")["signed"] == 0.4
    # no up or down: it counts in how much, not in the signed number
    assert parts4.part("graph", 0.4, "redirected")["signed"] == 0.0
    assert parts4.part("graph", 0.4, "nonsense")["direction"] == "redirected"
    # nothing changed is "unchanged", whatever was said
    assert parts4.part("graph", 0.0, "eased")["direction"] == "unchanged"
    for bad in (None, "big", True, -3):
        assert parts4.part("x", bad, "eased")["size"] == 0.0
    assert parts4.part("x", 7, "eased")["size"] == 1.0


def test_the_parts_vote_by_size_and_are_counted():
    parts = [parts4.part("a", 0.2, "eased"), parts4.part("b", 0.5, "eased"),
             parts4.part("c", 0.4, "redirected"), parts4.part("d", 0.0, "eased"), None, "x"]
    got = parts4.agree(parts)
    assert (got["eased"], got["redirected"], got["unchanged"], got["of"]) == (2, 1, 1, 4)
    assert got["direction"] == "eased"
    # one big escalation outweighs two small easings
    assert parts4.vote([parts4.part("a", 0.1, "eased"), parts4.part("b", 0.1, "eased"),
                        parts4.part("c", 0.9, "escalated")]) == "escalated"
    assert parts4.vote([parts4.part("c", 0.4, "redirected")]) == "redirected"
    assert parts4.vote([]) == "unchanged" and parts4.vote(None) == "unchanged"
    assert parts4.agree(None)["of"] == 0


def test_a_moved_person_the_model_no_longer_counts():
    """Sunny, 2026-11-24: the man is moved; he keeps his state. What the
    model said about him: no longer at risk, nothing threatens him, the
    recommendation that protected him is gone."""
    b = _scene()
    a = _scene(man_at_risk=False, man_arrows=False, man_rec=False)
    p = parts4.itself(b, (_one("person_1"), a))
    by = _lines(p)
    assert (by["at risk"]["size"], by["at risk"]["direction"]) == (1.0, "eased")
    assert by["at risk"]["before"] == "at risk (proximity)" and by["at risk"]["after"] == "not at risk"
    assert (by["hazards that threaten it"]["size"],
            by["hazards that threaten it"]["direction"]) == (1.0, "eased")
    assert by["hazards that threaten it"]["before"] == "car_1"
    assert by["hazards that threaten it"]["after"] == "nothing"
    assert (by["recommendations that protect it"]["size"],
            by["recommendations that protect it"]["direction"]) == (1.0, "eased")
    assert (by["its state"]["size"], by["its state"]["direction"]) == (0.0, "unchanged")
    assert p["size"] == 0.75 and p["direction"] == "eased" and p["name"] == "itself"
    assert p["victim"] is True and p["listed"] is True


def test_a_moved_person_the_model_reads_the_same():
    b = _scene()
    p = parts4.itself(b, (_one("person_1"), _scene()))
    assert p["size"] == 0.0 and p["direction"] == "unchanged"
    assert all(x["size"] == 0.0 for x in p["lines"])


def test_the_kind_of_at_risk_has_a_direction():
    b = _scene(man_kind="distress")
    p = parts4.itself(b, (_one("person_1"), _scene(man_kind="proximity")))
    assert (_lines(p)["at risk"]["size"], _lines(p)["at risk"]["direction"]) == (0.5, "eased")
    p = parts4.itself(_scene(), (_one("person_1"), _scene(man_kind="distress")))
    assert _lines(p)["at risk"]["direction"] == "escalated"
    # a kind word we do not know: changed, no direction
    p = parts4.itself(_scene(), (_one("person_1"), _scene(man_kind="wobbly")))
    assert _lines(p)["at risk"]["direction"] == "changed"
    # entering the list is an escalation
    p = parts4.itself(_scene(man_at_risk=False), (_one("person_1"), _scene()))
    assert (_lines(p)["at risk"]["size"], _lines(p)["at risk"]["direction"]) == (1.0, "escalated")


def test_a_hazard_dealt_with_reads_its_own_four_lines():
    b = _scene()
    a = _scene(car_state="burnt", man_arrows=False, man_rec=False)
    p = parts4.itself(b, (_one("car_1"), a))
    by = _lines(p)
    assert p["victim"] is False
    assert by["in the threats list"]["size"] == 0.0
    # it threatened the house and the man, now the house only
    assert by["who it threatens"]["size"] == 0.5 and by["who it threatens"]["direction"] == "eased"
    assert by["who it threatens"]["before"] == "house_1, person_1"
    assert by["recommendations that act on it"]["direction"] == "eased"
    assert (by["its state"]["size"], by["its state"]["direction"]) == (1.0, "eased")
    assert p["direction"] == "eased"
    # no longer a threat at all
    p = parts4.itself(b, (_one("car_1"), _scene(car_state="burnt", car_threat=False)))
    assert _lines(p)["in the threats list"]["size"] == 1.0
    assert _lines(p)["in the threats list"]["direction"] == "eased"


def test_a_removed_thing_has_no_such_part():
    b, a = _scene(), _scene(man_listed=False, man_at_risk=False, man_arrows=False, man_rec=False)
    assert parts4.itself(b, (_one("person_1", "removed"), a)) is None
    assert parts4.itself(b, ({"target": "person_1", "signals": SIGNALS}, a)) is None   # an older try
    # and the shift is the average of the other four
    rd = declared4.try_reading(b, (_one("person_1", "removed"), a))
    assert "itself" not in [p["name"] for p in rd["parts"]]
    assert rd["size"] == pytest.approx(sum(p["size"] for p in rd["parts"]) / len(rd["parts"]),
                                       abs=0.002)


def test_dealt_with_but_not_listed_after_is_read_and_marked():
    b = _scene()
    a = _scene(man_listed=False, man_at_risk=False, man_arrows=False, man_rec=False)
    p = parts4.itself(b, (_one("person_1"), a))
    assert p["listed"] is False
    assert "its state" not in _lines(p)                  # no state to read
    assert p["size"] == 1.0 and p["direction"] == "eased"


def test_the_fifth_part_is_in_the_shift_of_a_dealt_with_try():
    b = _scene()
    a = _scene(man_at_risk=False, man_arrows=False, man_rec=False)
    rd = declared4.try_reading(b, (_one("person_1"), a))
    names = [p["name"] for p in rd["parts"]]
    assert names[-1] == "itself" and len(names) == 5
    assert rd["size"] == pytest.approx(sum(p["size"] for p in rd["parts"]) / 5, abs=0.002)
    assert rd["agree"]["of"] == 5
    # it reaches the feature row too
    row = {r["object_id"]: r for r in declared4.compare(b, [(_one("person_1"), a)])["victims"]}
    assert [p["name"] for p in row["person_1"]["parts"]][-1] == "itself"
    assert row["person_1"]["agree"]["of"] == 5


def test_the_state_part_has_a_direction():
    b = _scene()
    assert parts4.state_direction_of(b, _scene(car_state="burnt"), {"car_1", "person_1"}) == "eased"
    assert parts4.state_direction_of(b, _scene(man_state="unconscious"),
                                     {"car_1", "person_1"}) == "escalated"
    # one each way, or a word with no severity: changed
    assert parts4.state_direction_of(b, _scene(car_state="burnt", man_state="unconscious"),
                                     {"car_1", "person_1"}) == "changed"
    assert parts4.state_direction_of(b, _scene(car_state="wobbling"), {"car_1"}) == "changed"
    assert parts4.state_direction_of(b, b, {"car_1"}) == "unchanged"
    rd = declared4.try_reading(b, (_one("car_1", "removed"), _scene(car_state="burnt")))
    assert {p["name"]: p["direction"] for p in rd["parts"]}["state"] == "eased"


@pytest.mark.parametrize("bad", [None, "x", 3, [], {}, {"detected_objects": "zz"},
                                 {"detected_objects": [None, 3], "at_risk": "no",
                                  "threats": 7, "graph_b": [1], "recommendations": "r"},
                                 {"detected_objects": [{"object_id": "person_1"}],
                                  "at_risk": [None, {"kind": 3}, {"object_id": 9}],
                                  "graph_b": {"edges": [None, {"source": 1}]},
                                  "recommendations": [None, {"structured_reasoning": 5}]}])
def test_malformed_runs_never_raise(bad):
    good = _scene()
    for b, a in ((bad, good), (good, bad), (bad, bad)):
        for one in (_one("person_1"), _one("car_1"), None, "x", {"target": 5, "kind": "dealt with"},
                    {"target": "nobody_9", "kind": "dealt with"}):
            got = parts4.itself(b, (one, a))
            assert got is None or 0.0 <= got["size"] <= 1.0
        assert parts4.state_direction_of(b, a, {"car_1"}) in parts4.DIRECTIONS
        assert parts4.state_direction_of(b, a, None) == "unchanged"
    assert parts4.itself(good, None) is None and parts4.itself(good, "pair") is None


# ---- the bar picture on the mechanistic cell ---------------------------------
def test_the_mechanistic_cell_carries_the_bars_of_its_parts():
    import json

    from agentic import ui_intervene as ui_iv
    b = _scene()
    a = _scene(man_at_risk=False, man_arrows=False, man_rec=False)
    row = {r["object_id"]: r for r in declared4.compare(b, [(_one("person_1"), a)])["victims"]}
    cell = ui_iv.mech_cell(row["person_1"])
    data = json.loads(getattr(cell, "data-bars"))
    names = [r["name"] for r in data["rows"]]
    # "all recommendations": the whole list; those that name the thing are a
    # line of their own under "itself", and the two can go opposite ways
    assert names == ["level", "all recommendations", "graph", "state", "person_1 itself"]
    assert all(set(r) >= {"name", "size", "direction"} for r in data["rows"])
    own = data["rows"][-1]
    assert [x["name"] for x in own["lines"]][0] == "at risk"
    assert own["lines"][0]["was"] == "at risk (proximity)" and own["lines"][0]["now"] == "not at risk"
    assert "average of the 5 parts" in data["foot"] and "eased" in data["foot"]
    # the picture replaces the text pop-up (they would cover each other);
    # the text stays on the element for a reader of the page source
    assert not cell.title and "person_1 itself: 0.75 eased" in getattr(cell, "data-parts")
    # a second try's faded cell says so inside the picture
    faded = json.loads(getattr(ui_iv.mech_cell(row["person_1"], faded=True), "data-bars"))
    assert faded["note"] == ui_iv.ONE_TRY_TIP


def test_not_listed_after_is_said_in_the_picture():
    import json

    from agentic import ui_intervene as ui_iv
    b = _scene()
    a = _scene(man_listed=False, man_at_risk=False, man_arrows=False, man_rec=False)
    row = {r["object_id"]: r for r in declared4.compare(b, [(_one("person_1"), a)])["victims"]}
    data = json.loads(ui_iv.bars_of(row["person_1"]))
    assert data["rows"][-1]["note"] == "not listed in the after picture"


def test_parts_without_a_direction_keep_the_text_pop_up():
    from agentic import ui_intervene as ui_iv
    old = {"does": 0.5, "shift": 0.47, "parts": [{"name": "level", "signed": 0.1}, None,
                                                 {"name": "x", "signed": "big"}]}
    cell = ui_iv.mech_cell(old)
    assert cell.title == "level: +0.10" and not hasattr(cell, "data-bars")
    assert ui_iv.bars_of(old) == ""
    for bad in (None, "x", {}, {"parts": "zz"}, {"parts": [None, 3, {"name": "a"}]},
                {"parts": [{"name": "a", "signed": 0.1, "size": "big", "direction": 3}]},
                {"parts": [{"name": "itself", "signed": 0.1, "size": 0.1, "direction": "eased",
                            "lines": [None, "x", {"name": "q"}]}], "agree": "no", "shift": "s"}):
        assert isinstance(ui_iv.bars_of(bad), str)
        ui_iv.mech_cell(bad)
    assert ui_iv.part_label({"name": "itself"}) == "the tried object itself"
    assert ui_iv.part_label({"name": "itself", "target": "car_1"}) == "car_1 itself"
    assert ui_iv.part_label(None) == ""


def test_the_page_has_the_bar_picture_script():
    from agentic import ui
    page = ui.app.index_string
    assert "parts-pop" in page and "data-bars" in page


# ---- the row's direction comes from ALL the parts (2026-11-25) ---------------
def _p(name, size, direction, **more):
    return parts4.part(name, size, direction, **more)


def test_the_direction_of_a_row_is_what_all_its_parts_say():
    """Sunny, 2026-11-25: one part's word (the danger level) beside a number
    made of all of them "will miscommunicate"."""
    # H, the house fire dealt with: the level eased a little, the
    # recommendations and the graph escalated a lot
    house = [_p("level", 0.10, "eased"), _p("recommendations", 0.87, "escalated"),
             _p("graph", 0.50, "escalated"), _p("state", 0.40, "eased"),
             _p("itself", 0.51, "eased", target="house_1")]
    got = parts4.agree(house)
    assert got["word"] == "mixed"                # 3 against 2 by count, the other way by size
    assert got["text"] == "mixed (3 eased, 2 escalated)"
    # most one way, by count and by size; a part that stood still is not counted
    car = [_p("level", 0.0, "eased"), _p("recommendations", 1.0, "eased"),
           _p("graph", 0.75, "escalated"), _p("state", 0.2, "eased"),
           _p("itself", 0.67, "eased", target="car_1")]
    got = parts4.agree(car)
    assert (got["word"], got["text"], got["moved"], got["of"]) == ("eased", "eased (3 of 4)", 4, 5)
    # all that moved went one way: the word alone
    assert parts4.agree([_p("level", 0.2, "eased"), _p("graph", 0.5, "eased"),
                         _p("state", 0.0, "eased")])["text"] == "eased"
    assert parts4.agree([_p("level", 0.2, "escalated")])["text"] == "escalated"
    # one moved with no up or down
    assert parts4.agree([_p("level", 0.2, "eased"), _p("graph", 0.5, "redirected")]
                        )["text"] == "eased (1 of 2)"
    # a tie by count is a split
    assert parts4.agree([_p("level", 0.2, "eased"), _p("graph", 0.5, "escalated")]
                        )["text"] == "mixed (1 eased, 1 escalated)"
    # nothing went up or down
    assert parts4.agree([_p("graph", 0.5, "redirected"), _p("state", 0.3, "changed")]
                        )["text"] == "redirected"
    assert parts4.agree([_p("level", 0.0, "eased")])["text"] == "unchanged"
    for bad in (None, "x", [], [None, 3, {"name": "a"}, {"direction": "banana", "size": 1}]):
        got = parts4.agree(bad)
        assert got["text"] == "" and got["word"] == "unchanged" and got["of"] == 0
    assert parts4.label({"name": "itself"}) == "the tried object itself" and parts4.label(3) == ""


def test_the_row_shows_that_sentence_and_an_older_record_keeps_its_own():
    from agentic import ui_intervene as ui_iv
    mixed = {"direction": "eases", "direction_by": "level", "type": "keeps",
             "agree": {"word": "mixed", "text": "mixed (1 eased, 1 escalated)"}}
    assert ui_iv.words_of(mixed) == "mixed (1 eased, 1 escalated) \u00b7 its recommendations kept"
    assert ui_iv.direction_colour(mixed) == "#b45309"
    cell = ui_iv.way_cell(mixed, {})
    assert cell.style["color"] == "#b45309" and "what ALL the parts" in cell.title
    # the bottom row of the recommendations table reads the same
    rd = {**mixed, "size": 0.47, "counts": {}, "parts": []}
    assert ui_iv.reading_cell(rd)[0].style["color"] == "#b45309"
    # no sentence (before 2026-11-25, or nonsense): the level's word, as before
    for ag in (None, "x", {}, {"text": ""}, {"text": 3}):
        old = {"direction": "eases", "direction_by": "level", "agree": ag}
        assert ui_iv.words_of(old) == "danger level eased"
        assert ui_iv.direction_colour(old) == "#16a34a"
    assert ui_iv.direction_colour(None) == "#64748b"


# ---- a followed recommendation that acted on two things ----------------------
def _two_officers(moved=False):
    objs = [{"object_id": "building_1", "label": "building", "state": "collapsed"},
            {"object_id": "officer_1", "label": "person", "state": "standing"},
            {"object_id": "officer_2", "label": "person", "state": "standing"}]
    edges = [] if moved else [
        {"source": "building_1", "effect": "may_harm", "target": "officer_1"},
        {"source": "building_1", "effect": "may_harm", "target": "officer_2"}]
    recs = [] if moved else [_rec(1, "Move officer_1 and officer_2 back from building_1.",
                                  "building_1", ["officer_1", "officer_2"], state="collapsed")]
    return {"detected_objects": objs, "threats": [{"object_id": "building_1"}],
            "at_risk": [] if moved else [{"object_id": "officer_1", "kind": "proximity"},
                                         {"object_id": "officer_2", "kind": "proximity"}],
            "graph_b": {"nodes": [], "edges": copy.deepcopy(edges)},
            "causal_graph": {"nodes": [], "edges": copy.deepcopy(edges)},
            "recommendations": recs, "disaster_level": 7}


def test_both_things_a_followed_recommendation_moved_count_as_tried():
    """E, 2026-11-25: 'move both officers back' is one try on two people."""
    from agentic import grounding4
    b, a = _two_officers(), _two_officers(moved=True)
    one = {**_one("officer_1"), "also": ["officer_2"],
           "follows": {"rank": 1, "action": "Move officer_1 and officer_2 back from building_1."}}
    tbl = grounding4.table(b, [(one, a)])
    col = tbl["columns"][0]
    assert col["follows"]["rank"] == 1 and col["also"] == ["officer_2"] and col["victim"] is True
    cell = tbl["rows"][0]["cells"][0]
    assert cell["named_removed"] is True and cell["state"] == grounding4.GONE
    # each of the two has its own row in the feature table, with that try's shift
    rows = {r["object_id"]: r for r in declared4.compare(b, [(one, a)])["victims"]}
    assert rows["officer_1"]["shift"] == rows["officer_2"]["shift"] is not None
    # without "also", only the picked one is the tried one
    rows = {r["object_id"]: r for r in declared4.compare(b, [(_one("officer_1"), a)])["victims"]}
    assert rows["officer_1"]["shift"] is not None and rows["officer_2"]["shift"] is None
    # nonsense in the record never breaks either table
    for also in ("officer_2", 3, None, [None, 4, "ghost_9", "officer_1"]):
        odd = {**_one("officer_1"), "also": also, "follows": "zz"}
        grounding4.table(b, [(odd, a)])
        declared4.compare(b, [(odd, a)])
