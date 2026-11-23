"""WHO IS WHO: the pictures of a run and its tries side by side, the same
object in the same colour (agentic/ui_who.py). Hermetic."""
from PIL import Image

from agentic import align4, grounding4 as g, ui_intervene as ui_iv, ui_who
from agentic.test_ui_intervene import _by_class, _flat, _rec


def _scene(tmp_path, name, objs, recs=(), threats=()):
    """A run folder with a picture and one saved shape per object."""
    d = tmp_path / name
    d.mkdir()
    Image.new("RGB", (400, 200), (40, 90, 200)).save(d / "s.png")
    out = []
    for oid, box, family in objs:
        m = Image.new("L", (400, 200), 0)
        m.paste(255, tuple(box))
        m.save(d / f"s__{oid}_mask.png")
        out.append({"object_id": oid, "label": oid.rsplit("_", 1)[0], "family": family,
                    "state": "s", "bbox": list(box), "mask_path": str(d / f"s__{oid}_mask.png")})
    return {"run_dir": str(d), "scene_image": "s.png", "image_size": [400, 200],
            "detected_objects": out, "threats": [{"object_id": t} for t in threats],
            "recommendations": list(recs)}


def _pair(tmp_path):
    before = _scene(tmp_path, "b", [("child_1", (20, 20, 80, 80), "person"),
                                    ("child_2", (300, 20, 360, 80), "person"),
                                    ("pool_1", (0, 100, 400, 200), "infrastructure"),
                                    ("fence_1", (0, 0, 400, 10), "infrastructure")],
                    recs=[_rec(1, "pool_1", ["child_1"]), _rec(2, "pool_1", ["child_2"])])
    # the edited run swapped the names and moved one child; a handrail is new
    raw = _scene(tmp_path, "a", [("child_1", (300, 20, 360, 80), "person"),
                                 ("child_2", (150, 120, 210, 180), "person"),
                                 ("pool_1", (0, 100, 400, 200), "infrastructure"),
                                 ("fence_1", (0, 0, 400, 10), "infrastructure"),
                                 ("handrail_1", (380, 100, 395, 180), "object")])
    # the moved child is wrapped in a white towel, so it no longer looks the same
    for run, boxes in ((before, [((20, 20, 80, 80), (220, 30, 30)),
                                 ((300, 20, 360, 80), (30, 200, 60))]),
                       (raw, [((150, 120, 210, 180), (240, 240, 240)),
                              ((300, 20, 360, 80), (30, 200, 60))])):
        path = run["run_dir"] + "/s.png"
        img = Image.open(path)
        for box, colour in boxes:
            img.paste(colour, box)
        img.save(path)
    return before, align4.aligned(before, raw)


def test_only_what_matters_gets_a_colour_and_the_same_one_in_every_picture(tmp_path):
    before, after = _pair(tmp_path)
    pairs = [({"index": 1, "target": "child_1", "kind": "dealt with"}, after)]
    col = ui_who.colours(before, pairs)
    assert list(col) == ["child_1", "child_2", "pool_1"]          # the fence is no threat, not at risk
    assert len(set(col.values())) == 3
    url_b = ui_who.draw(before, col)
    url_a = ui_who.draw(after, col, after["unsure"], after["tracks"]["new"],
                        ui_who._its_names(after))
    assert url_b.startswith("data:image/jpeg;base64,") and url_a != url_b
    assert ui_who.draw(before, col) is url_b                      # made once, kept
    # the edited run's own names, the other way round
    assert ui_who._its_names(after) == {"child_2": "child_1", "child_1": "child_2"}


def test_the_strip_has_the_original_then_one_picture_per_try_and_says_how_sure(tmp_path):
    before, after = _pair(tmp_path)
    one = {"index": 1, "target": "child_1", "kind": "dealt with"}
    (fold,) = ui_who.component(before, [(one, after)], {"fontSize": "17px"})
    cards = _by_class(fold, "who-card")
    assert len(cards) == 2 and _flat(cards[0]).startswith("before")
    assert _flat(cards[1]).startswith("#1 child_1 dealt with")
    # the moved child is a guess: the fold opens by itself and says so
    assert fold.open is True and _flat(_by_class(fold, "who-head")) == "· 1 not sure"
    assert _flat(_by_class(cards[1], "who-sure")) == "1 not sure"
    assert "child_1" in _flat(_by_class(cards[1], "who-unsure"))
    assert _by_class(cards[1], "who-new") == []           # a new handrail is no threat: left out
    assert "child_1 is child_2" in _flat(_by_class(cards[1], "who-names"))
    assert ui_who.not_sure([(one, after)]) == 1 and ui_who.says(after) == "1 not sure"
    # every match sure: closed, "all sure"
    same = align4.aligned(before, before)
    (fold,) = ui_who.component(before, [(one, same), ({"index": 2, "target": "pool_1"}, same)])
    assert fold.open is False and _flat(_by_class(fold, "who-head")) == "· all sure"
    assert len(_by_class(fold, "who-card")) == 3
    assert _flat(_by_class(fold, "who-card")[2]).startswith("#2 removed pool_1")


def test_no_try_no_picture_or_nonsense_shows_nothing_and_never_breaks(tmp_path):
    before, after = _pair(tmp_path)
    assert ui_who.component(before, []) == [] and ui_who.component(before, None) == []
    assert ui_who.component({"detected_objects": []}, [({}, after)]) == []     # no picture
    for bad in (None, "x", {}, {"run_dir": str(tmp_path / "nope"), "scene_image": "s.png"}):
        assert ui_who.draw(bad, {"a_1": "#e6194b"}) is None
        assert ui_who.important(bad, None) == []
    odd = {**before, "detected_objects": before["detected_objects"] + [
        None, {"object_id": "x_1", "bbox": "zz", "mask_path": "/no/such.png"},
        {"object_id": "y_1", "bbox": [1, 2, 3, "q"]}]}
    assert ui_who.draw(odd, {"x_1": "#e6194b", "y_1": "#3cb44b", "child_1": "#4363d8"}) is not None
    # a try whose edited run has no picture still gets a card
    gone = {**after, "run_dir": str(tmp_path / "nope")}
    (fold,) = ui_who.component(before, [({"index": 1, "target": "child_1"}, gone)])
    assert "picture not found" in _flat(_by_class(fold, "who-card")[1])
    assert ui_who.says(None) == "all sure" and ui_who.not_sure([(None, None)]) == 0


def test_the_table_marks_the_recommendations_that_lean_on_a_guess(tmp_path):
    before, after = _pair(tmp_path)
    after = {**after, "recommendations": [_rec(1, "pool_1", ["child_2"])]}
    t = g.table(before, [({"index": 1, "target": "child_1", "kind": "dealt with"}, after)])
    col = t["columns"][0]
    assert col["unsure"] == ["child_1"] and col["tracked"] is True
    by = {r["rank"]: r["cells"][0] for r in t["rows"]}
    assert by[1]["unsure"] == ["child_1"] and by[2]["unsure"] == []
    comp = ui_iv.table_component(t)
    assert _flat(_by_class(comp, "iv-who")) == "#1 child_1: who is who: 1 not sure"
    marks = _by_class(comp, "iv-unsure")
    assert len(marks) == 1 and _flat(marks[0]) == "not sure which is child_1"
    # a run that was never tracked (an older caller) says nothing
    t = g.table(before, [({"index": 1, "target": "child_1"},
                          {k: v for k, v in after.items() if k not in ("tracks", "unsure")})])
    assert t["columns"][0]["tracked"] is False
    assert _by_class(ui_iv.table_component(t), "iv-who") == []


def test_only_threats_and_the_at_risk_are_shown(tmp_path):
    """Sunny, 2026-11-20: "for picture... only show threats and risks/victims"."""
    objs = [("house_1", (0, 0, 100, 100), "structure"), ("car_1", (120, 0, 200, 60), "vehicle"),
            ("man_1", (220, 0, 260, 100), "person"), ("tree_1", (300, 0, 340, 100), "vegetation"),
            ("house_2", (350, 0, 399, 100), "structure")]
    # the recommendation also protects a tree and a second house: not shown
    b = _scene(tmp_path, "b", objs, threats=["house_1", "car_1"],
               recs=[_rec(1, "house_1", ["man_1", "tree_1", "house_2"])])
    b["at_risk"] = [{"object_id": "man_1", "kind": "proximity"}, None, "x"]
    assert ui_who.important(b, []) == ["house_1", "car_1", "man_1"]
    # what a try was aimed at is always shown
    assert ui_who.important(b, [({"target": "tree_1"}, {})]) == ["house_1", "car_1", "man_1",
                                                                  "tree_1"]
    # a run that carries no at-risk list at all: the things its arrows end at
    old = {k: v for k, v in b.items() if k != "at_risk"}
    assert set(ui_who.important(old, [])) == {"house_1", "car_1", "man_1", "tree_1", "house_2"}
    # a new object is shown only when the edited run calls it a threat or at risk
    after = _scene(tmp_path, "a", objs + [("fire_9", (0, 150, 60, 199), "hazard_media"),
                                          ("pole_1", (380, 150, 399, 199), "object")],
                   threats=["house_1", "car_1", "fire_9"])
    after["at_risk"] = b["at_risk"]
    al = align4.aligned(b, after)
    (fold,) = ui_who.component(b, [({"index": 1, "target": "car_1"}, al)])
    assert _flat(_by_class(fold, "who-new")) == "new: fire_9"
    assert "only the threats and those at risk are shown" in _flat(_by_class(fold, "who-key"))
