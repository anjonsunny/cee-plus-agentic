"""The picture beside the NEW TRY form: every thing that can be picked,
boxed; a click picks it (Sunny, 2026-11-23). Hermetic."""
from PIL import Image

from agentic import ui, ui_intervene as ui_iv
from agentic.test_ui_intervene import _by_class, _flat
from agentic.test_ui_views import _find


def _info(tmp_path, same=None):
    img = tmp_path / "s.png"
    Image.new("RGB", (400, 200), (40, 90, 200)).save(img)
    objs = [("house_1", [0, 0, 200, 150]), ("person_1", [80, 90, 100, 120]),
            ("car_1", [220, 100, 320, 160]), ("fire_1", [10, 10, 60, 60]),
            ("tree_1", [350, 0, 399, 150])]
    result = {"image_size": [400, 200], "same_hazards": same or [],
              "detected_objects": [{"object_id": o, "label": o.rsplit("_", 1)[0], "bbox": b}
                                   for o, b in objs]}
    cands = [{"object_id": "house_1", "state": "burning"},
             {"object_id": "fire_1", "state": "spreading"},
             {"object_id": "car_1", "state": "burning"},
             {"object_id": "person_1", "state": "standing", "variable_role": "victim"},
             {"object_id": "ghost_9", "state": "x"}, None]
    return {"image": str(img), "result": result, "candidates": cands, "picked_by": {}}


def test_every_thing_that_can_be_picked_is_boxed_and_the_pick_is_thick(tmp_path):
    info = _info(tmp_path)
    (wrap,) = ui_iv.picker_component(info, "car_1", lambda oid: {"type": "iv-pick", "oid": oid})
    assert wrap.children[0].src.startswith("data:image/jpeg;base64,")
    boxes = _by_class(wrap, "iv-pick-box")
    by = {getattr(b, "data-pick"): b for b in boxes}
    assert set(by) == {"house_1", "fire_1", "car_1", "person_1"}     # not the tree, not a ghost
    assert "picked" in by["car_1"].className and "picked" not in by["house_1"].className
    assert by["house_1"].style["borderColor"] == "#ef4444"           # a hazard
    assert by["person_1"].style["borderColor"] == "#f97316"          # a victim
    assert by["car_1"].style == {"left": "55.00%", "top": "50.00%", "width": "25.00%",
                                 "height": "30.00%", "borderColor": "#ef4444"}
    assert by["car_1"].id == {"type": "iv-pick", "oid": "car_1"}
    # a small box inside a big one is drawn after it, so it can be clicked
    order = [getattr(b, "data-pick") for b in boxes]
    assert order.index("person_1") > order.index("house_1")
    # the picture is made once
    assert ui_iv.small_image(info["image"]) is ui_iv.small_image(info["image"])


def test_a_same_hazard_pair_is_picked_and_drawn_together(tmp_path):
    info = _info(tmp_path, same=[["fire_1", "house_1"]])
    assert ui_iv.pick_unit(info, "fire_1") == ["fire_1", "house_1"]
    (wrap,) = ui_iv.picker_component(info, "fire_1")
    thick = {getattr(b, "data-pick") for b in _by_class(wrap, "iv-pick-box")
             if "picked" in b.className}
    assert thick == {"fire_1", "house_1"}
    # a click on either member picks the pair's one option
    opts = ui_iv.target_options(info)
    values = [o["value"] for o in opts]
    assert ("house_1" in values) != ("fire_1" in values)             # one option for the two
    pair = "house_1" if "house_1" in values else "fire_1"
    assert ui_iv.option_for(info, opts, "fire_1") == pair
    assert ui_iv.option_for(info, opts, "house_1") == pair
    assert ui_iv.option_for(info, opts, "car_1") == "car_1"
    assert ui_iv.option_for(info, opts, "tree_1") == ""              # cannot be picked


def test_no_picture_no_size_or_nonsense_shows_nothing(tmp_path):
    info = _info(tmp_path)
    assert ui_iv.picker_component(None, "car_1") == []
    assert ui_iv.picker_component({**info, "image": ""}, "car_1") == []
    assert ui_iv.picker_component({**info, "image": str(tmp_path / "nope.png")}, "car_1") == []
    for bad in (None, "x", [0, 5], [400]):
        assert ui_iv.picker_component(
            {**info, "result": {**info["result"], "image_size": bad}}, "car_1") == []
    odd = {**info, "result": {**info["result"], "detected_objects": [
        None, {"object_id": "car_1", "bbox": "zz"}, {"object_id": "house_1", "bbox": [5, 5, 1, 1]},
        {"object_id": "fire_1", "bbox": [1, 2, 3, True]}]}}
    (wrap,) = ui_iv.picker_component(odd, None)
    assert _by_class(wrap, "iv-pick-box") == []                       # the picture, no boxes
    assert ui_iv.pick_unit(None, None) == [] and ui_iv.option_for(None, None, None) == ""
    assert ui_iv.small_image(None) is None


def test_the_three_rows_are_on_one_side_and_the_picture_across_them():
    top, _ = _find(ui.app.layout, "iv-picker")
    rows, anc = _find(ui.app.layout, "iv-target")
    form_top = [a for a in anc if getattr(a, "className", "") == "iv-form-top"]
    assert form_top, "WHAT sits inside the two-column row"
    left, right = form_top[0].children
    assert left.className == "iv-form-rows" and right.id == "iv-picker"
    for wid in ("iv-target", "iv-kind", "iv-modality"):
        assert _find(left, wid) is not None                           # all three rows on the left
    assert _find(left, "iv-picker") is None
    # a click picks; the dropdown follows
    from unittest.mock import patch
    opts = [{"label": " car_1", "value": "car_1"}, {"label": " house_1", "value": "house_1"}]
    Trig = type("T", (), {"triggered_id": {"type": "iv-pick", "oid": "house_1"}})()
    with patch.object(ui, "ctx", Trig), patch.object(ui, "_iv_info", lambda r: None):
        assert ui.iv_pick([0, 1], opts, "base") == "house_1"
        assert ui.iv_pick([0, 0], opts, "base") is ui.dash.no_update    # nothing clicked
    Trig2 = type("T", (), {"triggered_id": {"type": "iv-pick", "oid": "tree_1"}})()
    with patch.object(ui, "ctx", Trig2), patch.object(ui, "_iv_info", lambda r: None):
        assert ui.iv_pick([1], opts, "base") is ui.dash.no_update       # not an option
    with patch.object(ui, "_iv_info", lambda r: None):
        assert ui.iv_picker("car_1", opts, "nope") == []


def test_the_pick_has_its_own_colour_and_the_upload_sits_under_the_caption():
    """Sunny, 2026-11-23: "thicker bbox not working. Make the color of the
    picked one different", and "make the upload edited image button below
    caption... so the picker picture can be bigger"."""
    page = ui.app.index_string
    assert ".iv-pick-box.picked { border-width:4px; border-color:#facc15 !important;" in page
    assert ".iv-pick-box.picked span { background:#facc15 !important;" in page
    left = [a for a in _find(ui.app.layout, "iv-target")[1]
            if getattr(a, "className", "") == "iv-form-rows"][0]
    ids = [getattr(k, "id", None) for k in left.children]
    assert ids.index("iv-caption-wrap") + 1 == ids.index("iv-upload-wrap")   # stacked, upload below
    assert _find(left, "iv-instructions") is not None
    picker, _ = _find(ui.app.layout, "iv-picker")
    assert picker.style["flex"] == "0 0 46%"                                # a large picture
    # the channel switch still hides the box that is not needed
    cap, img = ui.iv_show_channel("caption")
    assert cap["display"] == "block" and img["display"] == "none"
    cap, img = ui.iv_show_channel("image")
    assert cap["display"] == "none" and img["display"] == "block"
