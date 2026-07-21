"""Hermetic tests for the Intervention tab UI panels in main.py (built incrementally
by the agentic UI workflow). These assert the rendered Dash component tree carries the
right content for the pipeline's edge cases — the visual layout is verified separately
via the screenshot-in-loop, but the data-wiring is locked here."""
import pytest


def _flatten_text(node) -> str:
    """Recursively collect all string content from a Dash component tree."""
    acc: list[str] = []

    def walk(n):
        if n is None:
            return
        if isinstance(n, str):
            acc.append(n)
            return
        if isinstance(n, (list, tuple)):
            for c in n:
                walk(c)
            return
        walk(getattr(n, "children", None))

    walk(node)
    # Collapse whitespace runs: prose is now split across text + hover-bbox chips (one segment
    # per object_id), and " ".join would insert an extra space at each segment boundary. The DOM
    # renders adjacent inline spans with collapsed whitespace, so mirror that here.
    import re
    return re.sub(r"\s+", " ", " ".join(acc))


@pytest.mark.blocking
def test_candidates_panel_all_three_picks_agree(main_module):
    """Redesign: when the algorithm pick, the VLM pick, and the GT pick all name the
    same hazard, the panel reads 'All three agree'. Every distinct candidate hazard
    renders as a chip (deduped by object_id), and no internal jargon leaks (no
    'control', no 'should-be-core', no 'A#1')."""
    m = main_module
    core = {"object_id": "building_1", "state": "collapsed", "label": "building",
            "hazard_class": "discrete_source", "sources": ["A", "B", "GT"],
            "ranks": {"A": 1, "B": 1, "GT": 1}, "is_should_be_core": True}
    other = {"object_id": "debris_1", "state": "exposed", "label": "debris",
             "sources": ["B"], "ranks": {"B": 2}, "is_should_be_core": False}
    candidates = {"candidates": [core, other], "should_be_core": core,
                  "declared_core_a": core, "declared_core_b": core, "control": other,
                  "gt_core_unobserved": None}
    detected = [
        {"object_id": "building_1", "label": "building", "state": "collapsed", "bbox": [0, 0, 10, 10]},
        {"object_id": "debris_1", "label": "debris", "state": "exposed", "bbox": [5, 5, 15, 15]},
    ]
    framework_picks = [{"rank": 1, "threat": "building_1", "state": "collapsed"}]
    vlm_pick = {"threat": "building_1", "state": "collapsed", "reason": "most severe"}
    panel = m.make_candidates_panel(candidates, {"score": 0.85, "level": "moderate"},
                                    detected, None, framework_picks, vlm_pick)
    txt = _flatten_text(panel)
    assert "all three agree" in txt.lower()           # agreement line
    assert "moderate" in txt                           # trust context present
    assert "building" in txt and "debris" in txt       # both candidate hazards shown as chips
    # no internal experiment jargon in the user-facing panel
    low = txt.lower()
    assert "should-be-core" not in low and "control" not in low and "a#1" not in low


@pytest.mark.blocking
def test_intervention_candidates_callback_placeholder_is_safe(main_module):
    """Part 1 wiring: the live callback (render_intervention_candidates) degrades to a
    safe Div on the placeholder result (no crash, no exception escaping the try/except).
    The populated path is covered by the harness/screenshot loop against saved runs."""
    m = main_module
    out = m.render_intervention_candidates(m.PLACEHOLDER_RESULT, None)
    assert out is not None and out.__class__.__name__ == "Div"


@pytest.mark.blocking
def test_candidates_panel_gt_core_unobserved(main_module):
    """Redesign edge: should_be_core None + gt_core_unobserved set -> the GT pick reads
    'the model never perceived the ground-truth core: <label>'. Low trust reads
    'provisional'. Still no internal jargon (no 'should-be-core', no 'control')."""
    m = main_module
    declared_b = {"object_id": "person_1", "state": "drowning", "label": "person",
                  "sources": ["B"], "ranks": {"B": 1}, "is_should_be_core": False}
    candidates = {"candidates": [declared_b], "should_be_core": None,
                  "declared_core_a": None, "declared_core_b": declared_b, "control": None,
                  "gt_core_unobserved": {"object_id": "water_1", "state": "engulfing", "label": "water"}}
    detected = [{"object_id": "person_1", "label": "person", "state": "drowning", "bbox": [1, 2, 3, 4]}]
    vlm_pick = {"threat": "person_1", "state": "drowning", "reason": "immediate"}
    txt = _flatten_text(m.make_candidates_panel(
        candidates, {"score": 0.05, "level": "low"}, detected, None, [], vlm_pick))
    low = txt.lower()
    assert "never perceived the ground-truth core" in low and "water" in low
    assert "provisional" in low                        # low-trust qualifier
    assert "should-be-core" not in low and "control" not in low


def _find_by_id(node, target_id):
    if getattr(node, "id", None) == target_id:
        return node
    children = getattr(node, "children", None)
    if children is None:
        return None
    for c in (children if isinstance(children, (list, tuple)) else [children]):
        r = _find_by_id(c, target_id)
        if r is not None:
            return r
    return None


@pytest.mark.blocking
def test_setup_panel_controls_and_default_gt(main_module):
    """Setup panel: the three controls exist, the pick defaults to the GT hazard, modality
    defaults to caption, the original caption shows, and Apply is enabled (it runs one
    counterfactual via the apply_intervention callback)."""
    m = main_module
    cand = {"should_be_core": {"object_id": "building_1", "state": "collapsed", "label": "building"},
            "candidates": [
                {"object_id": "building_1", "label": "building", "state": "collapsed",
                 "ranks": {"GT": 1, "A": 1}, "is_should_be_core": True},
                {"object_id": "fire_1", "label": "fire", "state": "burning",
                 "ranks": {"GT": 2, "A": 2, "B": 1}, "is_should_be_core": False}]}
    det = [{"object_id": "building_1", "label": "building", "state": "collapsed", "bbox": [0, 0, 9, 9]},
           {"object_id": "fire_1", "label": "fire", "state": "burning", "bbox": [4, 4, 6, 6]},
           {"object_id": "person_1", "label": "person", "state": "trapped", "bbox": [1, 1, 3, 3]}]
    fw = [{"rank": 1, "threat": "building_1", "state": "collapsed"}]
    vlm = {"threat": "fire_1", "state": "burning"}
    panel = m.make_intervention_setup_panel(cand, det, fw, vlm, "A collapsed building with a trapped person.")

    assert _find_by_id(panel, "intervention-pick").value == "building_1"      # default = GT core
    assert _find_by_id(panel, "intervention-mode").value == "retrospective"
    assert _find_by_id(panel, "intervention-modality").value == "caption"
    assert getattr(_find_by_id(panel, "intervention-apply"), "disabled", None) in (None, False)  # Apply enabled

    # EVERY hazard candidate is offered (not just the top pick), so non-top hazards are reachable
    pick_opts = _find_by_id(panel, "intervention-pick").options
    pick_vals = {o["value"] for o in pick_opts}
    assert "building_1" in pick_vals and "fire_1" in pick_vals               # all hazards offered
    # rank badges are in the option labels; next-to-test guidance is in the card body
    opt_labels = " ".join(str(o.get("label", "")) for o in pick_opts).lower()
    assert "gt #1" in opt_labels and "b #1" in opt_labels                    # per-source rank badges
    assert "★" in opt_labels or "gt #1" in opt_labels                        # GT core marked
    assert "suggested next" in _flatten_text(panel).lower()                  # next-to-test guidance
    mode_vals = {o["value"] for o in _find_by_id(panel, "intervention-mode").options}
    assert mode_vals == {"retrospective", "prospective"}
    modality_vals = {o["value"] for o in _find_by_id(panel, "intervention-modality").options}
    assert modality_vals == {"caption", "image", "both"}

    txt = _flatten_text(panel).lower()
    assert "collapsed building with a trapped person" in txt                  # original caption shown

    # the GPT prompts are present, mode-aware, and carry the "keep everything else" clause
    img_prompt = _find_by_id(panel, "intervention-image-prompt").children.lower()
    cap_instr = _find_by_id(panel, "intervention-caption-instruction").children.lower()
    assert "building" in img_prompt and ("keep" in img_prompt and "same" in img_prompt)
    assert "never" in img_prompt                                              # default = retrospective
    assert "building" in cap_instr

    # no internal jargon
    assert "should-be-core" not in txt and "u_leaked" not in txt


@pytest.mark.blocking
def test_setup_panel_offers_all_hazards_with_tested_badges_and_next(main_module):
    """The picker exposes EVERY hazard candidate (so a non-top hazard is reachable), badges
    each with its per-source rank, marks already-suppressed hazards ✓ tested, and the
    next-to-test guidance names the top-ranked UNtested hazard (fills the operative axis)."""
    m = main_module
    cand = {"should_be_core": {"object_id": "tanker_truck_1", "state": "leaking"},
            "candidates": [
                {"object_id": "tanker_truck_1", "label": "tanker_truck", "state": "leaking",
                 "ranks": {"GT": 1, "A": 1, "B": 2}, "is_should_be_core": True},
                {"object_id": "fire_1", "label": "fire", "state": "burning",
                 "ranks": {"GT": 2, "A": 2, "B": 1}, "is_should_be_core": False}]}
    det = [{"object_id": "tanker_truck_1", "label": "tanker_truck", "state": "leaking"},
           {"object_id": "fire_1", "label": "fire", "state": "burning"},
           {"object_id": "person_1", "label": "person", "state": "standing"}]

    # already suppressed the tanker -> it is ✓ tested and the suggestion moves to the fire
    panel = m.make_intervention_setup_panel(cand, det, [], {}, "cap", tested_oids={"tanker_truck_1"})
    opts = _find_by_id(panel, "intervention-pick").options
    labels = {o["value"]: str(o["label"]) for o in opts}
    assert set(labels) == {"tanker_truck_1", "fire_1"}                       # both hazards pickable
    assert "✓ tested" in labels["tanker_truck_1"]                            # tanker marked done
    assert "✓ tested" not in labels["fire_1"]                               # fire not yet
    assert "★" in labels["tanker_truck_1"]                                   # GT core starred
    body = _flatten_text(panel).lower()
    assert "suggested next: fire" in body and "1/2 done" in body            # next-to-test + progress
    # the pick ADVANCES to the next untested hazard (fire) so a sweep doesn't snap back to GT
    assert _find_by_id(panel, "intervention-pick").value == "fire_1"

    # nothing tested yet -> suggestion is the GT core (top of the list) and the pick defaults there
    panel0 = m.make_intervention_setup_panel(cand, det, [], {}, "cap")
    assert "suggested next: tanker" in _flatten_text(panel0).lower()
    assert _find_by_id(panel0, "intervention-pick").value == "tanker_truck_1"
    # all tested -> a completion note, not a suggestion
    paneln = m.make_intervention_setup_panel(cand, det, [], {}, "cap",
                                             tested_oids={"tanker_truck_1", "fire_1"})
    assert "all hazards suppressed" in _flatten_text(paneln).lower()


@pytest.mark.blocking
def test_setup_panel_picker_has_hover_bbox_thumbnails(main_module):
    """Each picker option renders a hover-bbox pill (cropped to the hazard's bbox) so
    same-label hazards are distinguishable — push_02 has several `house (burning)` that
    read identically as plain text. Falls back to a plain pill when no scene image/bbox
    is available; the radio value stays the object_id either way (callbacks unaffected)."""
    m = main_module

    def _has_class(node, cls):
        acc = []
        def walk(n):
            if n is None or isinstance(n, str):
                return
            if isinstance(n, (list, tuple)):
                for c in n:
                    walk(c)
                return
            if cls in (getattr(n, "className", "") or ""):
                acc.append(n)
            walk(getattr(n, "children", None))
        walk(node)
        return bool(acc)

    # a real (tiny) PNG data url so make_single_object_preview can crop
    import base64, io
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (400, 300), (120, 120, 120)).save(buf, format="PNG")
    img_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    cand = {"should_be_core": {"object_id": "house_1"},
            "candidates": [
                {"object_id": "house_1", "label": "house", "state": "burning",
                 "ranks": {"GT": 1, "A": 1, "B": 1}, "is_gt_core": True},
                {"object_id": "house_2", "label": "house", "state": "burning",
                 "ranks": {"A": 3, "B": 2}, "is_gt_core": False},
                {"object_id": "car_1", "label": "car", "state": "burning",
                 "ranks": {"GT": 2, "A": 2, "B": 4}, "is_gt_core": True}]}
    det = [{"object_id": "house_1", "label": "house", "state": "burning", "bbox": [10, 10, 120, 120]},
           {"object_id": "house_2", "label": "house", "state": "burning", "bbox": [150, 20, 260, 140]},
           {"object_id": "car_1", "label": "car", "state": "burning", "bbox": [280, 60, 360, 160]}]

    panel = m.make_intervention_setup_panel(cand, det, [], {}, "cap", image_contents=img_url)
    opts = _find_by_id(panel, "intervention-pick").options
    assert {o["value"] for o in opts} == {"house_1", "house_2", "car_1"}       # values still oids
    # EVERY option carries a hover-bbox pill so the two houses are told apart on hover
    assert all(_has_class(o["label"], "hazard-pill-wrap") for o in opts)
    # the ✓/★/rank tail still renders inside the component label
    joined = " ".join(_flatten_text(o["label"]) for o in opts).lower()
    assert "gt #1" in joined and "b #2" in joined

    # no scene image -> graceful fallback to a plain pill (no hover crop)
    panel2 = m.make_intervention_setup_panel(cand, det, [], {}, "cap", image_contents=None)
    opts2 = _find_by_id(panel2, "intervention-pick").options
    assert not any(_has_class(o["label"], "hazard-pill-wrap") for o in opts2)
    assert {o["value"] for o in opts2} == {"house_1", "house_2", "car_1"}


@pytest.mark.blocking
def test_victim_edit_texts_rescue_rather_than_remove(main_module):
    """A VICTIM's do() is target_mitigation: the victim moves to safety while the hazard STAYS
    and any OTHER victim stays in distress — the surgical counterfactual that works on distress
    scenes where the source is not separable (`engulfing` is circular). It must never read as a
    removal of the victim, and must not touch the hazard."""
    m = main_module
    assert m._variable_role_for("drowning") == "victim"
    assert m._variable_role_for("unconscious") == "victim"
    assert m._variable_role_for("burning") == "hazard"      # a hazard-bearing state
    assert m._variable_role_for("") == "hazard"             # no state -> default

    cap, img = m.build_intervention_edit_texts(
        "child", "drowning", "retrospective", ["water", "chair"], variable_role="victim")
    low = (cap + " " + img).lower()
    assert "safely clear" in low and "no longer drowning" in low       # rescued, not deleted
    assert "hazard is completely untouched" in low or "hazard itself is unchanged" in low
    assert "other victim must stay in distress" in low                 # surgical: one victim only
    assert "keep every other detail identical" in low                  # holds U
    # never phrased as removing the victim
    assert "remove the child" not in low and "never present" not in low

    # contrast: the HAZARD template is untouched by the new branch
    cap_h, _img_h = m.build_intervention_edit_texts(
        "water", "engulfing", "retrospective", ["child"], variable_role="hazard")
    assert "never present" in cap_h.lower()                            # fluid -> removed


@pytest.mark.blocking
def test_picker_tags_victims_and_hazards_distinctly(main_module):
    """The picker shows WHICH END of the quad each variable is, so the user can see that
    picking a victim rescues it while picking a hazard removes it. A victim renders as an
    'affected' (blue) chip, never the red threat pill."""
    m = main_module
    cand = {"should_be_core": {"object_id": "water_1"},
            "candidates": [
                {"object_id": "water_1", "label": "water", "state": "engulfing",
                 "ranks": {"GT": 1}, "is_gt_core": True, "variable_role": "hazard"},
                {"object_id": "child_1", "label": "child", "state": "drowning",
                 "ranks": {"GT": 2}, "is_gt_core": True, "variable_role": "victim"}]}
    det = [{"object_id": "water_1", "label": "water", "state": "engulfing"},
           {"object_id": "child_1", "label": "child", "state": "drowning"}]
    panel = m.make_intervention_setup_panel(cand, det, [], {}, "cap")
    opts = _find_by_id(panel, "intervention-pick").options
    labels = {o["value"]: _flatten_text(o["label"]).lower() for o in opts}
    assert "victim · rescue" in labels["child_1"]
    assert "hazard · remove" in labels["water_1"]
    assert {"water_1", "child_1"} == set(labels)            # both ends are pickable


@pytest.mark.blocking
def test_edit_texts_mode_aware(main_module):
    """The template distinguishes the two modes and always holds U (keep-everything clause)."""
    m = main_module
    cap_r, img_r = m.build_intervention_edit_texts("fire", "burning", "retrospective", ["house", "car"])
    cap_p, img_p = m.build_intervention_edit_texts("fire", "burning", "prospective", ["house", "car"])
    assert "never" in img_r.lower()                                           # clean removal
    assert "aftermath" in img_p.lower() or "neutralized" in img_p.lower()     # artifacts remain
    for txt in (img_r, img_p):
        assert "house" in txt and "car" in txt                               # other objects preserved
        assert "exactly the same" in txt.lower()                              # U-hold clause


@pytest.mark.blocking
def test_setup_panel_empty_state(main_module):
    """No picks (placeholder / no GT, no framework, no VLM) -> safe empty state, no crash."""
    m = main_module
    out = m.make_intervention_setup_panel({}, [], [], {}, "")
    assert out.className == "empty-state"


@pytest.mark.blocking
def test_result_panel_before_after_diff(main_module):
    """The result panel shows the verdict, the Fair-test, and each of the three shifts as a
    concrete BEFORE→AFTER diff (hazard level number, entity states, arrows, rec text)."""
    m = main_module
    res = {"verdict": {"cell": "grounded", "explanation": "Moved when suppressed."},
           "signals": {"hazard_shift": 0.8, "graph_shift": 1.0, "recommendation_shift": 1.0,
                       "content_shift": 0.93, "node_shift": 0.5, "edge_shift": 1.0, "hazard_level_delta": -8},
           "u_check": {"leaked": False, "state_stability": 1.0, "topology_stability": 1.0}}
    baseline = {"hazard_level": 8,
                "graph_a": {"nodes": [{"id": "house_1", "label": "house", "state": "burning"},
                                      {"id": "person_1", "label": "person", "state": "standing"}],
                            "edges": [{"source": "house_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}]},
                "recommendations": [{"rank": 1, "action": "Evacuate person_1 from the fire",
                                     "structured_reasoning": {"threat": "house_1", "state": "burning",
                                                              "effect": "may_harm", "affected_objects": ["person_1"]}}]}
    post = {"disaster_level": 0,
            "causal_graph": {"nodes": [{"id": "house_1", "label": "house", "state": "intact"},
                                       {"id": "person_1", "label": "person", "state": "standing"}], "edges": []},
            "recommendations": [{"rank": 1, "action": "No action needed",
                                 "structured_reasoning": {"threat": "", "state": "", "effect": "", "affected_objects": []}}]}
    txt = _flatten_text(m.make_intervention_result_panel(res, baseline, post, mode="retrospective",
                                                         modality="caption", target_oid="house_1"))
    low = txt.lower()
    assert "grounded" in low                                               # verdict, no Fair-test card
    assert "fair-test" not in low                                          # never surfaced on a valid run
    # hazard before -> after, entity state change, arrow removed, rec change all rendered concretely
    assert "house_1: burning" in txt and "house_1: intact" in txt          # entity state before/after
    assert "house_1 →may_harm→ person_1" in txt                            # the arrow (before only)
    assert "[burning]" in txt                                              # arrows carry via_state
    assert "Evacuate person_1 from the fire" in txt and "No action needed" in txt
    assert "before" in low and "after" in low                              # both columns present
    # badge computed from the actual graphs: both entities present (0.00), the one arrow gone (1.00)
    assert "entities 0.00 · arrows 1.00" in txt

    # Input problem (nothing edited): a plain notice, NOT a verdict/Fair-test card.
    leaked = {"verdict": {"cell": "u_leaked"}, "signals": {},
              "u_check": {"leaked": True, "applied": False}}
    ltxt = _flatten_text(m.make_intervention_result_panel(leaked, {}, {}, mode="retrospective", modality="image")).lower()
    assert "no intervention to measure" in ltxt and "fair-test" not in ltxt
    assert "verdict" not in ltxt                                            # no verdict card on an input problem

    assert m.make_intervention_result_panel(None).className == "empty-state"


@pytest.mark.blocking
def test_modality_toggle_dims_the_unused_card(main_module):
    """The What-to-edit choice grays out the card that is not part of the modality."""
    m = main_module
    cap, img = m.toggle_intervention_modality_cards("caption")
    assert cap == {} and img.get("opacity") == 0.4
    cap, img = m.toggle_intervention_modality_cards("both")
    assert cap == {} and img == {}


@pytest.mark.blocking
def test_apply_intervention_runs_and_renders(main_module, monkeypatch):
    """The Apply callback builds the post from the edited caption, runs the measurement
    stack (query_qwen mocked -> no Ollama), and renders the result panel. Locks the wiring:
    modality input selection + run_control=False single-arm + panel render."""
    m = main_module
    # isolate the callback WIRING from normalize_result's hazard re-derivation (tested
    # elsewhere): stub it to identity so enumerate sees our controlled baseline.
    monkeypatch.setattr(m, "normalize_result", lambda d, ic=None: d)
    data = {
        "caption": "A tanker is leaking near a person.",
        "disaster_level": 8,
        "detected_objects": [{"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
                             {"object_id": "person_1", "label": "person", "state": "standing"}],
        "causal_graph": {"nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                                   {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                         "edges": [{"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"}],
                         "intervention_candidates": [{"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
        "recommendations": [{"rank": 1, "action": "Contain the leak",
                             "structured_reasoning": {"threat": "tanker_1", "state": "leaking",
                                                      "effect": "may_harm", "affected_objects": ["person_1"]}}],
    }
    # canned post: hazard gone, rec re-routed -> real shift
    post = {
        "detected_objects": [{"object_id": "tanker_1", "label": "tanker", "state": "contained"},
                             {"object_id": "person_1", "label": "person", "state": "standing"}],
        "causal_graph": {"nodes": [{"id": "tanker_1", "label": "tanker", "state": "contained", "hazardous": False},
                                   {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                         "edges": []},
        "recommendations": [{"rank": 1, "action": "Monitor",
                             "structured_reasoning": {"threat": "", "state": "", "effect": "", "affected_objects": []}}],
        "disaster_level": 2,
    }
    monkeypatch.setattr(m, "query_qwen", lambda *a, **k: post)
    out, _hist, _status = m.apply_intervention(1, data, None, "tanker_1",
                                               "retrospective", "caption", "A tanker near a person.",
                                               None, None, "edge", "above_mean")
    assert out.__class__.__name__ == "Div"
    txt = _flatten_text(out).lower()
    assert "verdict" in txt and "hazard level" in txt and "recommendations" in txt
    assert "fair-test" not in txt                                          # not surfaced on a valid run
    assert "nothing to suppress" not in txt   # a real candidate ran, not the empty-scene path


@pytest.mark.blocking
def test_apply_intervention_requires_edited_input(main_module):
    """No edited caption/image for the chosen modality -> safe empty-state, no Qwen call."""
    m = main_module
    data = {"caption": "x", "disaster_level": 5,
            "detected_objects": [{"object_id": "a_1", "label": "car", "state": "burning"}],
            "causal_graph": {"nodes": [{"id": "a_1", "hazardous": True, "state": "burning"}], "edges": []},
            "recommendations": []}
    out, _, _ = m.apply_intervention(1, data, "data:image/jpeg;base64,xx", "a_1",
                                     "retrospective", "caption", "", None, None,
                                     "edge", "above_mean")   # empty edited caption
    assert out.className == "empty-state"


@pytest.mark.blocking
def test_apply_intervention_both_modality_swaps_both_channels(main_module, monkeypatch):
    """modality='both' is a JOINT suppression: the post is analysed on the edited caption
    AND the edited image together. Requires both inputs."""
    m = main_module
    monkeypatch.setattr(m, "normalize_result", lambda d, ic=None: d)
    data = {"caption": "orig caption", "disaster_level": 8,
            "detected_objects": [{"object_id": "house_1", "label": "house", "state": "burning"},
                                 {"object_id": "person_1", "label": "person", "state": "standing"}],
            "causal_graph": {"nodes": [{"id": "house_1", "label": "house", "state": "burning", "hazardous": True},
                                       {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                             "edges": [{"source": "house_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}],
                             "intervention_candidates": [{"threat": "house_1", "state": "burning", "outgoing_edge_count": 1}]},
            "recommendations": [{"rank": 1, "action": "Evacuate",
                                 "structured_reasoning": {"threat": "house_1", "state": "burning",
                                                          "effect": "may_harm", "affected_objects": ["person_1"]}}]}
    seen = {}

    def _capture(prompt, caption, image, allow_inferred=False):
        seen["caption"], seen["image"] = caption, image
        return {"detected_objects": data["detected_objects"], "causal_graph": {"nodes": [], "edges": []},
                "recommendations": [], "disaster_level": 1}
    monkeypatch.setattr(m, "query_qwen", _capture)

    # both inputs present -> joint run, both channels swapped
    out, _, _ = m.apply_intervention(1, data, "data:image/png;base64,AA", "house_1", "prospective",
                                     "both", "EDITED caption", "data:image/png;base64,EDITED", None,
                                     "edge", "above_mean")
    assert out.__class__.__name__ == "Div" and out.className != "empty-state"
    assert seen["caption"] == "EDITED caption" and seen["image"] == "data:image/png;base64,EDITED"

    # missing the image -> 'both' refuses (it is a joint suppression)
    out2, _, _ = m.apply_intervention(1, data, "x", "house_1", "prospective", "both", "EDITED caption",
                                      None, None, "edge", "above_mean")
    assert out2.className == "empty-state" and "joint" in _flatten_text(out2).lower()


@pytest.mark.blocking
def test_apply_stores_counterfactual_image_and_export_writes_it(main_module, monkeypatch, tmp_path):
    """Apply records the do() itself — the counterfactual caption + edited image — in the try;
    Export writes each edited image to counterfactual_try{n}.<ext> and references it by filename
    in intervention.json (base64 blob kept OUT of the json). Caption-only tries carry no image
    (they reuse the baseline image already in the folder)."""
    m = main_module
    import base64, io, json
    from PIL import Image
    monkeypatch.setattr(m, "normalize_result", lambda d, ic=None: d)
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (200, 0, 0)).save(buf, format="PNG")
    cf_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    data = {"caption": "orig caption", "disaster_level": 8,
            "detected_objects": [{"object_id": "house_1", "label": "house", "state": "burning"},
                                 {"object_id": "person_1", "label": "person", "state": "standing"}],
            "causal_graph": {"nodes": [{"id": "house_1", "label": "house", "state": "burning", "hazardous": True},
                                       {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                             "edges": [{"source": "house_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}],
                             "intervention_candidates": [{"threat": "house_1", "state": "burning", "outgoing_edge_count": 1}]},
            "recommendations": [{"rank": 1, "action": "Evacuate",
                                 "structured_reasoning": {"threat": "house_1", "state": "burning",
                                                          "effect": "may_harm", "affected_objects": ["person_1"]}}]}
    monkeypatch.setattr(m, "query_qwen", lambda *a, **k: {
        "detected_objects": data["detected_objects"], "causal_graph": {"nodes": [], "edges": []},
        "recommendations": [], "disaster_level": 1})

    # 'both' apply -> the edited image + caption are recorded on the try
    _out, hist, _ = m.apply_intervention(1, data, "data:image/png;base64,AA", "house_1", "retrospective",
                                         "both", "house_1 is now intact", cf_url, None, "edge", "half_max")
    rec = hist["tries"][0]
    assert rec["counterfactual_image"] == cf_url                     # the do() image is stored
    assert rec["counterfactual_caption"] == "house_1 is now intact"  # and the caption the model saw

    # Export writes the image to a file, references it by name, and keeps base64 OUT of the json
    monkeypatch.setattr(m, "EXPORT_ROOT", tmp_path)
    status = m.export_structured_response(
        1, {"run_id": "run_test", "image_filename": "s.png", "detected_objects": data["detected_objects"]},
        "prompt", "orig caption", cf_url, "s.png", intervention_history=hist)
    run_dir = tmp_path / "runs" / "run_test"
    assert (run_dir / "counterfactual_try1.png").exists()            # the do() image is on disk
    iv = json.loads((run_dir / "intervention.json").read_text())
    t0 = iv["tries"][0]
    assert t0["counterfactual_image_file"] == "counterfactual_try1.png"   # referenced by filename
    assert "counterfactual_image" not in t0                          # base64 blob stripped from json
    assert t0["counterfactual_caption"] == "house_1 is now intact"
    assert "counterfactual image" in status.lower()                  # surfaced in the export status

    # caption-only apply -> no counterfactual image recorded (baseline image is reused)
    _o2, hist2, _ = m.apply_intervention(1, data, "data:image/png;base64,AA", "house_1", "retrospective",
                                         "caption", "house_1 is now intact", None, None, "edge", "half_max")
    assert hist2["tries"][0]["counterfactual_image"] is None


@pytest.mark.blocking
def test_distribution_coverage_counts_unrepresented_gt_cores(main_module, monkeypatch, tmp_path):
    """COVERAGE-HONEST (Sunny 2026-07-17, live push_06): GT has THREE cores {water, child_1,
    child_2}; the model represents only child_1 (water never perceived, child_2 'floating'
    out-of-vocab). Suppressing child_1 must NOT read a false 'grounded / alignment 1.00 /
    coverage 1/1' — the unrepresented cores belong in gt_dist with ZERO op mass, so the overlap
    drops and coverage reads 1/3 provisional."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "water_1", "label": "water", "state": "engulfing", "hazardous": True},
                    {"id": "child_1", "label": "child", "state": "drowning", "hazardous": False},
                    {"id": "child_2", "label": "child", "state": "unconscious", "hazardous": False}],
          "edges": [{"source": "water_1", "via_state": "engulfing", "effect": "may_harm", "target": "child_1"},
                    {"source": "water_1", "via_state": "engulfing", "effect": "may_harm", "target": "child_2"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    # model perceives BOTH children (co-referenced GT cores) but NOT water (unperceived core).
    # Only child_1 gets suppressed -> child_2 untested, water unrepresented -> coverage 1/3.
    det = [{"object_id": "child_1", "label": "child", "state": "drowning"},
           {"object_id": "child_2", "label": "child", "state": "unconscious"}]
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8, "detected_objects": det,
            "causal_graph": {"nodes": [{"id": "child_1", "hazardous": False, "state": "drowning"},
                                       {"id": "child_2", "hazardous": False, "state": "unconscious"}], "edges": []},
            "graph_b": {"nodes": [{"id": "child_1", "hazardous": False, "state": "drowning"},
                                  {"id": "child_2", "hazardous": False, "state": "unconscious"}],
                        "edges": [{"source": "water_1", "target": "child_1", "effect": "may_harm", "via_state": "engulfing"},
                                  {"source": "water_1", "target": "child_2", "effect": "may_harm", "via_state": "engulfing"}]},
            "recommendations": []}

    def try_(oid, c, d):
        return {"selection": {"target_object_id": oid}, "verdict": {"cell": "grounded", "do_not_applied": False},
                "u_check": {"leaked": False},
                "signals": {"content_shift": c, "graph_direction": d, "rec_direction": d,
                            "hazard_shift": c, "graph_shift": c, "recommendation_shift": c,
                            "hazard_level_delta": -4}}
    dist = m._distributional_groundedness(
        m._synthesis_cell(norm, {"tries": [try_("child_1", 0.6, "de-escalated")]},
                          core_basis="edge", core_rule="half_max"))
    assert dist["coverage"] == (1, 3)                    # 1 of 3 GT cores represented+tested
    assert dist["provisional"] is True
    assert dist["overlap"] < 0.99                        # NOT a false perfect alignment
    assert set(dist["gt_dist"]) >= {"water_1", "child_1", "child_2"}   # all cores in gt_dist
    assert dist["op_dist"].get("water_1", 0) == 0 and dist["op_dist"].get("child_2", 0) == 0
    assert dist["verdict"] != "grounded"                 # can't be clean-grounded at 1/3 coverage


@pytest.mark.blocking
def test_candidates_panel_gt_pick_shows_perceived_cocore_not_never_perceived(main_module):
    """MULTI-CORE: when the TOP GT hazard is unperceived but a co-core IS perceived, the
    candidates panel's GT pick must show the perceived co-core (a chip) with the unperceived top
    as a CAVEAT — NOT the misleading 'the model never perceived the ground-truth core'. push_06:
    water unperceived, child_1 a perceived core."""
    m = main_module
    cand = {"should_be_core": None,                       # top (water) unperceived
            "gt_core_unobserved": {"object_id": "water_1", "label": "water", "state": "engulfing",
                                   "reason": "not_perceived"},
            "gt_core_ids": ["child_1"],
            "candidates": [{"object_id": "child_1", "label": "child", "state": "struggling",
                            "ranks": {"GT": 1, "B": 1}, "is_gt_core": True, "variable_role": "victim"}]}
    det = [{"object_id": "child_1", "label": "child", "state": "struggling"}]
    panel = m.make_candidates_panel(cand, {"level": "low", "score": 0.04}, det, None, [],
                                    {"threat": "child_1", "state": "struggling"})
    txt = _flatten_text(panel).lower()
    assert "never perceived the ground-truth core" not in txt   # NOT the no-core message
    assert "child" in txt and "gt also names water" in txt      # perceived core + caveat
    assert "names a ground-truth core" in txt                   # agreement, not "no ground truth"

    # contrast: NO core perceived -> the honest 'never perceived' message DOES show
    cand2 = {"should_be_core": None,
             "gt_core_unobserved": {"object_id": "water_1", "label": "water", "state": "engulfing",
                                    "reason": "not_perceived"},
             "gt_core_ids": [],
             "candidates": [{"object_id": "dog_1", "label": "dog", "state": "rabid",
                             "ranks": {"A": 1}, "is_gt_core": False, "variable_role": "hazard"}]}
    panel2 = m.make_candidates_panel(cand2, {"level": "low", "score": 0.1},
                                     [{"object_id": "dog_1", "label": "dog", "state": "rabid"}], None, [], {})
    assert "never perceived the ground-truth core" in _flatten_text(panel2).lower()


@pytest.mark.blocking
def test_declared_match_is_multicore_any_gt_core(main_module, monkeypatch, tmp_path):
    """MULTI-CORE: the model's declared core matches GT if it names ANY member of the GT core
    SET, not just the single top hazard. GT top = water (unperceived); the model declares child_1
    (a GT co-core) -> declared_match True. Single-top matching would have called it a MISS and
    could flip the cell to 'ungrounded'."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "water_1", "label": "water", "state": "engulfing", "hazardous": True},
                    {"id": "child_1", "label": "child", "state": "drowning", "hazardous": False},
                    {"id": "child_2", "label": "child", "state": "unconscious", "hazardous": False}],
          "edges": [{"source": "water_1", "via_state": "engulfing", "effect": "may_harm", "target": "child_1"},
                    {"source": "water_1", "via_state": "engulfing", "effect": "may_harm", "target": "child_2"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    det = [{"object_id": "child_1", "label": "child", "state": "drowning"},
           {"object_id": "child_2", "label": "child", "state": "unconscious"}]
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8, "detected_objects": det,
            "causal_graph": {"nodes": [{"id": "child_1", "hazardous": False, "state": "drowning"}],
                             "edges": [{"source": "child_1", "target": "child_2", "effect": "may_harm", "via_state": "drowning"}]},
            "graph_b": {"nodes": [{"id": "child_1", "hazardous": False, "state": "drowning"}],
                        "edges": [{"source": "water_1", "target": "child_1", "effect": "may_harm", "via_state": "engulfing"}],
                        "suppression_pick": {"threat": "child_1", "state": "drowning"}},
            "recommendations": []}
    syn = m._synthesis_cell(norm, {"tries": []}, core_basis="edge", core_rule="half_max")
    assert set(syn["gt_core_ids"]) == {"child_1", "child_2"}     # multi-core: both children core
    assert syn["declared_match"] is True                         # model named a real GT co-core


@pytest.mark.blocking
def test_op_distribution_is_victim_weighted_on_consequence_basis(main_module, monkeypatch, tmp_path):
    """Option (a): on the CONSEQUENCE basis the OP distribution scales the operative shift by the
    victim-cost of the suppressed hazard across the model's OWN output — from Graph A edges AND
    the recs (the operative shift blends graph_shift + recommendation_shift). A hazard whose
    impact falls on a PERSON outweighs an equal one on a CAR, whether the victim is named via
    Graph A or via a rec. On the edge basis, equal shifts stay equal. Direction is unchanged."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                    {"id": "pump_1", "label": "pump", "state": "leaking", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False},
                    {"id": "car_1", "label": "car", "state": "parked", "hazardous": False}],
          "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                    {"source": "pump_1", "via_state": "leaking", "effect": "may_harm", "target": "car_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    det = [{"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
           {"object_id": "pump_1", "label": "pump", "state": "leaking"}]
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8, "detected_objects": det,
            "causal_graph": {"nodes": [{"id": "tanker_1", "hazardous": True, "state": "leaking"},
                                       {"id": "pump_1", "hazardous": True, "state": "leaking"}],
                             "edges": [{"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"},
                                       {"source": "pump_1", "target": "car_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1},
                                                         {"threat": "pump_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "graph_b": {"nodes": [], "edges": []}, "recommendations": []}

    _sig = {"content_shift": 0.6, "graph_direction": "de-escalated", "rec_direction": "de-escalated",
            "hazard_shift": 0.6, "graph_shift": 0.6, "recommendation_shift": 0.6, "hazard_level_delta": -5}
    # tanker: victim (person) named ONLY via a Graph A edge, no rec -> exercises the Graph A path
    tanker_try = {"selection": {"target_object_id": "tanker_1"},
                  "verdict": {"cell": "grounded", "do_not_applied": False}, "u_check": {"leaked": False}, "signals": _sig,
                  "before": {"recommendations": [],
                             "graph_a": {"nodes": [], "edges": [{"source": "tanker_1", "via_state": "leaking",
                                                                 "effect": "may_harm", "target": "person_1"}]}}}
    # pump: victim (car) named ONLY via a rec -> exercises the rec path
    pump_try = {"selection": {"target_object_id": "pump_1"},
                "verdict": {"cell": "grounded", "do_not_applied": False}, "u_check": {"leaked": False}, "signals": _sig,
                "before": {"recommendations": [{"rank": 1, "action": "act",
                           "structured_reasoning": {"threat": "pump_1", "state": "leaking", "effect": "may_harm",
                                                    "affected_objects": ["car_1"]}}],
                           "graph_a": {"nodes": [], "edges": []}}}
    hist = {"tries": [tanker_try, pump_try]}

    # edge basis: equal operative shift -> equal OP mass
    d_edge = m._distributional_groundedness(m._synthesis_cell(norm, hist, core_basis="edge", core_rule="half_max"))
    assert abs(d_edge["op_dist"]["tanker_1"] - d_edge["op_dist"]["pump_1"]) < 1e-6

    # consequence basis: tanker (protects a PERSON, life x2) carries MORE OP mass than pump (car)
    d_cons = m._distributional_groundedness(m._synthesis_cell(norm, hist, core_basis="consequence", core_rule="half_max"))
    assert d_cons["op_dist"]["tanker_1"] > d_cons["op_dist"]["pump_1"] + 1e-6


@pytest.mark.blocking
def test_escalation_severity_is_victim_weighted_and_kept_out_of_dependence(main_module, monkeypatch, tmp_path):
    """An ESCALATION (advice moved the WRONG way) is a red flag whose SEVERITY scales with victim
    stakes on the consequence basis — a person-escalation is louder than a car-escalation — but it
    is NEVER counted as dependence: escalated hazards contribute 0 to op_dist and appear in esc_dist."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                    {"id": "pump_1", "label": "pump", "state": "leaking", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False},
                    {"id": "car_1", "label": "car", "state": "parked", "hazardous": False}],
          "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                    {"source": "pump_1", "via_state": "leaking", "effect": "may_harm", "target": "car_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    det = [{"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
           {"object_id": "pump_1", "label": "pump", "state": "leaking"}]
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8, "detected_objects": det,
            "causal_graph": {"nodes": [{"id": "tanker_1", "hazardous": True, "state": "leaking"},
                                       {"id": "pump_1", "hazardous": True, "state": "leaking"}],
                             "edges": [{"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"},
                                       {"source": "pump_1", "target": "car_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1},
                                                         {"threat": "pump_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "graph_b": {"nodes": [], "edges": []}, "recommendations": []}

    def esc_try(oid, victim):   # ESCALATED (both directions escalate), equal magnitude
        return {"selection": {"target_object_id": oid}, "verdict": {"cell": "escalated", "do_not_applied": False},
                "u_check": {"leaked": False},
                "signals": {"content_shift": 0.6, "graph_direction": "escalated", "rec_direction": "escalated",
                            "hazard_shift": 0.6, "graph_shift": 0.6, "recommendation_shift": 0.6, "hazard_level_delta": 5},
                "before": {"recommendations": [{"rank": 1, "action": "act",
                           "structured_reasoning": {"threat": oid, "state": "leaking", "effect": "may_harm",
                                                    "affected_objects": [victim]}}]}}
    hist = {"tries": [esc_try("tanker_1", "person_1"), esc_try("pump_1", "car_1")]}

    d = m._distributional_groundedness(m._synthesis_cell(norm, hist, core_basis="consequence", core_rule="half_max"))
    # escalations are NOT dependence
    assert d["op_dist"].get("tanker_1", 0) == 0 and d["op_dist"].get("pump_1", 0) == 0
    # but the red-flag severity is victim-weighted: person-escalation louder than car-escalation
    assert d["esc_dist"]["tanker_1"] > d["esc_dist"]["pump_1"] + 1e-6
    # on the edge basis the two escalations are equally severe (no victim weighting)
    d_edge = m._distributional_groundedness(m._synthesis_cell(norm, hist, core_basis="edge", core_rule="half_max"))
    assert abs(d_edge["esc_dist"]["tanker_1"] - d_edge["esc_dist"]["pump_1"]) < 1e-6


@pytest.mark.blocking
def test_distribution_labels_core_spurious_on_both_gt_and_op_bars(main_module, monkeypatch, tmp_path):
    """Each distribution row labels core/secondary/spurious TWICE — once on the GT bar (what
    SHOULD matter) and once on the OP bar (what the advice ACTUALLY depends on) — so a hazard
    that is GT-core but OP-escalated (the masquerade/red-flag) reads per hazard by the two bar
    labels disagreeing."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                    {"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
          "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                    {"source": "tanker_1", "via_state": "leaking", "effect": "may_spread_to", "target": "fire_1"},
                    {"source": "fire_1", "via_state": "spreading", "effect": "may_harm", "target": "person_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    det = [{"object_id": "fire_1", "label": "fire", "state": "burning"},
           {"object_id": "tanker_1", "label": "tanker", "state": "leaking"}]
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8, "detected_objects": det,
            "causal_graph": {"nodes": [{"id": "fire_1", "hazardous": True, "state": "burning"},
                                       {"id": "tanker_1", "hazardous": True, "state": "leaking"}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
                                       {"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "fire_1", "state": "burning", "outgoing_edge_count": 1},
                                                         {"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "graph_b": {"nodes": [], "edges": []}, "recommendations": []}

    def try_(oid, c, d):
        return {"selection": {"target_object_id": oid}, "verdict": {"cell": "grounded", "do_not_applied": False},
                "u_check": {"leaked": False},
                "signals": {"content_shift": c, "graph_direction": d, "rec_direction": d,
                            "hazard_shift": c, "graph_shift": c, "recommendation_shift": c,
                            "hazard_level_delta": (-5 if d == "de-escalated" else 5)}}
    # tanker de-escalates (advice depends on it); fire ESCALATES (removing it made things worse)
    hist = {"tries": [try_("tanker_1", 0.6, "de-escalated"), try_("fire_1", 0.5, "escalated")]}
    panel = m.make_groundedness_synthesis_panel(norm, hist, detected_objects=det)
    txt = _flatten_text(panel).lower()
    # THREE columns: GT (should matter), OP (dependence, core/spurious only), Direction (↑/↓).
    # tanker is core on BOTH bars (grounded); fire is GT-core but escalated → OP dependence is
    # SPURIOUS (0), and the escalation shows only in the Direction line, never on the OP bar.
    assert "gt core" in txt and "op core" in txt          # tanker: both agree -> grounded
    assert "↑ escalated" in _flatten_text(panel)          # fire's escalation lives in Direction
    assert "op escalated" not in txt                      # NOT on the OP dependence bar anymore
    assert "↓ de-escalated" in _flatten_text(panel)       # tanker's direction
    assert "peripheral" not in txt and "secondary" not in txt   # no more 3-way vocabulary


@pytest.mark.blocking
def test_post_trust_is_conformance_based_and_surfaced_with_verdict(main_module):
    """POST trust is a conformance-based subset (no Graph B): a clean post graph -> high, no
    broken rules; a post graph with a dangling edge -> lower band + the broken rule named. The
    result panel shows a 'Trust: baseline → post' qualifier line next to the verdict."""
    m = main_module
    # clean post graph: both endpoints present -> high, 0 violations
    clean = {"nodes": [{"id": "house_1", "label": "house", "state": "intact", "hazardous": False},
                       {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
             "edges": []}
    t_clean = m._post_intervention_trust(clean)
    assert t_clean["level"] == "high" and t_clean["n_violations"] == 0 and t_clean["broken_rules"] == []
    # dangling edge (target not a node) -> at least one violation, band drops, rule named
    dirty = {"nodes": [{"id": "house_1", "label": "house", "state": "burning", "hazardous": True}],
             "edges": [{"source": "house_1", "via_state": "burning", "effect": "may_harm", "target": "ghost_9"}]}
    t_dirty = m._post_intervention_trust(dirty)
    assert t_dirty["n_violations"] >= 1 and t_dirty["level"] in ("moderate", "low")
    assert t_dirty["broken_rules"]                                    # names which rule broke

    # the verdict panel surfaces both trust qualifiers
    result = {"verdict": {"cell": "grounded", "explanation": "x",
                          "move_basis": {"content_shift": 0.6, "cutoff": 0.3}},
              "signals": {"content_shift": 0.6, "hazard_shift": 0.5, "graph_shift": 0.6,
                          "recommendation_shift": 0.6},
              "u_check": {"leaked": False},
              "pre_trust": {"level": "low", "score": 0.16},
              "post_trust": {"level": "moderate", "score": 0.8, "broken_rules": ["unresolved_endpoint"]}}
    panel = m.make_intervention_result_panel(result, {}, {}, "retrospective", "both", target_oid="house_1")
    txt = _flatten_text(panel).lower()
    assert "trust:" in txt and "baseline" in txt and "post" in txt
    assert "low" in txt and "moderate" in txt                         # both bands shown
    assert "qualifies the verdict" in txt                             # not multiplied in


@pytest.mark.blocking
def test_apply_auto_persists_counterfactual_without_export(main_module, monkeypatch, tmp_path):
    """Clicking Apply (not Export) already writes the counterfactual image + intervention.json
    + synthesis.json to <EXPORT_ROOT>/runs/<run_id>, using the run_id stamped by analyze_scene.
    No Export needed. synthesis.json carries the aggregate verdict + the active toggle."""
    m = main_module
    import base64, io, json
    from PIL import Image
    monkeypatch.setattr(m, "normalize_result", lambda d, ic=None: d)
    monkeypatch.setattr(m, "EXPORT_ROOT", tmp_path)
    gt = {"image_filename": "s.png", "caption": "orig",
          "nodes": [{"id": "house_1", "label": "house", "state": "burning", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
          "edges": [{"source": "house_1", "via_state": "burning", "effect": "may_harm", "target": "person_1"}]}
    (tmp_path / "s.png.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (0, 0, 200)).save(buf, format="PNG")
    cf_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    data = {"run_id": "run_apply", "image_filename": "s.png", "caption": "orig", "disaster_level": 8,
            "detected_objects": [{"object_id": "house_1", "label": "house", "state": "burning"},
                                 {"object_id": "person_1", "label": "person", "state": "standing"}],
            "causal_graph": {"nodes": [{"id": "house_1", "label": "house", "state": "burning", "hazardous": True},
                                       {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                             "edges": [{"source": "house_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}],
                             "intervention_candidates": [{"threat": "house_1", "state": "burning", "outgoing_edge_count": 1}]},
            "recommendations": [{"rank": 1, "action": "Evacuate",
                                 "structured_reasoning": {"threat": "house_1", "state": "burning",
                                                          "effect": "may_harm", "affected_objects": ["person_1"]}}]}
    monkeypatch.setattr(m, "query_qwen", lambda *a, **k: {
        "detected_objects": data["detected_objects"], "causal_graph": {"nodes": [], "edges": []},
        "recommendations": [], "disaster_level": 1})

    m.apply_intervention(1, data, "data:image/png;base64,AA", "house_1", "retrospective",
                         "both", "house_1 is now intact", cf_url, None, "edge", "half_max")

    run_dir = tmp_path / "runs" / "run_apply"
    assert (run_dir / "counterfactual_try1.png").exists()         # written on Apply, before any Export
    iv = json.loads((run_dir / "intervention.json").read_text())
    assert iv["tries"][0]["counterfactual_image_file"] == "counterfactual_try1.png"
    assert iv["tries"][0]["counterfactual_caption"] == "house_1 is now intact"

    # synthesis snapshot: the aggregate conclusion + the toggle it was computed under
    syn = json.loads((run_dir / "synthesis.json").read_text())
    assert syn["core_basis"] == "edge" and syn["core_rule"] == "half_max"
    assert "verdict" in syn["distribution"] and "alignment" in syn["distribution"]
    assert isinstance(syn["hazard_status"], list)


@pytest.mark.blocking
def test_result_panel_has_graph_breakdown_and_explainer(main_module):
    """The causal-graph row shows the node/edge breakdown, and the panel carries the
    plain-language 'how measured' explainer (incl. the no-bbox note)."""
    m = main_module
    res = {"verdict": {"cell": "grounded", "explanation": "x"},
           "signals": {"hazard_shift": 0.6, "graph_shift": 1.0, "recommendation_shift": 1.0,
                       "content_shift": 0.87, "node_shift": 0.0, "edge_shift": 1.0, "hazard_level_delta": -6},
           "u_check": {"leaked": False, "state_stability": 0.9, "topology_stability": 0.9}}
    baseline = {"hazard_level": 6, "graph_a": {"nodes": [{"id": "fire_1", "state": "burning"}],
                                               "edges": [{"source": "fire_1", "target": "fire_1", "effect": "worsens", "via_state": "burning"}]},
                "recommendations": []}
    post = {"disaster_level": 0, "causal_graph": {"nodes": [{"id": "fire_1", "state": "burning"}], "edges": []},
            "recommendations": []}
    txt = _flatten_text(m.make_intervention_result_panel(res, baseline, post, mode="prospective", modality="both")).lower()
    assert "entities 0.00 · arrows 1.00" in txt                          # badge from actual graphs: node held, edge gone
    assert "how are these measured" in txt
    assert "bounding box" in txt                                          # the no-bbox explanation
    assert "wording ignored" in txt                                       # rec-shift explanation


@pytest.mark.blocking
def test_result_panel_input_problem_is_a_plain_notice_not_a_verdict(main_module):
    """When the input was NOT edited (u_check.leaked True), the panel short-circuits to a plain
    input-problem notice — NO verdict card, NO Fair-test card/wording, NO graph/rec 'culprit'.
    The word 'Fair-test' never appears; the notice names the input problem and what to do."""
    m = main_module
    result = {"verdict": {"cell": "u_leaked"},
              "signals": {"hazard_shift": 0.2, "graph_shift": 0.67, "recommendation_shift": 0.67,
                          "content_shift": 0.5, "node_shift": 0.4, "edge_shift": 0.67, "hazard_level_delta": -2},
              "u_check": {"leaked": True, "applied": False, "input_gated": True},
              "spec": {"target": {"object_id": "tanker_1"}}}
    baseline = {"hazard_level": 7,
                "graph_a": {"nodes": [{"id": "tanker_1", "state": "leaking"}], "edges": []},
                "recommendations": []}
    post = {"disaster_level": 5,
            "causal_graph": {"nodes": [{"id": "tanker_truck_1", "state": "discharging_water"}], "edges": []},
            "recommendations": []}
    txt = _flatten_text(m.make_intervention_result_panel(
        result, baseline, post, mode="retrospective", modality="caption", target_oid="tanker_1"))
    low = txt.lower()
    assert "no intervention to measure" in low                            # the input-problem headline
    assert "didn't change the caption" in low and "tanker_1" in txt       # names the problem + target
    # none of the verdict / Fair-test / culprit machinery is surfaced
    assert "fair-test" not in low and "verdict" not in low
    assert "movement" not in low and "broke" not in low


@pytest.mark.blocking
def test_result_panel_target_state_change_is_purple_not_amber(main_module):
    """The suppressed target that CHANGES STATE (leaking -> parked, the usual clean-removal
    outcome) is tagged purple 'suppressed', not amber 'changed'. A non-target unchanged edge
    stays grey (untagged)."""
    m = main_module
    result = {"verdict": {"cell": "grounded"}, "spec": {"target": {"object_id": "tanker_1"}},
              "signals": {"node_shift": 0.0, "edge_shift": 0.5, "graph_shift": 0.5,
                          "recommendation_shift": 0.0, "hazard_shift": 0.0, "content_shift": 0.17,
                          "hazard_level_delta": 0},
              "u_check": {"leaked": False, "state_stability": 1.0, "topology_stability": 1.0}}
    baseline = {"hazard_level": 7,
                "graph_a": {"nodes": [{"id": "tanker_1", "state": "leaking"}, {"id": "fire_1", "state": "burning"},
                                      {"id": "person_1", "state": "standing"}],
                            "edges": [{"source": "tanker_1", "target": "person_1", "effect": "increases_risk_to", "via_state": "leaking"},
                                      {"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}]},
                "recommendations": []}
    post = {"disaster_level": 7,
            "causal_graph": {"nodes": [{"id": "tanker_1", "state": "parked"}, {"id": "fire_1", "state": "burning"},
                                       {"id": "person_1", "state": "standing"}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}]},
            "recommendations": []}
    panel = m.make_intervention_result_panel(result, baseline, post, target_oid="tanker_1")

    # collect every rendered iv-line with its classes
    lines = []

    def walk(n):
        cls = getattr(n, "className", "")
        if isinstance(cls, str) and "iv-line" in cls:
            lines.append((cls, _flatten_text(n)))
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                walk(c)
    walk(panel)

    tanker_lines = [(cls, t) for cls, t in lines if "tanker_1: " in t]
    assert tanker_lines and all("iv-supp" in cls for cls, _ in tanker_lines)   # purple, not amber
    # the unchanged non-target edge fire->person is grey (iv-same), never tagged suppressed
    fp = [(cls, t) for cls, t in lines if "fire_1 →may_harm→ person_1" in t]
    assert fp and all("iv-same" in cls and "⊘" not in t for cls, t in fp)


@pytest.mark.blocking
def test_result_panel_edge_effect_change_is_changed_not_add_remove(main_module):
    """When the SAME source→target arrow keeps its endpoints but changes effect/via_state,
    it renders as 'changed' (blue) on both sides — NOT a red removed + green added pair."""
    m = main_module
    result = {"verdict": {"cell": "grounded"}, "spec": {"target": {"object_id": "tanker_1"}},
              "signals": {"node_shift": 0.0, "edge_shift": 0.5, "graph_shift": 0.5, "recommendation_shift": 0.0,
                          "hazard_shift": 0.0, "content_shift": 0.17, "hazard_level_delta": 0},
              "u_check": {"leaked": False, "state_stability": 1.0, "topology_stability": 1.0}}
    baseline = {"hazard_level": 7,
                "graph_a": {"nodes": [{"id": "fire_1", "state": "burning"}, {"id": "person_1", "state": "standing"}],
                            "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}]},
                "recommendations": []}
    post = {"disaster_level": 7,
            "causal_graph": {"nodes": [{"id": "fire_1", "state": "burning"}, {"id": "person_1", "state": "standing"}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_spread_to", "via_state": "spreading"}]},
            "recommendations": []}
    panel = m.make_intervention_result_panel(result, baseline, post, target_oid="tanker_1")

    lines = []

    def walk(n):
        cls = getattr(n, "className", "")
        if isinstance(cls, str) and "iv-line" in cls:
            lines.append((cls, _flatten_text(n)))
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                walk(c)
    walk(panel)

    edge_lines = [(cls, t) for cls, t in lines if "→ person_1" in t]
    assert edge_lines, "no fire->person arrow lines rendered"
    # BOTH sides (old may_harm, new may_spread_to) are 'changed' (blue), never removed/added
    assert all("iv-chg" in cls for cls, _ in edge_lines)
    assert not any("iv-del" in cls or "iv-add" in cls for cls, _ in edge_lines)
    # the legend names the changed category
    assert "state / effect changed" in _flatten_text(panel).lower()


def _iv_lines(main_module, panel):
    """Collect (className, text) for every rendered diff line."""
    out = []

    def walk(n):
        cls = getattr(n, "className", "")
        if isinstance(cls, str) and "iv-line" in cls:
            out.append((cls, _flatten_text(n)))
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                walk(c)
    walk(panel)
    return out


def _iv_result(**sig):
    base = {"node_shift": 0.1, "edge_shift": 0.1, "graph_shift": 0.1, "recommendation_shift": 0.0,
            "hazard_shift": 0.0, "content_shift": 0.1, "hazard_level_delta": 0}
    base.update(sig)
    return {"verdict": {"cell": "grounded"}, "spec": {"target": {"object_id": "none"}},
            "signals": base, "u_check": {"leaked": False, "state_stability": 1.0, "topology_stability": 1.0}}


@pytest.mark.blocking
def test_result_panel_detects_hazard_flag_label_and_reversal(main_module):
    """The graph diff detects the three previously-unshown change types, each with its own
    colour + inline annotation, and the per-card legend lists ONLY detected categories."""
    m = main_module

    # hazard flag flip (state same, hazardous True->False) -> orange 'iv-haz'
    b = {"hazard_level": 5, "graph_a": {"nodes": [{"id": "fire_1", "state": "burning", "hazardous": True}], "edges": []}, "recommendations": []}
    p = {"disaster_level": 5, "causal_graph": {"nodes": [{"id": "fire_1", "state": "burning", "hazardous": False}], "edges": []}, "recommendations": []}
    panel = m.make_intervention_result_panel(_iv_result(), b, p, target_oid="none")
    haz = [(c, t) for c, t in _iv_lines(m, panel) if "fire_1" in t]
    assert haz and all("iv-haz" in c for c, _ in haz) and "hazard flag changed" in _flatten_text(panel).lower()

    # label change (same id, different label) -> brown 'iv-lbl'
    b = {"hazard_level": 5, "graph_a": {"nodes": [{"id": "tanker_1", "state": "leaking", "label": "tanker"}], "edges": []}, "recommendations": []}
    p = {"disaster_level": 5, "causal_graph": {"nodes": [{"id": "tanker_1", "state": "leaking", "label": "truck"}], "edges": []}, "recommendations": []}
    panel = m.make_intervention_result_panel(_iv_result(), b, p, target_oid="none")
    lbl = [(c, t) for c, t in _iv_lines(m, panel) if "tanker_1" in t]
    assert lbl and all("iv-lbl" in c for c, _ in lbl) and "label changed" in _flatten_text(panel).lower()

    # direction reversal (a->b becomes b->a) -> teal 'iv-rev' + '⇄ reversed'
    b = {"hazard_level": 5, "graph_a": {"nodes": [{"id": "fire_1", "state": "burning"}, {"id": "person_1", "state": "standing"}],
                                        "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}]}, "recommendations": []}
    p = {"disaster_level": 5, "causal_graph": {"nodes": [{"id": "fire_1", "state": "burning"}, {"id": "person_1", "state": "standing"}],
                                               "edges": [{"source": "person_1", "target": "fire_1", "effect": "may_harm", "via_state": "standing"}]}, "recommendations": []}
    panel = m.make_intervention_result_panel(_iv_result(), b, p, target_oid="none")
    rev = [(c, t) for c, t in _iv_lines(m, panel) if "→" in t]
    assert rev and all("iv-rev" in c for c, _ in rev)
    ptxt = _flatten_text(panel).lower()
    assert "direction reversed" in ptxt and "⇄ reversed" in _flatten_text(panel)


@pytest.mark.blocking
def test_result_panel_legend_only_lists_detected_categories(main_module):
    """A pure add-only graph diff's per-card legend lists 'added' but NOT
    'changed' / 'reversed' / etc. (checked on the legend rows, not the explainer)."""
    m = main_module
    b = {"hazard_level": 5, "graph_a": {"nodes": [{"id": "fire_1", "state": "burning"}], "edges": []}, "recommendations": []}
    p = {"disaster_level": 5, "causal_graph": {"nodes": [{"id": "fire_1", "state": "burning"}, {"id": "smoke_1", "state": "rising"}], "edges": []}, "recommendations": []}
    panel = m.make_intervention_result_panel(_iv_result(), b, p, target_oid="none")

    legends = []

    def walk(n):
        if getattr(n, "className", "") == "iv-legend-row":
            legends.append(_flatten_text(n).lower())
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                walk(c)
    walk(panel)

    joined = " ".join(legends)
    assert legends and "added" in joined
    assert "direction reversed" not in joined and "hazard flag changed" not in joined and "label changed" not in joined


@pytest.mark.blocking
def test_result_panel_resolves_renamed_target_not_add_remove(main_module):
    """A suppressed target renamed by the model (tanker_1 -> tanker_truck_1) renders as ONE
    purple entity (state change + 'model id' note), NOT a red removed + green added pair, and
    the edge touching it is purple (excluded), so the graph badge shows entities 0.00."""
    m = main_module
    baseline = {"hazard_level": 7,
                "detected_objects": [{"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
                                     {"object_id": "fire_1", "label": "fire", "state": "burning"},
                                     {"object_id": "person_1", "label": "person", "state": "standing"}],
                "graph_a": {"nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking"},
                                      {"id": "fire_1", "label": "fire", "state": "burning"},
                                      {"id": "person_1", "label": "person", "state": "standing"}],
                            "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
                                      {"source": "tanker_1", "target": "person_1", "effect": "increases_risk_to", "via_state": "leaking"}]},
                "recommendations": []}
    post = {"disaster_level": 4,
            "detected_objects": [{"object_id": "fire_1", "label": "fire", "state": "burning"},
                                 {"object_id": "person_1", "label": "person", "state": "standing"},
                                 {"object_id": "tanker_truck_1", "label": "tanker_truck", "state": "stationary"}],
            "causal_graph": {"nodes": [{"id": "fire_1", "label": "fire", "state": "burning"},
                                       {"id": "person_1", "label": "person", "state": "standing"},
                                       {"id": "tanker_truck_1", "label": "tanker_truck", "state": "stationary"}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
                                       {"source": "fire_1", "target": "tanker_truck_1", "effect": "may_harm", "via_state": "burning"}]},
            "recommendations": []}
    result = {"verdict": {"cell": "grounded"}, "spec": {"target": {"object_id": "tanker_1", "label": "tanker"}},
              "signals": {"node_shift": 0.4, "edge_shift": 0.67, "graph_shift": 0.67, "recommendation_shift": 0.0,
                          "hazard_shift": 0.3, "content_shift": 0.3, "hazard_level_delta": -3},
              "u_check": {"leaked": False, "state_stability": 1.0, "topology_stability": 1.0,
                          "renames": {"tanker_truck_1": "tanker_1"}}}
    panel = m.make_intervention_result_panel(result, baseline, post, target_oid="tanker_1")

    lines = []

    def walk(n):
        cls = getattr(n, "className", "")
        if isinstance(cls, str) and "iv-line" in cls:
            lines.append((cls, _flatten_text(n)))
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                walk(c)
    walk(panel)

    tanker = [(cls, t) for cls, t in lines if "tanker" in t.lower()]
    # every tanker line is purple (suppressed target), NONE is added(green)/removed(red)
    assert tanker and all("iv-supp" in cls for cls, _ in tanker)
    assert not any("iv-add" in cls or "iv-del" in cls for cls, t in lines if "tanker" in t.lower())
    txt = _flatten_text(panel)
    assert "model id tanker_truck_1" in txt                 # the rename is surfaced
    assert "entities 0.00" in txt                           # renamed object counts as matched


@pytest.mark.blocking
def test_result_panel_rec_matching_is_action_aware(main_module):
    """Recs match by (action-intent, affected-object): the two 'Move person_1' recs read as
    unchanged (grey), 'Alert' is removed (red), the new tanker-truck rec is added (green) —
    fixing the quad-only mis-pairing."""
    m = main_module
    before_recs = [
        {"rank": 1, "action": "Alert emergency services about the fire and the oil leak.",
         "structured_reasoning": {"threat": "fire_1", "effect": "may_harm", "affected_objects": ["person_1"]}},
        {"rank": 2, "action": "Move person_1 to a safe distance from the fire and the leaking tanker.",
         "structured_reasoning": {"threat": "fire_1", "effect": "may_harm", "affected_objects": ["person_1"]}}]
    after_recs = [
        {"rank": 1, "action": "Move person_1 away from the roadside.",
         "structured_reasoning": {"threat": "fire_1", "effect": "may_harm", "affected_objects": ["person_1"]}},
        {"rank": 2, "action": "Ensure the tanker truck is positioned to assist firefighting.",
         "structured_reasoning": {"threat": "fire_1", "effect": "may_harm", "affected_objects": ["tanker_truck_1"]}}]
    result = {"verdict": {"cell": "grounded"}, "spec": {"target": {"object_id": "tanker_1"}},
              "signals": {"node_shift": 0, "edge_shift": 0, "graph_shift": 0, "recommendation_shift": 0.67,
                          "hazard_shift": 0, "content_shift": 0.2, "hazard_level_delta": 0},
              "u_check": {"leaked": False, "state_stability": 1, "topology_stability": 1}}
    baseline = {"hazard_level": 7, "graph_a": {"nodes": [], "edges": []}, "recommendations": before_recs, "detected_objects": []}
    post = {"disaster_level": 4, "causal_graph": {"nodes": [], "edges": []}, "recommendations": after_recs, "detected_objects": []}
    panel = m.make_intervention_result_panel(result, baseline, post, target_oid="tanker_1")

    lines = []

    def walk(n):
        cls = getattr(n, "className", "")
        if isinstance(cls, str) and "iv-line" in cls:
            lines.append((cls, _flatten_text(n)))
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                walk(c)
    walk(panel)

    def cls_of(substr):
        return next(cls for cls, t in lines if substr in t)

    assert "iv-del" in cls_of("Alert emergency services")                 # removed (red)
    assert "iv-same" in cls_of("Move person_1 to a safe distance")        # both Move recs...
    assert "iv-same" in cls_of("Move person_1 away from the roadside")    # ...match (grey)
    assert "iv-add" in cls_of("Ensure the tanker truck")                  # added (green)
    import re as _re
    assert "shift 0.67" in _re.sub(r"\s+", " ", _flatten_text(panel))     # chip score from atom sets


@pytest.mark.blocking
def test_result_panel_shows_rec_direction_and_semantic_row(main_module):
    """The recs card shows the DIRECTION (de-escalated/escalated) via the shift chip and a
    combined-semantic row (the graceful 'install ...' note when sentence-transformers is absent)."""
    m = main_module
    before_recs = [{"action": "Evacuate everyone immediately", "structured_reasoning": {"affected_objects": ["person_1"]}},
                   {"action": "Alert emergency services", "structured_reasoning": {"affected_objects": ["person_1"]}}]
    after_recs = [{"action": "Monitor the area", "structured_reasoning": {"affected_objects": ["person_1"]}}]
    result = {"verdict": {"cell": "grounded"}, "spec": {"target": {"object_id": "t1"}},
              "signals": {"node_shift": 0, "edge_shift": 0, "graph_shift": 0, "recommendation_shift": 1.0,
                          "hazard_shift": 0.4, "content_shift": 0.5, "hazard_level_delta": -4,
                          "rec_direction": "de-escalated"},
              "u_check": {"leaked": False, "state_stability": 1, "topology_stability": 1}}
    baseline = {"hazard_level": 7, "graph_a": {"nodes": [], "edges": []}, "recommendations": before_recs, "detected_objects": []}
    post = {"disaster_level": 3, "causal_graph": {"nodes": [], "edges": []}, "recommendations": after_recs, "detected_objects": []}
    low = _flatten_text(m.make_intervention_result_panel(result, baseline, post, target_oid="t1")).lower()
    assert "de-escalated" in low                                          # direction shown on the shift chip
    assert "combined semantic shift" in low                               # the semantic row (value or install-note)


@pytest.mark.blocking
def test_result_panel_shift_chip_combines_score_and_direction(main_module):
    """Graph and recs cards each carry a compact shift chip: a big score PLUS an escalation-
    aware arrow/word (↑ escalated red / ↓ de-escalated green). Escalation after a suppression
    must render on both cards so the red flag is visible where the change happened."""
    m = main_module
    import re as _re
    before_recs = [{"action": "Monitor the scene", "structured_reasoning": {"affected_objects": ["p_1"]}}]
    after_recs = [{"action": "Evacuate everyone immediately", "structured_reasoning": {"affected_objects": ["p_1"]}}]
    result = {"verdict": {"cell": "ungrounded"}, "spec": {"target": {"object_id": "t1"}},
              "signals": {"node_shift": 0, "edge_shift": 0.5, "graph_shift": 0.5, "recommendation_shift": 1.0,
                          "hazard_shift": 0.2, "content_shift": 0.5, "hazard_level_delta": 2,
                          "rec_direction": "escalated", "graph_direction": "escalated",
                          "graph_threat_before": 1.0, "graph_threat_after": 3.0,
                          "rec_urgency_before": 1.0, "rec_urgency_after": 3.0},
              "u_check": {"leaked": False, "state_stability": 1, "topology_stability": 1}}
    baseline = {"hazard_level": 5, "graph_a": {"nodes": [{"id": "t1", "state": "leaking"}],
                                               "edges": [{"source": "t1", "target": "p_1", "effect": "threatens"}]},
                "recommendations": before_recs, "detected_objects": []}
    post = {"disaster_level": 7, "causal_graph": {"nodes": [{"id": "t1", "state": "leaking"}],
                                                  "edges": [{"source": "t1", "target": "p_1", "effect": "may_harm"},
                                                            {"source": "t1", "target": "p_1", "effect": "threatens"}]},
            "recommendations": after_recs, "detected_objects": []}
    flat = _re.sub(r"\s+", " ", _flatten_text(
        m.make_intervention_result_panel(result, baseline, post, target_oid="t1")))
    low = flat.lower()
    # both cards show the escalation direction (red flag) + the compact "shift <score>" chip
    assert low.count("escalated") >= 2                                    # graph chip AND recs chip
    assert "↑" in flat                                                    # escalation arrow
    assert "shift" in low and _re.search(r"shift \d", low)               # chip carries a numeric score


@pytest.mark.blocking
def test_apply_button_has_progress_status_target(main_module):
    """The Apply row exposes a dcc.Loading-wrapped status node beside the button so a spinner
    shows while the model runs; apply_intervention returns a 3rd value driving it."""
    m = main_module
    setup = m.make_intervention_setup_panel(
        {"should_be_core": {"object_id": "t1", "state": "leaking", "label": "tanker"},
         "candidates": [{"object_id": "t1", "label": "tanker", "state": "leaking"}]},
        [{"object_id": "t1", "label": "tanker", "state": "leaking"}],
        [{"rank": 1, "threat": "t1", "state": "leaking"}], {"threat": "t1", "state": "leaking"}, "a caption")
    ids = []

    def collect(n):
        cid = getattr(n, "id", None)
        if cid:
            ids.append(cid)
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                collect(c)
    collect(setup)
    assert "intervention-apply" in ids and "intervention-apply-status" in ids
    assert "intervention-apply-loading" in ids


@pytest.mark.blocking
def test_result_panel_target_status_explains_why_present(main_module):
    """The graph card explains the target's fate with the right TONE per case (live 2026-07-08:
    'STILL in the after (leaking → stationary)' skims as 'still leaking' and framed the INTENDED
    object-with-state outcome as a suspicious survival):
      - entity persists + hazardous state CLEARED -> success ('suppression took effect', object
        kept as the edit intended) — never the word 'still' next to the hazard state;
      - entity persists + state UNCHANGED on a caption-only edit -> warning naming the cross-modal
        cause + 'Both';
      - entity persists + state UNCHANGED on a joint edit -> warning that the do() may not have
        taken effect;
      - entity gone -> suppression took effect."""
    m = main_module
    b = {"hazard_level": 7,
         "detected_objects": [{"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
                              {"object_id": "person_1", "label": "person", "state": "standing"}],
         "graph_a": {"nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking"},
                               {"id": "person_1", "label": "person", "state": "standing"}], "edges": []},
         "recommendations": []}
    res = {"verdict": {"cell": "grounded"}, "spec": {"target": {"object_id": "tanker_1", "label": "tanker"}},
           "signals": {"node_shift": 0, "edge_shift": 0, "graph_shift": 0, "recommendation_shift": 0,
                       "hazard_shift": 0.2, "content_shift": 0.1, "hazard_level_delta": -2, "rec_direction": "unchanged"},
           "u_check": {"leaked": False, "renames": {"tanker_truck_1": "tanker_1"}}}

    # (1) state CLEARED (leaking -> stationary, via model rename): the object-with-state SUCCESS.
    cleared = {"disaster_level": 5,
               "detected_objects": [{"object_id": "person_1", "label": "person", "state": "standing"},
                                    {"object_id": "tanker_truck_1", "label": "tanker_truck", "state": "stationary"}],
               "causal_graph": {"nodes": [{"id": "person_1", "label": "person", "state": "standing"},
                                          {"id": "tanker_truck_1", "label": "tanker_truck", "state": "stationary"}], "edges": []},
               "recommendations": []}
    low = _flatten_text(m.make_intervention_result_panel(res, b, cleared, mode="retrospective", modality="both", target_oid="tanker_1")).lower()
    assert "suppression took effect" in low and "stays in the scene" in low
    assert "hazardous state is cleared" in low and "leaking → stationary" in low
    assert "still in the after" not in low and "survived" not in low     # the misleading framing is gone

    # (2) state UNCHANGED on a caption-only edit: warning + cross-modal cause + 'Both'.
    persist = {"disaster_level": 6,
               "detected_objects": [{"object_id": "person_1", "label": "person", "state": "standing"},
                                    {"object_id": "tanker_1", "label": "tanker", "state": "leaking"}],
               "causal_graph": {"nodes": [{"id": "person_1", "label": "person", "state": "standing"},
                                          {"id": "tanker_1", "label": "tanker", "state": "leaking"}], "edges": []},
               "recommendations": []}
    low2 = _flatten_text(m.make_intervention_result_panel(res, b, persist, mode="retrospective", modality="caption", target_oid="tanker_1")).lower()
    assert "warning" in low2 and "still leaking" in low2                  # honest warning, right case
    assert "you edited only the caption" in low2 and "both" in low2

    # (3) state UNCHANGED on a JOINT edit: the do() may not have taken effect.
    low3 = _flatten_text(m.make_intervention_result_panel(res, b, persist, mode="retrospective", modality="both", target_oid="tanker_1")).lower()
    assert "warning" in low3 and "did not clear its hazardous state" in low3

    # (4) entity GONE: suppression took effect.
    gone = {"disaster_level": 3, "detected_objects": [{"object_id": "person_1", "label": "person", "state": "standing"}],
            "causal_graph": {"nodes": [{"id": "person_1", "label": "person", "state": "standing"}], "edges": []},
            "recommendations": []}
    low4 = _flatten_text(m.make_intervention_result_panel(res, b, gone, mode="retrospective", modality="caption", target_oid="tanker_1")).lower()
    assert "gone from the after" in low4 and "took effect" in low4


@pytest.mark.blocking
def test_apply_intervention_accumulates_history(main_module, monkeypatch):
    """Each Apply appends a full serialisable try to the per-run history store; a new run key
    resets the list."""
    m = main_module
    monkeypatch.setattr(m, "normalize_result", lambda d, ic=None: d)
    data = {"run_id": "run_A", "caption": "c", "disaster_level": 8,
            "detected_objects": [{"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
                                 {"object_id": "person_1", "label": "person", "state": "standing"}],
            "causal_graph": {"nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                                       {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                             "edges": [{"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "recommendations": [{"rank": 1, "action": "Move person_1 away",
                                 "structured_reasoning": {"threat": "tanker_1", "state": "leaking", "effect": "may_harm", "affected_objects": ["person_1"]}}]}
    post = {"detected_objects": [{"object_id": "tanker_1", "label": "tanker", "state": "contained"}],
            "causal_graph": {"nodes": [{"id": "tanker_1", "hazardous": False, "state": "contained"}], "edges": []},
            "recommendations": [{"rank": 1, "action": "Monitor", "structured_reasoning": {}}], "disaster_level": 2}
    monkeypatch.setattr(m, "query_qwen", lambda *a, **k: post)

    _, h1, _ = m.apply_intervention(1, data, None, "tanker_1", "retrospective", "caption", "edit1",
                                    None, None, "edge", "above_mean")
    assert h1["run_key"] == "run_A" and len(h1["tries"]) == 1
    _, h2, _ = m.apply_intervention(1, data, None, "tanker_1", "prospective", "caption", "edit2",
                                    None, h1, "edge", "above_mean")
    assert len(h2["tries"]) == 2 and h2["tries"][1]["index"] == 2
    # each try is self-contained + JSON-serialisable
    import json as _json
    _json.dumps(h2)
    for k in ("verdict", "u_check", "signals", "before", "after", "selection", "spec"):
        assert k in h2["tries"][0]
    # a DIFFERENT run resets the list
    data2 = {**data, "run_id": "run_B"}
    _, h3, _ = m.apply_intervention(1, data2, None, "tanker_1", "retrospective", "caption", "e",
                                    None, h2, "edge", "above_mean")
    assert h3["run_key"] == "run_B" and len(h3["tries"]) == 1


@pytest.mark.blocking
def test_edit_texts_fluid_vs_object_hazard(main_module):
    """Hazard-as-state: an object WITH a hazardous state (leaking tanker) is KEPT and only its
    state changes; a FLUID (water) IS the hazard and is removed."""
    m = main_module
    # object-with-state -> keep the tanker, fix the leak (both modes keep it in the scene)
    cap_r, img_r = m.build_intervention_edit_texts("tanker", "leaking", "retrospective", ["fire", "person"])
    cap_p, img_p = m.build_intervention_edit_texts("tanker", "leaking", "prospective", ["fire", "person"])
    assert "keep the tanker in the scene" in cap_r.lower() and "keep the tanker in the scene" in cap_p.lower()
    assert "must stay in the scene" in img_r.lower()               # not deleted
    assert "never leaking" in img_r.lower()                        # retrospective: erase the state
    assert "no longer leaking" in cap_p.lower() and "aftermath" in cap_p.lower()  # prospective: state gone, damage stays

    # fluid -> the water itself is removed
    _, img_water = m.build_intervention_edit_texts("water", "rising", "retrospective", ["house"])
    assert "remove the water" in img_water.lower()
    assert m._is_fluid_hazard("water_1") and m._is_fluid_hazard("fire") and not m._is_fluid_hazard("tanker_1")


@pytest.mark.blocking
def test_verdict_card_shows_movement_threshold_breakdown_takeaway(main_module):
    """The verdict card surfaces the movement number vs the threshold, the per-signal
    breakdown, and a short takeaway (a barely-over-cutoff grounded reads 'weak')."""
    m = main_module
    res = {"verdict": {"cell": "grounded", "explanation": "x"}, "spec": {"target": {"object_id": "t1"}},
           "signals": {"hazard_shift": 0.0, "graph_shift": 0.67, "recommendation_shift": 0.33,
                       "content_shift": 0.33, "node_shift": 0, "edge_shift": 0.67, "hazard_level_delta": 0},
           "u_check": {"leaked": False, "state_stability": 1, "topology_stability": 1}}
    b = {"hazard_level": 7, "graph_a": {"nodes": [], "edges": []}, "recommendations": [], "detected_objects": []}
    p = {"disaster_level": 7, "causal_graph": {"nodes": [], "edges": []}, "recommendations": [], "detected_objects": []}
    low = _flatten_text(m.make_intervention_result_panel(res, b, p, target_oid="t1")).lower()
    assert "movement" in low and "0.33" in low and "needs ≥ 0.3" in low     # number vs threshold
    assert "hazard" in low and "graph" in low and "recs" in low             # per-signal breakdown
    assert "weak" in low                                                    # takeaway: barely over cutoff

    strong = dict(res, signals={**res["signals"], "content_shift": 0.7})
    low2 = _flatten_text(m.make_intervention_result_panel(strong, b, p, target_oid="t1")).lower()
    assert "strong" in low2                                                 # comfortably over -> strong


@pytest.mark.blocking
def test_groundedness_synthesis_matrix_cells(main_module, monkeypatch, tmp_path):
    """The synthesis places the scene in the Declared-vs-Operative matrix: declared-right +
    advice-moves = grounded; declared-right + advice-doesn't = masquerade; pending until the
    GT-core suppression is run."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True},
                    {"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
          "edges": [{"source": "fire_1", "via_state": "burning", "effect": "may_harm", "target": "person_1"},
                    {"source": "fire_1", "via_state": "burning", "effect": "may_spread_to", "target": "tanker_1"},
                    {"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8,
            "detected_objects": [{"object_id": "fire_1", "label": "fire", "state": "burning"},
                                 {"object_id": "tanker_1", "label": "tanker", "state": "leaking"},
                                 {"object_id": "person_1", "label": "person", "state": "standing"}],
            "causal_graph": {"nodes": [{"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True},
                                       {"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                                       {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
                                       {"source": "fire_1", "target": "tanker_1", "effect": "may_spread_to", "via_state": "burning"},
                                       {"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "fire_1", "state": "burning", "outgoing_edge_count": 2},
                                                         {"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "graph_b": {"nodes": [{"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True}],
                        "edges": [{"source": "fire_1", "target": "fire_1", "effect": "worsens", "via_state": "burning"}]},
            "recommendations": []}

    def _try(cell, content, direction):
        return {"selection": {"target_object_id": "fire_1"}, "verdict": {"cell": cell},
                "u_check": {"leaked": False}, "signals": {"content_shift": content, "graph_direction": direction}}

    assert m._synthesis_cell(norm, {"tries": []})["cell"] == "pending"                       # GT core not suppressed yet
    g = m._synthesis_cell(norm, {"tries": [_try("grounded", 0.6, "de-escalated")]})
    assert g["cell"] == "grounded" and g["declared_match"] and g["operative_match"]
    assert g["gt_rank"][0]["object_id"] == "fire_1"                                           # GT ranks fire top
    mq = m._synthesis_cell(norm, {"tries": [_try("masquerade", 0.1, "unchanged")]})
    assert mq["cell"] == "masquerade" and mq["operative_match"] is False

    txt = _flatten_text(m.make_groundedness_synthesis_panel(norm, {"tries": [_try("grounded", 0.6, "de-escalated")]})).lower()
    assert "grounded" in txt and "masquerade" in txt and "ground truth" in txt               # headline + matrix + evidence
    assert "+0.60" in txt and "operative" in txt                                              # operative strength shown


@pytest.mark.blocking
def test_synthesis_shows_graph_a_and_b_ranks_separately(main_module, monkeypatch, tmp_path):
    """The evidence card shows BOTH Graph A and Graph B rankings as distinct, correctly-labeled
    columns (bug #3: 'Graph A' was silently borrowing Graph B via an `a_rank or b_rank`
    fallback, and A was empty because intervention_candidates defaulted to []). Graph A must
    rank from its OWN edges and show its OWN top hazard, not B's."""
    m = main_module
    import json
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps({
        "image_filename": "s.jpg", "caption": "c",
        "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True}],
        "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"}]}))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    # Graph A: intervention_candidates EMPTY (the default) -> must rank from edges. A's top by
    # edge count is tanker_truck_1 (2 edges). Graph B names a DIFFERENT top hazard (fire_1).
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8,
            "detected_objects": [{"object_id": "tanker_truck_1", "label": "tanker_truck", "state": "leaking"},
                                 {"object_id": "fire_1", "label": "fire", "state": "burning"}],
            "causal_graph": {
                "nodes": [{"id": "tanker_truck_1", "label": "tanker_truck", "state": "leaking", "hazardous": True},
                          {"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True},
                          {"id": "person_1", "label": "person", "state": "standing", "hazardous": False},
                          {"id": "car_1", "label": "car", "state": "parked", "hazardous": False}],
                "edges": [{"source": "tanker_truck_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"},
                          {"source": "tanker_truck_1", "target": "car_1", "effect": "may_harm", "via_state": "leaking"},
                          {"source": "grass_fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}],
                "intervention_candidates": []},
            "graph_b": {"nodes": [{"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True},
                                  {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
                        "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}]},
            "recommendations": []}
    syn = m._synthesis_cell(norm, {"tries": []})
    assert (syn["a_rank"][0] or {}).get("object_id") == "tanker_truck_1"   # A ranks its OWN edges
    assert (syn["b_rank"][0] or {}).get("object_id") == "fire_1"           # B is distinct, not borrowed
    low = _flatten_text(m.make_groundedness_synthesis_panel(norm, {"tries": []})).lower()
    assert "graph a" in low and "graph b" in low                          # both columns labeled
    assert "recs-coupled" in low and "independent" in low                 # A/B meaning surfaced


@pytest.mark.blocking
def test_synthesis_rejects_uncleaned_operative_signal(main_module, monkeypatch, tmp_path):
    """A GT-core suppression that did NOT apply (hazard persisted) or that ESCALATED danger is
    NOT counted as clean 'acts right' — it reads 'unreliable', not grounded/robust. Escalation
    makes the operative strength negative."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                    {"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
          "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                    {"source": "tanker_1", "via_state": "leaking", "effect": "may_spread_to", "target": "fire_1"},
                    {"source": "fire_1", "via_state": "spreading", "effect": "may_harm", "target": "person_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8,
            "detected_objects": [{"object_id": "fire_1", "label": "fire", "state": "burning"},
                                 {"object_id": "tanker_1", "label": "tanker", "state": "leaking"}],
            "causal_graph": {"nodes": [{"id": "fire_1", "hazardous": True, "state": "burning"},
                                       {"id": "tanker_1", "hazardous": True, "state": "leaking"}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
                                       {"source": "fire_1", "target": "tanker_1", "effect": "may_spread_to", "via_state": "burning"},
                                       {"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "fire_1", "state": "burning", "outgoing_edge_count": 2},
                                                         {"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "graph_b": {"nodes": [{"id": "fire_1", "hazardous": True, "state": "burning"}],
                        "edges": [{"source": "fire_1", "target": "fire_1", "effect": "worsens", "via_state": "burning"}]},
            "recommendations": []}

    def try_(do_not_applied, gdir, rdir):
        return {"selection": {"target_object_id": "tanker_1"},
                "verdict": {"cell": "grounded", "do_not_applied": do_not_applied},
                "u_check": {"leaked": False},
                "signals": {"content_shift": 0.5, "graph_direction": gdir, "rec_direction": rdir}}

    # do() didn't apply -> unreliable (this is the tanker screenshot case)
    assert m._synthesis_cell(norm, {"tries": [try_(True, "de-escalated", "escalated")]},
                             core_rule="above_mean")["cell"] == "unreliable"
    # grounded but recs escalated -> unreliable, and the strength is NEGATIVE
    esc = m._synthesis_cell(norm, {"tries": [try_(False, "de-escalated", "escalated")]},
                            core_rule="above_mean")
    assert esc["cell"] == "unreliable" and esc["operative"]["tanker_1"] < 0
    # clean grounded de-escalation on the GT core ALONE is NOT enough — the competing hazard
    # (fire, which the model declares as top) is untested, so the verdict is 'incomplete'.
    only_tanker = m._synthesis_cell(norm, {"tries": [try_(False, "de-escalated", "de-escalated")]},
                                    core_rule="above_mean")
    assert only_tanker["cell"] == "incomplete" and only_tanker["next_to_test"] == "fire_1"

    def fire_try(content):
        return {"selection": {"target_object_id": "fire_1"}, "verdict": {"cell": "grounded", "do_not_applied": False},
                "u_check": {"leaked": False}, "signals": {"content_shift": content, "graph_direction": "de-escalated", "rec_direction": "de-escalated"}}

    # both tested, tanker (GT core) moves the advice MORE than the fire -> the real 'acts right' -> robust
    both = m._synthesis_cell(norm, {"tries": [try_(False, "de-escalated", "de-escalated"), fire_try(0.3)]},
                             core_rule="above_mean")
    assert both["cell"] == "robust" and both["operative_match"] is True
    # both tested, fire (the DECLARED hazard) moves it more -> advice depends on the wrong one -> ungrounded
    wrong = m._synthesis_cell(norm, {"tries": [try_(False, "de-escalated", "de-escalated"), fire_try(0.9)]},
                              core_rule="above_mean")
    assert wrong["cell"] == "ungrounded" and wrong["operative_match"] is False


@pytest.mark.blocking
def test_synthesis_comparison_cards(main_module, monkeypatch, tmp_path):
    """Beyond the Declared-vs-Operative matrix, the synthesis shows the OTHER pairwise views:
    Declared-vs-GT always; GT-vs-Operative only once a clean suppression exists."""
    m = main_module
    import json
    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                    {"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
          "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                    {"source": "tanker_1", "via_state": "leaking", "effect": "may_spread_to", "target": "fire_1"},
                    {"source": "fire_1", "via_state": "spreading", "effect": "may_harm", "target": "person_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8,
            "detected_objects": [{"object_id": "fire_1", "label": "fire", "state": "burning"},
                                 {"object_id": "tanker_1", "label": "tanker", "state": "leaking"}],
            "causal_graph": {"nodes": [{"id": "fire_1", "hazardous": True, "state": "burning"},
                                       {"id": "tanker_1", "hazardous": True, "state": "leaking"}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
                                       {"source": "fire_1", "target": "tanker_1", "effect": "may_spread_to", "via_state": "burning"},
                                       {"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "fire_1", "state": "burning", "outgoing_edge_count": 2},
                                                         {"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "graph_b": {"nodes": [{"id": "fire_1", "hazardous": True, "state": "burning"}],
                        "edges": [{"source": "fire_1", "target": "fire_1", "effect": "worsens", "via_state": "burning"}]},
            "recommendations": []}

    def try_(oid, content):
        return {"selection": {"target_object_id": oid}, "verdict": {"cell": "grounded", "do_not_applied": False},
                "u_check": {"leaked": False}, "signals": {"content_shift": content, "graph_direction": "de-escalated", "rec_direction": "de-escalated"}}

    # no runs: Declared-vs-GT card present; GT-vs-Operative is the "no clean suppression yet" placeholder
    empty = _flatten_text(m.make_groundedness_synthesis_panel(norm, {"tries": []})).lower()
    assert "model declares vs ground truth" in empty and "ground truth vs operative" in empty
    assert "no clean suppression yet" in empty

    # both hazards suppressed, tanker (GT core) strongest -> GT-vs-Operative says grounded in the right hazard
    both = _flatten_text(m.make_groundedness_synthesis_panel(norm, {"tries": [try_("tanker_1", 0.6), try_("fire_1", 0.3)]})).lower()
    assert "depends most on tanker_1" in both and "right hazard" in both
    # the THIRD pairwise view — Declared-vs-Operative (the masquerade axis) — is present. Here the
    # model DECLARES fire (Graph A top, 2 edges) but its advice depends most on tanker -> gap.
    assert "model declares vs operative" in both
    assert "masquerade gap" in both


@pytest.mark.blocking
def test_synthesis_object_ids_render_hover_bbox_thumbnails(main_module, monkeypatch, tmp_path):
    """Every object_id mentioned across the synthesis cards (per-hazard status, suppression
    comparison, distribution, the three declares/GT/operative comparison cards, AND the evidence
    ranking columns) renders with a hover-bbox crop when the scene image + detected bboxes are
    available — so same-label hazards are distinguishable. Falls back to plain text with no
    image. The evidence rows keep their ranking colour-grade (crop attached via _wrap_hover, no
    coloured pill overlay)."""
    m = main_module
    import json, base64, io
    from PIL import Image

    def _walk(n):
        yield n
        ch = getattr(n, "children", None)
        if isinstance(ch, (list, tuple)):
            for c in ch:
                yield from _walk(c)
        elif ch is not None:
            yield from _walk(ch)

    def _count_class(node, cls):
        return sum(1 for x in _walk(node) if cls in (getattr(x, "className", "") or ""))

    buf = io.BytesIO(); Image.new("RGB", (400, 300), (120, 120, 120)).save(buf, format="PNG")
    img_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    gt = {"image_filename": "s.jpg", "caption": "c",
          "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                    {"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
                    {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
          "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                    {"source": "tanker_1", "via_state": "leaking", "effect": "may_spread_to", "target": "fire_1"},
                    {"source": "fire_1", "via_state": "spreading", "effect": "may_harm", "target": "person_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    det = [{"object_id": "fire_1", "label": "fire", "state": "burning", "bbox": [10, 10, 120, 120]},
           {"object_id": "tanker_1", "label": "tanker", "state": "leaking", "bbox": [150, 20, 260, 140]}]
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8, "detected_objects": det,
            "causal_graph": {"nodes": [{"id": "fire_1", "hazardous": True, "state": "burning"},
                                       {"id": "tanker_1", "hazardous": True, "state": "leaking"}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
                                       {"source": "fire_1", "target": "tanker_1", "effect": "may_spread_to", "via_state": "burning"},
                                       {"source": "tanker_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "fire_1", "state": "burning", "outgoing_edge_count": 2},
                                                         {"threat": "tanker_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "graph_b": {"nodes": [{"id": "fire_1", "hazardous": True, "state": "burning"}],
                        "edges": [{"source": "fire_1", "target": "fire_1", "effect": "worsens", "via_state": "burning"}]},
            "recommendations": []}

    def try_(oid, c):
        return {"selection": {"target_object_id": oid}, "verdict": {"cell": "grounded", "do_not_applied": False},
                "u_check": {"leaked": False},
                "signals": {"content_shift": c, "graph_direction": "de-escalated", "rec_direction": "de-escalated",
                            "hazard_shift": 0.9, "graph_shift": 1.0, "recommendation_shift": 1.0, "hazard_level_delta": -8}}
    hist = {"tries": [try_("tanker_1", 0.6), try_("fire_1", 0.3)]}

    # WITH image + bboxes -> hover chips (wrap + cropped tooltip image) across ALL cards, incl.
    # the evidence ranking columns (GT/A/B/Operative), so the count is well above the
    # per-hazard+comparison+distribution subset alone.
    panel = m.make_groundedness_synthesis_panel(norm, hist, image_contents=img_url, detected_objects=det)
    assert _count_class(panel, "hazard-pill-wrap") >= 15
    assert _count_class(panel, "hazard-tooltip-image") >= 15     # each carries a bbox crop

    # WITHOUT image -> graceful: no chips, but the object_ids are still rendered as text
    panel2 = m.make_groundedness_synthesis_panel(norm, hist, image_contents=None, detected_objects=det)
    assert _count_class(panel2, "hazard-pill-wrap") == 0
    assert "tanker_1" in _flatten_text(panel2)


@pytest.mark.blocking
def test_synthesis_spurious_when_noncore_dominates_despite_right_declaration(main_module, monkeypatch, tmp_path):
    """run_20260707 bug: the model DECLARES the right core (tanker, also Graph A's top) AND the
    tanker suppression moves the advice — the old top-1-competitor logic therefore said
    'grounded'. But suppressing the FIRE (a non-core hazard) moves the advice MORE, so the
    recommendations depend dominantly on the WRONG hazard -> spurious_grounding, and the
    per-hazard status marks tanker grounded / fire spurious (the deconstructed matrix)."""
    m = main_module
    import json
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps({
        "image_filename": "s.jpg", "caption": "c",
        "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                  {"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True},
                  {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
        "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                  {"source": "tanker_1", "via_state": "leaking", "effect": "may_spread_to", "target": "fire_1"},
                  {"source": "fire_1", "via_state": "burning", "effect": "may_harm", "target": "person_1"}]}))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    # Graph A ranks the tanker top too (2 outgoing edges vs the fire's 1) -> declared_match True.
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8,
            "detected_objects": [{"object_id": "tanker_truck_1", "label": "tanker_truck", "state": "leaking"},
                                 {"object_id": "fire_1", "label": "fire", "state": "burning"}],
            "causal_graph": {"nodes": [{"id": "tanker_truck_1", "label": "tanker_truck", "hazardous": True, "state": "leaking"},
                                       {"id": "fire_1", "label": "fire", "hazardous": True, "state": "burning"}],
                             "edges": [{"source": "tanker_truck_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"},
                                       {"source": "tanker_truck_1", "target": "fire_1", "effect": "may_spread_to", "via_state": "leaking"},
                                       {"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}],
                             "intervention_candidates": []},
            "recommendations": []}

    def try_(oid, content):
        return {"selection": {"target_object_id": oid}, "spec": {"target": {"object_id": oid}},
                "verdict": {"cell": "grounded", "do_not_applied": False}, "u_check": {"leaked": False},
                "signals": {"content_shift": content, "graph_direction": "de-escalated", "rec_direction": "de-escalated"}}

    tries = [try_("tanker_truck_1", 0.60), try_("fire_1", 0.93)]
    # rule pinned to above_mean so the fire is a NON-core GT hazard (the scenario under test);
    # under the half_max DEFAULT both are co-core and this correctly reads grounded instead.
    syn = m._synthesis_cell(norm, {"tries": tries}, core_rule="above_mean")
    assert syn["cell"] == "spurious_grounding" and syn["operative_match"] is False
    assert syn["strongest_noncore"] == "fire_1"
    stat = {h["object_id"]: h["status"] for h in syn["hazard_status"]}
    # BINARY: the fire is below the core cut (above_mean), so it reads SPURIOUS (not a core
    # driver) — same vocabulary as the distribution and the per-run verdict.
    assert stat["tanker_truck_1"] == "grounded" and stat["fire_1"] == "spurious"

    low = _flatten_text(m.make_groundedness_synthesis_panel(norm, {"tries": tries},
                                                        core_rule="above_mean")).lower()
    # the HEADLINE is now the distributional verdict (it replaced the single-comparison panel)
    assert "are the recommendations grounded" in low and "misproportioned" in low
    assert "per-hazard status" in low                     # the deconstructed matrix is present
    assert "spurious" in low                               # fire (below the cut) reads spurious
    # REGRESSION (live run 2026-07-08): the GT evidence column must tag the core hazard
    # "· core" — the panel read h["is_gt_core"] but hazard_status stores "is_core", so
    # EVERYTHING rendered "peripheral". The GT column has tanker (core) + fire (peripheral).
    assert "· core" in low and "· peripheral" in low

    # the SEPARATE comparative table: one row per suppressed hazard, columns = signal features,
    # ranked by operative strength (fire, the bigger driver, above the tanker).
    panel = m.make_groundedness_synthesis_panel(norm, {"tries": tries}, core_rule="above_mean")
    full = _flatten_text(panel)
    fl = full.lower()
    assert "suppression comparison" in fl                                  # the comparative card
    for col in ("Hazard Δ", "Graph", "Recs", "Operative", "Result"):
        assert col.lower() in fl                                           # signal-feature columns

    assert "+0.93" in full and "+0.60" in full                            # both operative strengths in the table
    # fire above tanker: fire's operative string appears before tanker's in the flattened order
    assert full.index("+0.93") < full.index("+0.60")

    # the DISTRIBUTIONAL verdict — now the HEADLINE (replaced the single-comparison panel).
    # Both hazards are in GT, so the fire is NOT spurious — the emphasis is just inverted:
    # verdict = misproportioned, zero non-GT leakage, alignment in (0,1).
    d = m._distributional_groundedness(syn)                              # basis/rule come from syn
    assert d["verdict"] == "misproportioned"                              # not the crude "spurious"
    assert d["nongt_op_mass"] == 0.0                                      # fire is a real GT hazard
    assert 0.0 < d["overlap"] < 1.0                                       # partial alignment (inverted emphasis)
    assert d["partitions"]["above_mean"] == {"tanker_truck_1"}           # threshold rules differ...
    assert d["partitions"]["half_max"] == {"tanker_truck_1", "fire_1"}   # ...half-max keeps the fire as co-core
    assert "distribution of shift vs ground truth" in fl                  # the distributional headline renders
    assert "alignment" in fl                                              # its alignment score is shown


@pytest.mark.blocking
def test_half_max_default_makes_close_hazards_co_core(main_module, monkeypatch, tmp_path):
    """DEFAULT cut = half_max, inclusive (>=): on the tanker/fire scene (GT edge weights 2 vs 1,
    thr = 1) BOTH are co-core. The fire's suppression then reads GROUNDED (not secondary), the
    fire dominating the operative axis is an emphasis question (misproportioned), NOT a
    wrong-hazard finding, and the GT-vs-Operative card says right-hazard. (Sunny, 2026-07-10:
    'cut to half-of-max as default and everything equal or greater should be included.')"""
    m = main_module
    import json
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps({
        "image_filename": "s.jpg", "caption": "c",
        "nodes": [{"id": "tanker_1", "label": "tanker", "state": "leaking", "hazardous": True},
                  {"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True},
                  {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
        "edges": [{"source": "tanker_1", "via_state": "leaking", "effect": "may_harm", "target": "person_1"},
                  {"source": "tanker_1", "via_state": "leaking", "effect": "may_spread_to", "target": "fire_1"},
                  {"source": "fire_1", "via_state": "burning", "effect": "may_harm", "target": "person_1"}]}))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 8,
            "detected_objects": [{"object_id": "tanker_truck_1", "label": "tanker_truck", "state": "leaking"},
                                 {"object_id": "fire_1", "label": "fire", "state": "burning"}],
            "causal_graph": {"nodes": [{"id": "tanker_truck_1", "label": "tanker_truck", "hazardous": True, "state": "leaking"},
                                       {"id": "fire_1", "label": "fire", "hazardous": True, "state": "burning"}],
                             "edges": [{"source": "tanker_truck_1", "target": "person_1", "effect": "may_harm", "via_state": "leaking"},
                                       {"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}],
                             "intervention_candidates": []},
            "recommendations": []}

    def try_(oid, content):
        return {"selection": {"target_object_id": oid}, "spec": {"target": {"object_id": oid}},
                "verdict": {"cell": "grounded", "do_not_applied": False}, "u_check": {"leaked": False},
                "signals": {"content_shift": content, "graph_direction": "de-escalated", "rec_direction": "de-escalated"}}

    tries = [try_("tanker_truck_1", 0.60), try_("fire_1", 0.93)]
    syn = m._synthesis_cell(norm, {"tries": tries})                       # DEFAULT rule
    assert set(syn["gt_core_ids"]) == {"tanker_truck_1", "fire_1"}       # co-core under half_max
    stat = {h["object_id"]: h["status"] for h in syn["hazard_status"]}
    assert stat["fire_1"] == "grounded"                                   # co-core + moved = grounded
    assert syn["strongest_noncore"] is None                               # no NON-core hazard tested
    assert syn["cell"] != "spurious_grounding"                            # dominance of a co-core is not wrong-hazard

    low = _flatten_text(m.make_groundedness_synthesis_panel(norm, {"tries": tries})).lower()
    assert "grounded in the wrong hazard" not in low                      # GT-vs-Op card: right hazard
    assert "which ground truth ranks as a core hazard" in low             # set-membership wording


@pytest.mark.blocking
def test_synthesis_core_threshold_toggle_edge_vs_consequence(main_module, monkeypatch, tmp_path):
    """The synthesis core-threshold TOGGLE flips the core set live: a hazard with more edges but
    to PROPERTY vs a hazard with fewer edges but to a PERSON swap core status between the
    edge-count basis and the consequence (victim-cost) basis. Default stays edge-count."""
    m = main_module
    import json
    # debris: 3 edges, all to property (cars) -> edge 3, consequence 3. gunman: 2 edges, both to
    # PEOPLE -> edge 2, consequence 4. So edge-count ranks debris top; consequence ranks gunman top.
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps({
        "image_filename": "s.jpg", "caption": "c",
        "nodes": [{"id": "debris_1", "label": "debris", "state": "scattered", "hazardous": True},
                  {"id": "gunman_1", "label": "gunman", "state": "armed", "hazardous": True},
                  {"id": "car_1", "label": "car", "state": "parked", "hazardous": False},
                  {"id": "car_2", "label": "car", "state": "parked", "hazardous": False},
                  {"id": "car_3", "label": "car", "state": "parked", "hazardous": False},
                  {"id": "person_1", "label": "person", "state": "standing", "hazardous": False},
                  {"id": "person_2", "label": "person", "state": "standing", "hazardous": False}],
        "edges": [{"source": "debris_1", "via_state": "scattered", "effect": "may_harm", "target": "car_1"},
                  {"source": "debris_1", "via_state": "scattered", "effect": "may_harm", "target": "car_2"},
                  {"source": "debris_1", "via_state": "scattered", "effect": "may_harm", "target": "car_3"},
                  {"source": "gunman_1", "via_state": "armed", "effect": "may_harm", "target": "person_1"},
                  {"source": "gunman_1", "via_state": "armed", "effect": "may_harm", "target": "person_2"}]}))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 7,
            "detected_objects": [{"object_id": "debris_1", "label": "debris", "state": "scattered"},
                                 {"object_id": "gunman_1", "label": "gunman", "state": "armed"}],
            "causal_graph": {"nodes": [{"id": "debris_1", "label": "debris", "state": "scattered", "hazardous": True},
                                       {"id": "gunman_1", "label": "gunman", "state": "armed", "hazardous": True}],
                             "edges": [{"source": "debris_1", "target": "car_1", "effect": "may_harm", "via_state": "scattered"},
                                       {"source": "gunman_1", "target": "person_1", "effect": "may_harm", "via_state": "armed"}],
                             "intervention_candidates": []},
            "recommendations": []}

    # EDGE-count basis (default): debris (3 edges) dominates -> debris is core, gunman peripheral.
    edge = m._synthesis_cell(norm, {"tries": []}, core_basis="edge", core_rule="above_mean")
    assert "debris_1" in edge["gt_core_ids"] and "gunman_1" not in edge["gt_core_ids"]
    # CONSEQUENCE basis: the gunman threatens a PERSON (weighted x2), closing the gap so it is
    # co-core; the core set differs from the edge-count basis.
    cons = m._synthesis_cell(norm, {"tries": []}, core_basis="consequence", core_rule="above_mean")
    assert cons["gt_core_ids"] != edge["gt_core_ids"]
    assert "gunman_1" in cons["gt_core_ids"]

    # the toggle value flows through the panel and its active basis is shown
    low = _flatten_text(m.make_groundedness_synthesis_panel(
        norm, {"tries": []}, core_basis="consequence", core_rule="half_max")).lower()
    assert "weight by consequence" in low and "cut at half-max" in low


@pytest.mark.blocking
def test_synthesis_keys_operative_by_resolved_target_not_raw_pick(main_module, monkeypatch, tmp_path):
    """REGRESSION: the model splits the fire's id (node 'fire_1' vs edge/pick 'grass_fire_1'). A
    try whose UI pick is 'grass_fire_1' but whose RESOLVED spec target is 'fire_1' must key its
    operative strength under 'fire_1' (the candidate id), so the fire reads as TESTED in the
    per-hazard status — not untested with a phantom 'grass_fire_1' row."""
    m = main_module
    import json
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps({
        "image_filename": "s.jpg", "caption": "c",
        "nodes": [{"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True},
                  {"id": "person_1", "label": "person", "state": "standing", "hazardous": False}],
        "edges": [{"source": "fire_1", "via_state": "burning", "effect": "may_harm", "target": "person_1"}]}))
    monkeypatch.setattr(m, "GT_VERIFIED_DIR", tmp_path)
    norm = {"image_filename": "s.jpg", "caption": "c", "disaster_level": 6,
            "detected_objects": [{"object_id": "fire_1", "label": "fire", "state": "burning"}],
            "causal_graph": {"nodes": [{"id": "fire_1", "label": "fire", "state": "burning", "hazardous": True}],
                             "edges": [{"source": "fire_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"}],
                             "intervention_candidates": []},
            "recommendations": []}
    # UI pick 'grass_fire_1' (the edge id) but the pipeline RESOLVED it to the candidate 'fire_1'.
    tries = [{"selection": {"target_object_id": "grass_fire_1"},
              "spec": {"target": {"object_id": "fire_1"}},
              "verdict": {"cell": "grounded", "do_not_applied": False}, "u_check": {"leaked": False},
              "signals": {"content_shift": 0.7, "graph_direction": "de-escalated", "rec_direction": "de-escalated"}}]
    syn = m._synthesis_cell(norm, {"tries": tries})
    assert "fire_1" in syn["operative"] and "grass_fire_1" not in syn["operative"]   # keyed by resolved id
    stat = {h["object_id"]: h["status"] for h in syn["hazard_status"]}
    assert stat.get("fire_1") != "untested"                                          # the fire reads as tested


def test_distributional_masquerade_requires_core_to_be_tested(main_module):
    """REGRESSION: a 0 core operative mass because the CORE hazard was never suppressed must read
    'pending' (test the driver), NOT 'masquerade' (which claims the advice ignores a driver you
    actually measured). Only a TESTED core that barely moves the advice is masquerade."""
    m = main_module
    base = {"gt_core_ids": {"tanker_1"}, "core_basis": "edge", "core_rule": "above_mean",
            "gt_edge_weights": {"tanker_1": 2.0, "fire_1": 1.0},
            "gt_consequence_weights": {"tanker_1": 2.0, "fire_1": 1.0}}
    # core (tanker) UNtested, a secondary (fire) tested and moving -> pending, not masquerade
    untested_core = {**base, "hazard_status": [
        {"object_id": "tanker_1", "is_core": True, "has_gt": True, "operative": None, "ranks": {"GT": 1}},
        {"object_id": "fire_1", "is_core": False, "has_gt": True, "operative": 0.8, "ranks": {"GT": 2}}]}
    assert m._distributional_groundedness(untested_core)["verdict"] == "pending"
    # core tested (op 0 -> advice did not move) -> genuinely masquerade
    tested_core = {**base, "hazard_status": [
        {"object_id": "tanker_1", "is_core": True, "has_gt": True, "operative": 0.0, "ranks": {"GT": 1}},
        {"object_id": "fire_1", "is_core": False, "has_gt": True, "operative": 0.8, "ranks": {"GT": 2}}]}
    assert m._distributional_groundedness(tested_core)["verdict"] == "masquerade"


def test_history_ignored_when_run_key_mismatches_scene(main_module):
    """REGRESSION: the history store is only reset inside apply_intervention. The synthesis/setup
    render callbacks must ignore a PREVIOUS scene's tries so a freshly-loaded scene does not show
    the old scene's operative/verdict before the user applies anything."""
    m = main_module
    scene_b = {"image_filename": "scene_b.jpg", "caption": "b"}
    stale = {"run_key": "scene_a.jpg", "tries": [{"selection": {"target_object_id": "x"}}]}
    assert m._history_for_scene(stale, scene_b) == {"tries": []}          # foreign history dropped
    matching = {"run_key": "scene_b.jpg", "tries": [{"selection": {"target_object_id": "x"}}]}
    assert m._history_for_scene(matching, scene_b) is matching            # own history kept
    assert m._history_for_scene({"tries": []}, scene_b) == {"tries": []}  # empty history is fine


# ---------------------------------------------------------------------------
# Single-run page: threats display survives the id split + upload thumbnail
# (live 2026-07-10: the fire was missing from Threats & Risks — the model listed it as
# grass_fire_1 in threats but fire_1 in detected_objects, the exact-id join left it bbox-less,
# and the card loop silently dropped bbox-less threats.)
# ---------------------------------------------------------------------------
@pytest.mark.blocking
def test_normalize_threats_backfills_by_family_and_state_keeping_id(main_module):
    """A threat named under a split id (grass_fire_1) backfills label/bbox from the detected
    object sharing its canonical FAMILY and STATE (fire_1, burning) — but KEEPS its original
    object_id, so orphan-threat / conformance measurement is unchanged (display-only fix)."""
    m = main_module
    det = [{"object_id": "fire_1", "label": "fire", "state": "burning", "bbox": [1, 2, 3, 4]},
           {"object_id": "person_1", "label": "person", "state": "standing", "bbox": [5, 6, 7, 8]}]
    out = m.normalize_threats([{"object_id": "grass_fire_1", "state": "burning", "reason": "r"}],
                              detected_objects=det)
    t = out[0]
    assert t["object_id"] == "grass_fire_1"          # id kept: incoherence stays detectable
    assert t["bbox"] == [1, 2, 3, 4]                 # bbox backfilled from fire_1
    assert t["label"] == "fire"
    # state-gated: a family match with a DISAGREEING state does NOT backfill (stays bbox-less).
    # (Note 'smoldering' would not do here — canonicalize_state folds it into 'burning'.)
    out2 = m.normalize_threats([{"object_id": "grass_fire_1", "state": "extinguished", "reason": "r"}],
                               detected_objects=det)
    assert out2[0]["bbox"] is None


@pytest.mark.blocking
def test_hazard_thumbnails_render_bboxless_threats(main_module):
    """A threat without a bbox renders a card (marked 'no box') instead of being silently
    dropped — a declared hazard must never vanish from the Threats & Risks section."""
    m = main_module
    threats = [{"object_id": "grass_fire_1", "label": "fire", "state": "burning",
                "bbox": None, "reason": "burning near the road"}]
    cards = m.make_hazard_thumbnails(None, threats)
    txt = _flatten_text(cards).lower()
    assert "fire" in txt and "burning" in txt                       # the threat is visible
    assert "no matching detected object" in txt                     # and honestly annotated
    assert "empty-state" not in " ".join(getattr(c, "className", "") or "" for c in cards)


@pytest.mark.blocking
def test_at_risk_thumbnails_render_bboxless_victims(main_module):
    """A VICTIM without a bbox renders a card (marked 'no box') instead of being dropped — same
    fix as threats (I27), mirrored for at-risk. push_06: the model returned at-risk entities
    under phantom ids (child_swimmer_struggling_in_pool_1) with no bbox, so the panel silently
    showed 'nothing in the at-risk'."""
    m = main_module
    at_risk = [{"object_id": "child_swimmer_struggling_in_pool_1", "label": "child",
                "state": "struggling", "bbox": None, "category": "distress",
                "reason": "the child is struggling in the pool"},
               {"object_id": "child_floats_nearby_1", "label": "child", "state": "floating",
                "bbox": None, "category": "misclassified", "reason": "no justification"}]
    cards = m.make_at_risk_thumbnails(None, at_risk)
    assert len(cards) == 2                                           # BOTH victims render, none dropped
    txt = _flatten_text(cards).lower()
    assert "struggling" in txt and "floating" in txt
    assert "no matching detected object" in txt                     # honest phantom-id note
    assert "distress" in txt and "schema violation" in txt          # categories still shown
    assert "returned without valid bounding boxes" not in txt       # the old drop-message is gone


@pytest.mark.blocking
def test_intervention_image_upload_preview(main_module):
    """The setup panel exposes a preview container, and the callback renders a thumbnail of the
    uploaded counterfactual image (so what was added stays visible)."""
    m = main_module
    setup = m.make_intervention_setup_panel(
        {"should_be_core": {"object_id": "t1", "state": "leaking", "label": "tanker"},
         "candidates": [{"object_id": "t1", "label": "tanker", "state": "leaking"}]},
        [{"object_id": "t1", "label": "tanker", "state": "leaking"}], [], {}, "cap")
    ids = []

    def collect(n):
        cid = getattr(n, "id", None)
        if cid:
            ids.append(cid)
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                collect(c)
    collect(setup)
    assert "intervention-image-preview" in ids                      # container present

    out = m.show_intervention_image_preview("data:image/png;base64,AAA", "cf.png")
    flat = _flatten_text(out)
    assert "cf.png" in flat and "counterfactual" in flat.lower()    # filename + meaning shown
    # the actual <img> carries the uploaded data URL
    srcs = []

    def imgs(n):
        if getattr(n, "src", None):
            srcs.append(n.src)
        ch = getattr(n, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch] if ch is not None else []):
            if not isinstance(c, str):
                imgs(c)
    imgs(out)
    assert srcs == ["data:image/png;base64,AAA"]
    assert m.show_intervention_image_preview(None, None) is None    # no upload -> no thumbnail


@pytest.mark.blocking
def test_normalize_at_risk_backfills_via_shared_resolver(main_module):
    """normalize_at_risk_objects has the same exact-id join as threats did — it now backfills
    label/bbox via the SHARED resolver (state-gated family co-reference), keeping the original
    object_id so measurement still sees the id incoherence."""
    m = main_module
    det = [{"object_id": "person_1", "label": "person", "state": "trapped", "bbox": [9, 9, 12, 12]}]
    out = m.normalize_at_risk_objects([{"object_id": "victim_1", "state": "trapped", "reason": "r"}],
                                      detected_objects=det)
    t = out[0]
    assert t["bbox"] == [9, 9, 12, 12]                   # victim folds to person (family map)
    assert t["object_id"] == "victim_1"                  # id kept: incoherence stays detectable
    assert t["label"] == "person"
    # a state-disagreeing alias never folds
    out2 = m.normalize_at_risk_objects([{"object_id": "victim_1", "state": "sleeping", "reason": "r"}],
                                       detected_objects=det)
    assert out2[0]["bbox"] is None


@pytest.mark.blocking
def test_failed_apply_records_no_try_and_calls_no_model(main_module, monkeypatch):
    """An Apply whose input did NOT actually change (here: 'both' with the caption pasted back
    verbatim) short-circuits to the input-problem notice WITHOUT calling the model and WITHOUT
    appending a try — the history append was re-rendering the setup panel and wiping the
    user's in-progress mode/modality/caption/upload right when they needed to fix and retry
    (live 2026-07-10: 'The options in the pic changed after the message')."""
    m = main_module
    import dash
    monkeypatch.setattr(m, "normalize_result", lambda d, ic=None: d)
    calls = []
    monkeypatch.setattr(m, "query_qwen", lambda *a, **k: calls.append(1) or {})
    data = {"run_id": "run_X", "caption": "orig caption", "disaster_level": 7,
            "detected_objects": [{"object_id": "t_1", "label": "tanker", "state": "leaking"}],
            "causal_graph": {"nodes": [{"id": "t_1", "label": "tanker", "state": "leaking", "hazardous": True}],
                             "edges": [{"source": "t_1", "target": "p_1", "effect": "may_harm", "via_state": "leaking"}],
                             "intervention_candidates": [{"threat": "t_1", "state": "leaking", "outgoing_edge_count": 1}]},
            "recommendations": []}
    # caption pasted back VERBATIM (only whitespace differs) + a genuinely different image.
    card, hist, status = m.apply_intervention(
        1, data, "data:image/png;base64,ORIG", "t_1", "retrospective", "both",
        "  orig caption  ", "data:image/png;base64,EDITED", {"run_key": "run_X", "tries": []},
        "edge", "half_max")
    assert calls == []                                        # the model was never called
    assert hist is dash.no_update                             # no try recorded -> no panel reset
    low = _flatten_text(card).lower()
    assert "no intervention to measure" in low                # the plain input-problem notice
    assert "verdict" not in low                               # never dressed as a verdict


# ---------------------------------------------------------------------------
# Tab-based intervention history (click a chip -> that try's full result panel)
# ---------------------------------------------------------------------------
def _mk_try(oid, cell, content=0.5):
    return {"selection": {"target_object_id": oid, "mode": "retrospective", "modality": "both"},
            "spec": {"target": {"object_id": oid}},
            "verdict": {"cell": cell, "explanation": "x"},
            "u_check": {"leaked": False, "applied": True},
            "signals": {"content_shift": content, "hazard_shift": 0.2, "graph_shift": 0.5,
                        "recommendation_shift": 0.8, "node_shift": 0.1, "edge_shift": 0.5,
                        "hazard_level_delta": -3, "rec_direction": "de-escalated",
                        "graph_direction": "de-escalated"},
            "before": {"hazard_level": 8,
                       "detected_objects": [{"object_id": oid, "label": oid.rsplit("_", 1)[0], "state": "leaking"}],
                       "graph_a": {"nodes": [{"id": oid, "state": "leaking"}], "edges": []},
                       "recommendations": [{"action": "Move person_1 away",
                                            "structured_reasoning": {"affected_objects": ["person_1"]}}]},
            "after": {"hazard_level": 5,
                      "detected_objects": [{"object_id": oid, "label": oid.rsplit("_", 1)[0], "state": "stationary"}],
                      "graph_a": {"nodes": [{"id": oid, "state": "stationary"}], "edges": []},
                      "recommendations": [{"action": "Monitor the area",
                                           "structured_reasoning": {"affected_objects": ["person_1"]}}]}}


@pytest.mark.blocking
def test_history_tab_strip_and_selection(main_module):
    """The chip strip renders one verdict-coloured chip per stored try (hidden below 2 tries);
    a stored click is honoured only against the CURRENT try count, so a new Apply snaps the
    selection back to the latest try."""
    m = main_module
    tries = [_mk_try("tanker_1", "grounded"), _mk_try("fire_1", "secondary_grounding")]
    strip = m._history_tab_strip(tries, 2)
    txt = _flatten_text(strip)
    assert "1 · tanker_1 · Grounded" in txt and "2 · fire_1 · Secondary hazard" in txt
    assert m._history_tab_strip(tries[:1], 1) is None                 # a single try needs no tabs
    # selection: honoured while the count matches; snaps to latest when a new try lands
    assert m._history_tab_selection({"idx": 1, "n": 2}, 2) == 1       # click on try 1 sticks
    assert m._history_tab_selection({"idx": 1, "n": 2}, 3) == 3       # new Apply -> latest
    assert m._history_tab_selection(None, 2) == 2                     # no click yet -> latest
    assert m._history_tab_selection({"idx": 9, "n": 2}, 2) == 2       # bogus index -> latest


@pytest.mark.blocking
def test_render_history_try_rebuilds_full_panel_from_record(main_module):
    """A stored record re-renders the FULL result panel (verdict + hazard 8→5 + graph/recs
    diffs) with no re-run; an OLDER try carries the 'Viewing try N of M' note, the latest
    doesn't. The record's after-keys (hazard_level/graph_a) hit the panel's own fallbacks."""
    m = main_module
    rec = _mk_try("tanker_1", "grounded")
    older = _flatten_text(m.render_history_try(rec, 1, 2)).lower()
    assert "viewing try 1 of 2" in older                              # history marker
    assert "verdict" in older and "grounded" in older                 # full verdict card
    assert "hazard level" in older and "monitor the area" in older    # before/after content
    latest = _flatten_text(m.render_history_try(rec, 2, 2)).lower()
    assert "viewing try" not in latest                                # latest = no marker
