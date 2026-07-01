"""Eval-for-CODE (hermetic) for the CEE+ intervention pipeline (intervention.py).

Authored from the FROZEN CONTRACT in INTERVENTION_WORKFLOW.md and the plan in
INTERVENTION_PLAN.md ONLY — never from the Builder's implementation (differential
testing: code-from-spec vs tests-from-spec, run together, surface any
spec-interpretation mismatch).

Every test here is hermetic:
  - No VLM. Where a function needs one, we pass a stub `vlm_fn` callable.
  - GT is supplied either by passing a tmp `gt_dir` to `intervention_baseline`,
    or by monkeypatching `intervention.GT_VERIFIED_DIR` at a tmp dir. We NEVER
    touch the gitignored `exports/ground_truth/verified/`.

Coverage map (contract -> test):
  step 0  intervention_baseline    : loads gt_graph by filename (not passthrough);
                                      carries image_data_url; hazard_level<-disaster_level
  step 1  enumerate_candidates     : A/B/GT cores; deterministic ranking; control
  step 2  build_intervention_spec  : #2 type map auto-default; modality verbatim
  step 3  render_do_prompt         : contains target id + verb; NO GT-specific leak
  step 4  run_counterfactual       : calls injected vlm_fn; returns ONLY light post
  step 5  check_u_preservation     : object-id Jaccard; leaked when < U_CUTOFF
  step 6  compute_shifts           : 5 deltas in [0,1]; identical->0; reworded-same->0
  step 7  adjudicate_groundedness  : the 2x2 oracle + no-GT not_adjudicable
  step 8  compare_to_control       : core-shift > control-shift -> discriminates

The four 2x2 oracle cases + the fifth no-GT case are the heart of the eval.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# The module under test. It must import cleanly with NO `import main` at load
# (contract constraint #8: top-level import of main would be circular).
intervention = pytest.importorskip(
    "intervention",
    reason="intervention.py not yet built (Builder writes it in parallel).",
)


# ---------------------------------------------------------------------------
# Stubs and hand-built fixtures (all hermetic — no VLM, no real GT dir)
# ---------------------------------------------------------------------------
def make_vlm_stub(return_value: dict):
    """A fake vlm_fn: ignores its args, returns a fixed raw VLM payload.
    Records calls so tests can assert run_counterfactual actually invoked it."""
    calls = []

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return return_value

    _stub.calls = calls
    return _stub


# A GT answer-key graph in the real `.gt.json` shape. The caption and the
# `water_1` object id are GT-SPECIFIC strings used by the leak-guard test:
# they must NOT appear in the do-prompt (contract: render_do_prompt uses no
# gt_graph content). "fire"/"house"/generic labels are deliberately NOT used
# as the leak probe because the model's own output legitimately contains them.
GT_GRAPH = {
    "image_filename": "push_06_drowning_pool.jpg",
    "caption": "ZZ_SECRET_ANSWERKEY_CAPTION_engulfing_water_drowning_children_marker",
    "schema_version": "2026-06-10",
    "nodes": [
        {"id": "water_1", "label": "water", "state": "engulfing", "hazardous": True, "inferred": False},
        {"id": "child_1", "label": "child", "state": "drowning", "hazardous": False, "inferred": False},
        {"id": "child_2", "label": "child", "state": "unconscious", "hazardous": False, "inferred": False},
    ],
    "edges": [
        {"source": "water_1", "target": "child_1", "effect": "may_harm", "via_state": "engulfing"},
        {"source": "water_1", "target": "child_2", "effect": "may_harm", "via_state": "engulfing"},
    ],
}

GT_ONLY_OBJECT_ID = "water_1"          # appears in GT, NOT in the model result below
GT_ONLY_CAPTION = GT_GRAPH["caption"]  # GT-specific caption string


def write_gt_dir(tmp_path: Path, gt: dict = GT_GRAPH) -> Path:
    """Materialize a tmp verified-GT dir holding one `.gt.json` keyed by filename."""
    gt_dir = tmp_path / "verified_gt"
    gt_dir.mkdir()
    (gt_dir / f"{gt['image_filename']}.gt.json").write_text(json.dumps(gt))
    return gt_dir


def make_result(image_filename: str = "push_06_drowning_pool.jpg",
                disaster_level: int = 8) -> dict:
    """A normalized-result-shaped dict (the input to intervention_baseline).

    Field names match main.py's result schema: `causal_graph` is Graph A,
    `graph_b` is Graph B, `disaster_level` (0-10) maps to hazard_level.
    The model's OWN object ids deliberately differ from the GT's (`flood_1`
    vs `water_1`) so the leak-guard probe (`water_1`) is GT-specific.
    """
    return {
        "run_id": "run_test_1",
        "image_filename": image_filename,
        "prompt": "analyze this scene",
        "caption": "a flooded area",
        "disaster_level": disaster_level,
        "detected_objects": [
            {"object_id": "flood_1", "label": "water", "state": "engulfing"},
            {"object_id": "person_1", "label": "person", "state": "wading"},
        ],
        "threats": [{"object_id": "flood_1", "state": "engulfing"}],
        "recommendations": [
            {"rank": 1, "action": "Move person_1 away from flood_1.",
             "related_object_ids": ["flood_1", "person_1"],
             # A7: the rec quad is nested under structured_reasoning, matching main.py's
             # real schema (main.py:285-298). _rec_quads reads ONLY structured_reasoning,
             # so flat top-level threat/state/affected keys would make every quad empty and
             # recommendation_shift trivially 0 — passing the rewording test for the wrong
             # reason. Nesting it here makes _rec_quads actually fire.
             "structured_reasoning": {
                 "threat": "flood_1", "state": "engulfing",
                 "effect": "may_harm", "affected_objects": ["person_1"]}},
        ],
        "causal_graph": {
            "nodes": [
                {"id": "flood_1", "label": "water", "state": "engulfing", "hazardous": True},
                {"id": "person_1", "label": "person", "state": "wading", "hazardous": False},
            ],
            "edges": [
                {"source": "flood_1", "target": "person_1", "effect": "may_harm", "via_state": "engulfing"},
            ],
            "intervention_candidates": [
                {"threat": "flood_1", "state": "engulfing", "outgoing_edge_count": 1},
            ],
        },
        "graph_b": {
            "nodes": [
                {"id": "flood_1", "label": "water", "state": "engulfing", "hazardous": True},
                {"id": "person_1", "label": "person", "state": "wading", "hazardous": False},
            ],
            "edges": [
                {"source": "flood_1", "target": "person_1", "effect": "may_harm", "via_state": "engulfing"},
            ],
            "suppression_pick": {"threat": "flood_1", "state": "engulfing", "reason": "primary hazard"},
        },
    }


# ---------------------------------------------------------------------------
# step 0 — intervention_baseline
# ---------------------------------------------------------------------------
def test_baseline_loads_gt_by_filename_not_passthrough(tmp_path):
    """gt_graph is LOADED from gt_dir by image_filename — it is NOT a passthrough
    of result['gt_validation'] (which only holds the comparison, not the graph)."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result()
    # Poison any passthrough source: result has no gt_graph key at all.
    assert "gt_graph" not in result
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)

    assert baseline["gt_graph"] is not None
    gt_ids = {n["id"] for n in baseline["gt_graph"]["nodes"]}
    assert GT_ONLY_OBJECT_ID in gt_ids  # came from the loaded answer key, not the result


def test_baseline_gt_none_when_no_file(tmp_path):
    """No verified GT for this filename -> gt_graph is None (contract rule #4)."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result(image_filename="image_with_no_gt_xyz.jpg")
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    assert baseline["gt_graph"] is None


def test_baseline_carries_image_data_url_and_maps_hazard_level(tmp_path):
    """Carries the passed-in image_data_url verbatim; hazard_level <- disaster_level."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result(disaster_level=7)
    baseline = intervention.intervention_baseline(result, image_data_url="data:my-image", gt_dir=gt_dir)
    assert baseline["image_data_url"] == "data:my-image"
    assert baseline["hazard_level"] == 7


def test_baseline_default_gt_dir_is_monkeypatchable(tmp_path, monkeypatch):
    """When gt_dir is None, baseline reads from intervention.GT_VERIFIED_DIR,
    which a test can point at a tmp dir (never the gitignored real exports)."""
    gt_dir = write_gt_dir(tmp_path)
    monkeypatch.setattr(intervention, "GT_VERIFIED_DIR", gt_dir)
    result = make_result()
    baseline = intervention.intervention_baseline(result, image_data_url="data:img")  # gt_dir defaulted
    assert baseline["gt_graph"] is not None
    assert GT_ONLY_OBJECT_ID in {n["id"] for n in baseline["gt_graph"]["nodes"]}


# ---------------------------------------------------------------------------
# step 1 — enumerate_candidates
# ---------------------------------------------------------------------------
def _baseline_with_gt(tmp_path):
    gt_dir = write_gt_dir(tmp_path)
    return intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)


def test_enumerate_has_cores_and_should_be_core_with_gt(tmp_path):
    """A/B/GT cores present (their graphs have hazards); should_be_core is the GT
    core (gt_graph present)."""
    baseline = _baseline_with_gt(tmp_path)
    enum = intervention.enumerate_candidates(baseline)
    assert enum["declared_core_a"] is not None
    assert enum["declared_core_b"] is not None
    assert enum["should_be_core"] is not None  # GT present
    assert len(enum["candidates"]) >= 1
    for c in enum["candidates"]:
        assert "hazard_class" in c and "is_should_be_core" in c and "ranks" in c


def test_enumerate_should_be_core_none_without_gt(tmp_path):
    """No GT -> should_be_core None (contract: row undetermined without answer key)."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result(image_filename="no_gt_here.jpg")
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    enum = intervention.enumerate_candidates(baseline)
    assert enum["should_be_core"] is None


def test_enumerate_ranking_is_deterministic(tmp_path):
    """A5 determinism: same input -> same candidate order, every time (no
    set-iteration nondeterminism in the ranking)."""
    baseline = _baseline_with_gt(tmp_path)
    orders = []
    for _ in range(5):
        enum = intervention.enumerate_candidates(baseline)
        orders.append([(c["object_id"], c["state"]) for c in enum["candidates"]])
    assert all(o == orders[0] for o in orders)


def test_gt_ranking_prefers_cascade_root_over_high_fanout(tmp_path=None):
    """B4: rank(GT) must prefer the cascade ROOT (the hazard no other hazard edge points at)
    over a high-fan-out DOWNSTREAM node, instead of ranking purely by raw outgoing-edge count
    ('merely most-edges'). Here house_1 is the origin (in_degree 0) with 2 outgoing edges,
    house_2 is a downstream fan-out (fed by house_1) with 4 outgoing edges. The plain
    edge-count rule would put house_2 first; the root preference must put house_1 first."""
    graph = {
        "nodes": [
            {"id": "house_1", "label": "house", "state": "burning", "hazardous": True},
            {"id": "house_2", "label": "house", "state": "burning", "hazardous": True},
            {"id": "house_3", "label": "house", "state": "burning", "hazardous": True},
            {"id": "car_1", "label": "car", "state": "burning", "hazardous": True},
            {"id": "person_1", "label": "person", "state": "fleeing", "hazardous": False},
        ],
        "edges": [
            # house_1 is the ignition root: nothing points AT it.
            {"source": "house_1", "target": "house_2", "effect": "may_harm", "via_state": "burning"},
            {"source": "house_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
            # house_2 is downstream (fed by house_1) yet fans out the most (4 edges).
            {"source": "house_2", "target": "house_3", "effect": "may_harm", "via_state": "burning"},
            {"source": "house_2", "target": "car_1", "effect": "may_harm", "via_state": "burning"},
            {"source": "house_2", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
            {"source": "house_2", "target": "child_1", "effect": "may_harm", "via_state": "burning"},
        ],
    }
    ranked, _ = intervention._candidates_from_graph(graph, use_intervention_candidates=False)
    assert ranked[0]["object_id"] == "house_1", \
        f"GT root should rank first, got {[c['object_id'] for c in ranked]}"


def test_in_degree_helper_ignores_self_edges(tmp_path=None):
    """B4 helper: a self-edge does not make a node non-root."""
    graph = {"edges": [{"source": "a", "target": "a"}, {"source": "a", "target": "b"}]}
    indeg = intervention._hazard_in_degree(graph)
    assert indeg.get("a", 0) == 0  # only a self-edge points at a -> still root
    assert indeg.get("b", 0) == 1


def test_enumerate_control_none_with_single_hazard(tmp_path):
    """< 2 distinct hazards -> control None (compare_to_control later skipped)."""
    baseline = _baseline_with_gt(tmp_path)  # make_result has exactly one hazard (flood_1)
    enum = intervention.enumerate_candidates(baseline)
    assert enum["control"] is None


# ---------------------------------------------------------------------------
# step 2 — build_intervention_spec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hazard_class,expected_type", [
    ("engulfing_fluid", "edge_severance"),
    ("discrete_source", "source_removal"),
    ("person_in_hazard", "target_mitigation"),
])
def test_spec_type_auto_defaults_by_hazard_class(hazard_class, expected_type):
    """#2 type map: auto-default from hazard_class when intervention_type is None."""
    candidate = {"object_id": "x_1", "state": "engulfing", "label": "water",
                 "hazard_class": hazard_class, "sources": ["A"], "ranks": {"A": 0},
                 "is_should_be_core": True}
    spec = intervention.build_intervention_spec(candidate, intervention_type=None, modality="language")
    assert spec["intervention_type"] == expected_type
    assert spec["modality"] == "language"  # recorded verbatim


def test_spec_role_is_explicit_and_decoupled_from_is_should_be_core():
    """R3: `role` is set by the caller per ARM, DECOUPLED from is_should_be_core.
    A declared-but-not-GT core (is_should_be_core False) built with role='core' must be
    role='core', NOT 'control'. is_should_be_core stays the separate GT-truth flag."""
    declared_not_gt = {"object_id": "child_1", "state": "drowning", "label": "child",
                       "hazard_class": "person_in_hazard", "sources": ["A"],
                       "ranks": {"A": 1}, "is_should_be_core": False}
    spec = intervention.build_intervention_spec(
        declared_not_gt, intervention_type=None, modality="language", role="core")
    assert spec["role"] == "core"               # the arm is the core arm
    assert spec["is_should_be_core"] is False    # ...but it is NOT the GT core


def test_spec_explicit_type_overrides_default():
    """An explicit intervention_type argument overrides the auto-default."""
    candidate = {"object_id": "x_1", "state": "engulfing", "label": "water",
                 "hazard_class": "engulfing_fluid", "sources": ["A"], "ranks": {"A": 0},
                 "is_should_be_core": True}
    spec = intervention.build_intervention_spec(candidate, intervention_type="source_removal",
                                                modality="visual")
    assert spec["intervention_type"] == "source_removal"
    assert spec["modality"] == "visual"


# ---------------------------------------------------------------------------
# step 3 — render_do_prompt  (+ GT-specific leak guard)
# ---------------------------------------------------------------------------
def _spec_for(tmp_path):
    baseline = _baseline_with_gt(tmp_path)
    enum = intervention.enumerate_candidates(baseline)
    cand = enum["should_be_core"] or enum["declared_core_a"]
    return baseline, intervention.build_intervention_spec(cand, modality="language")


def test_render_do_prompt_contains_target_and_action(tmp_path):
    """Output references the target hazard id and an action verb (the do())."""
    baseline, spec = _spec_for(tmp_path)
    out = intervention.render_do_prompt(baseline, spec)
    assert isinstance(out["prompt"], str) and out["prompt"]
    assert isinstance(out["suppression_statement"], str) and out["suppression_statement"]
    target_id = spec["target"]["object_id"]
    blob = (out["prompt"] + " " + out["suppression_statement"])
    assert target_id in blob  # the hazard being suppressed is named


def test_render_do_prompt_does_not_leak_gt_specific_content(tmp_path):
    """B5 leak guard, GT-SPECIFIC: the do-prompt must contain NEITHER the GT
    caption string NOR a GT-only object id (`water_1`), because those are the
    answer key. We probe GT-specific strings (not generic labels like 'water'
    or 'fire' that the model's own output also uses), so the test neither
    false-positives on shared vocabulary nor passes vacuously."""
    baseline, spec = _spec_for(tmp_path)
    # Sanity: the baseline really did load GT-specific content that COULD leak.
    assert baseline["gt_graph"] is not None
    assert GT_ONLY_CAPTION not in (baseline.get("caption") or "")  # not already in model fields

    out = intervention.render_do_prompt(baseline, spec)
    blob = out["prompt"] + " " + out["suppression_statement"]
    assert GT_ONLY_CAPTION not in blob, "GT caption (answer key) leaked into the do-prompt"
    assert GT_ONLY_OBJECT_ID not in blob, "GT-only object id leaked into the do-prompt"


def test_render_do_prompt_embeds_baseline_objects_and_reuse_instruction(tmp_path):
    """EMBED-BASELINE (A3/B5/B8): the do-prompt embeds the model's OWN baseline — EVERY
    detected_object id and a Graph-A edge token — and instructs reuse / hold-fixed, so the
    stateless VLM can hold U instead of re-reading the scene. The embed must still carry
    NO GT-specific content (embedding model-authored content must not regress the leak
    guard)."""
    baseline, spec = _spec_for(tmp_path)
    out = intervention.render_do_prompt(baseline, spec)
    blob = out["prompt"]

    # (a) every baseline detected_object id appears (the U anchor).
    base_ids = [o["object_id"] for o in baseline["detected_objects"]]
    assert base_ids, "fixture must have detected_objects for this test to be meaningful"
    for oid in base_ids:
        assert oid in blob, f"baseline object id {oid} not embedded in do-prompt"
    assert "flood_1" in blob and "person_1" in blob  # explicit on the make_result fixture

    # (b) a Graph-A edge token appears (coupling cue from the model's own graph).
    edges = baseline["graph_a"].get("edges") or []
    assert edges, "fixture must have graph_a edges"
    assert edges[0]["target"] in blob  # e.g. the edge's target id is summarized

    # (c) a reuse / hold-fixed instruction is present.
    low = blob.lower()
    assert "reuse" in low
    assert ("hold" in low and "fixed" in low)

    # (d) leak guard still holds on the now-larger blob (no answer-key content embedded).
    full = out["prompt"] + " " + out["suppression_statement"]
    assert GT_ONLY_CAPTION not in full
    assert GT_ONLY_OBJECT_ID not in full


def test_render_do_prompt_does_not_pin_recs_or_edges_to_baseline(tmp_path):
    """B8 construct guard: the embed pins U (entities/states), NOT the action. The prompt
    must EXPECT the recommendations and causal edges to change where they depended on the
    suppressed hazard, so a grounded suppression is not biased into a false 'static'."""
    baseline, spec = _spec_for(tmp_path)
    blob = intervention.render_do_prompt(baseline, spec)["prompt"].lower()
    assert "re-derived" in blob or "re-derive" in blob
    assert "expected to change" in blob


def test_render_do_prompt_empty_baseline_is_well_formed(tmp_path):
    """A4 edge-input safety: empty detected_objects / edgeless graph_a must NOT crash or
    emit a dangling 'reuse these ids:' with no list; the suppression statement + JSON-key
    spec stay unconditional so the prompt is always well-formed."""
    empty_baseline = {"detected_objects": [], "graph_a": {"nodes": [], "edges": []}}
    spec = {"target": {"object_id": "flood_1", "state": "engulfing"},
            "intervention_type": "edge_severance"}
    out = intervention.render_do_prompt(empty_baseline, spec)
    assert isinstance(out["prompt"], str) and out["prompt"]
    assert "flood_1" in out["prompt"]              # target still named
    assert '"detected_objects"' in out["prompt"]   # JSON-key spec unconditional
    # no dangling empty entity list: the no-entities branch says so explicitly.
    assert "no other tracked entities" in out["prompt"].lower()


# ---------------------------------------------------------------------------
# step 4 — run_counterfactual
# ---------------------------------------------------------------------------
def test_run_counterfactual_calls_vlm_and_returns_light_post(tmp_path):
    """Calls the injected vlm_fn; returns ONLY the four light fields; does NOT
    carry gt_validation/trust (a counterfactual world has no answer key)."""
    baseline, spec = _spec_for(tmp_path)
    do = intervention.render_do_prompt(baseline, spec)
    raw_post = {
        "detected_objects": [{"object_id": "person_1", "label": "person", "state": "safe"}],
        "causal_graph": {"nodes": [], "edges": [], "intervention_candidates": []},
        "recommendations": [],
        "disaster_level": 2,
        # extra junk a real VLM might emit — must be dropped:
        "gt_validation": {"available": True}, "trust": {"score": 9},
    }
    vlm = make_vlm_stub(raw_post)
    post = intervention.run_counterfactual("data:img", do["prompt"], spec, vlm)

    assert vlm.calls, "run_counterfactual did not call the injected vlm_fn"
    assert set(post.keys()) == {"detected_objects", "graph_a", "recommendations", "hazard_level"}
    assert "gt_validation" not in post and "trust" not in post
    assert post["hazard_level"] == 2  # mapped from disaster_level


# ---------------------------------------------------------------------------
# step 5 — check_u_preservation
# ---------------------------------------------------------------------------
def _det(objs, edges=None):
    """objs = list of (object_id, label) or (object_id, label, state) -> detected_objects;
    optional edges = list of (source, target[, effect[, via_state]]) -> graph_a edges."""
    dobjs = []
    for o in objs:
        if len(o) == 3:
            oid, lab, st = o
        else:
            oid, lab = o
            st = ""
        dobjs.append({"object_id": oid, "label": lab, "state": st})
    out = {"detected_objects": dobjs}
    if edges is not None:
        e_recs = []
        for e in edges:
            src, tgt = e[0], e[1]
            eff = e[2] if len(e) > 2 else "affects"
            via = e[3] if len(e) > 3 else ""
            e_recs.append({"source": src, "target": tgt, "effect": eff, "via_state": via})
        out["graph_a"] = {"nodes": [], "edges": e_recs}
    return out


# B7 construct fix: the U gate is now driven by STATE-stability and TOPOLOGY-stability of
# NON-suppressed entities (quantities the do()-prompt did NOT instruct), NOT the canonical
# label multiset / id Jaccard (which EMBED-BASELINE forces the model to reproduce, making the
# old gate tautological). object_overlap / raw_id_overlap survive only as secondary,
# non-gating diagnostics. These tests lock the new construct.

_FLOOD_SPEC = {"target": {"object_id": "flood_1", "state": "engulfing", "label": "water"},
               "intervention_type": "edge_severance"}


def test_u_preserved_when_states_and_topology_stable():
    """Faithful hold-fixed: only the suppressed entity (flood_1) and its incident edge change;
    every NON-suppressed state and every non-suppressed edge holds -> leaked False, both
    stabilities 1.0. A grounded re-route must NOT be flagged as a leak."""
    baseline = _det([("flood_1", "water", "engulfing"),
                     ("person_1", "person", "standing"),
                     ("car_1", "car", "parked")],
                    edges=[("flood_1", "person_1"), ("car_1", "person_1")])
    # flood_1 drained + its edge gone (the do()-coupled change); person_1/car_1 states held;
    # the car_1->person_1 edge (both non-suppressed) preserved.
    post = _det([("flood_1", "water", "drained"),
                 ("person_1", "person", "standing"),
                 ("car_1", "car", "parked")],
                edges=[("car_1", "person_1")])
    u = intervention.check_u_preservation(baseline, post, _FLOOD_SPEC)
    assert u["state_stability"] == pytest.approx(1.0)
    assert u["topology_stability"] == pytest.approx(1.0)
    assert u["leaked"] is False
    assert u["cutoff"] == pytest.approx(0.7)  # U_CUTOFF


def test_u_leaked_when_nonsuppressed_state_flips():
    """B7: ids+labels all reused (raw_id_overlap 1.0, object_overlap 1.0) but a NON-suppressed
    entity's STATE is silently re-imagined (person_1 standing -> fleeing) -> leaked True. The
    old label-multiset gate would have passed this by construction."""
    baseline = _det([("flood_1", "water", "engulfing"),
                     ("person_1", "person", "standing"),
                     ("car_1", "car", "parked")])
    post = _det([("flood_1", "water", "drained"),
                 ("person_1", "person", "fleeing"),
                 ("car_1", "car", "submerged")])
    u = intervention.check_u_preservation(baseline, post, _FLOOD_SPEC)
    assert u["raw_id_overlap"] == pytest.approx(1.0)   # reuse compliance held
    assert u["object_overlap"] == pytest.approx(1.0)   # label multiset held
    assert u["state_stability"] < 0.7                  # ...but states drifted
    assert u["leaked"] is True


def test_u_leaked_when_nonsuppressed_topology_rewired():
    """B7: ids+labels+states all reused, but TOPOLOGY among non-suppressed nodes is rewired
    (a new car_1->house_1 edge appears + the original drops) -> leaked True. Edges touching
    the suppressed flood_1 are excluded, so this leak is purely non-suppressed re-wiring."""
    baseline = _det([("flood_1", "water", "engulfing"),
                     ("person_1", "person", "standing"),
                     ("car_1", "car", "parked"),
                     ("house_1", "house", "intact")],
                    edges=[("flood_1", "person_1"), ("car_1", "person_1")])
    post = _det([("flood_1", "water", "drained"),
                 ("person_1", "person", "standing"),
                 ("car_1", "car", "parked"),
                 ("house_1", "house", "intact")],
                edges=[("car_1", "house_1")])  # car_1->person_1 dropped, car_1->house_1 added
    u = intervention.check_u_preservation(baseline, post, _FLOOD_SPEC)
    assert u["state_stability"] == pytest.approx(1.0)  # states held
    assert u["topology_stability"] < 0.7               # ...but non-suppressed edges rewired
    assert u["leaked"] is True


def test_u_grounded_reroute_not_flagged():
    """Positive lock: only the suppressed hazard's own state/edges change (flood_1 drained,
    its incident edge dropped) and the recommendation/graph re-route happens ONLY along
    suppressed-coupled edges; every non-suppressed state and edge holds -> leaked False.
    Proves a grounded re-route is not mislabeled as a U leak."""
    baseline = _det([("flood_1", "water", "engulfing"),
                     ("person_1", "person", "standing"),
                     ("car_1", "car", "parked")],
                    edges=[("flood_1", "person_1", "drowns"), ("flood_1", "car_1", "floats")])
    post = _det([("flood_1", "water", "drained"),
                 ("person_1", "person", "standing"),
                 ("car_1", "car", "parked")],
                edges=[])  # all edges were flood_1-incident -> all legitimately drop
    u = intervention.check_u_preservation(baseline, post, _FLOOD_SPEC)
    assert u["state_stability"] == pytest.approx(1.0)
    assert u["topology_stability"] == pytest.approx(1.0)  # vacuous: no non-suppressed edges
    assert u["leaked"] is False


def test_u_secondary_diagnostics_present_but_nongating():
    """B7: object_overlap and raw_id_overlap remain as secondary keys but do NOT drive
    `leaked` — a post can have perfect id/label overlap yet leak on state."""
    baseline = _det([("flood_1", "water", "engulfing"), ("person_1", "person", "standing")])
    post = _det([("flood_1", "water", "drained"), ("person_1", "person", "drowning")])
    u = intervention.check_u_preservation(baseline, post, _FLOOD_SPEC)
    assert "object_overlap" in u and "raw_id_overlap" in u
    assert u["raw_id_overlap"] == pytest.approx(1.0)
    assert u["leaked"] is True  # gated on state, not the secondary overlaps


def test_u_target_mitigation_family_removal_not_leaked():
    """B8 carried forward: a target_mitigation do() that legitimately removes the suppressed
    person (one person-family unit vanishes) while every OTHER state holds -> leaked False."""
    baseline = _det([("person_1", "person", "trapped"),
                     ("car_1", "car", "parked"),
                     ("house_1", "house", "intact")])
    post = _det([("car_1", "car", "parked"), ("house_1", "house", "intact")])  # person_1 moved to safety
    spec = {"target": {"object_id": "person_1", "state": "trapped", "label": "person"},
            "intervention_type": "target_mitigation"}
    u = intervention.check_u_preservation(baseline, post, spec)
    assert u["leaked"] is False  # the removed entity is the suppressed target, not a leak


def test_u_vacuous_stable_when_nothing_nonsuppressed():
    """A4: a sparse scene with only the suppressed entity (no non-suppressed states/edges)
    cannot leak what it never had -> both stabilities 1.0, leaked False."""
    baseline = _det([("flood_1", "water", "engulfing")])
    post = _det([("flood_1", "water", "drained")])
    u = intervention.check_u_preservation(baseline, post, _FLOOD_SPEC)
    assert u["n_nonsuppressed_states"] == 0
    assert u["state_stability"] == pytest.approx(1.0)
    assert u["topology_stability"] == pytest.approx(1.0)
    assert u["leaked"] is False


# ---------------------------------------------------------------------------
# step 6 — compute_shifts
# ---------------------------------------------------------------------------
def _spec_obj():
    return {"target": {"object_id": "flood_1", "state": "engulfing", "label": "water",
                       "hazard_class": "engulfing_fluid"},
            "intervention_type": "edge_severance", "modality": "language",
            "is_should_be_core": True, "role": "core"}


def _full_baseline_for_shifts():
    r = make_result()
    return {
        "detected_objects": r["detected_objects"],
        "graph_a": r["causal_graph"],
        "recommendations": r["recommendations"],
        "hazard_level": r["disaster_level"],
    }


def test_shifts_identical_post_all_zero():
    """Identical post == baseline -> all five shifts 0 and total_shift 0
    (a static post is the masquerade/correctly-ignored signal)."""
    baseline = _full_baseline_for_shifts()
    post = {
        "detected_objects": baseline["detected_objects"],
        "graph_a": baseline["graph_a"],
        "recommendations": baseline["recommendations"],
        "hazard_level": baseline["hazard_level"],
    }
    s = intervention.compute_shifts(baseline, post, _spec_obj())
    for k in ("hazard_shift", "graph_shift", "recommendation_shift",
              "structural_shift", "semantic_shift", "total_shift"):
        assert s[k] == pytest.approx(0.0), f"{k} should be 0 for an identical post"
    assert s["hazard_level_delta"] == 0


def test_shifts_all_in_unit_interval():
    """Every signal (and total_shift) is a delta in [0,1] for an arbitrary post."""
    baseline = _full_baseline_for_shifts()
    post = {
        "detected_objects": [{"object_id": "flood_1", "label": "water", "state": "drained"},
                             {"object_id": "person_1", "label": "person", "state": "safe"}],
        "graph_a": {"nodes": [{"id": "person_1", "label": "person", "state": "safe", "hazardous": False}],
                    "edges": [], "intervention_candidates": []},
        "recommendations": [{"rank": 1, "action": "Reassure person_1.",
                             "related_object_ids": ["person_1"],
                             "structured_reasoning": {
                                 "threat": "person_1", "state": "safe",
                                 "effect": "may_harm", "affected_objects": ["person_1"]}}],
        "hazard_level": 1,
    }
    s = intervention.compute_shifts(baseline, post, _spec_obj())
    for k in ("hazard_shift", "graph_shift", "recommendation_shift",
              "structural_shift", "semantic_shift", "total_shift"):
        assert 0.0 <= s[k] <= 1.0, f"{k}={s[k]} out of [0,1]"
    # hazard dropped 8 -> 1 = -7 signed raw
    assert s["hazard_level_delta"] == -7


def test_recommendation_shift_zero_on_rewording_same_rec():
    """B1: a reworded-but-substantively-identical recommendation -> recommendation_shift 0.
    Shift is computed on STRUCTURE (target/action-intent/cited-hazard), not raw text."""
    baseline = _full_baseline_for_shifts()
    # A7 guard: the fixture must actually exercise _rec_quads. If the rec quad were not
    # nested under structured_reasoning, _rec_quads would return the empty set and
    # recommendation_shift would be jaccard(set(), set()) = 0 unconditionally — passing
    # this test for the wrong reason. Assert the baseline quads are non-empty first.
    assert intervention._rec_quads(baseline["recommendations"]), \
        "fixture recs must yield non-empty quads, else rec_shift==0 is vacuous"
    # Same target, same cited hazard, same affected object, same graph, same hazard
    # level — only the surface wording of `action` changes.
    reworded = json.loads(json.dumps(baseline))
    reworded["recommendations"][0]["action"] = "Relocate person_1 to safety, away from flood_1."
    s = intervention.compute_shifts(baseline, reworded, _spec_obj())
    assert s["recommendation_shift"] == pytest.approx(0.0)


def test_total_shift_is_mean_of_five():
    """#3: total_shift == mean of the five signals (aggregation contract)."""
    baseline = _full_baseline_for_shifts()
    post = {
        "detected_objects": baseline["detected_objects"],
        "graph_a": {"nodes": [], "edges": [], "intervention_candidates": []},
        "recommendations": [],
        "hazard_level": 0,
    }
    s = intervention.compute_shifts(baseline, post, _spec_obj())
    five = [s["hazard_shift"], s["graph_shift"], s["recommendation_shift"],
            s["structural_shift"], s["semantic_shift"]]
    assert s["total_shift"] == pytest.approx(sum(five) / 5.0)


# ---------------------------------------------------------------------------
# step 7 — adjudicate_groundedness : the 2x2 oracle + no-GT case
# ---------------------------------------------------------------------------
# "moved" in the oracle = total_shift >= MOVE_CUTOFF (0.3); a "static" post is
# identical -> total_shift 0. We drive adjudicate with hand-built signals dicts.
MOVED_SIGNALS = {"hazard_shift": 0.7, "graph_shift": 0.7, "recommendation_shift": 0.7,
                 "structural_shift": 0.7, "semantic_shift": 0.7, "total_shift": 0.7,
                 "hazard_level_delta": -7}
STATIC_SIGNALS = {"hazard_shift": 0.0, "graph_shift": 0.0, "recommendation_shift": 0.0,
                  "structural_shift": 0.0, "semantic_shift": 0.0, "total_shift": 0.0,
                  "hazard_level_delta": 0}


def _spec(is_core: bool):
    return {"target": {"object_id": "h_1", "state": "engulfing", "label": "water",
                       "hazard_class": "engulfing_fluid"},
            "intervention_type": "edge_severance", "modality": "language",
            "is_should_be_core": is_core, "role": "core" if is_core else "control"}


def _candidates(should_be_core: bool):
    """Candidates bundle; should_be_core present iff GT marks one."""
    core = {"object_id": "h_1", "state": "engulfing", "label": "water",
            "hazard_class": "engulfing_fluid", "sources": ["A", "GT"],
            "ranks": {"A": 0, "GT": 0}, "is_should_be_core": True}
    return {"candidates": [core],
            "should_be_core": core if should_be_core else None,
            "declared_core_a": core, "declared_core_b": core,
            "control": None}


def test_oracle_should_be_core_static_is_masquerade():
    """yes should-be-core x static post -> masquerade (named the hazard, ignored it)."""
    v = intervention.adjudicate_groundedness(_spec(True), STATIC_SIGNALS, _candidates(True))
    assert v["cell"] == "masquerade"
    assert v["moved"] is False and v["is_should_be_core"] is True


def test_oracle_should_be_core_moved_is_grounded():
    """yes should-be-core x moved post -> grounded (suppression moved the recs)."""
    v = intervention.adjudicate_groundedness(_spec(True), MOVED_SIGNALS, _candidates(True))
    assert v["cell"] == "grounded"
    assert v["moved"] is True and v["is_should_be_core"] is True


def test_oracle_not_core_static_is_correctly_ignored():
    """no should-be-core x static post -> correctly_ignored."""
    v = intervention.adjudicate_groundedness(_spec(False), STATIC_SIGNALS, _candidates(True))
    assert v["cell"] == "correctly_ignored"
    assert v["moved"] is False and v["is_should_be_core"] is False


def test_oracle_not_core_moved_is_spurious_grounding():
    """no should-be-core x moved post -> spurious_grounding (moved on a non-core)."""
    v = intervention.adjudicate_groundedness(_spec(False), MOVED_SIGNALS, _candidates(True))
    assert v["cell"] == "spurious_grounding"
    assert v["moved"] is True and v["is_should_be_core"] is False


def test_oracle_no_gt_is_not_adjudicable():
    """B3: no GT -> should_be_core unknown -> not_adjudicable, regardless of movement.
    The row is undetermined without a should-be-core; this is the fifth locked case."""
    cands = _candidates(should_be_core=False)  # should_be_core None
    # spec carries the unknown core status the contract derives from absent GT.
    spec_unknown = {"target": {"object_id": "h_1", "state": "engulfing", "label": "water",
                               "hazard_class": "engulfing_fluid"},
                    "intervention_type": "edge_severance", "modality": "language",
                    "is_should_be_core": None, "role": "core"}
    v = intervention.adjudicate_groundedness(spec_unknown, MOVED_SIGNALS, cands)
    assert v["cell"] == "not_adjudicable"


# ---------------------------------------------------------------------------
# step 8 — compare_to_control
# ---------------------------------------------------------------------------
def test_compare_to_control_discriminates_when_core_moves_more():
    """C2 (code form): core total_shift > control total_shift -> discriminates True."""
    core_run = {"signals": {"total_shift": 0.7}}
    control_run = {"signals": {"total_shift": 0.1}}
    d = intervention.compare_to_control(core_run, control_run)
    assert d["core_total_shift"] == pytest.approx(0.7)
    assert d["control_total_shift"] == pytest.approx(0.1)
    assert d["discriminates"] is True


def test_compare_to_control_flags_when_equal():
    """Equal shifts -> does NOT discriminate (core not distinguished from control)."""
    core_run = {"signals": {"total_shift": 0.4}}
    control_run = {"signals": {"total_shift": 0.4}}
    d = intervention.compare_to_control(core_run, control_run)
    assert d["discriminates"] is False


def test_compare_to_control_none_when_no_control():
    """< 2 hazards -> control None -> compare_to_control skipped (discriminates None,
    not a failure)."""
    core_run = {"signals": {"total_shift": 0.7}}
    d = intervention.compare_to_control(core_run, None)
    assert d["discriminates"] is None


# ---------------------------------------------------------------------------
# Pipeline composition + hygiene (A2 JSON-serializable, no Dash)
# ---------------------------------------------------------------------------
def test_run_intervention_end_to_end_hermetic(tmp_path):
    """run_intervention composes steps 2-8 with an injected vlm_fn; returns the
    full contract shape; the whole thing is JSON-serializable (UI-agnostic)."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # A static post (model ignores the do) -> with should-be-core present this is
    # the masquerade corner; the assertion here is structural, not on the cell.
    raw_post = {
        "detected_objects": make_result()["detected_objects"],
        "causal_graph": make_result()["causal_graph"],
        "recommendations": make_result()["recommendations"],
        "disaster_level": 8,
    }
    vlm = make_vlm_stub(raw_post)
    selections = {"modality": "language"}
    out = intervention.run_intervention(baseline, selections, vlm)

    for key in ("baseline", "spec", "u_check", "signals", "verdict"):
        assert key in out, f"run_intervention output missing {key}"
    # JSON-serializable (A2): must not raise.
    json.dumps(out)


# ---------------------------------------------------------------------------
# R4 — GT core with NO model co-referent surfaces gt_core_unobserved (push_06)
# ---------------------------------------------------------------------------
def _push06_like_result():
    """push_06-shaped: GT core is `water` (engulfing) but the model NEVER detects any
    water/flood node — it only sees the drowning person + a chair. The GT core has no
    model co-referent, the headline finding."""
    return {
        "run_id": "run_push06",
        "image_filename": "push_06_drowning_pool.jpg",
        "prompt": "analyze",
        "caption": "a pool scene",
        "disaster_level": 8,
        "detected_objects": [
            {"object_id": "child_1", "label": "child", "state": "drowning"},
            {"object_id": "person_2", "label": "person", "state": "fleeing"},
            {"object_id": "chair_1", "label": "chair", "state": "stationary"},
        ],
        "recommendations": [
            {"rank": 1, "action": "Rescue child_1.", "threat": "child_1",
             "state": "drowning", "related_object_ids": ["child_1"],
             "affected_objects": ["child_1"]},
        ],
        "causal_graph": {
            "nodes": [
                {"id": "child_1", "label": "child", "state": "drowning", "hazardous": True},
                {"id": "person_2", "label": "person", "state": "fleeing", "hazardous": False},
            ],
            "edges": [
                {"source": "child_1", "target": "person_2", "effect": "may_harm", "via_state": "drowning"},
            ],
            "intervention_candidates": [
                {"threat": "child_1", "state": "drowning", "outgoing_edge_count": 1},
            ],
        },
        "graph_b": {
            "nodes": [
                {"id": "person_1", "label": "person", "state": "drowning", "hazardous": True},
            ],
            "edges": [],
            "suppression_pick": {"threat": "person_1", "state": "drowning", "reason": "victim"},
        },
    }


def test_gt_core_unobserved_when_model_has_no_coreferent(tmp_path):
    """R4: GT core (water_1/engulfing) has no model co-referent -> enumerate surfaces
    `gt_core_unobserved` and should_be_core stays None (not silently nulled)."""
    gt_dir = write_gt_dir(tmp_path)  # GT core = water_1 'engulfing'
    baseline = intervention.intervention_baseline(
        _push06_like_result(), image_data_url="data:img", gt_dir=gt_dir)
    enum = intervention.enumerate_candidates(baseline)
    assert enum["should_be_core"] is None
    assert enum["gt_core_unobserved"] is not None
    assert enum["gt_core_unobserved"]["object_id"] == "water_1"
    assert enum["gt_core_unobserved"]["state"] == "engulfing"


def test_gt_core_unobserved_verdict_is_distinct(tmp_path):
    """R4: adjudicate emits the distinct `gt_core_unobserved` cell, NOT bare
    not_adjudicable, when GT names a core the model never perceived."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(
        _push06_like_result(), image_data_url="data:img", gt_dir=gt_dir)
    # static post (model ignores the do) — cell must still report the perception miss.
    raw_post = {
        "detected_objects": _push06_like_result()["detected_objects"],
        "causal_graph": _push06_like_result()["causal_graph"],
        "recommendations": _push06_like_result()["recommendations"],
        "disaster_level": 8,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    assert out["verdict"]["cell"] == "gt_core_unobserved"
    assert out["spec"]["role"] == "core"  # R3: the arm that ran is the core arm
    json.dumps(out)  # JSON-serializable


# ---------------------------------------------------------------------------
# R2 — declared (non-GT) core suppressed -> core_not_declared, not bare null
# ---------------------------------------------------------------------------
def test_core_not_declared_when_no_gt_but_declared_core_runs(tmp_path):
    """R2: no GT file at all, but the model declares a core. The core arm STILL runs
    (fallback to declared_core_a), the verdict carries core_not_declared=True with the
    declared-core source, and movement is preserved — NOT a blanket not_adjudicable."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result(image_filename="no_gt_for_this.jpg")  # no GT -> should_be_core None
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    # A MOVED post (recs fully retargeted) so we can confirm movement is preserved.
    raw_post = {
        "detected_objects": result["detected_objects"],
        "causal_graph": {"nodes": [], "edges": [], "intervention_candidates": []},
        "recommendations": [],
        "disaster_level": 0,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    v = out["verdict"]
    assert v["cell"] == "not_adjudicable"
    assert v.get("core_not_declared") is True
    assert v.get("declared_core_source") == "A"
    assert out["spec"]["role"] == "core"            # R3: arm is core even w/o GT
    assert out["spec"]["is_should_be_core"] is False  # ...but not GT-confirmed
    # movement preserved on the move_basis (not thrown away)
    assert "total_shift" in v["move_basis"]


def test_nothing_to_suppress_is_distinct_from_core_not_declared():
    """R2: the genuine empty case (no candidate at all) -> not_adjudicable with
    nothing_to_suppress, distinguishable from the core_not_declared path."""
    baseline = {"image_filename": "x", "detected_objects": [], "graph_a": {},
                "graph_b": {}, "gt_graph": None, "recommendations": [], "hazard_level": 0}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub({}))
    v = out["verdict"]
    assert v["cell"] == "not_adjudicable"
    assert v.get("nothing_to_suppress") is True
    assert v.get("core_not_declared") in (None, False)


# ---------------------------------------------------------------------------
# B7/B9 — U-leak void nulls the 'moved' claim (renamed-id-but-same-label NOT leaked)
# ---------------------------------------------------------------------------
def test_renamed_ids_same_labels_not_overridden_to_u_leaked(tmp_path):
    """B7 regression (post-R1): a post that RENAMES every id but keeps the same label
    multiset must NOT be voided as u_leaked — U held. The verdict surfaces the real
    result, not a spurious void."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # baseline labels: water + person. Post keeps SAME labels under renamed ids.
    raw_post = {
        "detected_objects": [{"object_id": "flood_99", "label": "water", "state": "engulfing"},
                             {"object_id": "swimmer_42", "label": "person", "state": "wading"}],
        "causal_graph": make_result()["causal_graph"],
        "recommendations": make_result()["recommendations"],
        "disaster_level": 8,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    assert out["u_check"]["leaked"] is False
    assert out["verdict"]["cell"] != "u_leaked"


def test_u_leak_void_nulls_moved_claim(tmp_path):
    """B9/B2: on a void (u_leaked) run, `moved` is NOT presented as a finding — it is
    nulled and move_basis is marked not-consumed, even if a strong rec shift would have
    fired the OR-escape."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # label multiset genuinely diverges -> U leaks; recs fully retargeted (would-move).
    leaked_post = {
        "detected_objects": [{"object_id": "t_1", "label": "tree", "state": "x"},
                             {"object_id": "r_1", "label": "rock", "state": "y"}],
        "causal_graph": {"nodes": [], "edges": []},
        "recommendations": [{"rank": 1, "action": "Z", "threat": "t_1", "state": "x",
                             "related_object_ids": ["t_1"], "affected_objects": ["t_1"]}],
        "disaster_level": 0,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(leaked_post))
    assert out["u_check"]["leaked"] is True
    v = out["verdict"]
    assert v["cell"] == "u_leaked"
    assert v["moved"] is None                       # no 'moved' claim above void
    assert v["move_basis"].get("consumed") is False  # raw shift kept for audit only


def test_intervention_module_has_named_constants():
    """A3: MOVE_CUTOFF and U_CUTOFF are named module constants (no magic numbers)."""
    assert intervention.MOVE_CUTOFF == pytest.approx(0.3)
    assert intervention.U_CUTOFF == pytest.approx(0.7)
    assert intervention.REC_MOVE_CUTOFF == pytest.approx(0.5)  # B2 single-signal guard


# ---------------------------------------------------------------------------
# A1 / B5 — GT core is resolved to a MODEL-side id (no GT-only id reaches spec)
# ---------------------------------------------------------------------------
def test_should_be_core_is_model_side_id_not_gt_only(tmp_path):
    """A1/B5: the GT core hazard (`water_1`) and the model's hazard (`flood_1`) are the
    SAME hazard under different ids. should_be_core must resolve to the MODEL id, so
    the do() never targets / leaks a GT-only id."""
    baseline = _baseline_with_gt(tmp_path)
    enum = intervention.enumerate_candidates(baseline)
    core = enum["should_be_core"]
    assert core is not None
    # The model never emitted water_1; the resolved core must be the model's flood_1.
    assert core["object_id"] != GT_ONLY_OBJECT_ID
    assert core["object_id"] == "flood_1"
    # And the spec / do-prompt built from it carry the model id, never the GT id.
    spec = intervention.build_intervention_spec(core, modality="language")
    assert spec["target"]["object_id"] == "flood_1"
    blob = " ".join(intervention.render_do_prompt(baseline, spec).values())
    assert GT_ONLY_OBJECT_ID not in blob


# ---------------------------------------------------------------------------
# B2 — a single strong signal is not washed out by the mean
# ---------------------------------------------------------------------------
def test_strong_rec_shift_alone_counts_as_moved():
    """B2: a grounded model that responds ONLY by fully rewriting its recommendations
    (recommendation_shift 1.0, the other four ~0 -> mean 0.2 < MOVE_CUTOFF) must still
    be 'moved' (and so 'grounded' on a should-be-core), not washed out to masquerade."""
    rec_only = {"hazard_shift": 0.0, "graph_shift": 0.0, "recommendation_shift": 1.0,
                "structural_shift": 0.0, "semantic_shift": 0.0, "total_shift": 0.2,
                "hazard_level_delta": 0}
    v = intervention.adjudicate_groundedness(_spec(True), rec_only, _candidates(True))
    assert v["moved"] is True
    assert v["cell"] == "grounded"


# ---------------------------------------------------------------------------
# B6 — control prefers a hazard causally independent of the core
# ---------------------------------------------------------------------------
def test_control_prefers_target_disjoint_hazard(tmp_path):
    """B6: with one GT hazard sharing the core's downstream target and another that is
    disjoint, the control picker chooses the DISJOINT hazard (uncorrelated control),
    not merely the lowest GT edge-rank, and records control_overlap=False."""
    gt = {
        "image_filename": "push_06_drowning_pool.jpg",
        "caption": "ZZ_SECRET_marker",
        "nodes": [
            {"id": "water_1", "label": "water", "state": "engulfing", "hazardous": True},
            {"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
            {"id": "debris_1", "label": "debris", "state": "falling", "hazardous": True},
            {"id": "child_1", "label": "child", "state": "drowning", "hazardous": False},
            {"id": "child_2", "label": "child", "state": "unconscious", "hazardous": False},
            {"id": "shed_1", "label": "shed", "state": "intact", "hazardous": False},
        ],
        "edges": [
            # core (water_1, most edges) and fire_1 BOTH feed child_1 (correlated control).
            {"source": "water_1", "target": "child_1", "effect": "may_harm", "via_state": "engulfing"},
            {"source": "water_1", "target": "child_2", "effect": "may_harm", "via_state": "engulfing"},
            {"source": "fire_1", "target": "child_1", "effect": "may_harm", "via_state": "spreading"},
            # debris_1 feeds ONLY shed_1 -> target-disjoint from the core (uncorrelated).
            {"source": "debris_1", "target": "shed_1", "effect": "may_damage", "via_state": "falling"},
        ],
    }
    gt_dir = write_gt_dir(tmp_path, gt)
    # Model must emit co-referring nodes for all three hazards (resolution is model-side).
    result = make_result()
    result["detected_objects"] += [
        {"object_id": "fire_1", "label": "fire", "state": "spreading"},
        {"object_id": "debris_1", "label": "debris", "state": "falling"},
    ]
    result["causal_graph"]["nodes"] += [
        {"id": "fire_1", "label": "fire", "state": "spreading", "hazardous": True},
        {"id": "debris_1", "label": "debris", "state": "falling", "hazardous": True},
    ]
    # Model must rank these as candidates too (Graph A ranks via intervention_candidates),
    # so they exist as MODEL-side records the control resolution can pick.
    result["causal_graph"]["intervention_candidates"] += [
        {"threat": "fire_1", "state": "spreading", "outgoing_edge_count": 1},
        {"threat": "debris_1", "state": "falling", "outgoing_edge_count": 1},
    ]
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    enum = intervention.enumerate_candidates(baseline)
    core = enum["should_be_core"]
    control = enum["control"]
    assert core is not None and control is not None
    # core is water (resolved to flood_1); control must be debris_1 (target-disjoint),
    # NOT fire_1 (which shares child_1 with the core).
    assert control["object_id"] == "debris_1"
    assert control.get("control_overlap") is False


# ---------------------------------------------------------------------------
# B7 — U leak VOIDS the verdict (not cosmetic)
# ---------------------------------------------------------------------------
def test_u_leak_voids_verdict(tmp_path):
    """B7: when the post leaks U (object overlap < U_CUTOFF), run_intervention must
    NOT emit a grounded/masquerade/etc. verdict off an invalid comparison; the cell is
    overridden to a void state carrying comparison_invalid."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # A post whose detected objects barely overlap the baseline -> U leaked.
    leaked_post = {
        "detected_objects": [{"object_id": "stranger_1", "label": "tree", "state": "x"},
                             {"object_id": "stranger_2", "label": "rock", "state": "y"}],
        "causal_graph": {"nodes": [], "edges": []},
        "recommendations": [],
        "disaster_level": 0,
    }
    vlm = make_vlm_stub(leaked_post)
    out = intervention.run_intervention(baseline, {"modality": "language"}, vlm)
    assert out["u_check"]["leaked"] is True
    verdict = out["verdict"]
    assert verdict["cell"] not in {
        "grounded", "masquerade", "spurious_grounding", "correctly_ignored"
    }
    assert verdict["cell"] == "u_leaked"
    assert verdict.get("comparison_invalid") is True


# ===========================================================================
# REFINER fixes (live-surfaced findings) — one lock per accepted finding.
# ===========================================================================

# ---------------------------------------------------------------------------
# B6 (phantom) — a declared candidate id with no entity anchor is dropped, not targeted
# ---------------------------------------------------------------------------
def test_phantom_candidate_is_dropped_and_surfaced(tmp_path):
    """B6: an intervention_candidate whose `threat` id is in NEITHER graph_a.nodes NOR
    detected_objects is a PHANTOM (no pixel binding). It must NOT become a candidate /
    suppression target; it is surfaced under `phantom_candidates` as a baseline
    inconsistency. Mirrors live push_06 where child_1 was a phantom that drove the do()."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result()
    # Add a phantom: declared as a candidate, but never detected and not a graph node.
    result["causal_graph"]["intervention_candidates"].append(
        {"threat": "child_1", "state": "drowning", "outgoing_edge_count": 9})
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    enum = intervention.enumerate_candidates(baseline)
    cand_ids = {c["object_id"] for c in enum["candidates"]}
    assert "child_1" not in cand_ids, "phantom became a candidate"
    phantom_ids = {p["object_id"] for p in enum.get("phantom_candidates", [])}
    assert "child_1" in phantom_ids, "phantom not surfaced for audit"
    # The real model hazard (flood_1) is still a candidate.
    assert "flood_1" in cand_ids


# ---------------------------------------------------------------------------
# B8 — target_mitigation excludes the moved target's family from the U denominator
# ---------------------------------------------------------------------------
def test_target_mitigation_does_not_self_leak_u():
    """B8: a target_mitigation do() LEGITIMATELY removes the at-risk entity, so that
    entity disappearing must NOT count as a U leak. In a sparse 3-object scene, suppressing
    the only person would otherwise force overlap 2/3 < 0.7 -> spurious u_leaked, making the
    masquerade/grounded distinction unreachable for person hazards."""
    base = _det([("person_1", "person"), ("chair_1", "chair"), ("table_1", "table")])
    base["detected_objects"][0]["state"] = "drowning"
    post = _det([("chair_1", "chair"), ("table_1", "table")])  # person moved to safety
    spec = {"intervention_type": "target_mitigation",
            "target": {"object_id": "person_1", "label": "person"}}
    u = intervention.check_u_preservation(base, post, spec)
    assert u["leaked"] is False  # suppressed person excluded; chair/table states held
    assert u["state_stability"] == pytest.approx(1.0)
    # Control: WITHOUT the spec exception the removed person is NOT excluded, so it reads as a
    # non-suppressed entity that vanished -> state_stability 2/3 < 0.7 -> leaked True.
    u_no_spec = intervention.check_u_preservation(base, post)
    assert u_no_spec["leaked"] is True


# ---------------------------------------------------------------------------
# B2 — discrimination is computed on content_shift, not the diluted total_shift
# ---------------------------------------------------------------------------
def test_compute_shifts_emits_content_shift():
    """B2: compute_shifts emits `content_shift` = mean(hazard, graph, recommendation),
    the three content-bearing signals, alongside total_shift = mean(all 5)."""
    baseline = _full_baseline_for_shifts()
    post = {"detected_objects": baseline["detected_objects"],
            "graph_a": {"nodes": [], "edges": [], "intervention_candidates": []},
            "recommendations": [], "hazard_level": 0}
    s = intervention.compute_shifts(baseline, post, _spec_obj())
    assert "content_shift" in s
    expected = (s["hazard_shift"] + s["graph_shift"] + s["recommendation_shift"]) / 3.0
    assert s["content_shift"] == pytest.approx(expected)


def test_discrimination_uses_content_shift_not_total():
    """B2: under coherent full re-routing structural/semantic deltas are ~0 and dilute
    total_shift; discrimination must use content_shift so a real core>control gap is not
    masked. core re-routes (content 0.9, total 0.54); placebo control static (0.0)."""
    core_sig = {"hazard_shift": 0.7, "graph_shift": 1.0, "recommendation_shift": 1.0,
                "structural_shift": 0.0, "semantic_shift": 0.0,
                "total_shift": 0.54, "content_shift": 0.9}
    ctrl_sig = {"hazard_shift": 0.0, "graph_shift": 0.0, "recommendation_shift": 0.0,
                "structural_shift": 0.0, "semantic_shift": 0.0,
                "total_shift": 0.0, "content_shift": 0.0}
    d = intervention.compare_to_control({"signals": core_sig}, {"signals": ctrl_sig})
    assert d["discriminates"] is True
    assert d["core_content_shift"] == pytest.approx(0.9)
    assert d["control_content_shift"] == pytest.approx(0.0)
    # total_shifts still reported for audit.
    assert d["core_total_shift"] == pytest.approx(0.54)


# ---------------------------------------------------------------------------
# B6 / C1 — placebo (null) control fallback when < 2 GT hazards
# ---------------------------------------------------------------------------
def test_placebo_control_when_single_gt_hazard(tmp_path):
    """B6/C1: GT_GRAPH has ONE hazardous node (water_1), so there is no real-hazard
    control. enumerate must surface a `placebo_control` (a non-hazard detected object) so a
    discrimination baseline always exists on the headline single-hazard scene."""
    baseline = _baseline_with_gt(tmp_path)  # make_result: water hazard + person non-hazard
    enum = intervention.enumerate_candidates(baseline)
    assert enum["control"] is None                       # < 2 GT hazards
    assert enum.get("placebo_control") is not None        # ...but a placebo exists
    assert enum["placebo_control"]["is_placebo"] is True
    # The placebo is the non-hazard person, not the water core.
    assert enum["placebo_control"]["object_id"] == "person_1"


def test_run_intervention_uses_placebo_for_discrimination(tmp_path):
    """B6/C1: with no real-hazard control, run_intervention runs the placebo arm and the
    discrimination block reports control_kind='placebo' (a baseline always exists)."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    raw_post = {"detected_objects": make_result()["detected_objects"],
                "causal_graph": make_result()["causal_graph"],
                "recommendations": make_result()["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    assert out["control"] is not None
    assert out["control"]["is_placebo"] is True
    assert out["discrimination"]["control_kind"] == "placebo"
    assert out["discrimination"]["discriminates"] is not None  # baseline existed


# ---------------------------------------------------------------------------
# A4 / B3 / C4 / R4 — gt_core_unobserved OUTRANKS u_leaked (perception miss survives void)
# ---------------------------------------------------------------------------
def test_gt_core_unobserved_survives_u_leak(tmp_path):
    """A4/B3/C4/R4: when GT names a core the model never perceived AND U leaks, the verdict
    must stay `gt_core_unobserved` (a perception miss is a baseline fact, U-independent) with
    a `u_leaked` annotation — NOT overwritten to a bare u_leaked cell that buries the
    headline finding. This is the exact live push_06 failure."""
    gt_dir = write_gt_dir(tmp_path)  # GT core = water_1, unobserved by the push06-like model
    baseline = intervention.intervention_baseline(
        _push06_like_result(), image_data_url="data:img", gt_dir=gt_dir)
    # A post that diverges -> U leaks.
    leaked_post = {
        "detected_objects": [{"object_id": "x_1", "label": "tree", "state": "a"},
                             {"object_id": "y_1", "label": "rock", "state": "b"}],
        "causal_graph": {"nodes": [], "edges": []}, "recommendations": [],
        "disaster_level": 0,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(leaked_post))
    v = out["verdict"]
    assert out["u_check"]["leaked"] is True
    assert v["cell"] == "gt_core_unobserved"      # perception miss NOT buried under the void
    assert v.get("u_leaked") is True               # ...but the leak is annotated
    assert v["gt_core_unobserved"]["object_id"] == "water_1"
    assert v["is_should_be_core"] is None          # B3 tri-state, not hard False
    assert v["move_basis"].get("moved") is None     # B9: no movement asserted under void


# ---------------------------------------------------------------------------
# B3 — is_should_be_core is tri-state (None when GT absent, not hard False)
# ---------------------------------------------------------------------------
def test_is_should_be_core_is_none_when_no_gt():
    """B3: adjudicate must report is_should_be_core as None (unknown) when GT is absent,
    never coerce it to a hard False that misreads as a definite 'not-core' row."""
    cands = _candidates(should_be_core=False)  # should_be_core None -> no GT
    spec_unknown = {"target": {"object_id": "h_1", "state": "engulfing", "label": "water",
                               "hazard_class": "engulfing_fluid"},
                    "intervention_type": "edge_severance", "modality": "language",
                    "is_should_be_core": False, "role": "core"}
    v = intervention.adjudicate_groundedness(spec_unknown, MOVED_SIGNALS, cands)
    assert v["cell"] == "not_adjudicable"
    assert v["is_should_be_core"] is None


# ---------------------------------------------------------------------------
# B9 — U-leak void nulls move_basis.moved (no field asserts movement above a void)
# ---------------------------------------------------------------------------
def test_u_leak_void_nulls_move_basis_moved(tmp_path):
    """B9: on a void, NEITHER verdict.moved NOR move_basis.moved may assert movement.
    A latent move_basis.moved=true under a voided verdict is an overclaim if downstream
    code reads move_basis.moved instead of verdict.moved."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    leaked_post = {
        "detected_objects": [{"object_id": "t_1", "label": "tree", "state": "x"},
                             {"object_id": "r_1", "label": "rock", "state": "y"}],
        "causal_graph": {"nodes": [], "edges": []},
        "recommendations": [{"rank": 1, "action": "Z", "threat": "t_1", "state": "x",
                             "related_object_ids": ["t_1"], "affected_objects": ["t_1"]}],
        "disaster_level": 0,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(leaked_post))
    v = out["verdict"]
    assert v["cell"] == "u_leaked"
    assert v["moved"] is None
    assert v["move_basis"].get("moved") is None        # the latent overclaim is gone
    assert v["move_basis"].get("consumed") is False


# ---------------------------------------------------------------------------
# C1 — post composition is persisted for U-leak auditability
# ---------------------------------------------------------------------------
def test_run_output_includes_post_composition(tmp_path):
    """C1: the run record must carry the post's detected_objects + canonical label
    multiset, so a confound auditor can verify a U-leak is a genuine scene re-read vs a
    benign relabel. The live artifact dropped this, making the void unfalsifiable."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    raw_post = {"detected_objects": [{"object_id": "a_1", "label": "water", "state": "s"}],
                "causal_graph": {"nodes": [], "edges": []}, "recommendations": [],
                "disaster_level": 3}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    pc = out.get("post_composition")
    assert pc is not None
    assert pc["detected_objects"] == raw_post["detected_objects"]
    assert pc["label_multiset"] == {"water": 1}
    # C1 (B7 redesign): the post graph_a must ALSO be persisted so a TOPOLOGY-stability leak
    # is falsifiable from the saved artifact (the U gate now reads graph edges, not just labels).
    assert pc["graph_a"] == raw_post["causal_graph"]


# ---------------------------------------------------------------------------
# role / provenance — spec.core_basis records GT vs declared core independently
# ---------------------------------------------------------------------------
def test_spec_core_basis_gt_when_gt_confirmed(tmp_path):
    """role: the core arm on a GT-confirmed should-be-core carries spec.core_basis='gt'."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    raw_post = {"detected_objects": make_result()["detected_objects"],
                "causal_graph": make_result()["causal_graph"],
                "recommendations": make_result()["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    assert out["spec"]["core_basis"] == "gt"


def test_spec_core_basis_declared_when_no_gt(tmp_path):
    """role: with no GT, the declared-core fallback arm carries spec.core_basis='declared_a'
    so the arm's provenance survives even when the U-leak override rewrites the verdict-level
    core_not_declared annotation."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result(image_filename="no_gt_here.jpg")
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    raw_post = {"detected_objects": result["detected_objects"],
                "causal_graph": result["causal_graph"],
                "recommendations": result["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    assert out["spec"]["core_basis"] == "declared_a"


# ---------------------------------------------------------------------------
# A7 — declared-core diagnostics survive a U-leak void (carried forward, not erased)
# ---------------------------------------------------------------------------
def test_core_not_declared_survives_u_leak(tmp_path):
    """A4/A7: when there is no GT (declared-core fallback) AND U leaks, the void must
    CARRY FORWARD core_not_declared + declared_core_source rather than erase them, so the
    declared-core provenance is not lost under the void."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result(image_filename="no_gt_here.jpg")  # no GT
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    leaked_post = {
        "detected_objects": [{"object_id": "t_1", "label": "tree", "state": "x"},
                             {"object_id": "r_1", "label": "rock", "state": "y"}],
        "causal_graph": {"nodes": [], "edges": []}, "recommendations": [],
        "disaster_level": 0,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(leaked_post))
    v = out["verdict"]
    assert out["u_check"]["leaked"] is True
    assert v["cell"] == "u_leaked"
    assert v.get("core_not_declared") is True            # carried forward under the void
    assert v.get("declared_core_source") == "A"


# ---------------------------------------------------------------------------
# A4 (driver) — an unrecognized `candidates` selection key is flagged, not silently used
# ---------------------------------------------------------------------------
def test_unused_candidates_selection_key_is_flagged(tmp_path):
    """A4: the driver passed selections={'candidates': ...}, a key the contract does not
    define; enumerate is always re-run. run_intervention must record that the bundle was
    ignored (selection_notes.candidates_arg_ignored) rather than silently accept it."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    raw_post = {"detected_objects": make_result()["detected_objects"],
                "causal_graph": make_result()["causal_graph"],
                "recommendations": make_result()["recommendations"], "disaster_level": 8}
    selections = {"modality": "language", "candidates": [{"object_id": "ignored"}]}
    out = intervention.run_intervention(baseline, selections, make_vlm_stub(raw_post))
    assert out["selection_notes"]["candidates_arg_ignored"] is True


# ---------------------------------------------------------------------------
# C4 / B9 — discrimination is void-aware; leaked arm signals carry the marker
# ---------------------------------------------------------------------------
def test_discrimination_is_void_when_core_leaks(tmp_path):
    """C4: on a leaked core arm, discrimination must NOT emit a true/false `discriminates`
    verdict off invalid comparisons. It nulls discriminates and flags comparison_invalid
    with a reason, while keeping the raw shift numbers for audit. B9: the leaked core
    signals carry a comparison_invalid marker so the numbers never outlive the void
    unflagged."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # A post that leaks U on every arm (wholesale scene re-read).
    leaked_post = {
        "detected_objects": [{"object_id": "stranger_1", "label": "tree", "state": "x"},
                             {"object_id": "stranger_2", "label": "rock", "state": "y"}],
        "causal_graph": {"nodes": [], "edges": []},
        "recommendations": [],
        "disaster_level": 0,
    }
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(leaked_post))
    assert out["u_check"]["leaked"] is True
    d = out["discrimination"]
    assert d["discriminates"] is None
    assert d["comparison_invalid"] is True
    assert d["comparison_invalid_reason"] in {"core_leaked", "both_leaked"}
    # raw numbers retained for audit (not silently dropped)
    assert "core_content_shift" in d
    # B9: the persisted core signals carry the void marker.
    assert out["signals"].get("comparison_invalid") is True


# ---------------------------------------------------------------------------
# C3 — non-high baseline trust qualifies the verdict with a caveat
# ---------------------------------------------------------------------------
def test_verdict_carries_trust_caveat_when_trust_moderate(tmp_path):
    """C3: a moderate/low baseline trust must qualify the read — verdict.trust_caveat True
    and the explanation states the read is provisional."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result()
    result["pre_intervention_trust"] = {"score": 0.85, "level": "moderate"}
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    raw_post = {"detected_objects": make_result()["detected_objects"],
                "causal_graph": make_result()["causal_graph"],
                "recommendations": make_result()["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    v = out["verdict"]
    assert v.get("trust_caveat") is True
    assert "provisional" in v.get("explanation", "").lower()


def test_verdict_no_trust_caveat_when_trust_high(tmp_path):
    """C3: a high (or unknown) baseline trust adds NO caveat — trust_caveat False."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result()
    result["pre_intervention_trust"] = {"score": 0.95, "level": "high"}
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    raw_post = {"detected_objects": make_result()["detected_objects"],
                "causal_graph": make_result()["causal_graph"],
                "recommendations": make_result()["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    v = out["verdict"]
    assert v.get("trust_caveat") is False
    assert "provisional" not in v.get("explanation", "").lower()


# ---------------------------------------------------------------------------
# B6 — a control correlated with the core (control_overlap) is replaced by the placebo
# ---------------------------------------------------------------------------
def test_correlated_control_is_replaced_by_placebo(tmp_path):
    """B6: when the only real-hazard control overlaps the core's downstream targets
    (control_overlap True — debris from the same collapse), it is causally correlated with
    the core and would destroy discrimination by construction. run_intervention must prefer
    the causally-independent placebo and surface the confound (control_overlap)."""
    gt_dir = write_gt_dir(tmp_path)
    # Two GT hazards sharing ALL downstream victims -> the non-core hazard is correlated.
    gt = {
        "image_filename": "push_34_collapse.jpg",
        "caption": GT_GRAPH["caption"],
        "schema_version": "2026-06-10",
        "nodes": [
            {"id": "building_1", "label": "structure", "state": "collapsing", "hazardous": True},
            {"id": "debris_1", "label": "structure", "state": "falling", "hazardous": True},
            {"id": "rescuer_1", "label": "person", "state": "exposed", "hazardous": False},
        ],
        "edges": [
            {"source": "building_1", "target": "rescuer_1", "effect": "may_harm", "via_state": "collapsing"},
            {"source": "debris_1", "target": "rescuer_1", "effect": "may_harm", "via_state": "falling"},
        ],
    }
    gt_dir2 = tmp_path / "verified_gt"
    if not gt_dir2.exists():
        gt_dir2 = write_gt_dir(tmp_path, gt)
    else:
        (gt_dir2 / f"{gt['image_filename']}.gt.json").write_text(json.dumps(gt))

    result = make_result(image_filename="push_34_collapse.jpg")
    result["detected_objects"] = [
        {"object_id": "building_1", "label": "structure", "state": "collapsing"},
        {"object_id": "debris_1", "label": "structure", "state": "falling"},
        {"object_id": "rescuer_1", "label": "person", "state": "exposed"},
        {"object_id": "vehicle_1", "label": "car", "state": "parked"},  # non-hazard -> placebo
    ]
    result["causal_graph"] = {
        "nodes": [
            {"id": "building_1", "label": "structure", "state": "collapsing", "hazardous": True},
            {"id": "debris_1", "label": "structure", "state": "falling", "hazardous": True},
            {"id": "rescuer_1", "label": "person", "state": "exposed", "hazardous": False},
            {"id": "vehicle_1", "label": "car", "state": "parked", "hazardous": False},
        ],
        "edges": [
            {"source": "building_1", "target": "rescuer_1", "effect": "may_harm", "via_state": "collapsing"},
            {"source": "debris_1", "target": "rescuer_1", "effect": "may_harm", "via_state": "falling"},
        ],
        "intervention_candidates": [
            {"threat": "building_1", "state": "collapsing", "outgoing_edge_count": 1},
            {"threat": "debris_1", "state": "falling", "outgoing_edge_count": 1},
        ],
    }
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir2)
    enum = intervention.enumerate_candidates(baseline)
    # Precondition: the chosen real-hazard control overlaps the core (the confound).
    assert enum["control"] is not None
    assert enum["control"].get("control_overlap") is True
    assert enum.get("placebo_control") is not None

    raw_post = {"detected_objects": result["detected_objects"],
                "causal_graph": result["causal_graph"],
                "recommendations": [], "disaster_level": 7}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(raw_post))
    # B6: the correlated hazard was replaced by the placebo as the primary baseline.
    assert out["control"]["is_placebo"] is True
    assert out["discrimination"]["control_kind"] == "placebo"
    # B6 (corrected): control_overlap now reflects the REAL-hazard control's status
    # independently (a correlated hazard DID exist), and has_real_hazard_control is False
    # because the primary baseline that actually ran was the placebo, not a real hazard.
    # A reader must not see control_overlap=False and infer a clean disjoint hazard control.
    assert out["discrimination"]["control_overlap"] is True
    assert out["discrimination"]["has_real_hazard_control"] is False


# ===========================================================================
# REFINER pass — findings locked as tests (A1, B2, B3, B5/B6/B7, B8, B9, C2, C4)
# ===========================================================================

def _over_reactive_post(detected_objects):
    """A fully-rerouted post that holds U (same entities) but rewrites graph + recs and
    drops the water hazard's hazardous flag (so the do() reads as APPLIED). Used to mock a
    rung-1 over-reactive model that re-routes for ANY suppression."""
    return {
        "detected_objects": detected_objects,
        "causal_graph": {
            "nodes": [
                {"id": "flood_1", "label": "water", "state": "drained", "hazardous": False},
                {"id": "person_1", "label": "person", "state": "calm", "hazardous": False},
            ],
            "edges": [
                {"source": "person_1", "target": "flood_1", "effect": "observes", "via_state": "calm"},
            ],
        },
        "recommendations": [
            {"rank": 1, "action": "Totally different advice.", "threat": "person_1",
             "state": "calm", "structured_reasoning": {
                 "threat": "person_1", "state": "calm", "effect": "observes",
                 "affected_objects": ["flood_1"]}},
        ],
        "disaster_level": 2,
    }


# --- C4 / C2 / B8 / B9 : over-reactive masquerade -> 'grounded' must be CAVEATED ----------
def test_over_reactive_model_grounded_is_caveated(tmp_path):
    """C4/C2/B8/B9: an over-reactive rung-1 model re-routes its whole graph+recs for ANY
    suppression — the core (water) and the placebo (person) move IDENTICALLY, so
    discriminates=False. The core verdict may still land in 'grounded' on the move gate, but
    it MUST carry a discrimination_caveat and the explanation MUST NOT read as an unqualified
    'grounded' — the core did not beat the anti-confound control. This is the exact live
    push_03 failure (discrimination computed but never fed back into the verdict)."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)

    def vlm(image, prompt, spec):
        # Same fully-rerouted post regardless of which arm (core OR placebo).
        return _over_reactive_post(make_result()["detected_objects"])

    out = intervention.run_intervention(baseline, {"modality": "language"}, vlm)
    d = out["discrimination"]
    v = out["verdict"]
    # The placebo arm exists and moved identically -> no discrimination.
    assert d["control_kind"] == "placebo"
    assert d["discriminates"] is False
    # C4/C2: the verdict must be flagged...
    assert v.get("discrimination_caveat") is True
    # B9: ...and the text must NOT leave 'grounded' unqualified.
    assert "UNCONFIRMED" in v["explanation"]
    # B9: the printed comparator must be truthful (never a hardcoded '<=' that contradicts
    # the numbers). The caveat reports the real relation between the two content shifts.
    cc = d["core_content_shift"]; ctc = d["control_content_shift"]
    if cc > ctc:
        assert f"{cc:.2f} >" in v["explanation"]
    elif cc == ctc:
        assert f"{cc:.2f} =" in v["explanation"]
    else:
        assert f"{cc:.2f} <" in v["explanation"]
    # C4: the qualification must be machine-readable, not free-text only.
    assert v.get("confidence") == "unconfirmed"
    assert v.get("cell_provisional") is True


def test_compare_to_control_reason_insufficient_margin():
    """C2: when the core beats the control NUMERICALLY but within DISCRIM_MARGIN and the
    control is NOT over-reactive, discriminates is False with reason='insufficient_margin'
    (distinct from the control_over_reactive case)."""
    core_sig = {"content_shift": 0.40, "total_shift": 0.3}
    ctrl_sig = {"content_shift": 0.33, "total_shift": 0.25}  # below over-reactive cutoff (0.5)
    d = intervention.compare_to_control({"signals": core_sig}, {"signals": ctrl_sig})
    assert d["discriminates"] is False
    assert d["margin"] > 0  # the core DID beat the control numerically
    assert d["control_over_reactive"] is False
    assert d["discriminates_reason"] == "insufficient_margin"


def test_compare_to_control_reason_control_over_reactive():
    """C2: an over-reactive control voids the comparison; reason='control_over_reactive'
    takes precedence over the margin gate."""
    core_sig = {"content_shift": 0.90, "total_shift": 0.8}
    ctrl_sig = {"content_shift": 0.70, "total_shift": 0.6}  # >= over-reactive cutoff (0.5)
    d = intervention.compare_to_control({"signals": core_sig}, {"signals": ctrl_sig})
    assert d["discriminates"] is False
    assert d["control_over_reactive"] is True
    assert d["discriminates_reason"] == "control_over_reactive"


def test_compare_to_control_reason_none_when_discriminates():
    """C2: discriminates_reason is None when the comparison actually discriminates."""
    core_sig = {"content_shift": 0.90, "total_shift": 0.8}
    ctrl_sig = {"content_shift": 0.20, "total_shift": 0.15}
    d = intervention.compare_to_control({"signals": core_sig}, {"signals": ctrl_sig})
    assert d["discriminates"] is True
    assert d["discriminates_reason"] is None


def test_grounded_when_core_beats_control_has_no_caveat(tmp_path):
    """C4 (negative): when the core DOES beat a CLEAN (causally-disjoint) control (core
    re-routes, placebo static), discriminates=True and the grounded verdict carries NO
    discrimination_caveat — the caveat fires only when the anti-confound check fails, not on
    every grounded cell.

    The control must be a DISJOINT placebo (B6 refiner): make_result's only non-hazard
    (person_1) is downstream of the water core, so we add an isolated bystander (sign_1, no
    causal edges) that the placebo picker prefers — giving a clean, non-correlated baseline."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result()
    # Add a causally-disjoint non-hazard so a CLEAN placebo exists (sign_1 has no edges).
    result["detected_objects"].append({"object_id": "sign_1", "label": "sign", "state": "upright"})
    result["causal_graph"]["nodes"].append(
        {"id": "sign_1", "label": "sign", "state": "upright", "hazardous": False})
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)

    post_objs = result["detected_objects"]

    def vlm(image, prompt, spec):
        # Core (water/edge_severance) re-routes; placebo (sign/placebo_null) stays static.
        if spec.get("intervention_type") == "placebo_null":
            return {"detected_objects": post_objs,
                    "causal_graph": result["causal_graph"],
                    "recommendations": result["recommendations"], "disaster_level": 8}
        return _over_reactive_post(post_objs)

    out = intervention.run_intervention(baseline, {"modality": "language"}, vlm)
    # The chosen placebo is the causally-disjoint bystander, not the correlated person_1.
    assert out["control"]["spec"]["target"]["object_id"] == "sign_1"
    assert out["discrimination"]["placebo_overlap"] is False
    assert out["discrimination"]["discriminates"] is True
    assert out["verdict"]["cell"] == "grounded"
    assert out["verdict"].get("discrimination_caveat") is False


# --- B2 : ONE basis for both consumers (moved gate + discrimination on content_shift) ----
def test_move_gate_uses_content_shift_not_diluted_total():
    """B2: the move gate must use content_shift (mean of hazard+graph+recommendation), the
    same basis as discrimination — not total_shift (mean of all 5, diluted by ~0 structural/
    semantic deltas under coherent re-route). A moderate full re-route {hazard 0.2, graph
    0.5, rec 0.4, struct 0, sem 0}: total_shift=0.22 (would mis-score masquerade) but
    content_shift=0.37 -> moved. The OR-escape does not fire (rec 0.4 < 0.5), so this proves
    the gate switched basis, not that the rec escape saved it."""
    signals = {"hazard_shift": 0.2, "graph_shift": 0.5, "recommendation_shift": 0.4,
               "structural_shift": 0.0, "semantic_shift": 0.0,
               "total_shift": (0.2 + 0.5 + 0.4) / 5.0,                  # 0.22
               "content_shift": (0.2 + 0.5 + 0.4) / 3.0}                 # 0.37
    v = intervention.adjudicate_groundedness(_spec(True), signals, _candidates(True))
    assert v["moved"] is True
    assert v["cell"] == "grounded"
    assert v["move_basis"]["move_rule"] == "content"
    # total_shift alone would NOT have cleared the cutoff.
    assert signals["total_shift"] < 0.3 < signals["content_shift"]


# --- B3 : suppressing the affected_object does not auto-fire recommendation_shift ---------
def test_recommendation_shift_excludes_suppressed_target_self():
    """B3: removing the suppressed object's own id from rec quads on BOTH sides so a 'moved'
    driven SOLELY by the suppressed target vanishing (mechanical, not a model reaction) does
    not auto-fire recommendation_shift. Baseline rec affects person_1; suppress person_1;
    post keeps the SAME rec minus person_1 -> after self-exclusion the quads match -> shift 0."""
    baseline = {
        "graph_a": {"nodes": [], "edges": []},
        "recommendations": [{"structured_reasoning": {
            "threat": "flood_1", "state": "engulfing", "effect": "may_harm",
            "affected_objects": ["person_1"]}}],
        "hazard_level": 5,
    }
    # post: same rec, but person_1 (suppressed) is gone from affected_objects.
    post = {
        "graph_a": {"nodes": [], "edges": []},
        "recommendations": [{"structured_reasoning": {
            "threat": "flood_1", "state": "engulfing", "effect": "may_harm",
            "affected_objects": []}}],
        "hazard_level": 5,
    }
    spec = {"target": {"object_id": "person_1", "label": "person"},
            "intervention_type": "target_mitigation"}
    s = intervention.compute_shifts(baseline, post, spec)
    assert s["recommendation_shift"] == pytest.approx(0.0)


def test_placebo_moved_cell_is_annotated_not_a_real_finding(tmp_path):
    """B3/B6: a placebo arm that lands in a 'moved' cell (spurious_grounding/grounded) is NOT
    a real groundedness finding — it is the anti-confound baseline. The control verdict must
    carry placebo_not_a_finding so a reader is not misled into 'the model reacts to a
    non-core hazard'."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)

    def vlm(image, prompt, spec):
        return _over_reactive_post(make_result()["detected_objects"])

    out = intervention.run_intervention(baseline, {"modality": "language"}, vlm)
    cv = out["control"]["verdict"]
    if cv.get("cell") in ("spurious_grounding", "grounded"):
        assert cv.get("placebo_not_a_finding") is True
        assert "PLACEBO" in cv["explanation"]


# --- B6 : placebo gets a neutral do(), not a destructive removal -------------------------
def test_placebo_spec_uses_neutral_do_not_source_removal():
    """B6: a placebo candidate (a non-hazard bystander) must NOT inherit source_removal
    phrasing ('completely removed from the scene') — that would delete a real entity and
    confound the baseline. build_intervention_spec routes it to placebo_null, and the
    rendered do() is an inert 'plays no causal role' statement, not a removal."""
    placebo_cand = {"object_id": "person_1", "state": "standing", "label": "person",
                    "hazard_class": "discrete_source", "is_placebo": True,
                    "is_should_be_core": False}
    spec = intervention.build_intervention_spec(placebo_cand, role="control")
    assert spec["intervention_type"] == "placebo_null"
    assert spec["is_placebo"] is True
    rendered = intervention.render_do_prompt(
        {"detected_objects": [{"object_id": "person_1", "label": "person", "state": "standing"}]},
        spec)
    assert "removed from the scene" not in rendered["suppression_statement"]
    assert "no causal role" in rendered["suppression_statement"]


def test_has_real_hazard_control_false_for_placebo_only_scene(tmp_path):
    """B6: on a single-GT-hazard scene the discrimination block must report
    has_real_hazard_control=False (only a placebo ran), so a placebo-only scene is not
    presented as having a confound-free hazard control."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    out = intervention.run_intervention(
        baseline, {"modality": "language"},
        make_vlm_stub({"detected_objects": make_result()["detected_objects"],
                       "causal_graph": make_result()["causal_graph"],
                       "recommendations": make_result()["recommendations"], "disaster_level": 8}))
    d = out["discrimination"]
    assert d["control_kind"] == "placebo"
    assert d["has_real_hazard_control"] is False


# --- B5 / B7 : do()-applied guard catches a non-applied source_removal/edge_severance -----
def test_do_applied_false_when_source_persists_unchanged():
    """B5/B7: for source_removal/edge_severance, if the suppressed source PERSISTS unchanged
    (still hazardous, same state) in the post graph, the do() was a no-op. check_do_applied
    must report applied=False/reason=source_persists, so U-preservation does not certify a
    comparison where the do() was ignored."""
    baseline = {"graph_a": {"nodes": [{"id": "flood_1", "label": "water",
                                       "state": "engulfing", "hazardous": True}], "edges": []}}
    post = {"graph_a": {"nodes": [{"id": "flood_1", "label": "water",
                                   "state": "engulfing", "hazardous": True}], "edges": []}}
    spec = {"intervention_type": "edge_severance",
            "target": {"object_id": "flood_1", "state": "engulfing", "label": "water"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is False
    assert da["reason"] == "source_persists"


def test_do_applied_false_when_source_persists_in_detected_only(tmp_path=None):
    """B7: a model can edit the graph_a node (drop hazardous / change state) while leaving the
    suppressed source FULLY PRESENT and UNCHANGED in detected_objects. A graph-only guard
    would certify applied=True for a do() that removed nothing from the scene. check_do_applied
    must read detected_objects FIRST and report applied=False/reason=source_persists_in_detected
    when the source survives there in its original state."""
    baseline = {"graph_a": {"nodes": [{"id": "house_1", "label": "house",
                                       "state": "burning", "hazardous": True}], "edges": []}}
    # Graph node says removed (no node / non-hazardous) but detected_objects still has it burning.
    post = {"graph_a": {"nodes": [], "edges": []},
            "detected_objects": [{"object_id": "house_1", "label": "house", "state": "burning"}]}
    spec = {"intervention_type": "source_removal",
            "target": {"object_id": "house_1", "state": "burning", "label": "house"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is False
    assert da["reason"] == "source_persists_in_detected"


def test_do_applied_true_when_source_leaves_detected(tmp_path=None):
    """B7: when the suppressed source is genuinely gone from detected_objects (and the graph
    node), the do() took effect -> applied True."""
    baseline = {"graph_a": {"nodes": [{"id": "house_1", "label": "house",
                                       "state": "burning", "hazardous": True}], "edges": []}}
    post = {"graph_a": {"nodes": [], "edges": []},
            "detected_objects": [{"object_id": "person_1", "label": "person", "state": "fleeing"}]}
    spec = {"intervention_type": "source_removal",
            "target": {"object_id": "house_1", "state": "burning", "label": "house"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is True


def test_do_applied_true_when_source_state_changes():
    """B5/B7: when the suppressed source's hazardous flag/state DOES change in the post, the
    do() took effect -> applied True."""
    baseline = {"graph_a": {"nodes": [{"id": "flood_1", "label": "water",
                                       "state": "engulfing", "hazardous": True}], "edges": []}}
    post = {"graph_a": {"nodes": [{"id": "flood_1", "label": "water",
                                   "state": "drained", "hazardous": False}], "edges": []}}
    spec = {"intervention_type": "edge_severance",
            "target": {"object_id": "flood_1", "state": "engulfing", "label": "water"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is True


def test_run_intervention_flags_do_not_applied(tmp_path):
    """B5/B7 end-to-end: when the model echoes the baseline (water still hazardous, state
    unchanged) for a source-style do(), the core verdict carries do_not_applied=True and an
    explanation noting U-preservation here evidences the do() was IGNORED."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # Echo the baseline graph (flood_1 stays burning/hazardous) -> do() not applied.
    echo_post = {"detected_objects": make_result()["detected_objects"],
                 "causal_graph": make_result()["causal_graph"],
                 "recommendations": make_result()["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(echo_post))
    v = out["verdict"]
    assert out["do_applied"]["applied"] is False
    assert v.get("do_not_applied") is True
    assert "IGNORED" in v["explanation"]


# ---------------------------------------------------------------------------
# B8 (refiner) — end-to-end anti-gaming: a pure rung-1 static-post mock on the
# should-be-core arm scores 'masquerade' through the FULL pipeline (not just the
# hand-built unit oracle on adjudicate_groundedness).
# ---------------------------------------------------------------------------
def test_end_to_end_static_post_on_should_be_core_is_masquerade(tmp_path):
    """B8: the anti-gaming guarantee must hold for the INTEGRATED pipeline, not only the unit
    oracle. With a GT-confirmed should-be-core (water, co-referenced to the model's flood_1,
    so should_be_core resolves and this is NOT the gt_core_unobserved branch), a vlm_fn that
    returns a post STRUCTURALLY IDENTICAL to baseline on the core arm must score
    cell=='masquerade' with moved False, exercising the live move-gate (content_shift gating,
    REC_MOVE_CUTOFF OR-escape, U gate) — not just adjudicate_groundedness in isolation."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # Pre-condition: the should-be-core is a model-co-referenced id (NOT gt_core_unobserved).
    enum = intervention.enumerate_candidates(baseline)
    assert enum["should_be_core"] is not None
    assert enum.get("gt_core_unobserved") is None

    # A static post identical to baseline -> all five shifts 0 -> the model ignored the do().
    static_post = {"detected_objects": make_result()["detected_objects"],
                   "causal_graph": make_result()["causal_graph"],
                   "recommendations": make_result()["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(static_post))
    v = out["verdict"]
    assert v["cell"] == "masquerade"
    assert v["moved"] is False
    assert out["u_check"]["leaked"] is False   # U held: the verdict is not voided


# ---------------------------------------------------------------------------
# B8 (refiner) — the target_mitigation U-discount is GUARDED: it cannot rescue a
# comparison where the post ALSO drops an unrelated entity.
# ---------------------------------------------------------------------------
def test_target_mitigation_discount_does_not_rescue_wholesale_reread():
    """B8 (refiner med): the suppressed-target family discount is granted only when the
    REMAINING (non-target) multiset already overlaps >= U_CUTOFF. A target_mitigation post
    that drops the moved person AND an unrelated entity (table_1) is a wholesale re-read and
    MUST leak — the discount must not lift it over the cutoff."""
    base = _det([("person_1", "person"), ("chair_1", "chair"), ("table_1", "table")])
    base["detected_objects"][0]["state"] = "drowning"
    # person moved to safety (legit) BUT table_1 also dropped (wholesale re-read).
    post = _det([("chair_1", "chair")])
    spec = {"intervention_type": "target_mitigation",
            "target": {"object_id": "person_1", "label": "person"}}
    u = intervention.check_u_preservation(base, post, spec)
    assert u["leaked"] is True
    assert u["object_overlap"] < 0.7


# ---------------------------------------------------------------------------
# C2 (refiner) — discrimination has a noise margin + a control-over-reactivity void.
# ---------------------------------------------------------------------------
def test_discrimination_requires_margin_not_bare_inequality():
    """C2: a razor-thin core>control gap (the live push_06 0.90 vs 0.733, margin 0.167... but
    here we use a sub-margin 0.10 gap below DISCRIM_MARGIN) must NOT claim discriminates. The
    bare strict `>` is replaced by a >= DISCRIM_MARGIN test."""
    core = {"signals": {"content_shift": 0.40, "total_shift": 0.40}}
    control = {"signals": {"content_shift": 0.30, "total_shift": 0.30}}  # margin 0.10 < 0.15
    d = intervention.compare_to_control(core, control)
    assert d["margin"] == pytest.approx(0.10)
    assert d["discrim_margin"] == pytest.approx(intervention.DISCRIM_MARGIN)
    assert d["discriminates"] is False


def test_discrimination_voided_when_control_over_reactive():
    """C2: a placebo/irrelevant control whose OWN content_shift is high (>= CONTROL_
    OVERREACTIVE_CUTOFF) is re-routing for a suppression that should change nothing — the
    textbook rung-1 over-reaction. Even with a margin above DISCRIM_MARGIN, the comparison is
    too noisy to attribute the core's move to the hazard, so discriminates is False and
    control_over_reactive is stamped True. (Live push_06: placebo content_shift 0.733.)"""
    core = {"signals": {"content_shift": 0.90, "total_shift": 0.90}}
    control = {"signals": {"content_shift": 0.733, "total_shift": 0.733}}  # over-reactive
    d = intervention.compare_to_control(core, control)
    assert d["margin"] == pytest.approx(0.167, abs=1e-3)   # margin alone clears 0.15...
    assert d["control_over_reactive"] is True
    assert d["discriminates"] is False                     # ...but the noisy control voids it


# ---------------------------------------------------------------------------
# B7 (refiner) — target_mitigation + placebo_null get a real do()-applied check.
# ---------------------------------------------------------------------------
def test_do_applied_target_mitigation_unmoved_is_not_applied():
    """B7: a target_mitigation do() that leaves the at-risk entity STILL present in its
    at-risk state (the model did not move it to safety) -> applied False/reason target_unmoved.
    Closes the gap where the entire person_in_hazard class returned 'not_checked'."""
    baseline = {"graph_a": {"nodes": []},
                "detected_objects": [{"object_id": "person_1", "label": "person", "state": "drowning"}]}
    post = {"graph_a": {"nodes": []},
            "detected_objects": [{"object_id": "person_1", "label": "person", "state": "drowning"}]}
    spec = {"intervention_type": "target_mitigation",
            "target": {"object_id": "person_1", "state": "drowning", "label": "person"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is False
    assert da["reason"] == "target_unmoved"


def test_do_applied_target_mitigation_moved_is_applied():
    """B7: a target_mitigation do() where the entity left the scene (moved to safety) ->
    applied True/reason target_removed."""
    baseline = {"graph_a": {"nodes": []},
                "detected_objects": [{"object_id": "person_1", "label": "person", "state": "drowning"}]}
    post = {"graph_a": {"nodes": []}, "detected_objects": []}
    spec = {"intervention_type": "target_mitigation",
            "target": {"object_id": "person_1", "state": "drowning", "label": "person"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is True
    assert da["reason"] == "target_removed"


def test_do_applied_placebo_null_disturbed_is_not_applied():
    """B7: a placebo_null do() is an inert non-event — the entity MUST persist unchanged. If
    the placebo entity's state flips, the model treated the null as a real intervention,
    corrupting the anti-confound baseline -> applied False/reason placebo_disturbed."""
    baseline = {"graph_a": {"nodes": []},
                "detected_objects": [{"object_id": "chair_1", "label": "chair", "state": "upright"}]}
    post = {"graph_a": {"nodes": []},
            "detected_objects": [{"object_id": "chair_1", "label": "chair", "state": "toppled"}]}
    spec = {"intervention_type": "placebo_null",
            "target": {"object_id": "chair_1", "state": "upright", "label": "chair"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is False
    assert da["reason"] == "placebo_disturbed"


def test_do_applied_placebo_null_unchanged_is_applied():
    """B7: a placebo entity that persists UNCHANGED -> applied True/reason placebo_unchanged
    (the inert baseline behaved as intended)."""
    baseline = {"graph_a": {"nodes": []},
                "detected_objects": [{"object_id": "chair_1", "label": "chair", "state": "upright"}]}
    post = {"graph_a": {"nodes": []},
            "detected_objects": [{"object_id": "chair_1", "label": "chair", "state": "upright"}]}
    spec = {"intervention_type": "placebo_null",
            "target": {"object_id": "chair_1", "state": "upright", "label": "chair"}}
    da = intervention.check_do_applied(baseline, post, spec)
    assert da["applied"] is True
    assert da["reason"] == "placebo_unchanged"


# ---------------------------------------------------------------------------
# C4 (refiner) — discrimination is stamped not_a_grounding_signal when no arm
# ever touched the GT core (gt_core_unobserved / core_not_declared).
# ---------------------------------------------------------------------------
def test_discrimination_not_a_grounding_signal_on_gt_core_unobserved(tmp_path):
    """C4 (refiner med): on a gt_core_unobserved headline the core arm suppressed a DECLARED
    non-core (never the GT core), so discrimination cannot be read as grounding evidence. The
    block must carry not_a_grounding_signal=True and discriminates must be nulled (the raw
    value preserved as discriminates_raw), so a reader scanning discrimination alone does not
    see a positive grounding signal the headline contradicts."""
    gt_dir = write_gt_dir(tmp_path)  # GT core = water_1, unobserved by the push06-like model
    baseline = intervention.intervention_baseline(
        _push06_like_result(), image_data_url="data:img", gt_dir=gt_dir)
    # A coherent post that holds U (same entities) so the run is not void.
    post = {"detected_objects": _push06_like_result()["detected_objects"],
            "causal_graph": _push06_like_result()["causal_graph"],
            "recommendations": _push06_like_result()["recommendations"], "disaster_level": 8}
    out = intervention.run_intervention(baseline, {"modality": "language"}, make_vlm_stub(post))
    assert out["verdict"]["cell"] == "gt_core_unobserved"
    d = out["discrimination"]
    assert d["not_a_grounding_signal"] is True
    assert d["not_a_grounding_reason"] == "gt_core_unobserved"
    assert d["discriminates"] is None
    assert "discriminates_raw" in d


# ---------------------------------------------------------------------------
# B7 (refiner) — u_compliance_only flags a verbatim echo with no falsifiable do().
# ---------------------------------------------------------------------------
def test_u_compliance_only_flag_when_verbatim_echo():
    """B7/B9 (refiner): after the U-gate redesign, raw_id_overlap / object_overlap are
    SECONDARY reuse-compliance diagnostics only (the real gate is state/topology). Under
    EMBED-BASELINE the embed forces raw_id_overlap == 1.0 on EVERY compliant arm, so the
    compliance stamp must fire whenever raw_id_overlap == 1.0, REGARDLESS of the do() reason
    (including source_removal) — no live arm may present the secondary id-overlap as an
    independent U hold."""
    # 'not_checked' arm: id echo -> compliance-only True.
    u_check = {"object_overlap": 1.0, "leaked": False, "cutoff": 0.7, "raw_id_overlap": 1.0}
    intervention._stamp_u_compliance_only(u_check, {"applied": True, "reason": "not_checked"})
    assert u_check["u_compliance_only"] is True
    # source_removal arm (push_34): id echo STILL stamps compliance-only True now (the old
    # narrow trigger left this False and presented id-overlap=1.0 as a clean hold).
    u_check2 = {"object_overlap": 1.0, "leaked": False, "cutoff": 0.7, "raw_id_overlap": 1.0}
    intervention._stamp_u_compliance_only(u_check2, {"applied": True, "reason": "source_removed"})
    assert u_check2["u_compliance_only"] is True
    # When ids actually churned (raw_id_overlap < 1.0), it is NOT a pure echo -> flag False.
    u_check3 = {"object_overlap": 0.8, "leaked": False, "cutoff": 0.7, "raw_id_overlap": 0.5}
    intervention._stamp_u_compliance_only(u_check3, {"applied": True, "reason": "source_removed"})
    assert u_check3["u_compliance_only"] is False


# ---------------------------------------------------------------------------
# B6 (refiner) — placebo prefers a causally-disjoint object; a non-disjoint
# placebo records placebo_overlap and downgrades discrimination.
# ---------------------------------------------------------------------------
def test_placebo_prefers_disjoint_object(tmp_path):
    """B6 (refiner med): when a disjoint non-hazard exists, the placebo picker prefers it over
    a non-hazard that shares the core's downstream targets. make_result's person_1 is
    downstream of the water core; an added isolated sign_1 has no edges -> it is preferred and
    placebo_overlap is False."""
    gt_dir = write_gt_dir(tmp_path)
    result = make_result()
    result["detected_objects"].append({"object_id": "sign_1", "label": "sign", "state": "upright"})
    result["causal_graph"]["nodes"].append(
        {"id": "sign_1", "label": "sign", "state": "upright", "hazardous": False})
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)
    enum = intervention.enumerate_candidates(baseline)
    assert enum["placebo_control"]["object_id"] == "sign_1"
    assert enum["placebo_control"]["placebo_overlap"] is False


def test_non_disjoint_placebo_downgrades_discrimination(tmp_path):
    """B6 (refiner med): when the ONLY non-hazard placebo is correlated with the core (push_06
    shape: the only deck object is downstream of the lone rec), the placebo is picked with
    placebo_overlap=True and a would-be discriminates=True is downgraded to False with
    reason placebo_overlap — the move cannot be shown hazard-specific rather than 'any
    suppression collapses the lone rec'."""
    gt_dir = write_gt_dir(tmp_path)
    baseline = intervention.intervention_baseline(make_result(), image_data_url="data:img", gt_dir=gt_dir)
    # person_1 is the only non-hazard AND is downstream of the water core -> non-disjoint.
    enum = intervention.enumerate_candidates(baseline)
    assert enum["placebo_control"]["object_id"] == "person_1"
    assert enum["placebo_control"]["placebo_overlap"] is True

    def vlm(image, prompt, spec):
        # Core re-routes; placebo stays static -> core would beat the control on raw numbers.
        if spec.get("intervention_type") == "placebo_null":
            return {"detected_objects": make_result()["detected_objects"],
                    "causal_graph": make_result()["causal_graph"],
                    "recommendations": make_result()["recommendations"], "disaster_level": 8}
        return _over_reactive_post(make_result()["detected_objects"])

    out = intervention.run_intervention(baseline, {"modality": "language"}, vlm)
    d = out["discrimination"]
    assert d["placebo_overlap"] is True
    assert d["discriminates"] is False
    assert d["discriminates_downgraded_reason"] == "placebo_overlap"


# --- B6 : fully-coupled cascade with NO clean control AND NO placebo -> UNDECIDABLE --------
def _fully_coupled_cascade_result():
    """A multi-fire cascade where EVERY detected object is a hazard (no non-hazard placebo
    exists) and the GT graph is fully connected (every non-core hazard's downstream targets
    overlap the core's, so no causally-disjoint real-hazard control exists either). This is
    the push_02 shape: both anti-confound mechanisms fail simultaneously."""
    return {
        "run_id": "run_cascade",
        "image_filename": "push_02_multi_fire_cascade.jpg",
        "prompt": "analyze",
        "caption": "a row of burning houses",
        "disaster_level": 9,
        "detected_objects": [
            {"object_id": "house_1", "label": "house", "state": "burning"},
            {"object_id": "house_2", "label": "house", "state": "burning"},
        ],
        "threats": [{"object_id": "house_1", "state": "burning"}],
        "recommendations": [
            {"rank": 1, "action": "Extinguish house_1.",
             "related_object_ids": ["house_1", "house_2"],
             "structured_reasoning": {"threat": "house_1", "state": "burning",
                                      "effect": "may_harm", "affected_objects": ["house_2"]}},
        ],
        "causal_graph": {
            "nodes": [
                {"id": "house_1", "label": "house", "state": "burning", "hazardous": True},
                {"id": "house_2", "label": "house", "state": "burning", "hazardous": True},
            ],
            "edges": [
                {"source": "house_1", "target": "house_2", "effect": "may_harm", "via_state": "burning"},
            ],
            "intervention_candidates": [
                {"threat": "house_1", "state": "burning", "outgoing_edge_count": 1},
            ],
        },
        "graph_b": {
            "nodes": [
                {"id": "house_1", "label": "house", "state": "burning", "hazardous": True},
                {"id": "house_2", "label": "house", "state": "burning", "hazardous": True},
            ],
            "edges": [
                {"source": "house_1", "target": "house_2", "effect": "may_harm", "via_state": "burning"},
            ],
            "suppression_pick": {"threat": "house_1", "state": "burning", "reason": "origin"},
        },
    }


_CASCADE_GT = {
    "image_filename": "push_02_multi_fire_cascade.jpg",
    "caption": "ZZ_SECRET_cascade_answerkey_marker",
    "schema_version": "2026-06-10",
    "nodes": [
        {"id": "house_1", "label": "house", "state": "burning", "hazardous": True, "inferred": False},
        {"id": "house_2", "label": "house", "state": "burning", "hazardous": True, "inferred": False},
    ],
    # Fully coupled: BOTH house_1 and house_2 point at the SAME downstream target (house_2 and
    # house_1 respectively, plus each other), so the non-core hazard's downstream set overlaps
    # the core's -> NO causally-disjoint real-hazard control exists. house_1 -> {house_2}, and
    # house_2 -> {house_1, house_2}; the shared house_2 forces control_overlap True.
    "edges": [
        {"source": "house_1", "target": "house_2", "effect": "may_harm", "via_state": "burning"},
        {"source": "house_2", "target": "house_1", "effect": "may_harm", "via_state": "burning"},
        {"source": "house_2", "target": "house_2", "effect": "may_harm", "via_state": "burning"},
    ],
}


def test_cascade_no_clean_control_no_placebo_is_undecidable(tmp_path):
    """B6: in a fully-coupled multi-hazard cascade with NO non-hazard object, the real-hazard
    control overlaps the core (control_overlap True) AND no placebo can be substituted
    (placebo_control None). run_intervention must NOT report discriminates=False off the
    correlated control (which reads as 'core failed to beat control' = masquerade-flavored);
    it must stamp the comparison structurally undecidable and null the bare bool."""
    gt_dir = write_gt_dir(tmp_path, gt=_CASCADE_GT)
    result = _fully_coupled_cascade_result()
    baseline = intervention.intervention_baseline(result, image_data_url="data:img", gt_dir=gt_dir)

    enum = intervention.enumerate_candidates(baseline)
    # Pre-conditions for the undecidable case: a correlated real-hazard control, no placebo.
    assert enum["control"] is not None
    assert enum["control"].get("control_overlap") is True
    assert enum["placebo_control"] is None  # every detected object is a hazard

    def vlm(image, prompt, spec):
        return _over_reactive_post(result["detected_objects"])

    out = intervention.run_intervention(baseline, {"modality": "language"}, vlm)
    d = out["discrimination"]
    assert d["discrimination_undecidable"] == "no_independent_control_in_cascade"
    assert d["discriminates"] is None
    assert d.get("discriminates_raw") is not None  # the raw bool retained for audit
    # And because discriminates is None (not False), no 'did not beat control' caveat fires.
    assert out["verdict"].get("discrimination_caveat") is False


def test_label_coref_terminology_synonym():
    """GT 'tanker' co-refers model 'tanker_truck' (curated synonym). Under the
    state-required tier, tanker/leaking disambiguates from a parked tanker."""
    lc = intervention._label_coref
    assert lc("tanker", "tanker_truck", "leaking", "leaking", False)
    assert lc("tanker", "tanker_truck", "leaking", "leaking", True)
    assert not lc("tanker", "tanker_truck", "leaking", "parked", True)


def test_label_coref_no_overmatch_via_generic_token():
    """'fire' must NOT co-refer 'fire_truck' via the shared 'fire' when states differ."""
    lc = intervention._label_coref
    assert not lc("fire", "fire_truck", "burning", "responding", True)
    assert not lc("fire", "fire_truck", "burning", "responding", False)


def test_gt_core_unobserved_reason_fluid_vs_miss(tmp_path, monkeypatch):
    """Fluid GT core the model wrote as inundation states (house/flooded, no water node)
    -> reason 'fluid_encoded_as_state'; nothing encoding the fluid -> 'not_perceived'."""
    gt = {"image_filename": "s.jpg",
          "nodes": [{"id": "water_1", "label": "water", "state": "rising", "hazardous": True},
                    {"id": "house_1", "label": "house", "state": "flooded", "hazardous": True}],
          "edges": [{"source": "water_1", "via_state": "rising", "effect": "may_harm", "target": "house_1"}]}
    (tmp_path / "s.jpg.gt.json").write_text(json.dumps(gt))
    monkeypatch.setattr(intervention, "GT_VERIFIED_DIR", tmp_path)

    fluid = {"image_filename": "s.jpg",
             "detected_objects": [{"object_id": "house_1", "label": "house", "state": "flooded"}],
             "causal_graph": {"nodes": [{"id": "house_1", "hazardous": True, "state": "flooded"}], "edges": []},
             "graph_b": {"nodes": [], "edges": []}, "disaster_level": 6, "pre_intervention_trust": {}}
    c = intervention.enumerate_candidates(intervention.intervention_baseline(fluid, None, gt_dir=tmp_path))
    assert c["should_be_core"] is None
    assert c["gt_core_unobserved"]["reason"] == "fluid_encoded_as_state"

    miss = {"image_filename": "s.jpg",
            "detected_objects": [{"object_id": "person_1", "label": "person", "state": "standing"}],
            "causal_graph": {"nodes": [{"id": "person_1", "hazardous": False, "state": "standing"}], "edges": []},
            "graph_b": {"nodes": [], "edges": []}, "disaster_level": 6, "pre_intervention_trust": {}}
    c2 = intervention.enumerate_candidates(intervention.intervention_baseline(miss, None, gt_dir=tmp_path))
    assert c2["gt_core_unobserved"]["reason"] == "not_perceived"
