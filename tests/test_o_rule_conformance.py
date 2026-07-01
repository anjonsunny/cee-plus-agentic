"""Section O — Rule conformance checker (module M7).

The checker runs the schema rulebook against the MODEL'S graphs, no GT
needed. These tests feed it hand-built graphs that each break exactly one
rule, plus a clean graph that breaks none.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import main_module  # noqa: E402, F401


def _node(nid, label, state, hazardous=False):
    return {"id": nid, "label": label, "state": state, "hazardous": hazardous, "inferred": False}


def _edge(src, tgt, effect, via):
    return {"source": src, "target": tgt, "effect": effect, "via_state": via}


def _rules_of(main_module, graph):
    return {v["rule"] for v in main_module.check_graph_rule_conformance(graph, "test")}


# O1 — clean graph: zero violations.
@pytest.mark.blocking
def test_o1_clean_graph_no_violations(main_module):
    graph = {
        "nodes": [
            _node("fire_1", "fire", "spreading", True),
            _node("smoke_1", "smoke", "billowing", True),
            _node("house_1", "house", "intact"),
            _node("person_1", "person", "stationary"),
        ],
        "edges": [
            _edge("fire_1", "house_1", "may_spread_to", "spreading"),
            _edge("fire_1", "smoke_1", "increases_risk_to", "spreading"),
            _edge("smoke_1", "person_1", "may_harm", "billowing"),
        ],
    }
    assert main_module.check_graph_rule_conformance(graph, "test") == []


# O2 — empty graph (negative control): zero violations.
@pytest.mark.blocking
def test_o2_empty_graph_clean(main_module):
    assert main_module.check_graph_rule_conformance({"nodes": [], "edges": []}, "t") == []


# O3 — the label triad lie: fluid may_harm an already-hazardous target.
@pytest.mark.blocking
def test_o3_fluid_may_harm_hazardous_target(main_module):
    graph = {
        "nodes": [
            _node("water_1", "water", "rising", True),
            _node("house_1", "house", "flooded", True),
        ],
        "edges": [_edge("water_1", "house_1", "may_harm", "rising")],
    }
    assert "may_harm_hazardous_target" in _rules_of(main_module, graph)


# O4 — fluid uses a non-victim effect on a person.
@pytest.mark.blocking
def test_o4_fluid_wrong_effect_for_person(main_module):
    graph = {
        "nodes": [
            _node("water_1", "water", "rising", True),
            _node("person_1", "person", "stationary"),
        ],
        "edges": [_edge("water_1", "person_1", "increases_risk_to", "rising")],
    }
    assert "fluid_wrong_effect_for_person" in _rules_of(main_module, graph)


# O5 — may_spread_to between two already-hazardous entities.
@pytest.mark.blocking
def test_o5_spread_between_hazards(main_module):
    graph = {
        "nodes": [
            _node("house_1", "house", "burning", True),
            _node("house_2", "house", "burning", True),
        ],
        "edges": [
            _edge("house_1", "house_2", "may_spread_to", "burning"),
            _edge("house_1", "house_1", "worsens", "burning"),
            _edge("house_2", "house_2", "worsens", "burning"),
        ],
    }
    assert "spread_between_hazards" in _rules_of(main_module, graph)


# O6 — one-way worsens between distinct entities.
@pytest.mark.blocking
def test_o6_one_way_worsens(main_module):
    graph = {
        "nodes": [
            _node("fire_1", "fire", "spreading", True),
            _node("tanker_1", "tanker", "leaking", True),
        ],
        "edges": [
            _edge("fire_1", "tanker_1", "worsens", "spreading"),
            _edge("tanker_1", "tanker_1", "worsens", "leaking"),
        ],
    }
    assert "one_way_worsens" in _rules_of(main_module, graph)


# O7 — uncoupled obstruction edge to a person.
@pytest.mark.blocking
def test_o7_uncoupled_obstruction(main_module):
    graph = {
        "nodes": [
            _node("tree_1", "tree", "fallen", True),
            _node("person_1", "person", "stationary"),
        ],
        "edges": [
            _edge("tree_1", "person_1", "blocks_access_to", "fallen"),
            _edge("tree_1", "tree_1", "worsens", "fallen"),
        ],
    }
    assert "uncoupled_obstruction" in _rules_of(main_module, graph)


# O8 — entrapment pattern is NOT flagged (water isolates rooftop family).
@pytest.mark.blocking
def test_o8_entrapment_isolates_allowed(main_module):
    graph = {
        "nodes": [
            _node("water_1", "water", "rising", True),
            _node("man_1", "man", "stationary"),
        ],
        "edges": [_edge("water_1", "man_1", "isolates", "rising")],
    }
    assert "uncoupled_obstruction" not in _rules_of(main_module, graph)


# O9 — smoke-superset violation.
@pytest.mark.blocking
def test_o9_smoke_superset(main_module):
    graph = {
        "nodes": [
            _node("house_1", "house", "burning", True),
            _node("smoke_1", "smoke", "billowing", True),
            _node("person_1", "person", "stationary"),
        ],
        "edges": [
            _edge("house_1", "person_1", "may_harm", "burning"),
            _edge("house_1", "smoke_1", "increases_risk_to", "burning"),
            _edge("smoke_1", "smoke_1", "worsens", "billowing"),
        ],
    }
    assert "smoke_superset_violation" in _rules_of(main_module, graph)


# O10 — structural basics: self-loop effect, orphan hazard, via mismatch,
# non-hazardous source, bad effect, unresolved endpoint.
@pytest.mark.blocking
def test_o10_structural_basics(main_module):
    graph = {
        "nodes": [
            _node("fire_1", "fire", "spreading", True),
            _node("car_1", "car", "burning", True),  # orphan hazard
            _node("person_1", "person", "stationary"),
        ],
        "edges": [
            _edge("fire_1", "fire_1", "may_harm", "spreading"),       # self-loop not worsens
            _edge("fire_1", "person_1", "may_harm", "burning"),       # via mismatch
            _edge("person_1", "fire_1", "may_harm", "stationary"),    # non-hazardous source + via not hazard-bearing
            _edge("fire_1", "ghost_9", "may_harm", "spreading"),      # unresolved
            _edge("fire_1", "person_1", "zaps", "spreading"),         # bad effect
        ],
    }
    rules = _rules_of(main_module, graph)
    for expected in (
        "self_loop_not_worsens", "via_state_mismatch", "edge_from_non_hazardous",
        "unresolved_endpoint", "effect_not_in_vocabulary", "hazardous_node_no_edges",
    ):
        assert expected in rules, f"missing {expected}; got {rules}"


# O11 — aggregate wrapper counts both graphs.
@pytest.mark.blocking
def test_o11_aggregate_counts(main_module):
    bad = {
        "nodes": [
            _node("water_1", "water", "rising", True),
            _node("house_1", "house", "flooded", True),
        ],
        "edges": [_edge("water_1", "house_1", "may_harm", "rising")],
    }
    rc = main_module.compute_rule_conformance(bad, bad)
    assert rc["n_violations"] == 2  # one per graph
    assert rc["by_rule"].get("may_harm_hazardous_target") == 2


# O13 — redundant instancing: six causal clones flagged, three are fine.
@pytest.mark.blocking
def test_o13_redundant_instancing(main_module):
    def flood_scene(n_houses):
        nodes = [_node("water_1", "water", "rising", True)]
        edges = []
        for i in range(1, n_houses + 1):
            nodes.append(_node(f"house_{i}", "house", "flooded", True))
            edges.append(_edge("water_1", f"house_{i}", "increases_risk_to", "rising"))
        return {"nodes": nodes, "edges": edges}

    six = _rules_of(main_module, flood_scene(6))
    three = _rules_of(main_module, flood_scene(3))
    assert "redundant_instancing" in six
    assert "redundant_instancing" not in three


# O14 — causally DISTINCT nodes are never flagged as redundant.
@pytest.mark.blocking
def test_o14_distinct_nodes_not_flagged(main_module):
    # Six houses but in three different causal situations: flooded,
    # collapsing, intact-in-trajectory. Largest identical group is 2.
    nodes = [_node("water_1", "water", "rising", True)]
    edges = []
    states = ["flooded", "flooded", "collapsing", "collapsing", "intact", "intact"]
    effects = {"flooded": "increases_risk_to", "collapsing": "increases_risk_to", "intact": "may_spread_to"}
    for i, st in enumerate(states, 1):
        nodes.append(_node(f"house_{i}", "house", st, st != "intact"))
        edges.append(_edge("water_1", f"house_{i}", effects[st], "rising"))
    for i, st in enumerate(states, 1):
        if st == "collapsing":
            edges.append(_edge(f"house_{i}", f"house_{i}", "worsens", "collapsing"))
    rules = _rules_of(main_module, {"nodes": nodes, "edges": edges})
    assert "redundant_instancing" not in rules


# O15 — node budget: a 14-node graph trips the cap.
@pytest.mark.blocking
def test_o15_node_budget(main_module):
    nodes = [_node("water_1", "water", "rising", True)]
    edges = []
    for i in range(1, 14):
        st = "flooded" if i % 2 else "intact"
        nodes.append(_node(f"e_{i}", f"label{i}", st, st == "flooded"))
        eff = "increases_risk_to" if st == "flooded" else "may_spread_to"
        edges.append(_edge("water_1", f"e_{i}", eff, "rising"))
    rules = _rules_of(main_module, {"nodes": nodes, "edges": edges})
    assert "node_budget_exceeded" in rules


# O16 — the generalized rule: a NON-fluid source (flying sign) may_harm an
# already-hazardous house is caught too (push_18 case).
@pytest.mark.blocking
def test_o16_non_fluid_may_harm_hazardous_target(main_module):
    graph = {
        "nodes": [
            _node("sign_1", "sign", "approaching", True),
            _node("house_1", "house", "collapsing", True),
        ],
        "edges": [
            _edge("sign_1", "house_1", "may_harm", "approaching"),
            _edge("house_1", "house_1", "worsens", "collapsing"),
        ],
    }
    assert "may_harm_hazardous_target" in _rules_of(main_module, graph)


# O17 — distress states belong to living beings only: a "trapped car" is
# flagged; the trapped person inside it is not.
@pytest.mark.blocking
def test_o17_distress_state_on_non_living(main_module):
    graph = {
        "nodes": [
            _node("water_1", "water", "rising", True),
            _node("car_1", "car", "trapped"),       # wrong: vehicle in distress
            _node("driver_1", "driver", "trapped"), # right: living being
        ],
        "edges": [
            _edge("water_1", "car_1", "may_spread_to", "rising"),
            _edge("water_1", "driver_1", "may_harm", "rising"),
        ],
    }
    violations = main_module.check_graph_rule_conformance(graph, "test")
    flagged = [v["detail"] for v in violations if v["rule"] == "distress_state_on_non_living"]
    assert any("car_1" in d for d in flagged), "trapped car must be flagged"
    assert not any("driver_1" in d for d in flagged), "trapped driver is legitimate"


# O18 — people are never clone-flagged: six identical stranded people pass;
# six identical flooded houses (O13) still fail.
@pytest.mark.blocking
def test_o18_people_exempt_from_redundant_instancing(main_module):
    nodes = [_node("water_1", "water", "rising", True)]
    edges = []
    for i in range(1, 7):
        nodes.append(_node(f"person_{i}", "person", "stranded"))
        edges.append(_edge("water_1", f"person_{i}", "isolates", "rising"))
    rules = _rules_of(main_module, {"nodes": nodes, "edges": edges})
    assert "redundant_instancing" not in rules


# O19 — minimal self-loop rule: a loop alongside real edges is flagged; the
# placeholder loop on an otherwise edge-less hazard is not.
@pytest.mark.blocking
def test_o19_redundant_self_loop(main_module):
    graph = {
        "nodes": [
            _node("fire_1", "fire", "spreading", True),
            _node("smoke_1", "smoke", "billowing", True),
            _node("car_9", "car", "crushed", True),  # orphan: loop is correct
        ],
        "edges": [
            _edge("fire_1", "smoke_1", "increases_risk_to", "spreading"),
            _edge("fire_1", "fire_1", "worsens", "spreading"),   # redundant
            _edge("car_9", "car_9", "worsens", "crushed"),        # legitimate
        ],
    }
    violations = main_module.check_graph_rule_conformance(graph, "test")
    flagged = [v["detail"] for v in violations if v["rule"] == "redundant_self_loop"]
    assert any("fire_1" in d for d in flagged)
    assert not any("car_9" in d for d in flagged)


# O21 — fluid-as-object: an inundation state with no matching fluid node fires;
# the same state paired with the fluid node emitted is clean.
@pytest.mark.blocking
def test_o21_fluid_encoded_as_state(main_module):
    # house 'flooded', no water entity -> the fluid was collapsed into a state.
    bad = {
        "nodes": [
            _node("house_1", "house", "flooded", True),
            _node("person_1", "person", "stranded", True),
        ],
        "edges": [_edge("house_1", "house_1", "worsens", "flooded")],
    }
    assert "fluid_encoded_as_state" in _rules_of(main_module, bad)

    # same scene done right: water_1 nodalized -> no fluid-as-state violation.
    good = {
        "nodes": [
            _node("water_1", "water", "rising", True),
            _node("house_1", "house", "flooded", True),
        ],
        "edges": [
            _edge("water_1", "house_1", "increases_risk_to", "rising"),
            _edge("water_1", "water_1", "worsens", "rising"),
        ],
    }
    assert "fluid_encoded_as_state" not in _rules_of(main_module, good)

    # ambiguous inundation (engulfed) is deliberately NOT mapped -> never fires.
    amb = {"nodes": [_node("figure_1", "person", "engulfed", True)],
           "edges": [_edge("figure_1", "figure_1", "worsens", "engulfed")]}
    assert "fluid_encoded_as_state" not in _rules_of(main_module, amb)
