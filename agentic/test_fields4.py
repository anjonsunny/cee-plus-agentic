"""size / how_soon: the two fields the model declares beside the quad.
Hermetic: no model. Covers malformed model output at every boundary."""
import re
from types import SimpleNamespace

import pytest

from agentic import fields4, recommend


def _rec(quad, rank=1):
    return {"rank": rank, "action": "cool fire_1", "reason": "r",
            "structured_reasoning": quad}


def _quad(**extra):
    return {"threat": "fire_1", "state": "burning", "effect": "may_harm",
            "affected_objects": ["man_1"], **extra}


def _raw_graph(node_extra=None, edge_extra=None):
    return {"causal_graph": {
        "nodes": [{"id": "fire_1", "label": "fire", "state": "burning",
                   "hazardous": True, "at_risk": False, **(node_extra or {})},
                  {"id": "man_1", "label": "man", "state": "standing",
                   "hazardous": False, "at_risk": True, "size": "large"}],
        "edges": [{"source": "fire_1", "target": "man_1", "effect": "may_harm",
                   "via_state": "burning", **(edge_extra or {})}]},
        "suppression_pick": {}}


def _record():
    return SimpleNamespace(detected_objects=[
        SimpleNamespace(object_id="fire_1", label="fire", state="burning", bbox=[0, 0, 5, 5]),
        SimpleNamespace(object_id="man_1", label="man", state="standing", bbox=[6, 6, 9, 9])],
        caption="", scene_caption="")


# ── the prompts ask for the fields, in closed words, with no scene tokens ──

def test_recommend_prompt_asks_for_both_fields():
    p = recommend.RECOMMEND_PROMPT
    assert "exactly the six keys" in p and "four keys" not in p
    assert "[small, medium, large]" in p and "[now, minutes, later]" in p
    assert "Leave\n  `size` and `how_soon` empty too." in p


def test_graph_b_prompt_asks_for_both_fields():
    p = recommend._graph_b_prompt(_record(), SimpleNamespace(threats=[], at_risk=[]))
    assert "- size: for a hazardous node only, how big or intense it is" in p
    assert "- how_soon: how soon the source has this effect on the target" in p
    assert "[small, medium, large]" in p and "[now, minutes, later]" in p
    assert "- via_state:" in p and "- suppression_pick:" in p


def test_added_lines_carry_no_id_shaped_token():
    for line in ("how big or intense that threat is right now",
                 "how soon the threat has this effect on the affected"):
        assert line in recommend.RECOMMEND_PROMPT
        assert not re.search(r"\b[a-z]+_\d+\b", line)


# ── quads ──

def test_parse_keeps_valid_fields_on_the_quad():
    _f, recs, _a, notes = recommend.parse_recommend(
        {"recommendations": [_rec(_quad(size="Large", how_soon="now"))]})
    q = recs[0]["structured_reasoning"]
    assert q["size"] == "large" and q["how_soon"] == "now"
    assert q["threat"] == "fire_1" and q["affected_objects"] == ["man_1"]
    assert not [n for n in notes if "dropped" in n]


def test_old_answer_without_fields_parses_as_before():
    _f, recs, _a, _n = recommend.parse_recommend({"recommendations": [_rec(_quad())]})
    q = recs[0]["structured_reasoning"]
    assert "size" not in q and "how_soon" not in q
    assert set(q) == {"threat", "state", "effect", "affected_objects"}


@pytest.mark.parametrize("bad", ["huge", 3, ["large"], {"a": 1}, True, "  "])
def test_bad_value_is_dropped_with_a_note_and_the_quad_is_kept(bad):
    _f, recs, _a, notes = recommend.parse_recommend(
        {"recommendations": [_rec(_quad(size=bad, how_soon=bad))]})
    q = recs[0]["structured_reasoning"]
    assert "size" not in q and "how_soon" not in q
    assert q["threat"] == "fire_1" and q["effect"] == "may_harm"
    if bad not in ("", None) and not (isinstance(bad, str) and not bad.strip() and False):
        assert any(n.startswith("size_dropped_rank_1") for n in notes) or bad == ""


def test_empty_fields_are_absent_without_a_note():
    _f, recs, _a, notes = recommend.parse_recommend(
        {"recommendations": [_rec(_quad(size="", how_soon=None))]})
    assert "size" not in recs[0]["structured_reasoning"]
    assert not [n for n in notes if "dropped" in n]


@pytest.mark.parametrize("raw,recs", [(None, []), ("x", [{}]), ([1, "a"], [None, 3]),
                                      ([{"structured_reasoning": "a -> b"}], [{"structured_reasoning": None}])])
def test_attach_quads_never_raises(raw, recs):
    fields4.attach_quads(raw, recs, [])


def test_prose_quad_still_recovers():
    _f, recs, _a, _n = recommend.parse_recommend(
        {"recommendations": [_rec("fire_1 -> burning -> may_harm -> man_1")]})
    assert recs[0]["structured_reasoning"]["threat"] == "fire_1"


# ── the model's own graph ──

def _norm(raw):
    from main import normalize_graph_b
    g = normalize_graph_b(raw, {"fire_1", "man_1"})
    return g, fields4.attach_graph(raw, g)


def test_graph_gets_size_on_hazard_and_how_soon_on_arrow():
    g, notes = _norm(_raw_graph({"size": "medium"}, {"how_soon": "minutes"}))
    nodes = {n["id"]: n for n in g["nodes"]}
    assert nodes["fire_1"]["size"] == "medium"
    assert "size" not in nodes["man_1"]          # only a hazard carries a size
    assert g["edges"][0]["how_soon"] == "minutes"
    assert notes == [] and "field_notes" not in g
    assert fields4.counts(g) == {"sized": 1, "timed": 1}


def test_old_graph_without_fields_is_unchanged():
    from main import normalize_graph_b
    raw = _raw_graph()
    raw["causal_graph"]["nodes"][1].pop("size")
    plain = normalize_graph_b(raw, {"fire_1", "man_1"})
    g, notes = _norm(raw)
    assert g == plain and notes == []


def test_bad_graph_values_keep_the_arrow_and_leave_a_note():
    g, notes = _norm(_raw_graph({"size": "enormous"}, {"how_soon": "soonish"}))
    assert len(g["edges"]) == 1 and "how_soon" not in g["edges"][0]
    assert "size" not in {n["id"]: n for n in g["nodes"]}["fire_1"]
    assert len(notes) == 2 and g["field_notes"] == notes


@pytest.mark.parametrize("raw", [None, "x", [], {"causal_graph": "x"},
                                 {"causal_graph": {"nodes": "x", "edges": [1, None]}},
                                 {"causal_graph": {"nodes": [None, {"size": 1}], "edges": [{"how_soon": []}]}}])
def test_attach_graph_never_raises(raw):
    assert isinstance(fields4.attach_graph(raw, {"nodes": [None, {}], "edges": [3, {}]}), list)
    assert fields4.attach_graph(raw, None) == []


def test_run_graph_b_carries_the_fields_and_reports_bad_ones():
    events = []
    raw = _raw_graph({"size": "large"}, {"how_soon": "whenever"})
    g = recommend.run_graph_b(_record(), SimpleNamespace(threats=[], at_risk=[]),
                              query_fn=lambda _p: raw,
                              on_event=lambda *a, **k: events.append((a, k)))
    assert {n["id"]: n for n in g["nodes"]}["fire_1"]["size"] == "large"
    assert "how_soon" not in g["edges"][0]
    assert any("graph_fields_noted" in str(e) for e in events)


# ── the graph built from the advice ──

def _graph_a():
    return {"nodes": [{"id": "fire_1"}, {"id": "man_1"}],
            "edges": [{"source": "fire_1", "effect": "may_harm", "target": "man_1"}]}


def test_graph_a_carries_fields_from_the_quads():
    g = _graph_a()
    notes = fields4.carry_to_graph_a(g, [_rec(_quad(size="small", how_soon="later"))])
    assert g["nodes"][0]["size"] == "small" and g["edges"][0]["how_soon"] == "later"
    assert notes == []


def test_disagreeing_advice_keeps_larger_and_sooner_and_notes_it():
    g = _graph_a()
    notes = fields4.carry_to_graph_a(g, [
        _rec(_quad(size="small", how_soon="later")),
        _rec(_quad(size="large", how_soon="now"), rank=2)])
    assert g["nodes"][0]["size"] == "large" and g["edges"][0]["how_soon"] == "now"
    assert any(n.startswith("size_conflict_fire_1") for n in notes)
    assert any(n.startswith("how_soon_conflict_fire_1_man_1") for n in notes)
    assert g["field_notes"] == notes


def test_benign_advice_gives_no_fields():
    g = _graph_a()
    fields4.carry_to_graph_a(g, [{**_rec(_quad(size="large", how_soon="now")), "benign": True}])
    assert "size" not in g["nodes"][0] and "how_soon" not in g["edges"][0]


@pytest.mark.parametrize("g,recs", [(None, []), ({}, None), ({"nodes": "x", "edges": None}, [1, None, {}]),
                                    ({"nodes": [1], "edges": [2]}, [{"structured_reasoning": {"threat": 5, "affected_objects": "x", "size": 9}}])])
def test_carry_never_raises(g, recs):
    assert isinstance(fields4.carry_to_graph_a(g, recs), list)


def test_word_only_accepts_the_closed_words():
    assert fields4.word(" NOW ", fields4.HOW_SOON) == "now"
    assert fields4.word("soon", fields4.HOW_SOON) == ""
    assert fields4.word(None, fields4.SIZES) == ""


# ── the weights read the two fields; a graph without them weighs as before ──

def _result(edges, nodes=None, key="graph_b"):
    objs = [{"object_id": "fire_1", "label": "fire"}, {"object_id": "spill_1", "label": "spill"},
            {"object_id": "man_1", "label": "man"}, {"object_id": "tree_1", "label": "tree"}]
    return {"detected_objects": objs,
            key: {"nodes": nodes or [], "edges": [
                {"source": s, "effect": e, "target": t, **({"how_soon": w} if w else {})}
                for s, e, t, w in edges]}}


def test_missing_fields_weigh_exactly_as_before():
    from agentic import chain4, declared4
    plain = _result([("fire_1", "may_harm", "man_1", ""), ("spill_1", "may_spread_to", "tree_1", "")])
    assert declared4.declared(plain) == {"fire_1": 4.0, "spill_1": 1.5}
    a = chain4.weigh(_result([("fire_1", "may_harm", "man_1", "")], key="causal_graph"))
    assert a["total"] == 4.0 and a["arrows"][0]["fields"] == 1.0


def test_how_soon_and_size_scale_the_arrow():
    from agentic import chain4, declared4
    r = _result([("fire_1", "may_harm", "man_1", "now"), ("spill_1", "may_harm", "man_1", "later")],
                nodes=[{"id": "fire_1", "size": "large"}, {"id": "spill_1", "size": "small"}])
    d = declared4.declared(r)
    assert d["fire_1"] == pytest.approx(4.0 * 1.5 * 1.5) and d["spill_1"] == 4.0
    w = chain4.weigh({**r, "causal_graph": r["graph_b"]})
    top = max(w["arrows"], key=lambda x: x["weight"])
    assert (top["how_soon"], top["size"], top["weight"]) == ("now", "large", 9.0)
    # gentle on purpose: the strongest arrow is 2.25x a plain one, not 9x
    assert max(chain4.SOON_X.values()) * max(chain4.SIZE_X.values()) == 2.25


@pytest.mark.parametrize("g", [None, "x", {"nodes": "x", "edges": 3},
                               {"nodes": [None, {"id": 1, "size": 7}, {"size": "vast"}],
                                "edges": [None, {"how_soon": ["now"]}, {"how_soon": "soonish"}]}])
def test_fields_of_never_raises_and_ignores_other_words(g):
    from agentic import chain4
    assert chain4.fields_of(g) == ({}, {})
    assert chain4.field_factor("a", "may_harm", "b", chain4.fields_of(g)) == 1.0
    assert chain4.field_factor("a", "may_harm", "b", None) == 1.0
