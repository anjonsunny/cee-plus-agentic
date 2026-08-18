"""JUDGES.md step 1 — the runoff judge: candidate selection, twin verdicts,
agreement display, capture completeness. Hermetic: judges are scripted."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.judge_runoff import (ANSWER_A, ANSWER_B, EQUAL, IMAGE_PREFIX,
                                  GRAPH_RUNOFF_PROMPT, RECS_RUNOFF_PROMPT,
                                  _graph_key, _rec_key, pick_pair, runoff,
                                  runoff_graph_b, runoff_recommendations)


def _rec(threat, effect, affected, action="act", reason="because"):
    return {"rank": 1, "action": action, "reason": reason,
            "structured_reasoning": {"threat": threat, "state": "bad",
                                     "effect": effect,
                                     "affected_objects": list(affected)}}


def _g(*edges):
    return {"nodes": [], "edges": [{"source": s, "effect": e, "target": t}
                                   for s, e, t in edges]}


# ── candidate selection ─────────────────────────────────────────────────

def test_unanimous_probes_need_no_runoff():
    """Judging two copies of the same answer measures nothing."""
    same = [[_rec("fire_1", "may_harm", ["p_1"])]] * 5
    assert pick_pair(same, _rec_key) is None


def test_rewording_is_not_a_different_candidate():
    """Identity is the causal claim, not the prose — the same
    wording-invariance rule the intervention shift uses."""
    a = [_rec("fire_1", "may_harm", ["p_1"], action="Evacuate p_1 now")]
    b = [_rec("fire_1", "may_harm", ["p_1"], action="Get p_1 out immediately")]
    assert pick_pair([a, b, a, b, a], _rec_key) is None


def test_clear_top_two_by_votes():
    x = [_rec("fire_1", "may_harm", ["p_1"])]
    y = [_rec("dust_1", "worsens", ["p_1"])]
    z = [_rec("spill_1", "isolates", ["p_1"])]
    i, j = pick_pair([x, x, x, y, y, z], _rec_key)
    assert _rec_key([x, x, x, y, y, z][i]) == _rec_key(x)
    assert _rec_key([x, x, x, y, y, z][j]) == _rec_key(y)


def test_five_way_tie_picks_the_most_different_pair():
    """D_aerial live: five answers, one vote each. One comparison should carry
    the maximum information, so the most-different pair is judged."""
    near1 = [_rec("fire_1", "may_harm", ["p_1"])]
    near2 = [_rec("fire_1", "may_harm", ["p_2"])]
    far = [_rec("spill_1", "isolates", ["t_1"]),
           _rec("dust_1", "worsens", ["t_2"])]
    items = [near1, near2, far]
    i, j = pick_pair(items, _rec_key)
    assert {tuple(_rec_key(items[i])), tuple(_rec_key(items[j]))} != \
        {tuple(_rec_key(near1)), tuple(_rec_key(near2))}


# ── the runoff itself, with scripted twins ──────────────────────────────

def _judge_saying(verdict):
    return lambda prompt: f"reasoning here.\nVERDICT: [{verdict}]"


def test_twins_agree_and_disagree_are_computed():
    agree = runoff("recommendations", "p", "A", "B", {},
                   judge_fn=_judge_saying(ANSWER_A),
                   judge_image_fn=_judge_saying(ANSWER_A), n_probes=3)
    split = runoff("recommendations", "p", "A", "B", {},
                   judge_fn=_judge_saying(ANSWER_A),
                   judge_image_fn=_judge_saying(ANSWER_B), n_probes=3)
    assert agree["twins_agree"] is True
    assert split["twins_agree"] is False
    assert split["text"]["verdict"] == ANSWER_A
    assert split["image"]["verdict"] == ANSWER_B


def test_no_image_twin_is_unknown_not_agreement():
    r = runoff("graph_b", "p", "A", "B", {},
               judge_fn=_judge_saying(EQUAL), judge_image_fn=None, n_probes=3)
    assert r["image"] is None and r["twins_agree"] is None


def test_the_image_twin_gets_the_extraction_constraint():
    """Sunny's framing rides on the image twin's prompt: the entity list IS
    the extraction; the image is context."""
    seen = {}

    def image_fn(prompt):
        seen["prompt"] = prompt
        return "VERDICT: [answer_a]"
    runoff("graph_b", "the shared prompt", "A", "B", {},
           judge_fn=_judge_saying(ANSWER_A), judge_image_fn=image_fn,
           n_probes=1)
    assert seen["prompt"].startswith(IMAGE_PREFIX)
    assert "the shared prompt" in seen["prompt"]


def test_capture_fields_ride_on_every_verdict():
    """Each result is a §9.2 preference record in waiting: candidates
    verbatim, prompt version, reasoning text, intervention slot open."""
    r = runoff("recommendations", "p", "CAND_A_TEXT", "CAND_B_TEXT",
               {"invented_ids_a": [], "invented_ids_b": ["ghost_7"]},
               judge_fn=_judge_saying(ANSWER_A), n_probes=3)
    assert r["candidate_a"] == "CAND_A_TEXT"
    assert r["prompt_version"].startswith("runoff-v")
    assert "reasoning" in r["text"]
    assert r["verified_by_intervention"] is None
    assert r["code_facts"]["invented_ids_b"] == ["ghost_7"]


def test_a_judge_that_crashes_never_takes_the_run_down():
    def boom(prompt):
        raise RuntimeError("ollama down")
    r = runoff("graph_b", "p", "A", "B", {}, judge_fn=boom, n_probes=3)
    assert r["text"]["verdict"] == "unclear"
    assert r["text"]["errors"]


# ── prompt neutrality (iron rule 5) ─────────────────────────────────────

def test_templates_carry_no_scene_tokens():
    import re
    for tpl in (RECS_RUNOFF_PROMPT, GRAPH_RUNOFF_PROMPT):
        bare = re.sub(r"\{[a-z_]+\}", "", tpl)
        assert not re.search(r"\b[a-z]+_\d+\b", bare), "id-shaped token in template"
        for word in ("fire", "pool", "tanker", "aerial", "collapse", "park"):
            assert word not in bare.lower()


# ── end-to-end through both controls ────────────────────────────────────

def test_runoff_reaches_stage4_result_and_twin_stays_identical():
    from agentic.test_recommend import _asm, _rec_answer, _record, _script
    from agentic.test_recommend import _rec as _mk
    from agentic.graph_s4 import run_s4_graph
    from agentic.recommend import run_stage4

    seq = [[_mk("building_1", "collapsed", "may_harm", ["person_1"])],
           [_mk("dust_1", "rising", "worsens", ["person_1"])]]
    box = {"i": 0}

    def probe_fn(prompt):
        a = _rec_answer(seq[box["i"] % 2]); box["i"] += 1
        return a

    def judge(prompt):
        return "thinking.\nVERDICT: [answer_a]"

    kw = dict(query_fn=_script(), probe_fn=probe_fn, judge_fn=judge,
              n_probes=4)
    ev1, ev2 = [], []
    box["i"] = 0
    r1 = run_stage4(_record(), _asm(), "", on_event=ev1.append, **kw)
    box["i"] = 0
    r2 = run_s4_graph(_record(), _asm(), "", on_event=ev2.append, **kw)
    ro = r1.runoff_judge
    assert ro.get("recommendations", {}).get("text", {}).get("verdict") == "answer_a"
    assert r1.model_dump() == r2.model_dump()          # byte-identical twin
    assert [e["type"] for e in ev1] == [e["type"] for e in ev2]
    assert "runoff_judged" in [e["type"] for e in ev1]
    # the event is capture-complete: candidates + reasoning ride in it
    e = next(x for x in ev1 if x["type"] == "runoff_judged")
    assert e["candidate_a"] and e["text_reasoning"]


def test_runoff_off_without_a_judge():
    from agentic.test_recommend import _asm, _record, _script
    from agentic.recommend import run_stage4
    r = run_stage4(_record(), _asm(), "", query_fn=_script())
    assert r.runoff_judge == {}


# ── F51: the loophole, closed and pinned ────────────────────────────────

def test_invented_is_defined_in_both_prompts():
    """F51, live on A_fire run one: rule 3 said "every entity ID" and the
    judge argued — correctly, by the letter — that 'trees' is not an id, so
    naming it broke no rule. It then preferred the candidate code had counted
    two invented entities against. Code and prompt defined "invented"
    differently; now the prompt defines it the way the code counts it."""
    import re

    def flat(t):
        return re.sub(r"\s+", " ", t)
    for tpl in (RECS_RUNOFF_PROMPT, GRAPH_RUNOFF_PROMPT):
        assert "INVENTED" in tpl
        assert "whether or not it looks like an id" in flat(tpl)
    assert ("reason that merely mentions a danger does not act on it"
            in flat(RECS_RUNOFF_PROMPT))


def test_the_audience_is_declared_and_delegation_is_not_invention():
    """F52 (Sunny): the recommendations are FOR the emergency response team —
    "the point is they can use it." A team summoning the specialist unit
    ("alert the fire department") is coordination, not invention: v2's blanket
    invented rule punished it, contradicting F27, and judge-vs-code split in
    the OPPOSITE direction from F51 — same class of defect, our definition."""
    import re
    flat = re.sub(r"\s+", " ", RECS_RUNOFF_PROMPT)
    assert "FOR the emergency response team" in flat
    assert "delegation is a legitimate response" in flat
    assert "are NOT scene entities and are not inventions" in flat
    # and the example that taught delegation must NOT be scene-flavored —
    # my first draft wrote "alert the fire department about the burning
    # house" into the TEMPLATE, and the neutrality test caught it (F2's
    # lesson: worked examples leak into unrelated scenes)


def test_probe_graphs_are_deduped_before_the_judge_sees_them():
    """A_fire ui_6ddd5df6 verbatim: the same edge three times in one probe."""
    from agentic.judge_runoff import _fmt_graph
    g = {"edges": [{"source": "smoke_1", "effect": "may_harm",
                    "target": "person_1"}] * 3}
    assert _fmt_graph(g).count("smoke_1") == 1


def test_prompt_version_bumped_so_v1_pairs_are_separable():
    from agentic.judge_runoff import PROMPT_VERSION
    assert PROMPT_VERSION == "runoff-v3"


def test_every_votes_reasoning_is_captured():
    """A majority exemplar cannot show whether a loophole reading was
    unanimous or a 2-1 lean. All votes ride, verdict-tagged."""
    seq = iter(["blah A.\nVERDICT: [answer_a]", "blah B.\nVERDICT: [answer_b]",
                "blah A2.\nVERDICT: [answer_a]"])
    r = runoff("graph_b", "p", "A", "B", {}, judge_fn=lambda p: next(seq),
               n_probes=3)
    ar = r["text"]["all_reasoning"]
    assert [x["verdict"] for x in ar] == ["answer_a", "answer_b", "answer_a"]
    assert ar[1]["text"].startswith("blah B")
