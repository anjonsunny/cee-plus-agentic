"""The Stage 4 card judge (F24/F28) — advisory, display-only, 5-probe vote.

Hermetic: every judge call is a scripted judge_fn. No models.

Run:  pytest agentic/test_judge_card.py -q
"""
from __future__ import annotations

from agentic.judge_card import (ACTION_ALIGNMENT_PROMPT, ALIGNED, DIFFERENT,
                                NOT_ALIGNED, REASON_QUAD_PROMPT, SAME,
                                UNCLEAR, judge_card, judge_cards, read_verdict)

PROMPTS = (ACTION_ALIGNMENT_PROMPT, REASON_QUAD_PROMPT)

CARD = {"rank": 1, "action": "Evacuate person_1 from house_1.",
        "reason": "Because house_1 is burning, it may_harm person_1.",
        "structured_reasoning": {"threat": "house_1", "state": "burning",
                                 "effect": "may_harm",
                                 "affected_objects": ["person_1"]}}


def _answer(prose, structure, same=SAME,
            preamble="Let me think step by step. ..."):
    """Each question now has its own prompt, so a scripted judge answers by
    looking at which prompt it was handed."""
    def reply(prompt):
        if "same claim" in prompt:
            return f"{preamble}\n\nVERDICT: {same}"
        return f"{preamble}\n\nPROSE: {prose}\nSTRUCTURE: {structure}"
    return reply


def _cycle_prompts(replies):
    """Rotate through per-probe answers, dispatching each by prompt.

    judge_card asks each QUESTION's probes consecutively, so a plain rotation
    over the call counter gives every question the same sequence of replies."""
    box = {"i": -1}
    def fn(prompt):
        box["i"] += 1
        return replies[box["i"] % len(replies)](prompt)
    fn.calls = box
    return fn


# ── three prompts, one question each ───────────────────────────────────

def test_the_questions_are_grouped_by_what_they_measure_against():
    """Two prompts: both alignment verdicts share one, because each is measured
    against the SAME action. The same-claim question compares the explanations
    to EACH OTHER, so it gets its own — mixed in, the alignment framing bled
    into it."""
    seen = []
    def fn(p):
        seen.append(p)
        return ("VERDICT: " + SAME if "same claim" in p
                else f"PROSE: {ALIGNED}\nSTRUCTURE: {ALIGNED}")
    judge_card(CARD, "scene", judge_fn=fn, n_probes=5)
    assert len(seen) == 10          # 2 prompts x 5 probes
    assert len(set(seen)) == 2


def test_every_prompt_showing_the_structure_explains_the_notation():
    """The same-claim question asks about "the same source of harm" and "the
    direction reversed"; neither is answerable in a notation whose direction
    was never stated."""
    for tpl in PROMPTS:
        assert "source of harm" in tpl


def test_five_probes_are_asked_per_question_and_the_majority_wins():
    """At temperature 0 a wrong verdict is wrong 5/5 and looks confident. The
    boundary cases are where a single sample is least informative."""
    fn = _cycle_prompts([_answer(ALIGNED, ALIGNED), _answer(ALIGNED, ALIGNED),
                         _answer(NOT_ALIGNED, ALIGNED),
                         _answer(ALIGNED, ALIGNED), _answer(ALIGNED, ALIGNED)])
    v = judge_card(CARD, "scene", judge_fn=fn, n_probes=5)
    assert v["prose"]["verdict"] == ALIGNED
    assert v["prose"]["votes"] == 4 and v["prose"]["n"] == 5


def test_the_vote_split_rides_beside_the_verdict():
    """'aligned 5/5' and 'aligned 3/5' are different findings — reporting only
    the winner throws away the more useful half."""
    fn = _cycle_prompts([_answer(ALIGNED, ALIGNED)] * 5)
    assert judge_card(CARD, "scene", judge_fn=fn,
                      n_probes=5)["prose"]["unanimous"] is True
    fn2 = _cycle_prompts([_answer(ALIGNED, ALIGNED)] * 3
                         + [_answer(NOT_ALIGNED, ALIGNED)] * 2)
    v2 = judge_card(CARD, "scene", judge_fn=fn2, n_probes=5)
    assert v2["prose"]["unanimous"] is False
    assert v2["prose"]["counts"] == {ALIGNED: 3, NOT_ALIGNED: 2}


def test_a_tie_reports_the_size_of_the_winning_bloc():
    """A 2-2 split used to render as '1/5' because the reported count was how
    many probes literally said 'unclear' — a real disagreement dressed up as
    near-total confusion."""
    fn = _cycle_prompts([_answer(ALIGNED, ALIGNED),
                         _answer(NOT_ALIGNED, ALIGNED)])
    v = judge_card(CARD, "scene", judge_fn=fn, n_probes=4)
    assert v["prose"]["verdict"] == UNCLEAR
    assert v["prose"]["votes"] == 2 and v["prose"]["tie"] is True


def test_the_three_verdicts_are_independent():
    fn = _cycle_prompts([_answer(ALIGNED, NOT_ALIGNED, DIFFERENT)] * 5)
    v = judge_card(CARD, "scene", judge_fn=fn, n_probes=5)
    assert v["prose"]["verdict"] == ALIGNED
    assert v["structure"]["verdict"] == NOT_ALIGNED
    assert v["same_claim"]["verdict"] == DIFFERENT


def test_the_criterion_says_specificity_is_not_a_difference():
    """The observed failure: the judge called a card different because the
    prose said "nearby structures" where the quad named one structure by id.
    That is the quad being MORE SPECIFIC, not a second claim."""
    assert "EITHER side" in REASON_QUAD_PROMPT
    assert "may be more specific than the other" in REASON_QUAD_PROMPT
    # default is SAME, and the ways to differ are exhaustive
    assert "Assume they do unless" in REASON_QUAD_PROMPT
    assert "ONLY when" in REASON_QUAD_PROMPT


def test_the_reason_quad_prompt_shows_the_action():
    """It was dropped when the prompts were split, on the reasoning that this
    question compares the explanations to EACH OTHER. But they are explanations
    OF an action, and without it the judge was asked whether two statements
    agree with no anchor for what they are about — the narrowing card flipped
    between yes and no across rounds."""
    assert "THE ACTION A RESPONDER PROPOSES" in REASON_QUAD_PROMPT
    assert "{action}" in REASON_QUAD_PROMPT


# ── reading a free-text answer ─────────────────────────────────────────

def test_the_last_sign_off_wins():
    """The model often restates the format line from the prompt partway
    through its reasoning before committing at the end."""
    text = ("...I will end with VERDICT: causally_aligned as instructed...\n"
            "Now the analysis.\n\nVERDICT: not_causally_aligned")
    assert read_verdict(text, (ALIGNED, NOT_ALIGNED)) == NOT_ALIGNED


def test_an_unchosen_template_line_is_not_a_verdict():
    """A model that echoes "[a | b]" rather than choosing has not answered, and
    reading its first option would invent a verdict it never gave."""
    echo = "VERDICT: [causally_aligned | not_causally_aligned]"
    assert read_verdict(echo, (ALIGNED, NOT_ALIGNED)) == UNCLEAR


def test_an_unreadable_answer_is_unclear_never_invented():
    for junk in (None, 42, "", "no verdict here", "VERDICT: maybe",
                 "VERDICT:", ["a"], {"verdict": "x"}):
        assert read_verdict(junk, (ALIGNED, NOT_ALIGNED)) == UNCLEAR


def test_a_judge_that_cannot_be_reached_does_not_take_the_run_down():
    def boom(_):
        raise RuntimeError("ollama is not running")
    v = judge_card(CARD, "scene", judge_fn=boom, n_probes=3)
    assert v["prose"]["verdict"] == UNCLEAR
    assert v["prose"]["errors"]


def test_malformed_cards_do_not_crash_the_judge():
    fn = _cycle_prompts([_answer(ALIGNED, ALIGNED)])
    for bad in ({}, {"structured_reasoning": "nope"},
                {"action": 5, "reason": None},
                {"structured_reasoning": {"affected_objects": "person_1"}}):
        out = judge_cards([bad], "s", judge_fn=fn, n_probes=2)
        assert out["n_cards"] == 1


# ── what it must never do ──────────────────────────────────────────────

def test_the_verdict_never_carries_a_score():
    """Judges advise; only reflection carries the message and only the model
    revises. A number here would leak into calibration."""
    import json
    out = judge_cards([CARD], "scene",
                      judge_fn=_cycle_prompts([_answer(ALIGNED, ALIGNED)]))
    assert out["advisory"] is True
    blob = json.dumps(out)
    assert "score" not in blob and "trust" not in blob


def test_every_prompt_is_neutral_and_carries_its_criterion():
    """F28: one sentence defining 'causally aligned' took the same judge from
    0/8 to 7/8 on a fixed discrimination set. Everything else we had built was
    scaffolding around a missing definition — so the criterion is the one thing
    that must not be lost, and nothing scene-specific may creep in beside it."""
    import re
    assert "would REDUCE" in ACTION_ALIGNMENT_PROMPT
    assert "still just as able to harm" in ACTION_ALIGNMENT_PROMPT
    for tpl in PROMPTS:
        body = re.sub(r"\{[a-z_]+\}", "", tpl)
        assert not re.search(r"\b[a-z][a-z0-9_]*_\d+\b", body)
        for word in ("fire", "flood", "quake", "spill", "drown", "burn",
                     "hazmat", "ambulance", "photograph"):
            assert word not in body.lower(), word


def test_the_prompt_does_not_tell_the_judge_what_to_look_for():
    """The first rubric said 'do not reward it for being plausible' and 'that
    is exactly what you are looking for', and returned the suspicious verdict
    every time. Iron rule 5 applies to our own instrument too."""
    for tpl in PROMPTS:
        low = tpl.lower()
        for phrase in ("you are looking for", "do not reward", "audit",
                       "find", "suspicious"):
            assert phrase not in low, phrase


# ── the discrimination set itself (F28) ────────────────────────────────

def test_every_judge_only_card_is_clean_under_the_rule_tier():
    """The set's whole point. A card the code already decides measures nothing
    about the judge — the first cut of this set was three quarters wasted on
    reversed direction, a different victim and a different source, all caught
    by subject_mismatch / object_mismatch / at_risk_used_as_hazard."""
    from agentic.evals4 import explanation_alignment
    from agentic.judge_set import CARDS
    from agentic.assessment import AtRiskEntry, SceneAssessment, ThreatEntry
    from agentic.perception import DetectedObject, PerceptionResult

    def obj(oid, state, kind):
        return DetectedObject(object_id=oid, label=oid.rsplit("_", 1)[0],
                              family="x", state=state, state_kind=kind,
                              bbox=[0, 0, 9, 9], box_source="dino_matched",
                              box_confidence=0.9, anchor_bbox=[0, 0, 9, 9])

    rec = PerceptionResult(
        image_path="/x", image_size=[10, 10], entity_source="vlm",
        detected_objects=[obj("person_1", "standing", "at_risk"),
                          obj("dog_1", "standing", "at_risk"),
                          obj("car_1", "parked", "normal"),
                          obj("house_1", "burning", "hazard_bearing"),
                          obj("house_2", "intact", "normal"),
                          obj("road_1", "paved", "normal")])
    asm = SceneAssessment(
        disaster_scenario="Yes", disaster_type="fire", disaster_level=7,
        severity_bucket="serious",
        threats=[ThreatEntry(object_id="house_1")],
        at_risk=[AtRiskEntry(object_id="person_1", kind="proximity"),
                 AtRiskEntry(object_id="dog_1", kind="proximity")])
    for entry in CARDS:
        fails = [f for f in explanation_alignment(
            rec, asm, [entry["card"]])["failures"] if f["severity"] > 0]
        assert bool(fails) == bool(entry.get("code_covered")), entry["name"]


def test_the_set_covers_the_shapes_identity_cannot_see():
    from agentic.judge_set import CARDS
    kinds = {c["kind"] for c in CARDS if not c.get("code_covered")}
    assert {"hollow", "good", "narrowing", "hedge", "added_fact",
            "contradiction", "partial"} <= kinds


def test_the_partial_card_is_deliberately_unscored():
    """Neither 'aligned' nor 'not aligned' is right for an action that reduces
    the harm without removing it. Forcing an expected value on it would bury
    the question it exists to ask."""
    from agentic.judge_set import CARDS
    partial = [c for c in CARDS if c["kind"] == "partial"][0]
    assert partial["expect_align"] is None


def test_unclear_is_never_scored_as_correct():
    """A non-answer is not an answer. The first cut of this harness credited
    'unclear' whenever the expected verdict was negative, which turned a
    degraded verdict into a pass."""
    from agentic.judge_set import score
    res = score(judge_fn=lambda p: "no verdict here", n_probes=1)
    assert res["hits"] == 0 and res["total"] > 0


def test_code_covered_cards_are_excluded_from_the_score_by_default():
    from agentic.judge_set import CARDS, score
    fn = lambda p: ("VERDICT: yes" if "same claim" in p
                    else "PROSE: causally_aligned\nSTRUCTURE: causally_aligned")
    assert not any(r["code_covered"] for r in score(judge_fn=fn, n_probes=1)["rows"])
    withem = score(judge_fn=fn, n_probes=1, include_code_covered=True)
    assert any(r["code_covered"] for r in withem["rows"])
    # ...and including them must not change the score
    assert withem["total"] == score(judge_fn=fn, n_probes=1)["total"]


# ── F30: the set-level rollup ──────────────────────────────────────────

def _judged(*per_card, n_probes=5):
    """Drive judge_cards with one scripted answer per card."""
    box = {"i": -1}
    def fn(prompt):
        box["i"] += 1
        # two prompts per card, n_probes each
        card = box["i"] // (2 * n_probes)
        return per_card[min(card, len(per_card) - 1)](prompt)
    cards = [{"rank": i + 1, "action": "a", "reason": "r",
              "structured_reasoning": {"threat": "x_1", "state": "s",
                                       "effect": "may_harm",
                                       "affected_objects": ["y_1"]}}
             for i in range(len(per_card))]
    return judge_cards(cards, "scene", judge_fn=fn, n_probes=n_probes)["rollup"]


def _reply(prose, structure, same):
    def fn(prompt):
        if "same claim" in prompt:
            return f"VERDICT: {same}"
        return f"PROSE: {prose}\nSTRUCTURE: {structure}"
    return fn


def test_the_rollup_names_the_card_not_just_a_count():
    """"2 of 4 explanations not tied to their action" sends you looking;
    "rec 1's reason" tells you where."""
    r = _judged(_reply(NOT_ALIGNED, ALIGNED, SAME),
                _reply(ALIGNED, ALIGNED, SAME))
    texts = [f["text"] for f in r["findings"]]
    assert any("rec 1" in t and "the reason" in t for t in texts)
    assert not any("rec 2" in t for t in texts)
    assert r["clean_ranks"] == [2]


def test_a_non_answer_is_not_rendered_as_a_defect():
    """'unclear' is a finding about the JUDGE — it read the card and could not
    decide. Reporting our instrument's confusion as the model's defect is
    exactly the mistake F26 and F28 were."""
    r = _judged(_reply("maybe", ALIGNED, SAME))     # unparseable -> unclear
    kinds = {f["kind"] for f in r["findings"]}
    assert "undecided" in kinds and "not_aligned" not in kinds


def test_a_scraped_majority_is_marked_thin():
    """A 5/5 is a verdict; a 3/5 is a coin flip that landed. Every judge
    finding on D_aerial round 4 was 3/5 or 2/5."""
    box = {"i": -1}
    def fn(prompt):
        box["i"] += 1
        if "same claim" in prompt:
            return f"VERDICT: {SAME}"
        # 3 of 5 probes say not aligned
        v = NOT_ALIGNED if box["i"] % 5 < 3 else ALIGNED
        return f"PROSE: {v}\nSTRUCTURE: {ALIGNED}"
    card = {"rank": 1, "action": "a", "reason": "r",
            "structured_reasoning": {"threat": "x_1", "state": "s",
                                     "effect": "may_harm",
                                     "affected_objects": ["y_1"]}}
    r = judge_cards([card], "s", judge_fn=fn, n_probes=5)["rollup"]
    thin = [f for f in r["findings"] if f["kind"] == "not_aligned"]
    assert thin and thin[0]["thin"] is True and thin[0]["votes"] == 3


def test_a_unanimous_verdict_is_not_marked_thin():
    r = _judged(_reply(NOT_ALIGNED, ALIGNED, SAME))
    bad = [f for f in r["findings"] if f["kind"] == "not_aligned"]
    assert bad and bad[0]["thin"] is False and bad[0]["votes"] == 5


def test_an_unreachable_judge_is_reported_once_as_itself():
    """A judge that cannot reach its model returns 'unclear' for every card,
    which reads identically to genuine indecision. Two different things."""
    def boom(_):
        raise RuntimeError("ollama is not running")
    r = _judged(boom, boom)
    assert r["unreachable"] == 2 and r["n_judged"] == 0
    assert r["findings"] == []
    assert "could not be reached" in r["headline"]


def test_a_clean_set_says_so_without_listing_every_card():
    """Above two cards, itemising the clean ones is noise."""
    r = _judged(*[_reply(ALIGNED, ALIGNED, SAME)] * 5)
    assert r["findings"] == [] and len(r["clean_ranks"]) == 5
    assert "all 5 card(s) clean" in r["headline"]


def test_the_rollup_never_carries_a_score():
    import json
    r = _judged(_reply(NOT_ALIGNED, ALIGNED, DIFFERENT))
    assert r["advisory"] is True
    blob = json.dumps(r)
    assert "score" not in blob and "trust" not in blob


# ── F38: the graph judge asks only what arithmetic cannot answer ────────

def test_it_asks_nothing_when_the_graphs_agree_on_the_victims():
    """Q1 exists for the case where the graphs agree on the hazard and differ
    on who it harms. With no such difference there is nothing to judge."""
    from agentic.judge_graph import judge_graphs
    g = {"edges": [{"source": "h_1", "effect": "may_harm", "target": "p_1"}]}
    out = judge_graphs(g, g, {"hazards": 1.0, "victims_only_in_a": [],
                              "victims_only_in_b": []}, "scene",
                       judge_fn=lambda p: "VERDICT: graph_a")
    assert out["victims"] is None and out["mechanisms"] == []


def test_q1_only_runs_when_they_agree_on_the_hazard():
    """"Which victims are more exposed" needs a SHARED danger to be exposed
    to. Graphs blaming different hazards have no common ground."""
    from agentic.judge_graph import judge_graphs
    a = {"edges": [{"source": "h_1", "effect": "may_harm", "target": "p_1"}]}
    b = {"edges": [{"source": "x_1", "effect": "may_harm", "target": "d_1"}]}
    out = judge_graphs(a, b, {"hazards": 0.0, "victims_only_in_a": ["p_1"],
                              "victims_only_in_b": ["d_1"]}, "s",
                       judge_fn=lambda p: "VERDICT: graph_a")
    assert out["victims"] is None


def test_q2_runs_once_per_shared_pair_with_a_different_effect():
    import os
    os.environ["GRAPH_JUDGE_Q2"] = "1"          # retired live; on-demand here
    try:
        from agentic.judge_graph import judge_graphs
        a = {"edges": [{"source": "t_1", "effect": "exposes", "target": "s_1"},
                       {"source": "h_1", "effect": "may_harm", "target": "p_1"}]}
        b = {"edges": [{"source": "t_1", "effect": "may_spread_to", "target": "s_1"},
                       {"source": "h_1", "effect": "may_harm", "target": "p_1"}]}
        out = judge_graphs(a, b, {"hazards": 1.0}, "s",
                           judge_fn=lambda p: "VERDICT: same_response")
        assert len(out["mechanisms"]) == 1          # the identical pair is skipped
        assert out["mechanisms"][0]["source"] == "t_1"
        assert out["mechanisms"][0]["verdict"] == "same_response"

    finally:
        os.environ.pop("GRAPH_JUDGE_Q2", None)

def test_the_graph_judge_is_off_unless_asked_for():
    """The hermetic spine and the twin equivalence tests must stay
    model-free."""
    from agentic.judge_graph import judge_graphs
    assert judge_graphs({}, {}, {}, "s") == {}


def test_the_mechanism_criterion_anchors_on_the_actual_entities():
    """Sunny: "the source and target need to be added in the definition —
    depending on source and target it may differ." Asked about the effect
    WORDS, the judge answered 'different' for everything; asked what a
    responder would do about THESE two entities, the real case came back
    5/5 correct."""
    from agentic.judge_graph import MECHANISM_PROMPT
    assert "{source}" in MECHANISM_PROMPT and "{target}" in MECHANISM_PROMPT
    assert "Do not compare the two descriptions as words" in MECHANISM_PROMPT


def test_neither_graph_prompt_says_what_to_look_for():
    """Iron rule 5 applies to our own instrument. The first mechanism prompt
    carried a mapping from each effect to an action, which told the judge that
    different effects mean different actions — and it duly answered
    'different' every time."""
    from agentic.judge_graph import MECHANISM_PROMPT, VICTIM_PROMPT
    import re
    for tpl in (VICTIM_PROMPT, MECHANISM_PROMPT):
        low = re.sub(r"\{[a-z_]+\}", "", tpl).lower()
        for phrase in ("you are looking for", "do not reward", "audit"):
            assert phrase not in low, phrase
        assert not re.search(r"\b[a-z][a-z0-9_]*_\d+\b", low)


# ── F53 follow-up: Q1 convenes only in its designed case ────────────────

def test_graph_judge_sits_out_when_a_victim_set_has_no_at_risk_entity():
    """C_tanker live: Graph B's "victims" were a road and a spill, and "which
    set is more exposed, people or pavement" answered itself 3/3, informing
    nobody. When a set contains no declared at-risk entity the judge sits out
    and CODE states the informative fact — no model call spent."""
    from agentic.judge_graph import judge_graphs
    A = {"edges": [{"source": "fire_1", "effect": "may_harm",
                    "target": "person_1"}]}
    B = {"edges": [{"source": "fire_1", "effect": "may_spread_to",
                    "target": "road_1"}]}
    dc = {"hazards": 1.0, "victims_only_in_a": ["person_1"],
          "victims_only_in_b": ["road_1"]}
    calls = []
    out = judge_graphs(A, B, dc, "scene", judge_fn=lambda p: calls.append(p),
                       at_risk_ids={"person_1"})
    assert out.get("victims") is None
    assert "no declared at-risk entity" in out["victims_note"]
    assert "Graph B" in out["victims_note"]
    assert not calls                       # the judge was never called for Q1


def test_graph_judge_convenes_when_both_sets_hold_at_risk_entities():
    """D_aerial's shape — the case Q1 was designed for."""
    from agentic.judge_graph import judge_graphs
    A = {"edges": [{"source": "spill_1", "effect": "blocks_access_to",
                    "target": "fire_truck_1"},
                   {"source": "spill_1", "effect": "may_harm",
                    "target": "person_1"}]}
    B = {"edges": [{"source": "spill_1", "effect": "may_harm",
                    "target": "hazmat_worker_1"}]}
    dc = {"hazards": 1.0, "victims_only_in_a": ["fire_truck_1", "person_1"],
          "victims_only_in_b": ["hazmat_worker_1"]}
    out = judge_graphs(A, B, dc, "scene",
                       judge_fn=lambda p: "VERDICT: [graph_b]",
                       at_risk_ids={"person_1", "hazmat_worker_1"})
    assert out.get("victims", {}).get("verdict") == "graph_b"
    assert "victims_note" not in out


# ── the combined A-vs-B judge (ab-v1): one call, two questions ──────────

_AB_A = {"edges": [{"source": "fire_1", "effect": "may_harm",
                    "target": "person_1"}]}
_AB_B = {"edges": [{"source": "fire_1", "effect": "may_harm",
                    "target": "person_2"}]}


def test_the_ab_judge_is_off_unless_asked_for():
    from agentic.judge_graph import judge_ab
    assert judge_ab(_AB_A, _AB_B, {}, "s") == {}


def test_the_ab_judge_reads_the_account_verdict_and_keeps_reasoning():
    from agentic.judge_graph import judge_ab
    out = judge_ab(_AB_A, _AB_B, {}, "scene",
                   judge_fn=lambda p: "because B saw more.\n"
                                      "ACCOUNT: [account_b]",
                   at_risk_ids={"person_1", "person_2"}, n_probes=3)
    assert out["advisory"] is True
    assert out["text"]["account"]["verdict"] == "account_b"
    assert out["text"]["account"]["votes"] == 3
    # F51: every probe's prose survives — loopholes live in the reasoning
    assert len(out["text"]["all_reasoning"]) == 3
    assert "because B saw more" in out["text"]["all_reasoning"][0]["text"]


def test_victims_runs_only_on_its_designed_case():
    """Both exposed sets must contain a declared at-risk entity, else
    people-vs-pavement answers itself and the question is skipped."""
    from agentic.judge_graph import judge_ab
    fn = lambda p: "ACCOUNT: [equally_good]\nVICTIMS: [account_a]"
    # designed case: sets differ, both hold an at-risk entity -> asked
    out = judge_ab(_AB_A, _AB_B, {}, "s", judge_fn=fn,
                   at_risk_ids={"person_1", "person_2"})
    assert out["asked_victims"] is True
    assert out["text"]["victims"]["verdict"] == "account_a"
    # people-vs-pavement: B's set holds no at-risk entity -> skipped
    b_pave = {"edges": [{"source": "fire_1", "effect": "may_spread_to",
                         "target": "structure_1"}]}
    out = judge_ab(_AB_A, b_pave, {}, "s", judge_fn=fn,
                   at_risk_ids={"person_1"})
    assert out["asked_victims"] is False
    assert "victims" not in out["text"]
    # identical sets: nothing to weigh
    out = judge_ab(_AB_A, _AB_A, {}, "s", judge_fn=fn,
                   at_risk_ids={"person_1"})
    assert out["asked_victims"] is False


def test_the_ab_twins_agree_and_disagree_honestly():
    from agentic.judge_graph import judge_ab
    text_fn = lambda p: "ACCOUNT: [account_a]"
    out = judge_ab(_AB_A, _AB_B, {}, "s", judge_fn=text_fn,
                   judge_image_fn=lambda p: "ACCOUNT: [account_a]",
                   at_risk_ids=set(), n_probes=1)
    assert out["twins_agree"] is True
    out = judge_ab(_AB_A, _AB_B, {}, "s", judge_fn=text_fn,
                   judge_image_fn=lambda p: "ACCOUNT: [account_b]",
                   at_risk_ids=set(), n_probes=1)
    assert out["twins_agree"] is False
    assert out["image"]["account"]["verdict"] == "account_b"


def test_the_ab_prompt_carries_the_stats_and_both_graphs():
    """The arithmetic is handed to the judge as context, not recomputed."""
    from agentic.judge_graph import judge_ab
    seen = []

    def fn(p):
        seen.append(p)
        return "ACCOUNT: [equally_good]"

    judge_ab(_AB_A, _AB_B, {"hazards": 1.0, "victims": 0.0},
             "the scene block", judge_fn=fn, at_risk_ids=set(), n_probes=1)
    p = seen[0]
    assert "the scene block" in p
    assert "fire_1 --may_harm--> person_1" in p
    assert "fire_1 --may_harm--> person_2" in p


def test_the_ab_prompt_is_neutral():
    """F2/F52 scars: the template names no scene, ships no example answer,
    and never says what the right verdict is."""
    from agentic.judge_graph import AB_PROMPT
    low = AB_PROMPT.lower()
    for banned in ("fire_1", "pool", "tanker", "collapse", "for example",
                   "e.g."):
        assert banned not in low, banned
