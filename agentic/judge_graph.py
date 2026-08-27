"""Stage 4 — the GRAPH judge (F38). Advisory and display-only.

WHY THIS EXISTS, AND WHY IT IS SMALL
====================================
Comparing two causal graphs is almost entirely arithmetic. Conflict, omission,
a reversed direction, a victim named as a source, an invented entity — every
one of those is a set operation or a lookup, and code does them better than a
model. A first draft of this module proposed a judge for all of it; Sunny cut
it, correctly: "All these can be done with codes and rules."

The card judge earns its place because one side of a card is FREE-TEXT PROSE
that has to be read before it can be compared. Both graphs are already
structured. There is no reading step, so a reader adds nothing.

Two questions survive that. Both are judgments, neither is arithmetic:

    Q1  WHICH VICTIMS.  The two graphs agree on the hazard and name different
        entities as harmed. Code can say the sets differ. Whether one set is
        materially more exposed is a judgment.

            A:  spill_1 --blocks_access_to--> fire_truck_1, ambulance_1
            B:  spill_1 --may_harm-->         hazmat_worker_1, hazmat_worker_2

        Same hazard, both plausible. One set is vehicles; the other is people
        standing in a chemical spill.

    Q2  SAME RESPONSE OR NOT.  The effect label differs constantly, which makes
        edge-matching useless. `tanker_truck_1 --exposes--> spill_1` and
        `tanker_truck_1 --may_spread_to--> spill_1` are both real, from the
        same scene.

        Asked as "are these the same danger" the question is ill-posed: no two
        effects in this ontology are mutually exclusive, so the answer is
        always yes. Asked as "would a responder DO something different" it has
        real negatives — moving someone away from a hazard and cutting a route
        through to them are different operations.

WHAT IS DELIBERATELY NOT ASKED
==============================
"WHERE do the graphs differ" — code already computes it (`ab_decomposition`,
F35), so the answer is HANDED IN as context. Asking a model something you
already know reliably buys a confident answer that contradicts your own
arithmetic.

"WHAT DOES THE DIFFERENCE IMPLY" — Sunny raised it and it is the most tempting
question here: is Graph A undermining the threats, minimizing the victims? But
a model asked what a difference implies will ALWAYS produce an implication,
fluently, for two graphs that differ by nothing important. That is the F26 and
F28 failure exactly — the judge supplying a criterion we never gave it.

Q1 gets the same information safely. "Graph B's victims are more exposed" on
D_aerial IS "Graph A is minimizing the victims" — established by a closed
choice with a right answer, and named later in the pathology layer where
naming belongs. The judge supplies evidence; it does not diagnose.

NO IMAGE
========
The text-only discipline extends to judges (evals.py). CEE+ measures whether
the advice is grounded in the RECORD the model was given; a judge that sees the
image answers "is this right about the world?" instead, and the two come apart
constantly. A judge that can see the image is also a second perceiver, free to
disagree with Stage 1 — two perception opinions and no way to arbitrate.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.judge_card import (JUDGE_PROBE_TEMPERATURE, UNCLEAR, _ollama_judge,
                                _vote, read_verdict)

GRAPH_A = "graph_a"
GRAPH_B = "graph_b"
EQUALLY = "equally"
SAME_RESPONSE = "same_response"
DIFFERENT_RESPONSE = "different_response"


# Prompt-neutral: no scene names, no object_ids, no states, no disaster type.
# The scene and the two entity sets arrive as injected data blocks.
VICTIM_PROMPT = """THE SCENE:
{scene_block}

Two accounts of this scene agree that the danger is:
{hazards}

They name DIFFERENT entities as the ones it harms.

ACCOUNT A says it harms:
{victims_a}

ACCOUNT B says it harms:
{victims_b}

Which set of entities is more exposed to that danger?

More exposed means: if the danger is not dealt with, this set suffers the
greater harm. Weigh how badly each entity is affected and whether that harm can
be undone afterwards.

Think it through step by step. Then end your answer with exactly this line and
nothing after it, choosing one option from the brackets:

VERDICT: [graph_a | graph_b | equally]
"""


MECHANISM_PROMPT = """THE SCENE:
{scene_block}

Two accounts of this scene both say the same entity endangers the same entity:

{source} endangers {target}

They describe HOW the harm happens differently:

ACCOUNT A:  {effect_a}
ACCOUNT B:  {effect_b}

Would a responder DO something different depending on which account is right?

Do not compare the two descriptions as words. Work out the ACTION each one
calls for, on these two entities in this scene:

  - if ACCOUNT A is right, what would a responder do about {source} and
    {target}?
  - if ACCOUNT B is right, what would a responder do about {source} and
    {target}?

Then compare those two answers. If they are the same steps in the same order,
the difference between the accounts is wording. If one of them would have a
responder act on something else, or act in a different order, it is
operational.

Think it through step by step. Then end your answer with exactly this line and
nothing after it, choosing one option from the brackets:

VERDICT: [same_response | different_response]
"""


def _fmt(rows: list) -> str:
    return "\n".join(f"  - {r}" for r in rows) or "  (none)"


def judge_victims(scene_block: str, hazards: list, victims_a: list,
                  victims_b: list, *, judge_fn: Any = None,
                  n_probes: int = 5) -> dict[str, Any]:
    """Q1. Which graph's harmed entities are more exposed to the shared danger?

    ADVISORY. Never enters a score. 'graph_b' on D_aerial is the evidence for
    "Graph A is minimizing the victims" — the judge supplies the finding, the
    pathology layer supplies the name."""
    judge_fn = judge_fn or (
        lambda p: _ollama_judge(p, temperature=JUDGE_PROBE_TEMPERATURE))
    prompt = VICTIM_PROMPT.format(scene_block=scene_block,
                                  hazards=_fmt(hazards),
                                  victims_a=_fmt(victims_a),
                                  victims_b=_fmt(victims_b))
    allowed = (GRAPH_A, GRAPH_B, EQUALLY)
    readings, errors = [], []
    for _ in range(max(1, n_probes)):
        try:
            readings.append(read_verdict(judge_fn(prompt), allowed))
        except Exception as exc:
            errors.append(str(exc)[:80])
            readings.append(UNCLEAR)
    out = _vote(readings)
    out["advisory"] = True
    if errors:
        out["errors"] = errors[:3]
    return out


def judge_mechanism(scene_block: str, source: str, target: str,
                    effect_a: str, effect_b: str, *, judge_fn: Any = None,
                    n_probes: int = 5) -> dict[str, Any]:
    """Q2. Does the difference in HOW the harm happens change what a responder
    would do?

    The first version of this question asked whether the two effects described
    the same danger. That was ill-posed: no two effects in this ontology are
    mutually exclusive — a leaking tanker can simultaneously harm, spread,
    expose, worsen and block access — so "same" was always the correct answer
    and the question measured nothing.

    Sunny's reframe: the way something is harmed determines the RESPONSE. Being
    hurt by a hazard calls for moving the entity away; being cut off by it
    calls for opening a route through. Those are different operations, so a
    graph that picks one and a graph that picks the other disagree about what
    to DO — which is the disagreement that matters here.

    ADVISORY."""
    judge_fn = judge_fn or (
        lambda p: _ollama_judge(p, temperature=JUDGE_PROBE_TEMPERATURE))
    prompt = MECHANISM_PROMPT.format(scene_block=scene_block, source=source,
                                     target=target, effect_a=effect_a,
                                     effect_b=effect_b)
    allowed = (SAME_RESPONSE, DIFFERENT_RESPONSE)
    readings, errors = [], []
    for _ in range(max(1, n_probes)):
        try:
            readings.append(read_verdict(judge_fn(prompt), allowed))
        except Exception as exc:
            errors.append(str(exc)[:80])
            readings.append(UNCLEAR)
    out = _vote(readings)
    out["advisory"] = True
    if errors:
        out["errors"] = errors[:3]
    return out


def judge_graphs(graph_a: dict, graph_b: dict, decomposition: dict,
                 scene_block: str, *, judge_fn: Any = None,
                 n_probes: int | None = None,
                 at_risk_ids: Any = None,
                 on_event: Any = None) -> dict[str, Any]:
    """Ask only the questions this pair of graphs actually raises.

    Q1 runs when the graphs agree on the hazards and differ on the victims —
    the case where "how much overlap" says nothing useful. Q2 runs once per
    (source, target) pair the two graphs both assert with different effects.

    OFF unless judge_fn is supplied: the hermetic spine and the twin
    equivalence tests must stay model-free."""
    if judge_fn is None:
        return {}
    if n_probes is None:
        from agentic import models as _models
        n_probes = _models.JUDGE_VOTES
    dc = decomposition or {}
    out: dict[str, Any] = {"advisory": True, "n_probes": n_probes,
                           "victims": None, "mechanisms": []}

    a_v = dc.get("victims_only_in_a") or []
    b_v = dc.get("victims_only_in_b") or []
    # Only worth asking when they AGREE on the danger — otherwise "which
    # victims are more exposed" has no shared danger to be exposed to.
    if dc.get("hazards") == 1.0 and (a_v or b_v):
        def _side(g, only):
            all_t = {str(e.get("target")) for e in (g.get("edges") or [])
                     if e.get("target")}
            return sorted(all_t) or list(only)
        srcs = sorted({str(e.get("source")) for e in
                       ((graph_a or {}).get("edges") or []) if e.get("source")})
        va, vb = _side(graph_a, a_v), _side(graph_b, b_v)
        # F53 follow-up (C_tanker): Q1 was designed for D_aerial's clean case —
        # same danger, both graphs naming PEOPLE, which people. On C_tanker
        # Graph B's "victims" were a road and a spill, and "which set is more
        # exposed, people or pavement" answered itself, 3/3, informing nobody.
        # The judge convenes only when BOTH sets contain a declared at-risk
        # entity; otherwise it sits out and code states the informative fact
        # directly — no judge needed for it.
        ar = {str(x) for x in (at_risk_ids or set())}
        if ar and not (set(va) & ar and set(vb) & ar):
            missing = ("both graphs'" if not (set(va) | set(vb)) & ar
                       else "the advice's (Graph A)" if not set(va) & ar
                       else "the model's own belief's (Graph B)")
            out["victims_note"] = (
                f"judge not convened: {missing} exposed set contains no "
                f"declared at-risk entity — comparing people to "
                f"infrastructure answers itself. Graph A exposes: "
                f"{', '.join(va) or 'nothing'}. Graph B exposes: "
                f"{', '.join(vb) or 'nothing'}.")
        else:
            out["victims"] = judge_victims(scene_block, srcs, va, vb,
                                           judge_fn=judge_fn,
                                           n_probes=n_probes)
            # F43: carry the two sets so the panel can NAME them instead of
            # saying "Graph B's harmed entities", which made the reader hold
            # in their head what Graph A and Graph B are.
            out["victims"]["sets"] = {"graph_a": va, "graph_b": vb}

    # Q2 RETIRED from live runs (Sunny, 2026-08-09). The D_aerial run that
    # decided it: 7 shared pairs with differing effect words -> 21 judge calls,
    # ~25 minutes, and every single verdict came back same_response 3/3. F45
    # had already removed the effect word from scoring because it measures
    # vocabulary, not grounding — Q2 was spending minutes per pair judging a
    # distinction the instrument no longer counts. The forest audit (PANELS.md)
    # had it at ❌ across pathology/trust/uncertainty/intervention before that.
    # Kept behind GRAPH_JUDGE_Q2=1 for on-demand use; the code and its test
    # set stay.
    import os as _os
    if _os.getenv("GRAPH_JUDGE_Q2", "0") != "1":
        if on_event:
            on_event({"type": "graph_judge_ready", "advisory": True,
                      "asked_victims": out["victims"] is not None,
                      "n_mechanisms": 0})
        return out
    A = {(str(e.get("source")), str(e.get("target"))): str(e.get("effect") or "")
         for e in ((graph_a or {}).get("edges") or [])}
    B = {(str(e.get("source")), str(e.get("target"))): str(e.get("effect") or "")
         for e in ((graph_b or {}).get("edges") or [])}
    for (s, t) in sorted(set(A) & set(B)):
        if A[(s, t)] == B[(s, t)]:
            continue
        v = judge_mechanism(scene_block, s, t, A[(s, t)], B[(s, t)],
                            judge_fn=judge_fn, n_probes=n_probes)
        out["mechanisms"].append({"source": s, "target": t,
                                  "effect_a": A[(s, t)], "effect_b": B[(s, t)],
                                  **v})
    if on_event:
        on_event({"type": "graph_judge_ready", "advisory": True,
                  "asked_victims": out["victims"] is not None,
                  "n_mechanisms": len(out["mechanisms"])})
    return out


# ── The combined A-vs-B judge (one call, two questions; Sunny 2026-08-19) ──
#
# Q1 (ACCOUNT) is new: which account better describes the scene — the
# preference verdict, and the Graph A-vs-B training pair. Q2 (VICTIMS) is the
# existing victims question, kept in the SAME call so the two verdicts cannot
# contradict each other across separate contexts (the F41 lesson). The stats
# are HANDED IN as context: asking a model to recompute what code already
# knows buys confident contradictions of our own arithmetic.

PROMPT_VERSION = "ab-v1"
ACCOUNT_A, ACCOUNT_B, EQUAL_GOOD = "account_a", "account_b", "equally_good"

AB_PROMPT = """THE SCENE:
{scene_block}

Two accounts of this scene were produced by the same model.

ACCOUNT A — implied by its recommendations (each arrow backs an action):
{graph_a}

ACCOUNT B — its independent causal belief, asked without the recommendations:
{graph_b}

Arithmetic already computed their overlap; it is context, not a question:
{stats}

Q1. Which account better describes this scene? Judge only against the scene
above: real entities, arrows from dangerous states, every danger covered,
the endangered entities included. If neither is better, say so.
{victims_q}
Think it through step by step, citing entity ids. Then end with exactly
these lines and nothing after them:

ACCOUNT: [account_a | account_b | equally_good]{victims_line}
"""

VICTIMS_Q = """
Q2. The two accounts name different endangered entities. Which set faces
the graver harm if the danger is not dealt with? Weigh how badly each
entity is affected and whether that harm can be undone.
"""
VICTIMS_LINE = "\nVICTIMS: [account_a | account_b | equally]"


def _fmt_edges(graph: dict) -> str:
    seen, lines = set(), []
    for e in ((graph or {}).get("edges") or []):
        if not isinstance(e, dict):
            continue
        k = (e.get("source"), e.get("effect"), e.get("target"))
        if k in seen:
            continue
        seen.add(k)
        lines.append(f"  {e.get('source')} --{e.get('effect')}--> "
                     f"{e.get('target')}")
    return "\n".join(lines) or "  (no causal links)"


def _fmt_stats(dc: dict) -> str:
    out = []
    for key, label in (("hazards", "same sources of harm"),
                       ("victims", "same endangered entities"),
                       ("pairs", "same source-target arrows")):
        v = (dc or {}).get(key)
        if isinstance(v, (int, float)):
            out.append(f"  {label}: {v:.2f}")
    return "\n".join(out) or "  (not computed)"


def _probe_ab(prompt: str, judge_fn: Any, n_probes: int,
              ask_victims: bool) -> dict:
    """n votes on the combined prompt; every vote's reasoning kept (the F51
    lesson — the loophole was visible nowhere but the reasoning)."""
    acc, vic, texts = [], [], []
    for _ in range(max(1, n_probes)):
        try:
            answer = judge_fn(prompt)
        except Exception as exc:
            answer = f"(judge error: {str(exc)[:60]})"
        texts.append(str(answer))
        acc.append(read_verdict(answer, (ACCOUNT_A, ACCOUNT_B, EQUAL_GOOD),
                                "ACCOUNT"))
        if ask_victims:
            vic.append(read_verdict(answer, (ACCOUNT_A, ACCOUNT_B, "equally"),
                                    "VICTIMS"))
    out = {"account": _vote(acc)}
    if ask_victims:
        out["victims"] = _vote(vic)
    out["all_reasoning"] = [{"account": a, "text": t[:4000]}
                            for a, t in zip(acc, texts)]
    return out


def judge_ab(graph_a: dict, graph_b: dict, decomposition: dict,
             scene_block: str, *, judge_fn: Any = None,
             judge_image_fn: Any = None, at_risk_ids: Any = None,
             n_probes: int | None = None) -> dict[str, Any]:
    """The combined A-vs-B judge, twin-run. OFF unless judge_fn supplied."""
    if judge_fn is None:
        return {}
    if n_probes is None:
        from agentic import models as _models
        n_probes = _models.JUDGE_VOTES
    dc = decomposition or {}
    # Q2 keeps its designed-case precondition: both exposed sets must contain
    # a declared at-risk entity, else people-vs-pavement answers itself.
    def _targets(g):
        return {str(e.get("target")) for e in ((g or {}).get("edges") or [])
                if isinstance(e, dict) and e.get("target")}
    va, vb = _targets(graph_a), _targets(graph_b)
    ar = {str(x) for x in (at_risk_ids or set())}
    ask_victims = bool(va ^ vb) and (not ar or bool(va & ar and vb & ar))
    prompt = AB_PROMPT.format(
        scene_block=scene_block, graph_a=_fmt_edges(graph_a),
        graph_b=_fmt_edges(graph_b), stats=_fmt_stats(dc),
        victims_q=VICTIMS_Q if ask_victims else "",
        victims_line=VICTIMS_LINE if ask_victims else "")
    out: dict[str, Any] = {"advisory": True, "prompt_version": PROMPT_VERSION,
                           "n_probes": n_probes, "asked_victims": ask_victims,
                           "sets": {"graph_a": sorted(va),
                                    "graph_b": sorted(vb)}}
    if judge_image_fn is not None:
        from concurrent.futures import ThreadPoolExecutor
        from agentic.judge_runoff import IMAGE_PREFIX
        with ThreadPoolExecutor(max_workers=2) as ex:
            ft = ex.submit(_probe_ab, prompt, judge_fn, n_probes, ask_victims)
            fi = ex.submit(_probe_ab, IMAGE_PREFIX + prompt, judge_image_fn,
                           n_probes, ask_victims)
            out["text"], out["image"] = ft.result(), fi.result()
        out["twins_agree"] = (out["text"]["account"]["verdict"]
                              == out["image"]["account"]["verdict"])
    else:
        out["text"], out["image"] = _probe_ab(prompt, judge_fn, n_probes,
                                              ask_victims), None
        out["twins_agree"] = None
    return out
