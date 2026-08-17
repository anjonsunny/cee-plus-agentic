"""Stage 4 — the CARD judge (F24/F28). Advisory and display-only.

WHY THIS EXISTS
===============
The card rules are all IDENTITY comparisons: is this the same id, the same
token, a legal word. They are cheap, objective, deterministic — and blind to
whether the explanation MEANS anything. A card can name the right entities in
the right slots with the right vocabulary and still not justify its action:

    action:  Photograph the hazard for the incident report.
    reason:  Because <hazard> is <hazard state> it may_harm <victim>.

Every rule passes — the ids match, the state is declared, the effect is legal,
the action names the hazard. And the action makes nobody safer. The rule tier
catches INVERTED. This tier catches HOLLOW.

WHAT IT NEVER DOES
==================
It never changes a score. Not conformance, not internal alignment, not trust.
Judges advise; only the model revises, and only reflection carries the message.
Display-only is also what makes it safe to run during calibration: it cannot
contaminate the numbers the weights are being fitted to.

WHAT THE JUDGE WAS MISSING (F28)
================================
The first rubric asked three abstract questions (causal-or-circular, relevance,
necessity) across three calls with a forced JSON response. It scored 0 for 5 on
a run whose explanations were fine. A six-step "walk the causal chain" version
scored 0 for 4 on the case the tier exists for. Both were rewritten twice.

None of that was the problem. The judge reasons perfectly well — asked why a
hollow card was aligned, it answered:

    "Photographing ... directly addresses the immediate concern of DOCUMENTING
     the hazard that poses a risk to <victim>"

It was applying "the action is about the same hazard" as its test, because we
never said what causally aligned MEANS. One sentence defining it — carrying out
the action must REDUCE the harm the explanation describes — took the same model
from 0/8 to 7/8 on a fixed discrimination set. Everything else we had built was
scaffolding around a missing definition.

Two things follow, and both are load-bearing:

1. The prompt stays SMALL. Every clarification we added on top flipped the model
   wholesale rather than sharpening the boundary — the notation gloss moved the
   error to a different cell, and a second gloss on top drove it to 4/8, all
   verdicts negative. This prompt is frozen against a test set, not tuned by
   impression.
2. The prompt is scored against that set, never against our reading of a few
   verdicts. Judging by impression is how a 0-for-5 judge shipped.

THE THIRD QUESTION (reason vs quad)
===================================
The two alignment verdicts each measure an explanation against the ACTION. The
third asks whether the two explanations make the same CLAIM — the semantic
counterpart of the rule tier's subject/object/effect comparison, which can only
see identity. Identical ids with the roles swapped is caught in code; identical
ids meaning different things is not.

It rides in the SAME call, since both texts are already in the prompt, so it
costs nothing. It needed its own criterion for the same reason the first
question did: asked without one, the judge called a card different_claims
because the prose said "nearby structures" where the quad named one structure
by id. That is the quad being MORE SPECIFIC, not a second claim — so the
criterion says outright that specificity and extra wording are not differences,
and names the three things that are: a different source, different entities
harmed, or a reversed direction.

NAMING WHAT IT IS LOOKING AT (Sunny)
====================================
Two omissions, both ours, both the same shape as the missing definition above.

The prompt handed over "house_1 is burning --may_harm--> person_1" and never
said what that notation was — not that it is a causal quad, not which side is
the source of harm, not which side is harmed. SAME_CLAIM asks about "the same
source of harm" and "the direction reversed"; neither is answerable in a
notation whose direction was never stated. Both blocks are now labelled as
explanations, one in prose and one in structure.

And the sign-off block showed the format as
    PROSE: causally_aligned / STRUCTURE: causally_aligned / SAME_CLAIM: yes
— the positive answer, three times, sitting exactly where the answer goes. It
now offers both options in brackets, and the parser refuses to read an
un-chosen template line as a verdict.

WHY FIVE PROBES
===============
At temperature 0 the judge is deterministic, so a wrong verdict is wrong 5/5
and looks confident. The boundary cases are the ones that matter and they are
exactly where a single sample is least informative. So the same question is
asked five times at probe temperature and the majority wins — the same
machinery Stage 2 and the recommend step already use for measured uncertainty.

The VOTE SPLIT is reported next to the verdict, not just the winner. "aligned
4/5" and "aligned 5/5" are different findings, and a 3/2 split is the judge
telling us it cannot see the boundary — which is worth more than the verdict.
"""
from __future__ import annotations

import os

from agentic import models as _models
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

JudgeFn = Callable[[str], str]

JUDGE_PROBE_TEMPERATURE = float(os.getenv("CARD_JUDGE_TEMP", "0.7"))
DEFAULT_JUDGE_PROBES = int(os.getenv("CARD_JUDGE_PROBES", "5"))

ALIGNED = "causally_aligned"
NOT_ALIGNED = "not_causally_aligned"
SAME = "yes"
DIFFERENT = "no"
UNCLEAR = "unclear"


# Prompt-neutral: no scene names, no object_ids, no states, no disaster type.
# The card and the scene arrive as injected data blocks; no criterion says
# which answer is expected.
#
# TWO PROMPTS (Sunny). Grouped by what each question is measured AGAINST:
#
#   prompt 1   action <-> reason  AND  action <-> quad     (both vs the ACTION)
#   prompt 2   reason <-> quad                             (vs EACH OTHER)
#
# All three in ONE prompt let the alignment framing bleed into the same-claim
# question, and the structure notation could not be explained without that
# explanation also reaching the prose question. All three APART cost context:
# asked alone the prose question could not see the structure, and every
# alignment margin thinned (5/5 -> 3/5 on cards it still got right).
#
# Cost: 2 prompts x n_probes calls per card.

_STRUCTURE_LEGEND = ("It is written as:  source of harm · its state "
                     "--how it harms--> the entities it harms")

ACTION_ALIGNMENT_PROMPT = """THE SCENE:
{scene_block}

THE ACTION A RESPONDER PROPOSES:
{action}

THE EXPLANATION GIVEN FOR THAT ACTION, WRITTEN AS PROSE:
{reason}

THE SAME EXPLANATION, WRITTEN AS STRUCTURE.
""" + _STRUCTURE_LEGEND + """
{quad}

Is each explanation causally aligned with the action?

An explanation is causally aligned only if carrying out the action would REDUCE
the harm that explanation describes. If the action succeeds and the danger it
names is still just as able to harm the entities it names, the explanation is
not aligned with it — however true the danger is, and however much the action
relates to it.

Think it through step by step. Then end your answer with exactly these two
lines and nothing after them, choosing one option from each set of brackets:

PROSE: [causally_aligned | not_causally_aligned]
STRUCTURE: [causally_aligned | not_causally_aligned]
"""


REASON_QUAD_PROMPT = """THE SCENE:
{scene_block}

THE ACTION A RESPONDER PROPOSES:
{action}

Below are TWO explanations OF THAT ACTION, written by the same analyst.
One is prose, one is structure.

THE EXPLANATION, WRITTEN AS PROSE:
{reason}

THE EXPLANATION, WRITTEN AS STRUCTURE.
""" + _STRUCTURE_LEGEND + """
{quad}

Do these two explanations make the same claim?

Assume they do unless one of them rules out what the other says. EITHER side
may be more specific than the other — describing entities loosely where the
other names them exactly, or naming one entity where the other speaks of a
group it belongs to — and either side may carry extra wording. None of that is
a difference.

They make different claims ONLY when one blames a different source of harm than
the other, when one names as harmed an entity the other says is not harmed, or
when they reverse who harms whom.

Think it through step by step. Then end your answer with exactly this line and
nothing after it, choosing one option from the brackets:

VERDICT: [yes | no]
"""


def _ollama_judge(prompt: str, temperature: float = 0.0) -> str:
    """The judge model — a DIFFERENT training family than the subject VLM, so
    it does not share the tendencies it is being asked to catch.

    No response_format: forcing JSON from the first token leaves the model no
    room to think before answering, and the verdict lands before any reasoning
    happens. It answers in prose and signs off with two parseable lines."""
    import requests

    r = requests.post(os.getenv("QWEN_API_URL",
                                "http://localhost:11434/v1/chat/completions"),
                      json={"model": _models.JUDGE_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature},
                      timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


def read_verdict(answer: Any, allowed: tuple,
                 label: str = "VERDICT") -> str:
    """Pull the sign-off line out of a free-text answer. Boundary-guarded like
    every other model seam: anything unreadable becomes 'unclear', never a
    crash and never an invented verdict.

    The LAST matching line wins — the model often restates the format line from
    the prompt partway through its reasoning before committing at the end. And
    an un-chosen template line ("[a | b]") is not an answer: reading its first
    option as one would invent a verdict the judge never gave."""
    hits = re.findall(rf"^\s*{label}\s*:\s*\[?\s*([a-z_]+)(?!\s*\|)",
                      str(answer or ""), re.M | re.I)
    val = hits[-1].lower() if hits else ""
    return val if val in allowed else UNCLEAR


def _vote(readings: list[str]) -> dict[str, Any]:
    """Majority over the probes. A tie is UNCLEAR, not a coin flip — the judge
    failing to see a boundary is a finding, and resolving it arbitrarily would
    hide it."""
    counts = Counter(readings)
    if not counts:
        return {"verdict": UNCLEAR, "votes": 0, "n": 0, "counts": {}}
    top = counts.most_common()
    verdict, best = top[0]
    tie = len(top) > 1 and top[1][1] == best
    if tie:
        verdict = UNCLEAR
    # Report the size of the WINNING BLOC, not counts[verdict]: on a tie the
    # verdict becomes UNCLEAR, and counting how many probes literally said
    # "unclear" made a 2-2 split render as "1/5" — a real disagreement dressed
    # up as near-total confusion.
    return {"verdict": verdict, "votes": best, "n": len(readings),
            "counts": dict(counts), "tie": tie,
            "unanimous": (not tie) and best == len(readings)}


def _quad_text(q: Any) -> str:
    if not isinstance(q, dict):
        return str(q)
    aff = q.get("affected_objects") or []
    if not isinstance(aff, (list, tuple)):
        aff = [aff]
    return (f"{q.get('threat', '')} is {q.get('state', '')} "
            f"--{q.get('effect', '')}--> {', '.join(str(a) for a in aff)}")


def _probe(prompt: str, wanted: dict[str, tuple], judge_fn: Any,
           n_probes: int) -> dict[str, dict]:
    """Ask ONE prompt n_probes times and take a majority per sign-off line.

    `wanted` maps a result key to (sign-off label, allowed values), so one
    prompt can carry more than one verdict — the alignment prompt asks about
    both explanations at once."""
    readings: dict[str, list[str]] = {k: [] for k in wanted}
    errors: list[str] = []
    for _ in range(max(1, n_probes)):
        try:
            answer = judge_fn(prompt)
        except Exception as exc:            # a judge that cannot be reached
            errors.append(str(exc)[:80])    # must never take the run down
            answer = ""
        for key, (label, allowed) in wanted.items():
            readings[key].append(read_verdict(answer, allowed, label))
    out = {k: _vote(v) for k, v in readings.items()}
    if errors:
        for v in out.values():
            v["errors"] = errors[:3]
    return out


def judge_card(rec: dict, scene_block: str, *, judge_fn: Any = None,
               n_probes: int = DEFAULT_JUDGE_PROBES) -> dict[str, Any]:
    """Judge ONE recommendation card: three separate questions, each asked
    n_probes times, each taking its own majority.

        prose        does the PROSE explain the action?
        structure    does the STRUCTURE explain the action?
        same_claim   do the two explanations mean the same thing?

    ADVISORY. The return value is displayed and recorded; it never enters a
    score, so it cannot move the numbers the trust weights are calibrated
    against."""
    judge_fn = judge_fn or (
        lambda p: _ollama_judge(p, temperature=JUDGE_PROBE_TEMPERATURE))
    rec = rec if isinstance(rec, dict) else {}
    fields = {
        "scene_block": scene_block,
        "action": str(rec.get("action") or "") or "(none given)",
        "reason": str(rec.get("reason") or "") or "(none given)",
        "quad": _quad_text(rec.get("structured_reasoning")) or "(none given)",
    }
    ALIGN = (ALIGNED, NOT_ALIGNED)
    out: dict[str, Any] = {"rank": rec.get("rank"), "advisory": True}
    # TWO prompts. Both alignment verdicts share one, because each is measured
    # against the SAME action and splitting them cost context: asked alone, the
    # prose question could not see the structure and every alignment margin
    # thinned. The same-claim question keeps its own prompt — it compares the
    # two explanations to EACH OTHER, not to the action, and mixing it in let
    # the alignment framing bleed into it.
    out.update(_probe(ACTION_ALIGNMENT_PROMPT.format(**fields),
                      {"prose": ("PROSE", ALIGN),
                       "structure": ("STRUCTURE", ALIGN)},
                      judge_fn, n_probes))
    out.update(_probe(REASON_QUAD_PROMPT.format(**fields),
                      {"same_claim": ("VERDICT", (SAME, DIFFERENT))},
                      judge_fn, n_probes))
    return out


def _rollup(verdicts: list[dict], n_probes: int) -> dict[str, Any]:
    """F30 — what the judge found ACROSS the set, for the panel below the cards.

    The per-card footer already shows each verdict where it belongs. This says
    what they mean together, and it names the cards rather than counting them:
    "2 of 4 explanations not tied to their action" sends you looking, "rec 1's
    reason" tells you where.

    Three distinctions the counts alone cannot make:

      A PROBLEM vs A NON-ANSWER. "not causally aligned" is a finding about the
      recommendation. "unclear" is a finding about the JUDGE — it read the card
      and could not decide. Rendering them alike would report our instrument's
      confusion as the model's defect, which is the mistake F26 and F28 were.

      DECIDED vs SCRAPED THROUGH. A 5/5 is a verdict; a 3/5 is a coin flip that
      landed. Every judge finding on D_aerial round 4 was 3/5 or 2/5, and that
      context matters more than the verdicts did.

      COULD NOT DECIDE vs COULD NOT RUN. A judge that cannot reach its model
      returns 'unclear' for every card, which reads identically to genuine
      indecision. It is reported once, as itself.
    """
    SURFACE = {"prose": "the reason", "structure": "the quad"}
    findings: list[dict] = []
    unreachable = 0
    clean_ranks: list[Any] = []
    for v in verdicts:
        rank = v.get("rank")
        errored = any((v.get(k) or {}).get("errors")
                      for k in ("prose", "structure", "same_claim"))
        if errored:
            unreachable += 1
            continue
        rows: list[dict] = []
        for key, name in SURFACE.items():
            d = v.get(key) or {}
            if d.get("verdict") == NOT_ALIGNED:
                rows.append({"rank": rank, "kind": "not_aligned",
                             "text": f"rec {rank}: {name} is not causally "
                                     f"aligned with its action",
                             "votes": d.get("votes"), "n": d.get("n")})
            elif d.get("verdict") == UNCLEAR:
                rows.append({"rank": rank, "kind": "undecided",
                             "text": f"rec {rank}: the judge could not decide "
                                     f"about {name}",
                             "votes": d.get("votes"), "n": d.get("n")})
        sc = v.get("same_claim") or {}
        if sc.get("verdict") == DIFFERENT:
            rows.append({"rank": rank, "kind": "different_claims",
                         "text": f"rec {rank}: reason and quad make different "
                                 f"claims",
                         "votes": sc.get("votes"), "n": sc.get("n")})
        elif sc.get("verdict") == UNCLEAR:
            rows.append({"rank": rank, "kind": "undecided",
                         "text": f"rec {rank}: the judge could not decide "
                                 f"whether reason and quad agree",
                         "votes": sc.get("votes"), "n": sc.get("n")})
        if rows:
            findings.extend(rows)
        else:
            clean_ranks.append(rank)

    # thin = the majority scraped through. Marked, never hidden: a 3/5 that
    # happened to land is not the same finding as a 5/5.
    for f in findings:
        n, votes = f.get("n") or 0, f.get("votes") or 0
        f["thin"] = bool(n) and votes <= (n // 2) + 1 and votes < n

    judged = len(verdicts) - unreachable
    n_problems = sum(1 for f in findings if f["kind"] != "undecided")
    headline = ""
    if unreachable and not judged:
        headline = "the judge could not be reached — no card was judged"
    elif judged and not findings:
        headline = (f"all {judged} card(s) clean — every explanation tied to "
                    f"its action, and prose and quad agree")
    elif judged:
        bits = []
        if n_problems:
            bits.append(f"{n_problems} finding(s) across {judged} card(s)")
        undecided = len(findings) - n_problems
        if undecided:
            bits.append(f"{undecided} the judge could not decide")
        if clean_ranks:
            bits.append(f"{len(clean_ranks)} card(s) clean")
        headline = " · ".join(bits)
    return {"findings": findings, "headline": headline,
            "clean_ranks": clean_ranks, "n_judged": judged,
            "unreachable": unreachable, "n_probes": n_probes,
            "advisory": True}


def judge_cards(recommendations: list[dict], scene_block: str, *,
                judge_fn: Any = None, n_probes: int = DEFAULT_JUDGE_PROBES,
                on_event: Any = None) -> dict[str, Any]:
    """Judge every card. OFF unless a judge_fn is supplied or the live default
    is wanted — the hermetic spine and the twin equivalence tests must stay
    model-free, so this is never called implicitly."""
    verdicts = [judge_card(r, scene_block, judge_fn=judge_fn,
                           n_probes=n_probes)
                for r in recommendations if isinstance(r, dict)]
    # Counts for the panel header. Deliberately NOT a score: "2 explanations
    # the judge could not tie to their action" is a prompt to go and read them,
    # never a number to fold into anything.
    flags = {
        "not_aligned": sum(1 for v in verdicts for k in ("prose", "structure")
                           if v[k]["verdict"] == NOT_ALIGNED),
        "unclear": sum(1 for v in verdicts for k in ("prose", "structure")
                       if v[k]["verdict"] == UNCLEAR),
        "different_claims": sum(1 for v in verdicts
                                if v["same_claim"]["verdict"] == DIFFERENT),
        "split": sum(1 for v in verdicts
                     for k in ("prose", "structure", "same_claim")
                     if not v[k].get("unanimous")),
    }
    out = {"verdicts": verdicts, "flags": flags, "n_cards": len(verdicts),
           "n_probes": n_probes, "advisory": True,
           "rollup": _rollup(verdicts, n_probes)}
    if on_event:
        on_event({"type": "card_judge_ready", "n_cards": len(verdicts),
                  "flags": flags, "n_probes": n_probes, "advisory": True})
    return out
