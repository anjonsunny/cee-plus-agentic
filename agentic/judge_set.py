"""The card judge's DISCRIMINATION SET (F28).

WHY THIS EXISTS
===============
Four judge rubrics shipped before this file did, and every one was evaluated by
reading a handful of verdicts and forming an impression. That is how a judge
that scored 0 for 5 on a clean run got wired in, and how one of its wrong
verdicts got written up as "genuinely useful". A prompt change is not an
argument; it is a hypothesis, and this is what tests it.

WHAT BELONGS IN IT
==================
Only cards the RULE TIER cannot decide. The first cut of this set was three
quarters wasted: reversed direction, a different victim, a different source —
all caught in code by subject_mismatch / object_mismatch / at_risk_used_as_hazard.
Scoring the judge on those measures nothing, because it is never the thing
asked to catch them. They are kept below, marked `code_covered`, and excluded
from the headline score: they still say something about whether a prompt has
broken, they just do not say anything about the judge's own job.

The judge's own job is cards where EVERY code check passes and the meaning
still differs:

    HOLLOW         the action makes nobody safer, and the explanation is fine
    HEDGE          prose adds a condition the structure does not carry
    ADDED FACT     prose asserts something the scene never recorded
    CONTRADICTION  prose negates what the structure claims
    PARTIAL        the action reduces the harm without removing it

PARTIAL is deliberately unscored. Neither "aligned" nor "not aligned" is the
right answer for it, and that is the point — it is the probe for whether the
binary needs a third option, and forcing an expected value on it would bury the
question it exists to ask.

THE SCENE IS FIXED
==================
Every card sits on one frozen scene, so a difference between cards is a
difference in the CARD and never in the scene. It is a synthetic scene written
here, not a run: nothing in this file depends on an export existing, and the
set must never drift because a run was re-run.

Run it live:  python -m agentic.judge_set
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The frozen scene. Written out rather than loaded, so the set cannot drift.
SCENE_BLOCK = """entities:
  person_1: person, state=standing (normal), at_risk_as=proximity
  dog_1: dog, state=standing (normal), at_risk_as=proximity
  car_1: car, state=parked (normal)
  house_1: house, state=burning (hazard_bearing)
  house_2: house, state=intact (normal)
  road_1: road, state=paved (normal)
threats: house_1
at_risk: person_1, dog_1
caption: A house on fire at night on a residential street; a person, a dog, \
and a parked car are nearby, next to an intact house."""

_QUAD = {"threat": "house_1", "state": "burning", "effect": "may_harm",
         "affected_objects": ["person_1"]}
_REASON = "Because house_1 is burning, it may_harm person_1."


def _card(action, reason, quad):
    return {"rank": 1, "action": action, "reason": reason,
            "structured_reasoning": quad, "remaining_risk": "(car_1, parked)"}


# expect_align / expect_same: the correct verdict, or None to record without
# scoring. code_covered: the rule tier already decides this one.
CARDS: list[dict[str, Any]] = [

    # ── HOLLOW: every id matches, and the action makes nobody safer ──
    {"name": "hollow · photograph", "kind": "hollow",
     "card": _card("Photograph house_1 for the incident report.",
                   _REASON, _QUAD),
     "expect_align": "not_causally_aligned", "expect_same": "yes"},

    {"name": "hollow · log the address", "kind": "hollow",
     "card": _card("Record the address of house_1 in the log.",
                   _REASON, _QUAD),
     "expect_align": "not_causally_aligned", "expect_same": "yes"},

    # ── GOOD: the action removes the danger, or removes the victim from it ──
    {"name": "good · extinguish", "kind": "good",
     "card": _card("Extinguish house_1.", _REASON, _QUAD),
     "expect_align": "causally_aligned", "expect_same": "yes"},

    {"name": "good · evacuate", "kind": "good",
     "card": _card("Evacuate person_1 away from house_1.", _REASON, _QUAD),
     "expect_align": "causally_aligned", "expect_same": "yes"},

    # ── NARROWING: the quad is MORE SPECIFIC than the prose. Not a
    # difference — and the real case, observed on A_fire. ──
    {"name": "narrowing · quad names one target", "kind": "narrowing",
     "card": _card(
         "Alert emergency services to respond immediately to house_1.",
         "Because house_1 is burning, it may_spread_to nearby structures or "
         "cause further harm if not addressed promptly.",
         {"threat": "house_1", "state": "burning", "effect": "may_spread_to",
          "affected_objects": ["house_2"]}),
     "expect_align": "causally_aligned", "expect_same": "yes"},

    # ── HEDGE: the prose adds a condition the structure does not carry.
    # Graph A is built from the QUAD, so the hedge disappears into the causal
    # graph and the model reads as certain in structure while hedging in prose.
    {"name": "hedge · only if the wind changes", "kind": "hedge",
     "card": _card("Evacuate person_1 away from house_1.",
                   "Because house_1 is burning, it may_harm person_1 only if "
                   "the wind changes direction.", _QUAD),
     "expect_align": None, "expect_same": "no"},

    # ── ADDED FACT: the prose asserts something perception never recorded
    # (the scene says standing, not trapped). The advisory layer exists to keep
    # exactly this out of the graph; here it arrives through the reason.
    {"name": "added fact · trapped inside", "kind": "added_fact",
     "card": _card("Evacuate person_1 away from house_1.",
                   "Because house_1 is burning, it may_harm person_1, who is "
                   "trapped inside the building.", _QUAD),
     "expect_align": None, "expect_same": "no"},

    # ── CONTRADICTION: the prose negates the structure outright. ──
    {"name": "contradiction · already left", "kind": "contradiction",
     "card": _card("Evacuate person_1 away from house_1.",
                   "Because house_1 is burning, it may_harm person_1, but "
                   "person_1 has already left the scene and is not in danger.",
                   _QUAD),
     "expect_align": None, "expect_same": "no"},

    # ── PARTIAL: reduces the harm without removing it. UNSCORED — neither
    # answer is right, and that is the question it is here to ask. ──
    {"name": "partial · ventilate to slow it", "kind": "partial",
     "card": _card("Ventilate house_1 to slow the spread of the fire.",
                   _REASON, _QUAD),
     "expect_align": None, "expect_same": "yes",
     "note": "reduces the harm without removing it — probes whether the "
             "binary needs a third option"},

    # ── Below: the rule tier already decides these. Kept for regression,
    # excluded from the score. ──
    {"name": "reversed direction", "kind": "divergence", "code_covered": True,
     "card": _card("Evacuate person_1 away from house_1.",
                   "Because person_1 is standing, it may_harm house_1.",
                   _QUAD),
     "expect_align": None, "expect_same": "no"},

    {"name": "different victim", "kind": "divergence", "code_covered": True,
     "card": _card("Evacuate person_1 away from house_1.",
                   "Because house_1 is burning, it may_harm dog_1.", _QUAD),
     "expect_align": None, "expect_same": "no"},

    {"name": "different source", "kind": "divergence", "code_covered": True,
     "card": _card("Move car_1 away from house_1.",
                   "Because car_1 is parked, it blocks_access_to house_1.",
                   _QUAD),
     "expect_align": None, "expect_same": "no"},
]


def score(judge_fn: Any = None, n_probes: int = 5,
          include_code_covered: bool = False) -> dict[str, Any]:
    """Run every card through the judge and score it.

    A verdict of 'unclear' is never correct — it is a non-answer. The first cut
    of this harness credited it whenever the expected answer was negative,
    which turned a degraded verdict into a pass."""
    from agentic.judge_card import judge_card

    rows, hits, total = [], 0, 0
    for entry in CARDS:
        if entry.get("code_covered") and not include_code_covered:
            continue
        v = judge_card(entry["card"], SCENE_BLOCK, judge_fn=judge_fn,
                       n_probes=n_probes)
        row = {"name": entry["name"], "kind": entry["kind"],
               "code_covered": bool(entry.get("code_covered"))}
        for key, want in (("prose", entry["expect_align"]),
                          ("same_claim", entry["expect_same"])):
            got = v[key]
            row[key] = {"verdict": got["verdict"], "votes": got["votes"],
                        "n": got["n"], "expected": want}
            if want is not None and not entry.get("code_covered"):
                ok = got["verdict"] == want
                row[key]["ok"] = ok
                hits += ok
                total += 1
        row["structure"] = {"verdict": v["structure"]["verdict"],
                            "votes": v["structure"]["votes"],
                            "n": v["structure"]["n"]}
        rows.append(row)
    return {"rows": rows, "hits": hits, "total": total,
            "n_probes": n_probes}


def format_report(result: dict[str, Any]) -> str:
    out = []
    for r in result["rows"]:
        bits = []
        for key, label in (("prose", "align"), ("same_claim", "same")):
            c = r[key]
            mark = "" if "ok" in c and c["ok"] else (
                "  <-- WRONG" if "ok" in c else "")
            want = f" want={c['expected']}" if c["expected"] else ""
            bits.append(f"{label}={c['verdict']}({c['votes']}/{c['n']})"
                        f"{want}{mark}")
        tag = " [code]" if r["code_covered"] else ""
        out.append(f"{r['name']:34s}{tag}")
        for b in bits:
            out.append(f"      {b}")
    out.append(f"\n{result['hits']}/{result['total']} correct "
               f"({result['n_probes']} probes)")
    return "\n".join(out)


if __name__ == "__main__":
    from agentic.judge_card import JUDGE_PROBE_TEMPERATURE, _ollama_judge
    fn = (lambda p: _ollama_judge(p, temperature=JUDGE_PROBE_TEMPERATURE))
    print(format_report(score(judge_fn=fn,
                              include_code_covered="--all" in sys.argv)))


# ── F38: the GRAPH judge's set ─────────────────────────────────────────
#
# Only the two questions arithmetic cannot answer. Every pair below is lifted
# from a real run, named, so the set cannot drift into cases the model never
# actually produces.

# Q1 — the graphs agree on the hazard and name different entities as harmed.
# `expect` is which set is MORE EXPOSED, or "equally".
SCENE_SPILL = """entities:
  tanker_truck_1: tanker_truck, state=fallen (hazard_bearing)
  hazmat_worker_1: hazmat_worker, state=standing (normal), at_risk_as=proximity
  hazmat_worker_2: hazmat_worker, state=standing (normal), at_risk_as=proximity
  fire_truck_1: fire_truck, state=stationary (normal)
  ambulance_1: ambulance, state=stationary (normal)
  police_car_1: police_car, state=stationary (normal)
  spill_1: spill, state=chemical_spill (hazard_bearing)
threats: spill_1, tanker_truck_1
at_risk: hazmat_worker_1, hazmat_worker_2"""

SCENE_POOL = """entities:
  child_1: child, state=drowning (at_risk), at_risk_as=distress
  child_2: child, state=standing (normal), at_risk_as=proximity
  pool_1: pool, state=hazardous_in_context (hazard_bearing)
threats: pool_1
at_risk: child_1, child_2"""

VICTIM_PAIRS = [
    {"name": "people vs vehicles", "run": "ui_21f1cdad · D_aerial",
     "scene": SCENE_SPILL,
     "hazards": ["spill_1"],
     "victims_a": ["fire_truck_1", "ambulance_1", "police_car_1"],
     "victims_b": ["hazmat_worker_1", "hazmat_worker_2"],
     "expect": "graph_b"},

    {"name": "a dog vs an intact house", "run": "ui_be09616d · A_fire",
     "scene": SCENE_BLOCK,
     "hazards": ["house_1"],
     "victims_a": ["person_1", "house_2"],
     "victims_b": ["person_1", "dog_1", "car_1"],
     "expect": "graph_b"},

    {"name": "a child vs the pool itself", "run": "ui_3049cd31 · B_pool",
     "scene": SCENE_POOL,
     "hazards": ["pool_1"],
     "victims_a": ["child_1"],
     "victims_b": ["pool_1"],
     "expect": "graph_a"},

    {"name": "two sets of objects", "run": "ui_6d2b4f82 · C_tanker",
     "scene": """entities:
  fire_1: fire, state=burning (hazard_bearing)
  tanker_truck_1: tanker_truck, state=leaking (hazard_bearing)
  spill_1: spill, state=seeping (hazard_bearing)
  road_1: road, state=paved (normal)
  person_1: person, state=standing (normal), at_risk_as=proximity
threats: fire_1, tanker_truck_1
at_risk: person_1""",
     "hazards": ["fire_1"],
     "victims_a": ["tanker_truck_1"],
     "victims_b": ["road_1", "spill_1"],
     "expect": None},          # recorded, not scored — genuinely arguable
]

# Q2 — both graphs assert the same (source, target) with a different effect.
MECHANISM_PAIRS = [
    {"name": "exposes vs may_spread_to", "run": "ui_6df423fc · D_aerial",
     "scene": SCENE_SPILL, "source": "tanker_truck_1",
     "target": "spill_1", "effect_a": "exposes", "effect_b": "may_spread_to",
     "expect": "same_response"},
    # DROPPED: pool_1 --may_spread_to--> child_2. A pool does not spread to a
    # child; the edge is malformed, and asking what response it calls for is
    # not a test of the judge.
    # the control. These are NOT mutually exclusive — a fire both hurts you and
    # cuts you off — which is why the first version of Q2 was ill-posed. But
    # they call for different OPERATIONS: move the person away, versus open a
    # route through to them.
    {"name": "may_harm vs isolates (control)", "run": "constructed",
     "scene": SCENE_BLOCK, "source": "house_1", "target": "person_1",
     "effect_a": "may_harm", "effect_b": "isolates",
     "expect": "different_response"},
]


def score_graph_judge(judge_fn=None, n_probes: int = 5) -> dict:
    """Run both graph-judge questions over the pairs above."""
    from agentic.judge_graph import judge_mechanism, judge_victims
    rows, hits, total = [], 0, 0
    for c in VICTIM_PAIRS:
        v = judge_victims(c["scene"], c["hazards"], c["victims_a"],
                          c["victims_b"], judge_fn=judge_fn, n_probes=n_probes)
        ok = None
        if c["expect"] is not None:
            ok = v["verdict"] == c["expect"]
            hits += ok
            total += 1
        rows.append({"q": "victims", **c, "got": v["verdict"],
                     "votes": v["votes"], "n": v["n"], "ok": ok})
    for c in MECHANISM_PAIRS:
        v = judge_mechanism(c["scene"], c["source"], c["target"],
                            c["effect_a"], c["effect_b"], judge_fn=judge_fn,
                            n_probes=n_probes)
        ok = v["verdict"] == c["expect"]
        hits += ok
        total += 1
        rows.append({"q": "mechanism", **c, "got": v["verdict"],
                     "votes": v["votes"], "n": v["n"], "ok": ok})
    return {"rows": rows, "hits": hits, "total": total}


if __name__ == "__main__" and "--graphs" in sys.argv:
    from agentic.judge_card import JUDGE_PROBE_TEMPERATURE, _ollama_judge
    fn = (lambda p: _ollama_judge(p, temperature=JUDGE_PROBE_TEMPERATURE))
    res = score_graph_judge(judge_fn=fn)
    for r in res["rows"]:
        mark = "" if r["ok"] is None else ("OK" if r["ok"] else "XX")
        want = r["expect"] if r["expect"] else "(unscored)"
        print(f"{r['q']:10s} {r['name']:34s} got={r['got']:9s}"
              f"({r['votes']}/{r['n']})  want={want:9s} {mark}")
    print(f"\n{res['hits']}/{res['total']} correct")
