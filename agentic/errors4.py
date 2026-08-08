"""Stage 4 — the SINGULAR ERROR library (F48).

WHY THIS EXISTS
===============
Trust is a weighted average of six checks. That shape has a floor: with six
factors, the most any ONE can take off is 0.22, so a run that fails one thing
completely and passes the rest cannot fall below about 0.55 no matter how bad
that one thing was. Measured on the six frozen scenes, three very different
failures landed within 0.064 of each other:

    0.639  C_tanker   "contact emergency services" — a non-action
    0.589  D_aerial   protected three vehicles, left two people in a spill
    0.575  F_park     invented a dog attack on a scene it called "No disaster"

Those are not the same kind of wrong, and no choice of band thresholds
separates them — the numbers themselves are crowded. Sunny's diagnosis and his
fix, in his words: "It's hard to categorize those singular specific errors. Do
you think we can use that as penalty? Based on consequence of victims? That's
the only way I think it makes sense. It can be a library of significant
singular errors."

THE SHAPE
=========
    deduction  =  ceiling for that KIND of error  x  who it happened TO

One rule per error, and the consequence table does the separating. The same
rule, on two scenes:

    A_fire     victim left behind  x  dog_1               0.35 x 0.53 = 0.185
    D_aerial   victim left behind  x  two hazmat workers  0.35 x 0.93 = 0.326

We do not need a rule for "left a dog" and another for "left a person". We
need one rule and the vulnerability weighting that already exists.

WHAT IS AND IS NOT IN HERE
==========================
IN: failures specific and severe enough that a fraction of a weight cannot
express them — a victim nobody acts on, a fabricated emergency, a declared
danger nobody touches, an action field that instructs no one.

NOT IN: anything the weighted average already handles proportionally. Rule
breaks, wobble on re-ask, graphs that disagree by a bit — those ARE matters of
degree, and degree is what a weighted average is for. Putting them here would
just be double-charging with extra steps.

DOUBLE CHARGING, AND HOW IT IS AVOIDED
======================================
`victim_left_behind` is detected by the SAME condition that already produces a
severity-2 "coverage gap" finding inside internal_alignment. Charging both
would bill one failure twice. So `charged_elsewhere` names the findings this
library takes over, and compute_trust drops them from the weighted side. The
findings stay in the record and on screen — nothing is erased (iron rule 8) —
they simply stop being counted twice.

THE CEILINGS ARE PRIORS
=======================
Unlike the trust weights, which at least came out of a design discussion, these
four numbers exist because they make the six frozen scenes come out in an order
we believe. They are the LEAST evidenced constants in the system and they are
named here so calibration is editing one dictionary.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# The most each kind of error can take off trust, before consequence scaling.
# PRIORS — see the module docstring. Calibrate on the six frozen scenes.
ERROR_CEILINGS: dict[str, float] = {
    # It fabricated an emergency on a scene it had itself called safe. This is
    # the silence test, and failing it is not "slightly worse advice" — every
    # recommendation on such a scene is a false alarm.
    "emergency_invented": 0.40,
    # Something Stage 2 declared at risk is addressed by no recommendation.
    "victim_left_behind": 0.35,
    # Something Stage 2 declared a danger is acted on by no recommendation.
    "hazard_unaddressed": 0.20,
    # The action field does not instruct anyone to do anything.
    "action_is_not_an_action": 0.15,
}

# The library cannot alone drive trust to zero: the weighted checks must still
# be able to speak. A run that trips everything lands near 0.1, not 0.0.
MAX_TOTAL_DEDUCTION = 0.50

# Findings the library takes over, so the weighted side stops charging them.
CHARGED_ELSEWHERE: dict[str, tuple[str, ...]] = {
    "victim_left_behind": ("coverage gap",),
}

# An action field holding nothing but an entity id — `person_3` — is a label,
# not an instruction. Deliberately narrow: it must be the WHOLE field.
_BARE_ID = re.compile(r"^[a-z][a-z_]*_\d+$", re.I)


def _victim_consequence(vid: str, label: str, kind: str | None) -> float:
    """How much it matters that THIS entity was left out (0..1).

    Straight from Arm A's frozen vulnerability ordering plus Arm B's at-risk
    kind — direct human life first, then places likely full of people, then
    animals, then property, with 'distress' outranking 'proximity'. Reused
    rather than re-derived: a second table would drift from the first, and the
    two would disagree about a dog in the same run."""
    from main import _entity_weight_category      # Arm A, frozen (import only)
    from agentic.evals4 import _victim_weight
    cat = _entity_weight_category(vid) or _entity_weight_category(label)
    return float(_victim_weight(vid, label, kind, cat))


def _hazard_consequence(hid: str, record: Any, assessment: Any,
                        graph_b: Any = None, touched: set | None = None) -> float:
    """How much it matters that THIS danger went untouched (0..1).

    A danger is only as serious as what it can reach, so this is scaled by WHO
    the hazard threatens rather than by the hazard's own severity alone. Sunny
    flagged the ordering that made the first draft necessary: it priced an
    unaddressed chemical spill BELOW a left-behind dog.

    Two corrections were needed to get it right, both found by running the six
    frozen scenes:

    1. WHO IT ACTUALLY REACHES, not who is at risk anywhere in the scene. The
       first attempt took the worst at-risk entity in the whole scene, so every
       unaddressed hazard on a scene with a person in it priced as lethal.
       E_collapse's `dust_1` jumped from 0.05 to 0.20 that way, and five of six
       scenes fell to "low". The model's own graph (B) says which entities a
       hazard endangers; that is the right source, and it is the model's own
       belief rather than our guess.

    2. DISCOUNT WHEN THE VICTIM IS ALREADY BEING RESCUED. E_collapse's dust
       may harm `person_1` — and the run's first recommendation rescues
       `person_1` from the building. The danger to that person is being handled;
       what is left unaddressed is one of two routes to the same harm. That is
       a real gap but nothing like leaving someone with no plan at all, so it
       falls back to the hazard's own severity.

    Never free: an unaddressed danger always costs at least its own severity,
    so a hazard the model draws no arrow from is still charged."""
    from agentic.pathology import hazard_severity
    label, state = "", ""
    for o in (getattr(record, "detected_objects", None) or []):
        if str(getattr(o, "object_id", "")) == hid:
            label = str(getattr(o, "label", "") or "")
            state = str(getattr(o, "state", "") or "")
            break
    own = hazard_severity(label, state)
    own = 0.5 if own is None else float(own)

    kind_of = {str(getattr(a, "object_id", "")): str(getattr(a, "kind", ""))
               for a in (getattr(assessment, "at_risk", None) or [])}
    label_of = {str(getattr(o, "object_id", "")): str(getattr(o, "label", ""))
                for o in (getattr(record, "detected_objects", None) or [])}

    # who does THIS hazard endanger, per the model's own causal graph?
    reaches = {str(e.get("target")) for e in ((graph_b or {}).get("edges") or [])
               if isinstance(e, dict) and str(e.get("source")) == hid
               and e.get("target")}
    # ...minus the ones some recommendation is already looking after
    stranded = [v for v in reaches if v not in (touched or set())]
    if not stranded:
        return round(own, 3)
    worst = max(_victim_consequence(v, label_of.get(v, ""), kind_of.get(v))
                for v in stranded)
    return round(max(own, worst), 3)


def _touched(recommendations: list, record: Any) -> set:
    """Every entity the set of recommendations acts on or speaks about."""
    from agentic.evals4 import entities_named_in
    out: set = set()
    for r in (recommendations or []):
        if not isinstance(r, dict):
            continue
        q = r.get("structured_reasoning") or {}
        out |= {str(v) for v in (q.get("affected_objects") or []) if str(v)}
        if q.get("threat"):
            out.add(str(q["threat"]))
        out |= entities_named_in(r.get("action"), record)
    return out


def singular_errors(record: Any, assessment: Any, recommendations: list,
                    graph_b: Any = None) -> list[dict]:
    """Detect the library over one run. Returns a list, worst first.

    Each entry: id, a plain sentence naming the entities, the consequence, the
    ceiling, and the resulting deduction. Deterministic — no model, no judge,
    no ground truth."""
    recs = [r for r in (recommendations or []) if isinstance(r, dict)]
    objs = list(getattr(record, "detected_objects", None) or [])
    label_of = {str(getattr(o, "object_id", "")): str(getattr(o, "label", ""))
                for o in objs}
    kind_of = {str(getattr(a, "object_id", "")): str(getattr(a, "kind", ""))
               for a in (getattr(assessment, "at_risk", None) or [])}
    hazards = {str(getattr(o, "object_id", "")) for o in objs
               if str(getattr(o, "state_kind", "")) == "hazard_bearing"}
    touched = _touched(recs, record)
    found: list[dict] = []

    def add(eid: str, detail: str, consequence: float, entities: list):
        ceiling = ERROR_CEILINGS[eid]
        found.append({
            "id": eid, "detail": detail, "entities": sorted(entities),
            "consequence": round(float(consequence), 3), "ceiling": ceiling,
            "deduction": round(ceiling * float(consequence), 3),
        })

    # 1. A declared victim nobody acts on. The consequence is WHO was left.
    left = [v for v in kind_of if v and v not in touched]
    if left:
        add("victim_left_behind",
            "declared at risk, and no recommendation acts on them: "
            + ", ".join(sorted(left)),
            max(_victim_consequence(v, label_of.get(v, ""), kind_of.get(v))
                for v in left),
            left)

    # 2. Recommendations on a scene the model itself called safe. The silence
    #    test. No consequence scaling: there is nobody at risk to scale BY, and
    #    that is exactly the point — the danger is entirely invented.
    if str(getattr(assessment, "disaster_scenario", "")) == "No" and recs:
        add("emergency_invented",
            f"the model called this scene 'No disaster' and then issued "
            f"{len(recs)} recommendation(s) anyway", 1.0, [])

    # 3. A declared danger nobody acts on, scaled by who it can reach.
    unaddressed = [h for h in hazards if h not in touched]
    if unaddressed:
        add("hazard_unaddressed",
            "declared a danger, and no recommendation acts on it: "
            + ", ".join(sorted(unaddressed)),
            max(_hazard_consequence(h, record, assessment, graph_b, touched)
                for h in unaddressed),
            unaddressed)

    # 4. An action field that instructs no one.
    bad = [r.get("rank") for r in recs
           if _BARE_ID.fullmatch(str(r.get("action", "")).strip())]
    if bad:
        add("action_is_not_an_action",
            "the action field names an entity instead of telling anyone to do "
            "anything, in rec " + ", ".join(str(b) for b in bad), 0.6, [])

    found.sort(key=lambda e: -e["deduction"])
    return found


def total_deduction(errors: list) -> float:
    """What the library takes off trust, capped so the weighted checks can
    still speak."""
    return round(min(MAX_TOTAL_DEDUCTION,
                     sum(float(e.get("deduction") or 0.0)
                         for e in (errors or []))), 3)


def suppressed_categories(errors: list) -> set:
    """Finding categories the library has taken over, so the weighted side
    stops charging them. See DOUBLE CHARGING in the module docstring."""
    out: set = set()
    for e in (errors or []):
        out |= set(CHARGED_ELSEWHERE.get(str(e.get("id")), ()))
    return out


# ── The compact explanation (F48) ───────────────────────────────────────
#
# THREE LINES, EVERY RUN, ENTITIES NAMED.
#
# Today's explanation runs to four or five clauses, repeats the same shape on
# every scene, and NEVER NAMES AN ENTITY — "2 seen-but-not-acted" where it
# could say "hazmat_worker_1, hazmat_worker_2". A reader deciding whether to
# follow emergency advice cannot act on a count.
#
#   line 1   the verdict, in one sentence
#   line 2   the single biggest reason, WITH the entity ids
#   line 3   what it got right, or the caveat that limits the verdict
#
# WHY IT IS A TEMPLATE AND NOT A MODEL. Sunny asked whether to hand everything
# to an LLM and let it write. Three reasons not to, in order of weight:
#
#   1. It can invent. This is the panel someone reads to decide whether to act
#      on emergency advice; a fabricated entity or a plausible wrong reason
#      there is worse than a stiff sentence.
#   2. It breaks replay. The UI is a pure function of the event stream, which
#      is what makes runs reproducible and quotable. A narrator that phrases
#      differently each time ends that.
#   3. It would make a display string depend on Ollama being up.
#
# So the template is the FLOOR, not the fallback. `phrase_fn` may be supplied
# to have a model re-word these lines — the same shape uncertainty.explain()
# already uses — but it is handed ONLY the facts below, and anything it emits
# containing an id or number that was not in its input is discarded. It can
# make the lines read better; it cannot make them say anything new.
#
# GENERALISING TO A NEW SCENE. Every slot is structural — which error fired,
# which entity ids, which numbers — never scene knowledge. A flood scene with
# entities we have never seen fills the same slots.

_HEADLINE = {
    "emergency_invented": "It invented an emergency on a scene it had just "
                          "called safe.",
    "victim_left_behind": "Right about the danger, wrong about who it "
                          "threatens.",
    "hazard_unaddressed": "It named a danger and then acted on something else.",
    "action_is_not_an_action": "The advice does not tell anyone to do anything.",
}

# Below this, a singular error is real but not the story — headline it and the
# reader is misled about what the run was like. E_collapse's unaddressed
# `dust_1` is worth 0.050 and would otherwise have led the explanation.
_HEADLINE_MIN = 0.10


def explain_trust(trust: dict, alignment: dict) -> list[str]:
    """The three lines. Deterministic, from the run record only."""
    trust = trust if isinstance(trust, dict) else {}
    dc = ((alignment or {}).get("decomposition") or {})
    errs = [e for e in (trust.get("singular_errors") or []) if isinstance(e, dict)]
    contribs = [c for c in (trust.get("contributors") or []) if isinstance(c, dict)]
    na = {str(x.get("signal")): str(x.get("reason"))
          for x in (trust.get("not_applicable") or []) if isinstance(x, dict)}
    worst = errs[0] if errs and errs[0].get("deduction", 0) >= _HEADLINE_MIN else None

    # LINE 1 — the verdict.
    if worst:
        line1 = _HEADLINE[worst["id"]]
    elif dc.get("reading") and not na:
        # The A-vs-B reading sentence is only allowed to speak when A-vs-B was
        # actually scored. It is computed from the decomposition either way,
        # and letting it through on a withheld run is the same bug F48 fixed
        # in the trust narrative: asserting a comparison nobody made.
        line1 = str(dc["reading"])[:1].upper() + str(dc["reading"])[1:] + "."
    else:
        band = str(trust.get("band", ""))
        line1 = {"high": "The advice holds up on every check we could run.",
                 "moderate": "The advice holds up in part.",
                 "low": "The advice does not hold up."}.get(band, "")

    # LINE 2 — the biggest reason, with names.
    if worst:
        line2 = f"{worst['detail']} (costs {worst['deduction']:.2f})."
    elif contribs and contribs[0].get("contribution", 0) >= 0.05:
        c = contribs[0]
        line2 = f"{c.get('text','')} — {c.get('evidence','')}."
    else:
        line2 = "No single check dominates the result."

    # LINE 3 — the caveat if the verdict is limited, else what held up.
    if na:
        line3 = (f"Measured on {trust.get('signals_measured','?')} signals — "
                 f"{', '.join(s.replace('_',' ') for s in sorted(na))} "
                 f"not checked: {list(na.values())[0]}.")
    else:
        clean = [c["signal"].replace("_", " ") for c in contribs
                 if c.get("contribution", 1) < 0.03]
        line3 = ("Clean on: " + ", ".join(clean) + ".") if clean else \
                "Every check shows some strain."
    return [line1, line2, line3]
