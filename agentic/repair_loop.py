"""Loop 1 — local repair for Stage 1 perception.

WHAT THIS LOOP IS
=================
After the VLM names the scene's entities, some answers violate the
INTERFACE CONTRACT: a family name instead of a specific label ("vehicle"),
a label outside the vocabulary, a state word outside the state vocabulary,
a missing anchor box. These are all MACHINE-CHECKABLE. The loop shows the
model its own answer plus the specific rule each entry broke, and asks for
a corrected list. That is "evidence-triggered reflection": the model gets
something concrete to fix, never a vague "look again".

WHAT THIS LOOP IS NOT
=====================
It never questions the model's PERCEPTION. If the model says a drowning
child is "swimming", that is a legal state word and no rule fires; the
error stays in the record as evidence about the model. Repairing it would
coach the subject on exactly the ability CEE+ exists to measure. The loop
repairs how the model SPEAKS (vocabulary discipline), never what it SEES.

SHAPE OF THE LOOP
=================
    entities = first VLM answer
    for round in 1..MAX_REPAIR_ROUNDS:
        violations = detect_violations(entities)   # pure function, no model
        if no violations: stop                     # clean -> done
        if entities identical to last round: stop  # oscillation guard
        entities = ask_model(entities, violations) # one targeted call
    return entities + full trace of every round

Every round is logged in a RepairTrace: which violations fired, what was
asked, what came back. The trace is saved with the run record, so a scene
that needed three rounds is visibly different from one that was clean on
arrival, and the (violation -> fix) pairs become training data for the
repair SLM later (AGENTIC_PLAN Stage 25, track 1).

DETERMINISM NOTE
================
detect_violations is pure code: same entities in, same violations out.
The only nondeterminism is the VLM's answer itself, and every answer is
recorded. The loop cannot run away: hard cap + oscillation guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re

from agentic import rulebook  # noqa: E402
from agentic.vocabulary import (  # noqa: E402
    LABEL_FAMILIES,
    OTHER_LABEL,
    PURE_FAMILY_NAMES,
    canonicalize_label,
    family_of,
)

# The loop never runs more than this many repair calls per scene.
MAX_REPAIR_ROUNDS = 2
# ...unless it is making PROGRESS. A round that fixes something and surfaces
# something new is repairing, not circling, and the cap exists to stop
# circling. D_aerial 2026-07-28: round 1 asked for a missing caption entity,
# the model added it carrying a bad state, round 2 flagged that state, and the
# cap ended the loop before the model could fix what round 1 had caused. The
# defect it shipped then triggered a petition, which failed. So: allow extra
# rounds while the violation SET keeps changing, under a hard ceiling that
# still makes non-termination impossible.
MAX_REPAIR_ROUNDS_PROGRESSING = 4

# States the repair prompt may remind the model of. This is vocabulary
# DISCIPLINE (which words are legal), not perception guidance (which word
# is true for this entity). The model still chooses.
# Ordering is deliberate (F10 follow-up, the road·burning autopsy): VLMs
# have positional bias, and the old rendering chained "...what you see:
# hazard-bearing: burning" — the colon structure plus first-position
# hazard words primed scene-congruent drama ('paved' became 'burning').
# NORMAL leads now: most entities in most scenes ARE normal, so
# first-position bias pulls toward the statistically honest prior.
_STATE_WORD_REMINDER = (
    "(any group is a valid choice - most entities in most scenes are "
    "normal)\n"
    "  normal: intact, standing, upright, dry, sealed, stationary, "
    "resting, healthy, stable, swimming, walking, running, parked, paved\n"
    "  at-risk: injured, bleeding, fleeing, trapped, cowering, drowning, "
    "suffocating, unconscious\n"
    "  hazard-bearing: burning, burnt, collapsed, collapsing, fallen, "
    "crushed, flooded, leaking, spreading, billowing, rising, seeping, "
    "hazardous_in_context"
)


# ── The violation record ────────────────────────────────────────────────


class Violation(BaseModel):
    """One rule broken by one entity. `instruction` is the sentence the
    model will see; it always cites the evidence and the legal options."""

    entity_index: int          # position in the entity list (ids not yet assigned)
    raw_label: str             # what the model actually wrote
    kind: str                  # family_name_as_label | label_out_of_vocab
    #                          | state_out_of_vocab | missing_anchor_bbox
    instruction: str


class RepairRound(BaseModel):
    """What happened in one iteration of the loop (working memory + log)."""

    round_number: int
    violations: list[Violation]
    entities_after: list[dict[str, Any]] = Field(default_factory=list)
    changed: bool = True       # False when the model returned the same list


class RepairTrace(BaseModel):
    """The loop's full history, saved into the run record."""

    rounds: list[RepairRound] = Field(default_factory=list)
    clean_on_arrival: bool = False     # no violations in the first answer
    stopped_reason: str = ""           # clean | cap_reached | no_change


# ── Caption-entity completeness (rule 5's helper) ───────────────────────
#
# The caption is part of the system's INPUT, not an opinion: if it names a
# tanker truck and the entity list has none, that mismatch is checkable
# evidence (the C_tanker regression of 2026-07-21: the model returned ONE
# entity for a captioned three-hazard scene, and no rule fired). We resolve
# caption nouns through the same vocabulary + synonym map the labels use,
# so this stays mechanical.
#
# Matching rule: a caption noun is satisfied by an entity with the SAME
# canonical label, or, for living beings only (person / responder / animal
# families), by any entity of the same family ("driver" in the caption is
# satisfied by man_1). Vehicles and objects need the exact label: a caption
# tanker_truck is NOT satisfied by car_1; that leniency was the round 1 bug.

_LENIENT_FAMILIES = {"person", "responder", "animal"}

# P6: near-total overlap between two same-group living beings reads as
# one individual listed twice. 0.8 is a prior; calibrated on E_collapse
# (true duplicates measured IoU ≈ 0.9+; distinct adjacent officers ≈ 0.1).
DUPLICATE_IOU = 0.8


def _iou(a: Any, b: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
    except (TypeError, ValueError):
        return 0.0
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


def caption_labels(caption: str) -> dict[str, str]:
    """Extract canonical vocabulary labels mentioned in the caption.

    Returns {canonical_label: raw_phrase}. Bigrams are tried before
    unigrams so "tanker truck" resolves as tanker_truck, not truck.
    """
    words = re.findall(r"[a-z]+", str(caption or "").lower())
    found: dict[str, str] = {}
    used: set[int] = set()
    # Bigrams first (tanker truck, lifeguard chair, caution tape, ...).
    for i in range(len(words) - 1):
        if i in used or (i + 1) in used:
            continue
        phrase = f"{words[i]}_{words[i + 1]}"
        canon, _, in_vocab, is_family = canonicalize_label(phrase)
        if in_vocab and not is_family and canon != OTHER_LABEL:
            found.setdefault(canon, f"{words[i]} {words[i + 1]}")
            used.update({i, i + 1})
    for i, w in enumerate(words):
        if i in used:
            continue
        canon, _, in_vocab, is_family = canonicalize_label(w)
        if in_vocab and not is_family and canon != OTHER_LABEL:
            found.setdefault(canon, w)
    return found


def caption_danger_states(caption: str) -> dict[str, str]:
    """P7 (F21). Condition words the caption uses that are DANGER states in the
    closed state vocabulary. Returns {caption_word: canonical_state}.

    The lexicon is the existing vocabulary, deliberately: the check can then
    say "the caption names a condition you did not use" without inventing a
    single word of its own, and it can never propose which state is correct.
    Only hazard_bearing / at_risk states qualify — a caption saying 'swimming'
    is not evidence of anything.

    NORMALISATION DEPTH (Sunny, 2026-07-28). Both sides of the comparison go
    through `arm_b_canonical_state` — Arm B's declined folds first, then Arm
    A's frozen synonym map underneath, which is deeper than Arm B's own
    ('down' -> 'fallen', 'on fire' -> 'burning'). Comparing a shallow
    caption form against a deep declared form manufactures disagreements that
    are not there. ONE normaliser, both sides."""
    from agentic.perception import (  # local: cycle
        arm_b_canonical_state, state_kind)
    words = re.findall(r"[a-z]+", str(caption or "").lower())
    found: dict[str, str] = {}
    for w in words:
        canon = arm_b_canonical_state(w)
        if state_kind(canon) in ("hazard_bearing", "at_risk"):
            found.setdefault(w, canon)
    return found


def _caption_state_satisfied(canon_state: str,
                             entities: list[dict[str, Any]]) -> bool:
    """True when some entity already carries that state. The declared side is
    canonicalised through the SAME Arm A function as the caption side — one
    normaliser, both sides, or the comparison is not like-for-like."""
    from agentic.perception import arm_b_canonical_state  # local: cycle
    return any(arm_b_canonical_state(str(e.get("state", ""))) == canon_state
               for e in entities)


def _caption_label_satisfied(wanted: str, entities: list[dict[str, Any]]) -> bool:
    """Exact-label match; family-level match for living beings only."""
    wanted_family = family_of(wanted)
    for e in entities:
        canon, _, in_vocab, _ = canonicalize_label(str(e.get("label", "")))
        if canon == wanted:
            return True
        if (in_vocab and wanted_family in _LENIENT_FAMILIES
                and family_of(canon) == wanted_family):
            return True
    return False


# ── Fluid-convention-aware caption satisfaction (F9, 2026-07-22) ────────
#
# The A_fire regression: the caption "a house on fire" made the naive
# matcher demand a 'fire' entity, contradicting the fluid convention the
# perception prompt itself teaches (fire attached to a burning object is
# a STATE, not an entity). Loop 1 manufactured a redundant fire_1, S6
# then prosecuted the model for not listing it as a threat, and only the
# model's STOOD kept the answer GT-correct (at the cost of U 0.114->0.275).
# When two of our rules collide, the bug is in OUR rules, never the
# defendant.
#
# Fix, deliberately NARROW: a caption mention of a medium is satisfied by
# an entity already carrying the attached state — but only for media whose
# attachment the convention recognizes, and only for attached-sounding
# phrases. Free-burning phrasings ("brush fire", "wildfire") stay strict:
# no attached state accounts for a free fire. smoke/dust/gas stay strict
# (always diffuse). spill stays strict (producer-and-medium rule REQUIRES
# the spill entity beside the leaking producer). Residual risk, accepted
# and ledgered in F9: a caption saying just "fire" for a scene with BOTH a
# burning structure AND a separate free fire will not ticket the free
# fire — that gap belongs to the petition layer and the GT harness.

_ATTACHED_STATE_SATISFIERS: dict[str, dict[str, set[str]]] = {
    "fire": {
        "phrases": {"fire", "flame", "flames", "blaze"},
        "states": {"burning", "burnt"},
    },
    "water": {
        "phrases": {"water", "flood", "floodwater", "flood water"},
        "states": {"flooded"},
    },
}


def _satisfied_by_attached_state(
    wanted: str, raw_phrase: str, entities: list[dict[str, Any]]
) -> bool:
    spec = _ATTACHED_STATE_SATISFIERS.get(wanted)
    if spec is None or str(raw_phrase).lower() not in spec["phrases"]:
        return False
    # One normaliser across Arm B: arm_b_canonical_state applies our declined
    # folds and then Arm A's map. Calling Arm A directly here would bypass the
    # override and let two checks in the same file disagree about one word.
    from agentic.perception import arm_b_canonical_state
    for e in entities:
        if arm_b_canonical_state(str(e.get("state", ""))) in spec["states"]:
            return True
    return False


# ── Step 1: detect violations (pure code, no model call) ────────────────


def detect_violations(
    entities: list[dict[str, Any]], caption: str = ""
) -> list[Violation]:
    """Run the interface rules over a raw entity list.

    Works on the model's RAW answers (before canonicalization rewrites
    them), because the repair prompt must quote what the model actually
    wrote. Each check mirrors a flag the perception contract records.
    When a caption is given, rule 5 (completeness against the caption's
    named entities) also runs.
    """
    from agentic.perception import resolve_state, state_kind  # local: avoids cycle

    violations: list[Violation] = []
    for i, e in enumerate(entities):
        raw_label = str(e.get("label", "")).strip()
        canon, note, in_vocab, is_family = canonicalize_label(raw_label)

        # Rule P1: family names are not labels. The instruction text comes
        # from the rulebook (rule + rationale + worked example); this code
        # only supplies the evidence.
        if is_family:
            members = ", ".join(LABEL_FAMILIES[note.split(":", 1)[1]])
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="family_name_as_label",
                instruction=rulebook.instruction_for(
                    "family_name_as_label",
                    index=i, raw_label=raw_label, members=members,
                    # The model's own words — the evidence that unlocks
                    # the right label when it lives in ANOTHER family
                    # (E_collapse: 'infrastructure' described as "police
                    # car with flashing lights"; the old members-only
                    # menu was a locked room).
                    description=str(e.get("description", ""))[:80]),
            ))

        # Rule P2: label outside vocabulary and synonym map. The model may
        # legitimately keep 'other' (the escape hatch), but it must do so
        # deliberately, not by accident.
        elif not in_vocab and canon == OTHER_LABEL and raw_label.lower() != OTHER_LABEL:
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="label_out_of_vocab",
                instruction=rulebook.instruction_for(
                    "label_out_of_vocab", index=i, raw_label=raw_label),
            ))

        # Rule P3: state word outside the state vocabulary (after synonyms
        # and extensions — including LABEL-AWARE ones: spill·'active' is
        # legal English for seeping, so it must not draw a ticket). We
        # list the legal words; we NEVER suggest which one is true for
        # this entity - that would be coaching perception.
        if state_kind(resolve_state(canon, e.get("state", ""))) == "unknown":
            # P3b: the offending word is itself a legal LABEL. The model did
            # not reach for an unknown word — it answered "what is it?" twice.
            # That is diagnosable, so name the confusion instead of listing
            # thirty legal words at a model that was not short of vocabulary.
            _st = str(e.get("state", "")).strip().lower()
            _lc, _ln, _l_in, _l_fam = canonicalize_label(_st)
            if _st and _l_in and not _l_fam and _lc != OTHER_LABEL:
                violations.append(Violation(
                    entity_index=i, raw_label=raw_label,
                    kind="state_is_a_label",
                    instruction=rulebook.instruction_for(
                        "state_is_a_label", index=i, raw_label=raw_label,
                        state=e.get("state", ""),
                        state_words=_STATE_WORD_REMINDER),
                ))
                continue
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="state_out_of_vocab",
                instruction=rulebook.instruction_for(
                    "state_out_of_vocab",
                    index=i, raw_label=raw_label, state=e.get("state", ""),
                    state_words=_STATE_WORD_REMINDER),
            ))

        # Rule P4: every entity needs a rough anchor box; without one the
        # detector cannot be told WHICH instance the model meant.
        if not e.get("bbox") and not e.get("anchor_bbox"):
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="missing_anchor_bbox",
                instruction=rulebook.instruction_for(
                    "missing_anchor_bbox", index=i, raw_label=raw_label),
            ))

    # Rule P6: duplicate living individuals (E_collapse ui_20fb0754: the
    # model listed the two officers as person_2/person_3, then a P5
    # ticket for 'police_officer' made it ADD them again — five humans
    # in a three-human scene). Same life group + near-total box overlap
    # = one individual listed twice, absent an explicit reason. The
    # model resolves it (merge under the better label, or keep both and
    # disambiguate the descriptions) — code only points.
    lifelike = []
    for i, e in enumerate(entities):
        canon, _, _, _ = canonicalize_label(str(e.get("label", "")))
        fam = family_of(canon)
        group = ("human" if fam in ("person", "responder")
                 else "animal" if fam == "animal" else None)
        box = e.get("bbox") or e.get("anchor_bbox")
        if group and box:
            lifelike.append((i, e, group, box))
    for x in range(len(lifelike)):
        for y in range(x + 1, len(lifelike)):
            ia, ea, ga, ba = lifelike[x]
            ib, eb, gb, bb = lifelike[y]
            if ga != gb:
                continue
            iou = _iou(ba, bb)
            if iou >= DUPLICATE_IOU:
                violations.append(Violation(
                    entity_index=ia,
                    raw_label=str(ea.get("label", "")),
                    kind="duplicate_entity",
                    instruction=rulebook.instruction_for(
                        "duplicate_entity",
                        index_a=ia, label_a=ea.get("label", ""),
                        desc_a=str(ea.get("description", ""))[:60],
                        index_b=ib, label_b=eb.get("label", ""),
                        desc_b=str(eb.get("description", ""))[:60],
                        iou=f"{iou:.2f}"),
                ))

    # Rule 5: completeness against the caption. The caption is given input;
    # an entity it names that the list lacks is checkable evidence of a
    # dropped entity. The instruction explicitly allows the model to stand
    # its ground when the thing is genuinely not visible: an honest refusal
    # ends the loop via the no_change guard and stays flagged, it is never
    # forced. (This is completeness against the INPUT, not perception
    # coaching: we say what the caption names, never what to see in it.)
    for wanted, raw_phrase in caption_labels(caption).items():
        if _satisfied_by_attached_state(wanted, raw_phrase, entities):
            continue    # "a house on fire" + house·burning: already told
        if not _caption_label_satisfied(wanted, entities):
            violations.append(Violation(
                entity_index=-1, raw_label=wanted,
                kind="caption_entity_missing",
                instruction=rulebook.instruction_for(
                    "caption_entity_missing",
                    raw_phrase=raw_phrase, wanted=wanted),
            ))
    # Rule P7 (F21): the caption's CONDITION words, not just its entities.
    # B_pool: caption "floats motionless and unconscious", list says
    # child_2·swimming — a legal state, an entity that exists, so nothing
    # fired and an unconscious child entered the record as 'normal'. This is
    # the only check that reads a source outside the model's own answer, which
    # is why it can reach a model that is stably wrong. We quote the caption
    # and never name the right state; standing your ground is legal.
    #
    # ONE TICKET PER CAPTION, not one per word. B_pool's caption fires on
    # 'unconscious' (the real signal), on 'struggling' (arguably already
    # covered by child_1·drowning), and on 'down' (from "face down" — the
    # vocabulary carries 'down' as a hazard state, as in a downed power line).
    # Three tickets for one disagreement buries the signal in its own noise
    # and repeats the double-counting we corrected twice already. One ticket
    # lists every unmatched word and lets the model sort them out — including
    # standing its ground on the ones that are not states at all.
    if caption:
        declared = sorted({str(e.get("state", "")) for e in entities
                           if e.get("state")})
        unmatched = [w for w, canon in caption_danger_states(caption).items()
                     if not _caption_state_satisfied(canon, entities)]
        if unmatched:
            violations.append(Violation(
                entity_index=-1, raw_label=", ".join(unmatched),
                kind="caption_state_contradiction",
                instruction=rulebook.instruction_for(
                    "caption_state_contradiction",
                    word=", ".join(f'"{w}"' for w in unmatched),
                    declared=", ".join(declared) or "(none)"),
            ))
    return violations


# ── Step 2: the repair request (one targeted model call) ────────────────

REPAIR_PROMPT_TEMPLATE = """You listed entities for this image. Some entries
break the required format. Below is YOUR list, then the specific problems.
Fix ONLY the listed problems. Keep every judgment about what you SEE
(states, which entities exist, descriptions) unless a rule forces a change.

Your entity list:
{entities_json}

Problems to fix:
{problem_lines}

Return the FULL corrected list as valid JSON in the same schema:
{{"entities": [{{"label": "...", "state": "...", "description": "...",
"bbox": [x1, y1, x2, y2]}}]}}
"""


def build_repair_prompt(
    entities: list[dict[str, Any]], violations: list[Violation]
) -> str:
    """Assemble the repair message: the model's own answer + numbered,
    evidence-citing instructions. Nothing else changes between rounds, so
    any behavior shift is attributable to the cited violations."""
    slim = [
        {k: e.get(k) for k in ("label", "state", "description", "bbox") if k in e
         or k in ("label", "state")}
        for e in entities
    ]
    problems = "\n".join(f"{n}. {v.instruction}" for n, v in enumerate(violations, 1))
    return REPAIR_PROMPT_TEMPLATE.format(
        entities_json=json.dumps({"entities": slim}, indent=2),
        problem_lines=problems,
    )


# ── Step 3: the loop itself ─────────────────────────────────────────────

# A query function takes (repair_prompt_text) and returns the model's new
# entity list. Injected so tests run without a VLM and the live path reuses
# the perception module's Ollama call.
QueryFn = Callable[[str], list[dict[str, Any]]]


def repair_entities(
    entities: list[dict[str, Any]],
    query_fn: QueryFn,
    max_rounds: int = MAX_REPAIR_ROUNDS,
    caption: str = "",
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], RepairTrace]:
    """Run Loop 1. Returns (final entities, full trace).

    Stopping rules, in order:
      1. clean       - no violations detected
      2. no_change   - the model returned the same list twice (oscillation
                       guard; asking a third time would only add confidence,
                       not information)
      3. cap_reached - hard cap, non-termination made impossible
    Violations still present at stop are NOT hidden: they stay flagged in
    the perception record (family_name_as_label, vocab_extension, unknown
    state), so an unrepaired run is visibly unrepaired.
    """
    # Observability: every notable moment is emitted as a structured event
    # (Stage 23 in its embryonic form; the live UI is the first consumer).
    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"type": event_type, **data})

    trace = RepairTrace()
    current = [dict(e) for e in entities]

    violations = detect_violations(current, caption)
    for v in violations:
        emit("violation_found", round=1, **v.model_dump())
    if not violations:
        trace.clean_on_arrival = True
        trace.stopped_reason = "clean"
        emit("repair_stopped", reason="clean", rounds=0, remaining=[])
        return current, trace

    seen_sets: list[frozenset] = []
    round_number = 0
    while round_number < max_rounds:
        round_number += 1
        emit("repair_round_started", round=round_number,
             open_violations=len(violations))
        prompt = build_repair_prompt(current, violations)
        answer = query_fn(prompt)

        # Defensive: a broken answer (empty, wrong type) ends the loop with
        # the last good list rather than destroying it.
        if not isinstance(answer, list) or not answer:
            trace.rounds.append(RepairRound(
                round_number=round_number, violations=violations,
                entities_after=current, changed=False,
            ))
            trace.stopped_reason = "no_change"
            emit("repair_stopped", reason="no_change", rounds=round_number,
                 remaining=[v.model_dump() for v in violations])
            return current, trace

        changed = json.dumps(answer, sort_keys=True) != json.dumps(
            current, sort_keys=True
        )
        current = [dict(e) for e in answer]
        remaining = detect_violations(current, caption)
        trace.rounds.append(RepairRound(
            round_number=round_number, violations=violations,
            entities_after=current, changed=changed,
        ))
        emit("repair_round_done", round=round_number, changed=changed,
             remaining_violations=len(remaining),
             entities=[dict(e) for e in current])
        for v in remaining:
            emit("violation_found", round=round_number + 1, **v.model_dump())

        if not remaining:
            trace.stopped_reason = "clean"
            emit("repair_stopped", reason="clean", rounds=round_number,
                 remaining=[])
            return current, trace
        if not changed:
            trace.stopped_reason = "no_change"
            emit("repair_stopped", reason="no_change", rounds=round_number,
                 remaining=[v.model_dump() for v in remaining])
            return current, trace

        # PROGRESS, not circling: the violation set changed, so the model is
        # repairing — extend the budget rather than stopping mid-repair with a
        # known-bad record. A repeat of any set already seen is circling, and
        # the ordinary cap applies.
        sig = frozenset((v.kind, v.entity_index) for v in remaining)
        prev = frozenset((v.kind, v.entity_index) for v in violations)
        if (sig != prev and sig not in seen_sets
                and round_number >= max_rounds
                and max_rounds < MAX_REPAIR_ROUNDS_PROGRESSING):
            max_rounds = min(max_rounds + 1, MAX_REPAIR_ROUNDS_PROGRESSING)
            emit("repair_budget_extended", round=round_number,
                 reason="violation set changed — repairing, not circling",
                 now=[v.kind for v in remaining], cap=max_rounds)
        seen_sets.append(prev)
        violations = remaining

    trace.stopped_reason = "cap_reached"
    emit("repair_stopped", reason="cap_reached", rounds=max_rounds,
         remaining=[v.model_dump() for v in violations])
    return current, trace
