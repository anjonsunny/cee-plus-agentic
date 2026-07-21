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

from agentic.vocabulary import (  # noqa: E402
    LABEL_FAMILIES,
    OTHER_LABEL,
    PURE_FAMILY_NAMES,
    canonicalize_label,
)

# The loop never runs more than this many repair calls per scene.
MAX_REPAIR_ROUNDS = 2

# States the repair prompt may remind the model of. This is vocabulary
# DISCIPLINE (which words are legal), not perception guidance (which word
# is true for this entity). The model still chooses.
_STATE_WORD_REMINDER = (
    "hazard-bearing: burning, burnt, collapsed, collapsing, fallen, crushed, "
    "flooded, leaking, spreading, billowing, rising, seeping, engulfing / "
    "at-risk: injured, bleeding, fleeing, trapped, cowering, drowning, "
    "suffocating, unconscious / normal: intact, standing, upright, dry, "
    "sealed, stationary, resting, healthy, stable, swimming, walking, running"
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


# ── Step 1: detect violations (pure code, no model call) ────────────────


def detect_violations(entities: list[dict[str, Any]]) -> list[Violation]:
    """Run the interface rules over a raw entity list.

    Works on the model's RAW answers (before canonicalization rewrites
    them), because the repair prompt must quote what the model actually
    wrote. Each check mirrors a flag the perception contract records.
    """
    from agentic.perception import state_kind  # local import: avoids cycle

    violations: list[Violation] = []
    for i, e in enumerate(entities):
        raw_label = str(e.get("label", "")).strip()
        canon, note, in_vocab, is_family = canonicalize_label(raw_label)

        # Rule 1: family names are not labels. Cite the family's members so
        # the fix is a choice, not a guess.
        if is_family:
            members = ", ".join(LABEL_FAMILIES[note.split(":", 1)[1]])
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="family_name_as_label",
                instruction=(
                    f"Entity {i}: '{raw_label}' is a family name, not a label. "
                    f"Replace it with the specific noun that fits: {members}. "
                    f"If none fits, use 'other'."
                ),
            ))

        # Rule 2: label outside vocabulary and synonym map. The model may
        # legitimately keep 'other' (the escape hatch), but it must do so
        # deliberately, not by accident.
        elif not in_vocab and canon == OTHER_LABEL and raw_label.lower() != OTHER_LABEL:
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="label_out_of_vocab",
                instruction=(
                    f"Entity {i}: '{raw_label}' is not in the allowed labels. "
                    f"Pick the closest allowed label, or keep it as 'other' "
                    f"with a clear description if nothing fits."
                ),
            ))

        # Rule 3: state word outside the state vocabulary (after synonyms
        # and extensions). We list the legal words; we NEVER suggest which
        # one is true for this entity - that would be coaching perception.
        if state_kind(e.get("state", "")) == "unknown":
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="state_out_of_vocab",
                instruction=(
                    f"Entity {i} ('{raw_label}'): state "
                    f"'{e.get('state', '')}' is not a vocabulary state. "
                    f"Choose the single word that matches what you see: "
                    f"{_STATE_WORD_REMINDER}."
                ),
            ))

        # Rule 4: every entity needs a rough anchor box; without one the
        # detector cannot be told WHICH instance the model meant.
        if not e.get("bbox") and not e.get("anchor_bbox"):
            violations.append(Violation(
                entity_index=i, raw_label=raw_label,
                kind="missing_anchor_bbox",
                instruction=(
                    f"Entity {i} ('{raw_label}'): missing bbox. Add a rough "
                    f"pixel box [x1, y1, x2, y2] around it. Approximate is fine."
                ),
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
    trace = RepairTrace()
    current = [dict(e) for e in entities]

    violations = detect_violations(current)
    if not violations:
        trace.clean_on_arrival = True
        trace.stopped_reason = "clean"
        return current, trace

    for round_number in range(1, max_rounds + 1):
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
            return current, trace

        changed = json.dumps(answer, sort_keys=True) != json.dumps(
            current, sort_keys=True
        )
        current = [dict(e) for e in answer]
        remaining = detect_violations(current)
        trace.rounds.append(RepairRound(
            round_number=round_number, violations=violations,
            entities_after=current, changed=changed,
        ))

        if not remaining:
            trace.stopped_reason = "clean"
            return current, trace
        if not changed:
            trace.stopped_reason = "no_change"
            return current, trace
        violations = remaining

    trace.stopped_reason = "cap_reached"
    return current, trace
