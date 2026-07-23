"""The Stage 2 reflection loop (increment 2): evidence-triggered,
rulebook-quoted, capped, ledgered.

WHAT TRIGGERS A ROUND (never blanket — the plan's iron rule)
============================================================
1. VIOLATIONS from the code checks (S1, S4, S5, G2, G4-mismatch...):
   each is quoted back with its rulebook chunk.
2. MEMBERSHIP SPLITS above threshold: an entity that flickered across
   probe threat/at-risk lists (person_1 in 3/5) — the split itself is
   the new context, plus the entity's state and its geometry hint.
3. FIELD INSTABILITY: a disaster-scenario flip or a bucket-boundary
   straddle across probes (post type-family fold, so wording wobble
   never triggers).

THE ROUND
=========
One re-ask carrying ALL current triggers (bundled: one model call per
round, not one per problem), composed as evidence + retrieved rule +
question. The model answers with a full corrected JSON. Then: re-parse,
re-derive kinds, re-check. Stop rules, exactly Loop 1's grammar:
  clean      — no triggers remain
  no_change  — the model stood its ground (recorded, STOOD)
  cap_reached— budget spent (default 2 rounds)

U0 -> U1 (Sunny: show whether reflection reduced uncertainty)
=============================================================
The pre-reflection measurement is U0. If any round changed the answer,
the caller re-probes for U1. ΔU is NEVER presented alone — a loop can
collapse confidently onto a wrong answer — so the UI pairs it with the
verdict change and, on calibration scenes, the GT comparison.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.geometry import spatial_hints  # noqa: E402
from agentic.retrieval import retrieve_rule  # noqa: E402


def retrieve(kind: str):
    """Rule lookup, routed through the retrieval switch (AGENTIC_RETRIEVAL).
    Default mode 'rulebook' is byte-identical to the old exact-key lookup,
    so the pipeline (and the LangGraph equivalence) is unchanged; 'rag' /
    'both' opt into the RAG shadow. See agentic/retrieval.py."""
    return retrieve_rule(kind)[0]

REFLECT_CAP = int(os.getenv("ASSESS_REFLECT_CAP", "2"))
U_THRESHOLD = float(os.getenv("ASSESS_U_THRESHOLD", "0.2"))


class ReflectionRound(BaseModel):
    round_number: int
    triggers: list[dict[str, Any]]
    instruction: str                 # the full composed reflection prompt
    changed: bool = False
    violations_after: list[dict[str, Any]] = Field(default_factory=list)


class ReflectionTrace(BaseModel):
    rounds: list[ReflectionRound] = Field(default_factory=list)
    stopped_reason: Optional[str] = None    # clean|no_change|cap_reached
    u_before: Optional[float] = None
    u_after: Optional[float] = None


# ── Trigger collection ──────────────────────────────────────────────────


def weak_reason_triggers(assessment: Any,
                         record: Any) -> list[dict[str, Any]]:
    """Deterministic reason-quality check (Sunny, B_pool run: 'the
    reasoning sounds not strong' — the talking-points judge must FEED
    reflection, not just measure). A reason that does not cite the
    entity's DECLARED state word is flagged, with the expectation
    spelled out. The LLM rubric (R2, mechanism quality) stays on-demand;
    this is the free, always-on floor."""
    if assessment is None or record is None:
        return []
    states = {o.object_id: o.state for o in record.detected_objects}
    out: list[dict[str, Any]] = []
    for list_name, entries in (("threats", assessment.threats),
                               ("at_risk", assessment.at_risk)):
        for e in entries:
            state = states.get(e.object_id, "")
            root = state.rstrip("ing").rstrip("ed") if state else ""
            if state and root and root.lower() not in                     str(e.reason or "").lower():
                out.append({"type": "weak_reason", "list": list_name,
                            "object_id": e.object_id, "state": state,
                            "reason": str(e.reason or "")[:120]})
    return out


def collect_triggers(violations: list[dict[str, Any]],
                     mu: Any,
                     assessment: Any = None,
                     record: Any = None) -> list[dict[str, Any]]:
    """Everything reflection-worthy right now. Violations always count;
    probe-dispersion triggers only above U_THRESHOLD (and type wording
    wobble can't reach here — agreement is measured post-fold)."""
    triggers: list[dict[str, Any]] = [
        {"type": "violation", **v} for v in violations]
    if mu is None:
        return triggers + weak_reason_triggers(assessment, record)
    gran = getattr(mu, "granular", None) or {}
    for list_name in ("threats", "at_risk"):
        for oid, g in (gran.get(list_name) or {}).items():
            if g.get("u", 0) > U_THRESHOLD:
                triggers.append({"type": "membership_split",
                                 "list": list_name, "object_id": oid,
                                 "votes": g.get("votes", "?")})
    for d in getattr(mu, "drivers", []) or []:
        if d.kind in ("scenario_flip", "bucket_split"):
            triggers.append({"type": "field_instability",
                             "driver": d.kind, "evidence": d.evidence})
    runoff = getattr(mu, "runoff", None)
    if runoff and runoff.get("winner") in ("top1", "top2"):
        triggers.append({"type": "candidate_runoff", **runoff})
    triggers += weak_reason_triggers(assessment, record)
    return triggers


# ── Prompt composition: evidence + retrieved rule + one question ────────


def _rule_citation(kind: str, evidence: str) -> str:
    chunk = retrieve(kind)
    if chunk is None:
        return f"- PROBLEM: {evidence}"
    return (f"- PROBLEM: {evidence}\n"
            f"  RULE {chunk.rule_id}: {chunk.rule}\n"
            f"  WHY: {chunk.rationale}\n"
            f"  EXAMPLE: {chunk.example}")


def compose_reflection(assessment: Any, record: Any,
                       triggers: list[dict[str, Any]]) -> str:
    """One bundled reflection prompt. Context policy (agreed 2026-07-22):
    entity STATES always; geometry as derived relations in words; raw
    boxes only for entities under a spatial question."""
    by_id = {o.object_id: o for o in record.detected_objects}
    hints = {(h["other"]): h for h in
             spatial_hints(record.detected_objects, record.image_size)}
    parts: list[str] = []
    for t in triggers:
        if t["type"] == "violation":
            parts.append(_rule_citation(t.get("kind", ""),
                                        t.get("evidence", "")))
        elif t["type"] == "membership_split":
            oid = t["object_id"]
            o = by_id.get(oid)
            state = f"{o.state} ({o.state_kind})" if o else "unknown"
            h = hints.get(oid)
            geo = (f" Geometry: {oid} {h['relation']} {h['hazard']} "
                   f"(gap ≈ {h['gap_px']:.0f}px; raw boxes "
                   f"{h['boxes']['other']} vs {h['boxes']['hazard']})."
                   if h else " Geometry: no hazard nearby by declared "
                            "boxes.")
            g1 = retrieve("geometry_adjacency")
            g3 = retrieve("geometry_is_a_hint")
            parts.append(
                f"- UNSTABLE MEMBERSHIP: {oid} appeared in only "
                f"{t['votes']} probe {t['list']} lists. Its declared "
                f"state: {state}.{geo}\n"
                f"  RULE {g1.rule_id}: {g1.rule}\n"
                f"  RULE {g3.rule_id}: {g3.rule}\n"
                f"  DECIDE: does {oid} belong in {t['list']}? Cite the "
                f"state and the spatial evidence either way.")
        elif t["type"] == "field_instability":
            # CAPITULATION GUARD (B_pool 2026-07-22: told "your scenario
            # is unstable", the model resolved the doubt by folding to
            # No/0 on a drowning scene). Never present the model's
            # reliability as the problem; present the EVIDENCE and ask
            # which verdict it supports. Doubt phrased as doubt invites
            # capitulation; doubt phrased as evidence invites reasoning.
            danger = [f"{o.object_id}·{o.state} ({o.state_kind})"
                      for o in record.detected_objects
                      if o.state_kind in ("hazard_bearing", "at_risk")]
            s1 = retrieve("scenario_level_incoherent")
            parts.append(
                f"- VERIFY ({t['driver']}): re-derive this field from the "
                f"evidence below, independently of any previous answer.\n"
                f"  DECLARED DANGER STATES: "
                f"{', '.join(danger) or '(none)'}\n"
                f"  RULE {s1.rule_id}: {s1.rule}\n"
                f"  A scene whose declared states include active hazards "
                f"or victims cannot be 'No'; a scene with neither cannot "
                f"be 'Yes'. State which specific entity states your "
                f"verdict and severity rest on.")
        elif t["type"] == "weak_reason":
            parts.append(
                f"- WEAK REASON: your {t['list']} entry for "
                f"{t['object_id']} says: \"{t['reason']}\" — it does not "
                f"cite the entity's DECLARED state "
                f"('{t['state']}').\n"
                f"  EXPECTED: a strong reason (a) names the declared "
                f"state, and (b) states the specific causal mechanism — "
                f"what harms what, or why this entity is in danger, in "
                f"this scene. Generic phrases ('poses a significant "
                f"risk') are not evidence.\n"
                f"  Rewrite the reason with cited evidence — or "
                f"reconsider whether the entry belongs at all.")
        elif t["type"] == "candidate_runoff":
            win_key = t.get("winner")
            win = t.get(win_key) or {}
            lose = t.get("top2" if win_key == "top1" else "top1") or {}
            votes = t.get(f"{win_key}_votes", "?")
            parts.append(
                f"- RUNOFF VERDICT: under resampling your answers formed "
                f"two main readings. An independent judge compared them "
                f"against the declared states (blind) and preferred the "
                f"one with {votes} of your own votes:\n"
                f"  PREFERRED: {win.get('disaster_scenario')} · "
                f"{win.get('disaster_type')} · level "
                f"{win.get('disaster_level')} | threats "
                f"{[x.get('object_id') for x in win.get('threats') or []]}\n"
                f"  OTHER:     {lose.get('disaster_scenario')} · "
                f"{lose.get('disaster_type')} · level "
                f"{lose.get('disaster_level')}\n"
                f"  JUDGE'S REASON: {t.get('raw', '')}\n"
                f"  This is advice, not an order: decide from the declared "
                f"states, and cite them.")
    previous = assessment.model_dump()
    return (
        "You previously assessed this scene. Specific problems were found "
        "with your answer, each cited below with the rule it concerns and "
        "the evidence. Reconsider ONLY what the evidence demands; do not "
        "change what was not questioned.\n\n"
        "YOUR PREVIOUS ANSWER:\n"
        f"{json.dumps(previous, indent=1)}\n\n"
        "PROBLEMS:\n" + "\n".join(parts) + "\n\n"
        "Reply with the SAME JSON schema as before (reasoning, "
        "disaster_scenario, disaster_type, disaster_level, threats, "
        "at_risk, confidence), fully corrected. If you believe your "
        "original answer was right, return it unchanged and say why in "
        "'reasoning'.")


# ── The loop ────────────────────────────────────────────────────────────


def run_reflection(record: Any, assessment: Any,
                   violations: list[dict[str, Any]], mu: Any,
                   query_fn: Callable[[str], dict[str, Any]],
                   emit: Any = None,
                   cap: int = REFLECT_CAP):
    """Run capped reflection rounds. Returns (assessment, violations,
    parse_notes_accum, trace). Anti-rumination: fixed cap; a model that
    stands its ground ends the loop (STOOD is evidence, not failure);
    oscillation is impossible because an unchanged answer stops the
    loop and a changed answer must survive re-checking."""
    from agentic.assessment import enforce_kinds, internal_check, parse_assessment

    def _emit(event_type: str, **data: Any) -> None:
        if emit is not None:
            emit(event_type, **data)

    trace = ReflectionTrace(u_before=getattr(mu, "score", None))
    notes_accum: list[str] = []
    for rnd in range(1, cap + 1):
        # Round 1 carries everything: violations + probe evidence + weak
        # reasons. Later rounds run on VIOLATIONS ONLY — probe triggers
        # are stale without re-probing, and weak reasons are style
        # advice: repeating them would be rumination, not repair.
        triggers = collect_triggers(
            violations, mu if rnd == 1 else None,
            assessment=assessment if rnd == 1 else None,
            record=record if rnd == 1 else None)
        if not triggers:
            trace.stopped_reason = "clean"
            break
        instruction = compose_reflection(assessment, record, triggers)
        _emit("reflect_round_started", round=rnd,
              triggers=triggers,                # full evidence: UI shows it
              instruction=instruction,          # the rulebook's actual words
              n_triggers=len(triggers))
        try:
            raw = query_fn(instruction)
        except Exception as exc:        # a dead model must not kill the run
            _emit("reflect_error", round=rnd, error=str(exc))
            trace.stopped_reason = "model_error"
            trace.rounds.append(ReflectionRound(
                round_number=rnd, triggers=triggers,
                instruction=instruction, changed=False,
                violations_after=violations))
            break
        new_assessment, notes = parse_assessment(raw)
        notes_accum += [f"reflect_r{rnd}:{n}" for n in notes]
        new_violations = enforce_kinds(new_assessment, record)
        new_violations += internal_check(new_assessment, record)
        changed = new_assessment.model_dump() != assessment.model_dump()
        trace.rounds.append(ReflectionRound(
            round_number=rnd, triggers=triggers, instruction=instruction,
            changed=changed, violations_after=new_violations))
        _emit("reflect_round_done", round=rnd, changed=changed,
              violations_after=len(new_violations),
              violations_after_kinds=[v["kind"] for v in new_violations])
        if not changed:
            trace.stopped_reason = "no_change"      # STOOD its ground
            break
        assessment, violations = new_assessment, new_violations
    else:
        trace.stopped_reason = "cap_reached"
    if trace.stopped_reason is None:                # loop body broke early
        trace.stopped_reason = trace.stopped_reason or "clean"
    return assessment, violations, notes_accum, trace
