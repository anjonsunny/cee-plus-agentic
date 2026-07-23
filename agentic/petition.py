"""Increment 3 — contextual re-perception: Stage 2 petitions Stage 1.

THE LEDGERED IMAGE TRANSACTION
==============================
The one sanctioned way the pipeline looks at the image again. Sunny's
rule, verbatim intent: anyone may propose an update to the shared truth;
no one may hold a side-truth. A petition is that proposal — triggered by
evidence, capped at ONE per run, every step an event.

WHAT THE PETITION PROMPT DOES AND DOES NOT SAY (Sunny 2026-07-22)
=================================================================
It does NOT say what to look for. Telling perception "find the pool"
would be us planting the answer downstream wants — the rationalization
attack this design exists to measure, committed by the designers. The
prompt carries only:
  1. the ORIGINAL perception task (same prompt, same rules),
  2. the model's PREVIOUS ANSWER (its own entity list),
  3. WHY we are re-perceiving: the unresolved downstream problems,
     quoted as evidence, with the problem LOCATED in the previous
     answer where possible,
  4. permission to return the list UNCHANGED if it stands by it.
Fresh eyes, honest reason, no steering.

GT-FREE TRIGGERS (must work in production — no GT exists there)
===============================================================
Fire ONLY when reflection ended un-clean (no_change / cap_reached) AND a
surviving violation has a shape that implies the entity list itself may
be wrong:
  - threat_state_not_hazardous surviving: the assessor keeps citing
    non-hazard entities as threats -> it may believe a hazard exists
    that the list lacks (F3: B_pool kept drafting children).
  - false_alarm_incoherence surviving: the assessor insists Yes with
    zero declared danger states -> same implication from the other side.
  - Loop 1 caption ticket that STOOD: the caption names an entity
    nobody found (checkable from the frozen repair_trace).
On the six calibration scenes, GT then grades the trigger (B_pool should
fire; the other five should not).

THE RATIONALIZATION GUARD
=========================
Petitioned entities carry provenance ("petition"), so eval can ask
whether petitioned entities are disproportionately verdict-serving. The
re-perception runs through the FULL standard machinery (canonicalize ->
DINO bind -> Loop 1): an entity the detector cannot ground does not
merge — a failed petition is recorded pathology signal, never silently
retried. Success criterion, measurable without GT: did the violation
pressure vanish on its own in the re-assessment?
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PETITION_CAP = 1        # per run, hard

# Violation kinds whose survival implies the ENTITY LIST may be wrong
# (vs. kinds that indict the verdict itself, which reflection owns).
_PETITIONABLE = {
    "threat_state_not_hazardous":
        "the assessment kept citing non-hazard entities as threats — it "
        "may believe a source of harm exists that the entity list does "
        "not declare",
    "false_alarm_incoherence":
        "the assessment insists this is a disaster while the entity list "
        "declares no danger states at all",
    # S8 surviving reflection: the model keeps a threat it describes as a
    # victim — the contradiction indicts the entity's declared state
    # upstream (A_fire road·burning, 2026-07-22), so the evidence points
    # at the perception artifact, not the verdict.
    "threat_reason_victim_shaped":
        "the assessment keeps citing a threat whose own reason describes "
        "it as endangered — the entity's declared hazardous state may be "
        "a perception error",
}


class Petition(BaseModel):
    reasons: list[dict[str, Any]] = Field(default_factory=list)
    # each: {kind, evidence, implication, locates: object_id|None}
    # WHERE the complaint goes (routing, Sunny 2026-07-22 — E_collapse
    # ui_e45e9956 wasted an image look on a judgment error):
    #   "stage1" — the entity list may be wrong -> re-look at the IMAGE
    #   "stage2" — the facts are fine, the SORTING is wrong -> re-ask
    #              the assessment question once, fresh
    target: str = "stage1"


def detect_petition(reflection_trace: Optional[dict[str, Any]],
                    violations: list[dict[str, Any]],
                    record: Any) -> Optional[Petition]:
    """GT-free trigger. Returns a Petition or None.

    Routing rule (repair authority follows the evidence): a surviving
    violation indicts STAGE 1 only when the entity list could plausibly
    be the culprit — the caption names something nobody found, or the
    verdict wants a danger source and the record declares NONE. When the
    record already holds legal hazards and the states are fine, the
    mistake is the SORTING (which list each thing belongs in) — that is
    a stage-2 error, and re-looking at the image cannot fix it."""
    reasons: list[dict[str, Any]] = []
    has_hazard = any(o.state_kind == "hazard_bearing"
                     for o in record.detected_objects)
    stopped = (reflection_trace or {}).get("stopped_reason")
    if stopped in ("no_change", "cap_reached"):
        for v in violations:
            kind = v.get("kind")
            if kind in _PETITIONABLE:
                # locate the problem inside the previous answer if the
                # evidence names an entity id
                locates = next((o.object_id for o in record.detected_objects
                                if o.object_id in str(v.get("evidence", ""))),
                               None)
                reasons.append({"kind": kind,
                                "evidence": v.get("evidence", ""),
                                "implication": _PETITIONABLE[kind],
                                "locates": locates,
                                # no legal hazard anywhere -> the list
                                # may be missing the source (stage 1);
                                # hazards exist -> sorting error (stage 2)
                                "target": ("stage1" if not has_hazard
                                           else "stage2")})
    # Loop 1 caption ticket that stood its ground: the caption names an
    # entity nobody found — a perception-shaped disagreement by itself.
    trace = getattr(record, "repair_trace", None) or {}
    if trace.get("stopped_reason") in ("no_change", "cap_reached"):
        for rnd in trace.get("rounds", [])[-1:]:
            for v in rnd.get("violations", []):
                if v.get("kind") == "caption_entity_missing":
                    reasons.append({
                        "kind": "caption_entity_missing",
                        "evidence": f"caption names '{v.get('raw_label')}' "
                                    f"but no entity was declared for it",
                        "implication": "the given caption and the entity "
                                       "list disagree about what the "
                                       "scene contains",
                        "locates": None,
                        "target": "stage1"})
    if not reasons:
        return None
    # One image look covers everything; so if ANY reason points at the
    # entity list, the whole petition goes to stage 1.
    target = ("stage1" if any(r["target"] == "stage1" for r in reasons)
              else "stage2")
    return Petition(reasons=reasons, target=target)


def compose_petition(record: Any, petition: Petition) -> str:
    """The re-perception prompt: original task + previous answer with the
    problems LOCATED + the reason — and nothing about what to find."""
    from agentic.perception import build_perception_prompt
    base = build_perception_prompt()
    if record.caption:
        base += f"\n\nCaption:\n{record.caption}"

    located = {r["locates"] for r in petition.reasons if r.get("locates")}
    prev_lines = []
    for o in record.detected_objects:
        mark = "   <-- a downstream problem involves this entity" \
            if o.object_id in located else ""
        prev_lines.append(f"  - {o.object_id}: {o.label} · state={o.state} "
                          f"· bbox={o.bbox}{mark}")

    reason_lines = [f"  - {r['kind']}: {r['evidence']}\n"
                    f"    implication: {r['implication']}"
                    for r in petition.reasons]

    return (base
            + "\n\nYOU HAVE ANALYZED THIS IMAGE BEFORE. "
              "YOUR PREVIOUS ANSWER:\n" + "\n".join(prev_lines)
            + "\n\nWHY YOU ARE BEING ASKED TO LOOK AGAIN:\n"
              "The downstream scene assessment could not resolve these "
              "problems, and they suggest the entity list itself may be "
              "incomplete or mis-describe something:\n"
            + "\n".join(reason_lines)
            + "\n\nRe-examine the image with fresh eyes and return the "
              "FULL entity list in the same JSON format. Correct or "
              "extend it ONLY where the image itself warrants — do not "
              "add anything you cannot see. If, after looking again, you "
              "stand by your previous answer, return it unchanged.")


# ── Running the petition through the standard machinery ─────────────────

PerceiveFn = Callable[..., Any]     # (image_path, caption, entities) -> PerceptionResult


def diff_records(old: Any, new: Any) -> dict[str, list[str]]:
    """What the petition changed, by object identity (label+state pairs
    are the stable currency; ids can renumber across runs)."""
    def keyset(rec):
        return {(o.label, o.state) for o in rec.detected_objects}

    old_k, new_k = keyset(old), keyset(new)
    return {
        "added": sorted(f"{l}·{s}" for l, s in new_k - old_k),
        "removed": sorted(f"{l}·{s}" for l, s in old_k - new_k),
    }


def mark_provenance(old: Any, new: Any) -> None:
    """Tag petitioned entities in place: anything whose (label, state) was
    not in the pre-petition record carries provenance='petition' — the
    rationalization guard's handle."""
    old_k = {(o.label, o.state) for o in old.detected_objects}
    for o in new.detected_objects:
        if (o.label, o.state) not in old_k:
            o.provenance = "petition"


def run_petition(image_path: str, record: Any, petition: Petition,
                 on_event: Any = None,
                 perceive_fn: PerceiveFn | None = None,
                 query_fn: Any = None):
    """Execute ONE petition: re-ask the perception VLM with the petition
    prompt, then run the answer through the FULL standard machinery
    (canonicalize -> ground -> bind -> Loop 1) via run_perception's
    stand-in mode. Returns the new PerceptionResult, or None on failure
    (the run then proceeds on the original record — a failed petition is
    recorded, never fatal)."""
    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"type": event_type, **data})

    emit("petition_started", target="stage1",
         reasons=[{k: r.get(k) for k in ("kind", "evidence", "locates")}
                  for r in petition.reasons])
    try:
        prompt = compose_petition(record, petition)
        if perceive_fn is None:
            import base64
            import mimetypes

            from agentic.perception import _query_vlm_raw, run_perception
            mime = (mimetypes.guess_type(str(image_path))[0]
                    or "image/jpeg")
            data_url = (f"data:{mime};base64,"
                        + base64.b64encode(
                            Path(image_path).read_bytes()).decode())
            raw_fn = query_fn or _query_vlm_raw
            entities = raw_fn(prompt, data_url)
            new_record = run_perception(image_path, caption=record.caption,
                                        entities=entities,
                                        on_event=on_event)
        else:
            new_record = perceive_fn(image_path, record.caption, prompt)
        if not new_record or not new_record.detected_objects:
            emit("petition_failed", error="re-perception returned no "
                                          "entities")
            return None
        mark_provenance(record, new_record)
        # THE TWO-WITNESS RULE for additions: a petitioned entity exists
        # BECAUSE downstream wants it to — self-vouching (the claimant's
        # own fallback box) cannot admit it; only an independent
        # detector match (dino_matched) merges.
        additions = [o for o in new_record.detected_objects
                     if o.provenance == "petition"
                     and o.box_source == "dino_matched"]
        rejected = [o for o in new_record.detected_objects
                    if o.provenance == "petition"
                    and o.box_source != "dino_matched"]
        # THE NO-ERASURE RULE (B_pool live run 2026-07-22: the second
        # look DROPPED child_2, a drowning victim, and the old wholesale
        # merge accepted the deletion): the petition's mandate is "maybe
        # something is missing" — it has no authority to erase the
        # shared truth. Originals are PRESERVED; entities the second
        # look omitted are recorded as DISPUTES, not deletions.
        disputes = diff_records(record, new_record)["removed"]
        if not additions:
            if rejected:
                emit("petition_failed",
                     error="no petitioned entity survived detector "
                           "grounding: "
                           + ", ".join(f"{o.label}·{o.state} "
                                       f"({o.box_source})"
                                       for o in rejected))
            else:
                emit("petition_done", added=[], removed=[], rejected=[],
                     disputed=disputes, n_petitioned=0,
                     note=("model stood by its original perception"
                           if not disputes else
                           "second look omitted entities; recorded as "
                           "dispute, nothing erased"))
            return None
        merged = record.model_copy(deep=True)
        taken = {o.object_id for o in merged.detected_objects}
        for o in additions:
            if o.object_id in taken:          # id churn across runs
                base = o.label
                n = 2
                while f"{base}_{n}" in taken:
                    n += 1
                o.object_id = f"{base}_{n}"
            taken.add(o.object_id)
            merged.detected_objects.append(o)
        if disputes:
            merged.notes.append(
                f"petition dispute: the second look omitted {disputes}; "
                f"originals preserved (petitions add, never erase)")
        emit("petition_done",
             added=[f"{o.label}·{o.state}" for o in additions],
             removed=[],
             rejected=[f"{o.label}·{o.state}" for o in rejected],
             disputed=disputes,
             n_petitioned=len(additions))
        return merged
    except Exception as exc:
        emit("petition_failed", error=str(exc)[:200])
        return None


# ── Stage-2 petition: re-ask the QUESTION, not the image ────────────────


def compose_stage2_petition(record: Any, prev_assessment: Any,
                            petition: Petition) -> str:
    """One fresh re-ask of the assessment question. Same evidence, no
    dialogue history (the iterative channel is what produced the error),
    previous answer shown, problems quoted with their rules, and the
    standing right to keep the answer."""
    import json as _json

    from agentic.assessment import build_assessment_prompt
    from agentic.rulebook import retrieve
    base = build_assessment_prompt(record)
    prev = _json.dumps(prev_assessment.model_dump(), indent=1)
    lines = []
    for r in petition.reasons:
        lines.append(f"- {r['kind']}: {r['evidence']}")
        chunk = retrieve(r["kind"])
        if chunk is not None:
            lines.append(f"  Rule {chunk.rule_id}: {chunk.rule}")
    return (base
            + "\n\nYOU HAVE ANSWERED THIS QUESTION BEFORE. "
              "YOUR PREVIOUS ANSWER:\n" + prev
            + "\n\nWHY YOU ARE BEING ASKED AGAIN:\n"
              "These problems could not be resolved:\n" + "\n".join(lines)
            + "\n\nAnswer the question again, fresh, from the evidence "
              "above. If you stand by your previous answer, return it "
              "unchanged.")


def run_stage2_petition(record: Any, result: Any, petition: Petition,
                        on_event: Any = None, query_fn: Any = None):
    """Execute ONE stage-2 petition (cap 1, one model call). Returns the
    final AssessmentResult — the fresh answer if it parsed, else the
    original. The shared record is never touched."""
    from agentic.assessment import (_query_vlm_text, enforce_kinds,
                                    internal_check, parse_assessment)
    from agentic.evals import substantive_key

    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"type": event_type, **data})

    emit("petition_started", target="stage2",
         reasons=[{k: r.get(k) for k in ("kind", "evidence", "locates")}
                  for r in petition.reasons])
    try:
        prompt = compose_stage2_petition(record, result.assessment,
                                         petition)
        qf = query_fn or (lambda p: _query_vlm_text(p, 0.0))
        raw = qf(prompt)
        a, notes = parse_assessment(raw)
        violations = enforce_kinds(a, record) + internal_check(a, record)
        changed = (substantive_key(a.model_dump())
                   != substantive_key(result.assessment.model_dump()))
        emit("assess_verdict",
             scenario=a.disaster_scenario, disaster_type=a.disaster_type,
             level=a.disaster_level, bucket=a.severity_bucket,
             self_confidence=a.self_confidence,
             threats=[t.model_dump() for t in a.threats],
             at_risk=[r.model_dump() for r in a.at_risk],
             n_violations=len(violations))
        emit("petition_done", added=[], removed=[], rejected=[],
             disputed=[], n_petitioned=0,
             note=("fresh answer CHANGED the decision" if changed else
                   "model stood by its answer"))
        before = sorted(v["kind"] for v in result.violations
                        if v["kind"] in _PETITIONABLE)
        after = sorted(v["kind"] for v in violations
                       if v["kind"] in _PETITIONABLE)
        emit("petition_outcome", resolved=(not after),
             violations_before=before, violations_after=after)
        return result.model_copy(update={
            "assessment": a,
            "violations": violations,
            "parse_notes": list(result.parse_notes) + notes
            + [f"stage2_petition:{'changed' if changed else 'stood'}"]})
    except Exception as exc:
        emit("petition_failed", error=str(exc)[:200])
        return result


# ── The Stage-2 orchestrator: assess -> maybe petition -> re-assess ─────


def assess_with_petition(image_path: str, record: Any,
                         on_event: Any = None,
                         perceive_fn: PerceiveFn | None = None,
                         **assess_kwargs):
    """The full Stage-2 story. Returns (final_record, final_result,
    petitioned: bool). Cascade discipline: a granted petition changes the
    shared record, so everything downstream of it (the whole assessment,
    reflection included) re-runs on the merged truth."""
    from agentic.assessment import run_assessment

    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"type": event_type, **data})

    result = run_assessment(record, on_event=on_event, **assess_kwargs)
    petition = detect_petition(result.reflection_trace, result.violations,
                               record)
    if petition is None:
        return record, result, False

    if petition.target == "stage2":
        # The facts are fine; the sorting is wrong. Re-ask the question
        # once — the image is never touched, the record never changes.
        result2 = run_stage2_petition(
            record, result, petition, on_event=on_event,
            query_fn=assess_kwargs.get("query_fn"))
        return record, result2, False

    new_record = run_petition(image_path, record, petition,
                              on_event=on_event, perceive_fn=perceive_fn)
    if new_record is None:
        return record, result, False

    result2 = run_assessment(new_record, on_event=on_event, **assess_kwargs)
    # Success criterion, GT-free: did the petitionable pressure vanish?
    before = sorted(v["kind"] for v in result.violations
                    if v["kind"] in _PETITIONABLE)
    after = sorted(v["kind"] for v in result2.violations
                   if v["kind"] in _PETITIONABLE)
    emit("petition_outcome", resolved=(not after),
         violations_before=before, violations_after=after)
    return new_record, result2, True
