"""Stage 2 — MERGED scene assessment (old Stages 2+3 in one judgment).

Merged on Sunny's call (2026-07-22): severity depends on who is in
danger, so verdict, threats, and at-risk are ONE judgment over the same
declared state. The output keeps threats and at_risk as separate lists,
so hazard recall and at-risk recall still evaluate separately (push_06
lesson: 0/1 + 2/2 combined reads 67% and hides the zero).

WHAT THIS STAGE PRODUCES
========================
  - disaster_scenario : "Yes" | "No"
  - disaster_type     : short string ("hazmat fire", "N/A", ...)
  - disaster_level    : int 0-10; severity_bucket derived
                        (none / minor / serious / catastrophic)
  - threats           : [{object_id, reason}] — sources of harm
  - at_risk           : [{object_id, kind: distress|proximity, reason}]
Spatial hints (agentic/geometry.py — free, deterministic bbox math)
feed the prompt so proximity judgments have evidence; geometry
NOMINATES, the model decides (rule G3).

TEXT-ONLY, BY DESIGN (plan §3.5)
================================
This node NEVER receives the image. It reads caption + detected_objects
(labels, states, state kinds). If it could re-read the image it could
quietly hold beliefs about entities never declared in the shared record —
the exact grounding leak the project measures. Revisiting the image is
allowed only as a future LEDGERED transaction (contextual re-perception /
petition_perception, decided after the local loop lands). Sunny's rule:
anyone may propose an update to the shared truth; no one may hold a
side-truth.

INTERNAL CHECK (increment 1: detect + record; repair loop is increment 2)
=========================================================================
scenario No with level > 0, or Yes with level 0 — a self-contained
contradiction, checkable the moment this stage emits. Violations are
RECORDED on the result, not silently fixed: in Arm B every correction
must be a visible, capped loop round (built next), never a quiet patch.

MODEL
=====
The SUBJECT VLM (qwen2.5vl via Ollama), called text-only. This stage's
judgment is part of what the experiment measures, so it must come from
the model under evaluation — not from the dialogue model.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.perception import PerceptionResult  # noqa: E402
from agentic.uncertainty import (  # noqa: E402
    MeasuredUncertainty,
    _ollama_explain,
    explain,
    measure_merged,
)

# ── Contract ────────────────────────────────────────────────────────────

BUCKETS = ((0, 0, "none"), (1, 3, "minor"), (4, 6, "serious"),
           (7, 10, "catastrophic"))


def severity_bucket(level: int) -> str:
    """Deterministic 0-10 -> ordinal bucket map. The bucket, not the raw
    number, is what eval compares and what Stage 12 reads as
    escalation/de-escalation (a serious->minor move is legible; a 6->5
    wobble is noise)."""
    for lo, hi, name in BUCKETS:
        if lo <= level <= hi:
            return name
    raise ValueError(f"level out of range after clamping: {level}")


class ThreatEntry(BaseModel):
    """One entity that is a SOURCE of harm, with the model's reason."""
    object_id: str
    reason: str = ""


class AtRiskEntry(BaseModel):
    """One entity in danger. kind: 'distress' (its own state says so) or
    'proximity' (normal state, but near an active hazard — geometry
    nominated, model decided)."""
    object_id: str
    kind: str = "proximity"             # distress | proximity
    reason: str = ""


class SceneAssessment(BaseModel):
    """Validated merged-stage verdict (Stage 2+3 merged, Sunny 2026-07-22:
    severity depends on who is in danger — judging them apart put a wall
    between two halves of one judgment). Threats and at_risk stay
    SEPARATE lists in the output so hazard recall and at-risk recall are
    still evaluated separately (the push_06 lesson: 0/1 and 2/2 combined
    would read 67% and hide the zero)."""
    disaster_scenario: str              # "Yes" | "No"
    disaster_type: str
    disaster_level: int = Field(ge=0, le=10)
    severity_bucket: str
    threats: list[ThreatEntry] = Field(default_factory=list)
    at_risk: list[AtRiskEntry] = Field(default_factory=list)
    reasoning: str = ""
    # Channel 1: the model's OWN confidence claim, recorded as subject
    # data (its calibration is measured, never trusted). None = the model
    # did not provide one (itself a note-worthy fact).
    self_confidence: Optional[float] = Field(default=None, ge=0, le=1)


class AssessmentResult(BaseModel):
    """What Stage 2 appends to the shared record."""
    assessment: SceneAssessment
    parse_notes: list[str] = Field(default_factory=list)   # every coercion, logged
    violations: list[dict[str, Any]] = Field(default_factory=list)
    raw_answer: Optional[dict[str, Any]] = None            # evidence, kept verbatim
    # Channel 2: the instrument's measurement (probe dispersion) with its
    # causal drivers + narrative. None = probes not run (n_probes=0).
    # After a reflection round changes the answer, this is the RE-probe
    # (U1); U0 lives in reflection_trace.u_before.
    measured_uncertainty: Optional[MeasuredUncertainty] = None
    # The reflection ledger: every round's triggers, the composed
    # instruction, changed/stood, and the U0->U1 trajectory.
    reflection_trace: Optional[dict[str, Any]] = None


# ── Parsing the model's raw answer (guard at the boundary: Rule 1a) ─────


_YES = {"yes", "y", "true"}
_NO = {"no", "n", "false", "n/a", "none"}


def parse_assessment(raw: Any) -> tuple[SceneAssessment, list[str]]:
    """Normalize a raw model answer into the contract, LOGGING every
    coercion as a note. The model emits garbage under pressure (level
    "high", level 47, missing keys); notes make each rescue visible
    instead of silently cleaning the record."""
    notes: list[str] = []
    if not isinstance(raw, dict):
        notes.append(f"answer_not_object({type(raw).__name__})")
        raw = {}

    scen_raw = str(raw.get("disaster_scenario", "")).strip()
    low = scen_raw.lower()
    if low in _YES:
        scenario = "Yes"
    elif low in _NO:
        scenario = "No"
    else:
        scenario = "No"
        notes.append(f"scenario_unparseable({scen_raw!r})->No")

    dtype = str(raw.get("disaster_type", "") or "").strip() or "N/A"

    lvl_raw = raw.get("disaster_level", 0)
    try:
        level = int(lvl_raw)
    except (TypeError, ValueError):
        level = 0
        notes.append(f"level_unparseable({lvl_raw!r})->0")
    clamped = max(0, min(level, 10))     # same clamp semantics as main.py:3388
    if clamped != level:
        notes.append(f"level_clamped({level})->{clamped}")

    conf_raw = raw.get("confidence")
    conf: Optional[float]
    if conf_raw is None:
        conf = None
        notes.append("no_self_confidence_reported")
    else:
        try:
            conf = max(0.0, min(1.0, float(conf_raw)))
            if conf != float(conf_raw):
                notes.append(f"confidence_clamped({conf_raw})->{conf}")
        except (TypeError, ValueError):
            conf = None
            notes.append(f"confidence_unparseable({conf_raw!r})")

    threats: list[ThreatEntry] = []
    for i, t in enumerate(raw.get("threats") or []):
        if isinstance(t, dict) and t.get("object_id"):
            threats.append(ThreatEntry(object_id=str(t["object_id"]),
                                       reason=str(t.get("reason", "") or "")))
        elif isinstance(t, str) and t.strip():
            threats.append(ThreatEntry(object_id=t.strip()))
            notes.append(f"threat_{i}_bare_string({t!r})")
        else:
            notes.append(f"threat_{i}_malformed({t!r})")

    at_risk: list[AtRiskEntry] = []
    for i, r in enumerate(raw.get("at_risk") or []):
        if isinstance(r, dict) and r.get("object_id"):
            kind = str(r.get("kind", "") or "").strip().lower()
            if kind not in ("distress", "proximity"):
                notes.append(f"at_risk_{i}_kind_coerced({kind!r})->proximity")
                kind = "proximity"
            at_risk.append(AtRiskEntry(object_id=str(r["object_id"]),
                                       kind=kind,
                                       reason=str(r.get("reason", "") or "")))
        elif isinstance(r, str) and r.strip():
            at_risk.append(AtRiskEntry(object_id=r.strip()))
            notes.append(f"at_risk_{i}_bare_string({r!r})")
        else:
            notes.append(f"at_risk_{i}_malformed({r!r})")

    return SceneAssessment(
        disaster_scenario=scenario,
        disaster_type=dtype,
        disaster_level=clamped,
        severity_bucket=severity_bucket(clamped),
        threats=threats,
        at_risk=at_risk,
        reasoning=str(raw.get("reasoning", "") or ""),
        self_confidence=conf,
    ), notes


# ── Internal check (detect only; the repair loop is increment 2) ────────


# ── S8: victim-shaped reasons in the threat slot (2026-07-22) ────────────
#
# The A_fire road case (ui_34d8177e) and F5's car_1: under S6/judge
# pressure the model pays a check off by promoting an entity to threats
# while its own reason describes the entity RECEIVING harm ("could be at
# risk if the fire spreads"). Slot and story contradict — and the story
# is the model's honest belief, so the promotion (and possibly the
# upstream state) is in doubt. Direction-sensitive on purpose: "puts
# person_1 at risk" (causing) must NOT match; only copula/modal forms
# where the threat itself is the endangered subject do.

_VICTIM_SHAPED_RE = re.compile(
    r"\b(?:is|are|was|were|could\s+be|may\s+be|might\s+be|would\s+be|"
    r"appears?(?:\s+to\s+be)?|seems?(?:\s+to\s+be)?)\s+"
    r"(?:at\s+risk|in\s+danger|in\s+distress|endangered|vulnerable|"
    r"in\s+peril|threatened|in\s+harm'?s\s+way)\b",
    re.IGNORECASE)


def victim_shaped(reason: Any) -> bool:
    """True when a reason describes its entity as RECEIVING harm."""
    return bool(_VICTIM_SHAPED_RE.search(str(reason or "")))


def internal_check(a: SceneAssessment,
                   record: PerceptionResult | None = None
                   ) -> list[dict[str, Any]]:
    """Code-decidable checks on the merged verdict, run the moment it
    emits. S1 needs only the verdict; the rest compare it against the
    declared perception (the shared truth). Detect + record; the loop
    that acts on these is the next increment."""
    v: list[dict[str, Any]] = []
    # S1: the verdict cannot contradict its own severity number.
    if a.disaster_scenario == "No" and a.disaster_level > 0:
        v.append({"kind": "scenario_no_level_gt0",
                  "evidence": f"disaster_scenario=No but disaster_level="
                              f"{a.disaster_level}"})
    if a.disaster_scenario == "Yes" and a.disaster_level == 0:
        v.append({"kind": "scenario_yes_level_0",
                  "evidence": "disaster_scenario=Yes but disaster_level=0"})
    if record is None:
        return v

    by_id = {o.object_id: o for o in record.detected_objects}
    has_hazard = any(o.state_kind == "hazard_bearing"
                     for o in record.detected_objects)
    danger_states = [f"{o.object_id}·{o.state}" for o in
                     record.detected_objects
                     if o.state_kind in ("hazard_bearing", "at_risk")]
    # S2 — the push_06 catch, now IN CODE (B_pool 2026-07-22: reflection
    # capitulated to No/0 on a drowning scene and nothing fired because
    # S2 lived only in the rulebook): a No/0 verdict while the declared
    # perception carries hazard or at-risk states is a contradiction
    # with upstream evidence.
    if (a.disaster_scenario == "No" and a.disaster_level == 0
            and danger_states):
        v.append({"kind": "missed_disaster_incoherence",
                  "evidence": f"verdict is No/0 but the declared "
                              f"perception contains danger states: "
                              f"{', '.join(danger_states)}"})
    # S3 — the false-alarm direction (control-scene guard): Yes with no
    # supporting states anywhere in the declared perception.
    if a.disaster_scenario == "Yes" and not danger_states:
        v.append({"kind": "false_alarm_incoherence",
                  "evidence": "verdict is Yes but no entity carries a "
                              "hazard-bearing or at-risk state"})
    # S1 extension: 'No' means nothing to fight and nobody in danger —
    # a No verdict with populated threat/at-risk lists is incoherent.
    if a.disaster_scenario == "No" and (a.threats or a.at_risk):
        v.append({"kind": "scenario_no_with_entities",
                  "evidence": f"verdict is No but threats="
                              f"{[t.object_id for t in a.threats]} "
                              f"at_risk="
                              f"{[r.object_id for r in a.at_risk]}"})
    # Hazard coverage (flag-only, C_tanker's spill_1): a declared
    # hazard-bearing entity missing from threats. Flagged, never
    # auto-added — the model may argue an inactive hazard, but it must
    # argue, not forget.
    cited_threats = {t.object_id for t in a.threats}
    if a.disaster_scenario == "Yes":
        for o in record.detected_objects:
            if (o.state_kind == "hazard_bearing"
                    and o.object_id not in cited_threats):
                v.append({"kind": "hazard_not_in_threats",
                          "evidence": f"{o.object_id} carries "
                                      f"hazard-bearing state '{o.state}' "
                                      f"but is absent from threats"})
    # Every cited id must resolve to the declared perception.
    for e in list(a.threats) + list(a.at_risk):
        if e.object_id not in by_id:
            v.append({"kind": "id_not_in_perception",
                      "evidence": f"'{e.object_id}' is cited but not in "
                                  f"detected_objects "
                                  f"(known: {sorted(by_id)[:8]})"})
    # A threat must carry a hazard-bearing state.
    for t in a.threats:
        o = by_id.get(t.object_id)
        if o is not None and o.state_kind != "hazard_bearing":
            v.append({"kind": "threat_state_not_hazardous",
                      "evidence": f"{t.object_id} listed as threat but its "
                                  f"state '{o.state}' is "
                                  f"{o.state_kind}, not hazard_bearing"})
        # S8: the threat's own reason describes it as a victim.
        if victim_shaped(getattr(t, "reason", "")):
            v.append({"kind": "threat_reason_victim_shaped",
                      "evidence": f"{t.object_id} sits in the threat slot "
                                  f"but its reason describes RECEIVING "
                                  f"harm: \"{t.reason}\" — a source of "
                                  f"harm cannot be justified by its own "
                                  f"endangerment; either it does not "
                                  f"belong in threats, or its declared "
                                  f"state is in doubt"})
    for r in a.at_risk:
        o = by_id.get(r.object_id)
        if o is None:
            continue
        # S7: hazard-bearing entities belong in threats only — EXCEPT
        # living beings (Sunny: a burning person, a contagious person —
        # source of harm to others AND victim, simultaneously). Expected
        # dual role for person/animal; still a breach for objects (a
        # burning house is not its own victim; its neighbor is).
        if (o.state_kind == "hazard_bearing"
                and o.family not in ("person", "animal")):
            v.append({"kind": "hazard_as_at_risk",
                      "evidence": f"{r.object_id} carries hazard-bearing "
                                  f"state '{o.state}' but is listed in "
                                  f"at_risk; a source of harm belongs in "
                                  f"threats only"})
        # G2: proximity at-risk requires an active hazard to be near.
        if r.kind == "proximity" and not has_hazard:
            v.append({"kind": "proximity_without_hazard",
                      "evidence": f"{r.object_id} marked proximity at-risk "
                                  f"but no hazard-bearing entity exists "
                                  f"in the scene (G2)"})
    return v


def enforce_kinds(a: SceneAssessment,
                  record: PerceptionResult) -> list[dict[str, Any]]:
    """SCHEMA-STRICT kind, matching the baseline (main.py:26/1001): 'the
    entity's state alone tells which kind' — at-risk vocab -> distress,
    normal vocab -> proximity. The model's CLAIMED kind is subject data:
    a mismatch is recorded as a violation (it confused victim-now with
    victim-maybe — a tense error in the causal chain), but the RECORD
    carries the derived kind, so the confusion never propagates to
    Graph B or suppression. Mutates a.at_risk in place; returns the
    mismatch violations. C_tanker person_1 (claimed distress while
    standing) is the worked example that forced this."""
    v: list[dict[str, Any]] = []
    by_id = {o.object_id: o for o in record.detected_objects}
    for r in a.at_risk:
        o = by_id.get(r.object_id)
        if o is None:
            continue                    # unresolved ids are S4's business
        living = o.family in ("person", "animal")
        # Living being in a hazard state (burning person) = dual role:
        # source of harm to others AND victim of its own state -> its
        # at-risk kind is DISTRESS (harm actualized on itself).
        derived = ("distress" if (o.state_kind == "at_risk"
                                  or (o.state_kind == "hazard_bearing"
                                      and living))
                   else "proximity")
        if r.kind != derived:
            v.append({"kind": "at_risk_kind_mismatch",
                      "evidence": f"{r.object_id}: model claimed "
                                  f"kind={r.kind}, but its declared state "
                                  f"'{o.state}' ({o.state_kind}) derives "
                                  f"kind={derived} — the state alone "
                                  f"decides (G4, schema-strict)"})
            r.kind = derived            # the record carries the truth
    return v


# ── The prompt: built from STATE, never from pixels ─────────────────────

ASSESSMENT_PROMPT_TEMPLATE = """You are assessing an emergency scene. You do
NOT get the image: judge ONLY from the declared perception below (an
entity list produced by an earlier analysis stage) and the caption.

Caption: {caption}

Perceived entities ({n} total):
{entity_table}

Spatial hints (computed from declared boxes; 2D nominations only — image-
plane closeness is not proof of real closeness; you decide):
{spatial_hints}
{extras}
Answer in JSON, reasoning FIRST, then the verdict fields:
{{
  "reasoning": "<2-4 sentences: which entity states drive your verdict>",
  "disaster_scenario": "Yes" or "No",
  "disaster_type": "<short phrase, or N/A>",
  "disaster_level": <integer 0-10; 0 means no disaster at all; reserve
                     7-10 for catastrophic scenes; consider who is in
                     danger, not just what is hazardous>,
  "threats": [{{"object_id": "<id>", "reason": "<why it is a source of
               harm>"}}],
  "at_risk": [{{"object_id": "<id>", "kind": "distress" or "proximity",
               "reason": "<why it is in danger>"}}],
  "confidence": <number 0-1: how confident you are in this verdict>
}}
Rules:
- If disaster_scenario is "No", disaster_type must be "N/A",
  disaster_level must be 0, and threats and at_risk must be empty.
- A threat is an entity whose STATE is a source of harm (burning,
  leaking, spreading). Cite only listed object_ids. threats MAY be
  empty even on a "Yes" verdict: a victim in distress with no visible
  active hazard is a legitimate scene; never invent a harm source.
- at_risk kind "distress": the entity's own state says it is in danger
  (drowning, trapped). kind "proximity": normal state, but near an
  active hazard — use the spatial hints, and say which hazard.
- Do not invent entities that are not listed."""


def build_assessment_prompt(record: PerceptionResult) -> str:
    from agentic.geometry import hints_as_prompt_lines, spatial_hints
    rows = [f"  - {o.object_id}: {o.label} · state={o.state} "
            f"({o.state_kind})" for o in record.detected_objects]
    extras = ""
    if record.unlocalized:
        extras += f"Unlocalized (named but no box): {record.unlocalized}\n"
    if record.notes:
        extras += f"Perception notes: {record.notes}\n"
    hints = spatial_hints(record.detected_objects, record.image_size)
    return ASSESSMENT_PROMPT_TEMPLATE.format(
        caption=record.caption or "(none)",
        n=len(record.detected_objects),
        entity_table="\n".join(rows) or "  (no entities perceived)",
        spatial_hints=hints_as_prompt_lines(hints),
        extras=extras,
    )


# ── The subject-VLM call, text-only (injectable for tests) ──────────────


def _query_vlm_text(prompt: str, temperature: float = 0.0) -> dict[str, Any]:
    """One TEXT-ONLY completion against the subject VLM. Same endpoint and
    model as perception — same subject — but no image in the payload.
    temperature 0 for the canonical verdict; probes pass ~0.7."""
    import requests

    from main import extract_json_block  # lazy: keeps offline import light

    api_url = os.getenv("QWEN_API_URL",
                        "http://localhost:11434/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("QWEN_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": os.getenv("QWEN_MODEL_NAME", "qwen2.5vl:7b"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(api_url, headers=headers, json=payload,
                      timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    return extract_json_block(r.json()["choices"][0]["message"]["content"])


QueryFn = Callable[[str], dict[str, Any]]

PROBE_TEMPERATURE = float(os.getenv("ASSESS_PROBE_TEMP", "0.7"))
DEFAULT_N_PROBES = int(os.getenv("ASSESS_N_PROBES", "5"))


def measure_uncertainty(prompt: str, n_probes: int,
                        probe_fn: QueryFn | None = None,
                        emit: Any = None) -> MeasuredUncertainty:
    """Channel 2: re-ask the SAME prompt n_probes times at probe
    temperature and measure verdict dispersion. The canonical verdict is
    untouched — probes describe its stability, they never replace it.
    A probe whose answer is unparseable garbage still counts: it parses
    to the conservative default, and instability from garbage IS
    instability worth measuring."""
    probe_fn = probe_fn or (lambda p: _query_vlm_text(p, PROBE_TEMPERATURE))
    verdicts = []
    full_answers = []
    for i in range(n_probes):
        try:
            a, _notes = parse_assessment(probe_fn(prompt))
        except Exception as exc:               # transport failure ≠ dispersion
            if emit:
                emit("assess_probe_error", index=i, error=str(exc))
            continue
        verdicts.append({"scenario": a.disaster_scenario,
                         "disaster_type": a.disaster_type,
                         "level": a.disaster_level,
                         "bucket": a.severity_bucket,
                         "threat_ids": [t.object_id for t in a.threats],
                         "at_risk_ids": [r.object_id for r in a.at_risk]})
        full_answers.append(a.model_dump())
        if emit:
            emit("assess_probe", index=i, scenario=a.disaster_scenario,
                 level=a.disaster_level, bucket=a.severity_bucket,
                 threat_ids=[t.object_id for t in a.threats],
                 at_risk_ids=[r.object_id for r in a.at_risk])
    mu = measure_merged(verdicts)
    from agentic.uncertainty import probe_candidates
    mu.candidates = probe_candidates(full_answers)
    if n_probes > 0 and not verdicts:
        # Every probe died in transport: nothing was measured, and an
        # unmeasured verdict must read as maximally untrustworthy, not
        # as unanimous.
        from agentic.uncertainty import Driver
        mu.score = 1.0
        mu.drivers.append(Driver(
            kind="probes_failed",
            evidence=f"all {n_probes} probe calls failed in transport",
            action="fix the model endpoint and re-run; treat this verdict "
                   "as unmeasured until then"))
    return mu


# ── The node ────────────────────────────────────────────────────────────


def run_assessment(record: PerceptionResult,
                   query_fn: QueryFn | None = None,
                   on_event: Any = None,
                   n_probes: int = 0,
                   probe_fn: QueryFn | None = None,
                   explain_fn: Any = None,
                   reflect: bool = True,
                   runoff_judge_fn: Any = None) -> AssessmentResult:
    """The merged Stage 2, full increment-2 shape:

        canonical verdict -> derive kinds (schema-strict) -> checks
        -> probes (U0, granular) -> REFLECTION LOOP (evidence-triggered,
        rulebook-quoted, capped) -> re-probe if the answer changed (U1)

    n_probes > 0 turns on channel-2 uncertainty. reflect=False gives the
    detect-only behavior (the pre-loop baseline mode)."""
    query_fn = query_fn or (lambda p: _query_vlm_text(p, 0.0))

    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"type": event_type, **data})

    # Medium-bound hazard derivation at the assessment boundary: covers
    # frozen records loaded from disk and petition-merged records alike.
    # Idempotent — a record already derived (or with no water body) is
    # untouched. See perception.derive_medium_hazards for the argument.
    from agentic.perception import derive_medium_hazards
    for dv in derive_medium_hazards(record):
        emit("hazard_derived", **dv)

    hazards = [o.object_id for o in record.detected_objects
               if o.state_kind == "hazard_bearing"]
    at_risk = [o.object_id for o in record.detected_objects
               if o.state_kind == "at_risk"]
    from agentic.geometry import spatial_hints as _sh
    _hints = _sh(record.detected_objects, record.image_size)
    def _pair_line(h: dict[str, Any]) -> str:
        if h["relation"] == "overlap":
            return (f"{h['other']} ↔ {h['hazard']} "
                    f"(overlap {h.get('overlap_frac', 0):.0%}, "
                    f"centers {h.get('center_dist_px', 0):.0f}px)")
        return (f"{h['other']} ↔ {h['hazard']} "
                f"(adjacent, gap {h['gap_px']:.0f}px)")

    pairs = [_pair_line(h) for h in _hints]
    emit("stage_started", stage="assess")
    emit("assess_context", n_entities=len(record.detected_objects),
         hazard_ids=hazards, at_risk_ids=at_risk,
         n_spatial_hints=len(_hints), spatial_pairs=pairs)

    prompt = build_assessment_prompt(record)
    raw = query_fn(prompt)
    assessment, notes = parse_assessment(raw)
    for n in notes:
        emit("assess_parse_note", note=n)

    # Schema-strict kind derivation BEFORE checks (baseline main.py:1001:
    # the state alone decides; the model's claim is measured, not obeyed).
    violations = enforce_kinds(assessment, record)
    violations += internal_check(assessment, record)
    for v in violations:
        emit("assess_violation", **v)

    def emit_verdict() -> None:
        emit("assess_verdict",
             scenario=assessment.disaster_scenario,
             disaster_type=assessment.disaster_type,
             level=assessment.disaster_level,
             bucket=assessment.severity_bucket,
             self_confidence=assessment.self_confidence,
             threats=[t.model_dump() for t in assessment.threats],
             at_risk=[r.model_dump() for r in assessment.at_risk],
             n_violations=len(violations))

    emit_verdict()

    def measure_and_explain(tag: str):
        m = measure_uncertainty(prompt, n_probes, probe_fn=probe_fn,
                                emit=emit)
        verdict_line = (f"{assessment.disaster_scenario} · "
                        f"{assessment.disaster_type} · level "
                        f"{assessment.disaster_level} "
                        f"({assessment.severity_bucket})")
        m = explain(m, verdict_line, explain_fn=explain_fn)
        emit("assess_uncertainty", phase=tag, score=m.score,
             n_probes=m.n_probes,
             scenario_agreement=m.scenario_agreement,
             type_agreement=m.type_agreement,
             bucket_agreement=m.bucket_agreement,
             granular=m.granular,
             drivers=[d.kind for d in m.drivers],
             explanation=m.explanation, explainer=m.explainer)
        return m

    mu = measure_and_explain("initial") if n_probes > 0 else None

    # ── TOP-2 RUNOFF (Sunny 2026-07-22): when U is high, the model's own
    # probe readings disagree — blind-judge the two most-voted candidates
    # against the declared states. The winner is CONTEXT for reflection
    # (via collect_triggers), never the installed answer. Best-effort: no
    # judge model -> no runoff, run proceeds.
    if mu is not None and runoff_judge_fn is not None:
        from agentic.reflection import U_THRESHOLD
        if mu.score > U_THRESHOLD and len(mu.candidates) >= 2:
            try:
                from agentic.evals import judge_runoff
                top1, top2 = mu.candidates[0], mu.candidates[1]
                verdict = judge_runoff(
                    top1["answer"], top2["answer"], record.model_dump(),
                    Path(record.image_path).stem,
                    judge_fn=runoff_judge_fn)
                def _line(c):
                    ans = c["answer"]
                    th = ",".join(t.get("object_id", "?")
                                  for t in ans.get("threats") or []) or "-"
                    return (f"{ans.get('disaster_scenario')}·"
                            f"L{ans.get('disaster_level')} threats[{th}]")
                mu.runoff = {
                    "winner": verdict.get("winner"),
                    "raw": verdict.get("raw", ""),
                    "top1_votes": f"{top1['votes']}/{mu.n_probes}",
                    "top2_votes": f"{top2['votes']}/{mu.n_probes}",
                    "top1_line": _line(top1), "top2_line": _line(top2),
                    "top1": top1["answer"], "top2": top2["answer"],
                }
                emit("assess_runoff", **{k: v for k, v in mu.runoff.items()
                                         if k not in ("top1", "top2")})
            except Exception as exc:
                emit("assess_runoff_error", error=str(exc)[:200])

    reflection_trace = None
    if reflect:
        from agentic.reflection import run_reflection
        assessment, violations, r_notes, reflection_trace = run_reflection(
            record, assessment, violations, mu, query_fn, emit=emit)
        notes += r_notes
        changed = any(r.changed for r in reflection_trace.rounds)
        if changed:
            emit_verdict()                       # the corrected verdict
            if n_probes > 0:
                mu = measure_and_explain("post_reflection")   # U1
                reflection_trace.u_after = mu.score
        emit("reflect_stopped", reason=reflection_trace.stopped_reason,
             rounds=len(reflection_trace.rounds),
             u_before=reflection_trace.u_before,
             u_after=reflection_trace.u_after)

    emit("stage_done", stage="assess")

    return AssessmentResult(
        assessment=assessment, parse_notes=notes, violations=violations,
        raw_answer=raw if isinstance(raw, dict) else None,
        measured_uncertainty=mu,
        reflection_trace=(reflection_trace.model_dump()
                         if reflection_trace else None))


# ── Runner: assess the frozen Stage 1 records ───────────────────────────

PERCEPTION_DIR = REPO_ROOT / "experiments" / "agentic_scenes" / "perception"
ASSESSMENT_DIR = REPO_ROOT / "experiments" / "agentic_scenes" / "assessment"


def main(argv: list[str] | None = None) -> None:
    """Assess every frozen perception record (or the ones named on the
    command line) and freeze the verdicts alongside them. Text-only, so
    this runs from the records with no images needed.

        python -m agentic.assessment            # all scenes, 5 probes each
        python -m agentic.assessment C_tanker_fire
        ASSESS_N_PROBES=0 python -m agentic.assessment   # skip probes
    """
    names = list(argv if argv is not None else sys.argv[1:])
    records = sorted(PERCEPTION_DIR.glob("*__perception.json"))
    if names:
        records = [r for r in records
                   if r.name.replace("__perception.json", "") in names]
    if not records:
        print(f"no perception records found in {PERCEPTION_DIR}")
        return
    ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True)
    for path in records:
        name = path.name.replace("__perception.json", "")
        record = PerceptionResult.model_validate_json(path.read_text())
        print(f"── {name}: {len(record.detected_objects)} entities")
        from agentic.evals import _ollama_judge
        from agentic.graph_live import assess_with_control
        printer = (lambda e: print(f"   {e['type']}: "
                                   f"{ {k: v for k, v in e.items() if k != 'type'} }"))
        if Path(record.image_path).exists():
            record, result, petitioned = assess_with_control(
                record.image_path, record,
                n_probes=DEFAULT_N_PROBES,
                explain_fn=_ollama_explain,
                runoff_judge_fn=_ollama_judge,
                on_event=printer)
            if petitioned:
                (ASSESSMENT_DIR / f"{name}__petitioned_perception.json"
                 ).write_text(record.model_dump_json(indent=2))
        else:
            print("   (image not found — petition path disabled)")
            result = run_assessment(
                record, n_probes=DEFAULT_N_PROBES,
                explain_fn=_ollama_explain,
                runoff_judge_fn=_ollama_judge, on_event=printer)
        out = ASSESSMENT_DIR / f"{name}__assessment.json"
        out.write_text(result.model_dump_json(indent=2))
        a = result.assessment
        print(f"   -> {a.disaster_scenario} · {a.disaster_type} · "
              f"level {a.disaster_level} ({a.severity_bucket}) · "
              f"self-conf {a.self_confidence} · "
              f"{len(result.violations)} violation(s)  [{out.name}]")
        if result.measured_uncertainty:
            mu = result.measured_uncertainty
            print(f"   U={mu.score} ({mu.n_probes} probes) · "
                  f"{mu.explanation}")


if __name__ == "__main__":
    # `python -m agentic.assessment` loads this file as "__main__", while
    # reflection.py imports it again as "agentic.assessment" — two copies
    # of every class, and Pydantic rightly refuses the twin's instances.
    # Re-enter through the canonical module so exactly one copy runs.
    from agentic.assessment import main as _canonical_main
    _canonical_main()
