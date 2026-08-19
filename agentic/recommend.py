"""Stage 4 — RECOMMENDATION (the generation spine).

WHY THIS EXISTS
===============
Stages 1–2 perceive the scene and judge it. Stage 4 turns that judgment into
RECOMMENDATIONS — what a responder should do — and records WHY, as causal
quads. Those recommendations and quads are the MODEL'S own: they are the very
thing CEE+ measures for causal grounding, so nothing here may generate them on
the model's behalf. This module is the straight line only:

    recommend  →  build Graph A  →  Graph B  →  pick targets

No reflection, no judges, no petition yet (Phase 2). The checks (conformance,
alignment) run later, at the end, as evals — not here.

IRON RULES honored here
=======================
- NEVER edit main.py (Arm A). Import only — the causal-quad ontology, the
  Graph-A builder, the suppression ranker, and the state/effect vocabulary all
  live there and stay frozen.
- Perception is FROZEN. The model may look at the image to reason about layout,
  but it may only use object_ids that already exist in the Stage-1 record. It
  never re-perceives. detected_objects is the single source of truth.
- Two layers out of `recommend`: the HARD layer (reasoning frame + the
  recommendations whose quads become Graph A) stays grounded in frozen
  entities; the ADVISORY layer (assumptions about likely-unseen entities) is
  recorded ON TOP, never baked into the graph. Its promotion to a real entity
  is a Stage-1 petition — Phase 2, not here.
- Guard every boundary that eats a raw VLM answer: parse `Any`, log coercions
  as note strings, never crash on garbage.

WHAT GETS IMPORTED FROM ARM A
=============================
    build_causal_graph            recommendations' quads -> Graph A (code)
    pick_suppression_framework    rank intervention candidates by out-degree
    GRAPH_B_PROMPT / normalize_graph_b   the independent graph + suppression_pick
    normalize_recommendations     tolerant parse of one recommendations list
    extract_json_block            tolerant JSON extraction
    EFFECT_LABELS                 the 8 legal effect labels (for the prompt)

Model calls are injected as `query_fn: Callable[[str], dict]` so the whole
module is hermetic under test (the image, when live, is captured in the
query_fn closure — the node cores only ever see text prompts).
"""
from __future__ import annotations

import os

from agentic import models as _models
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


QueryFn = Callable[[str], dict]


# ── The subject-VLM call (image-bound at the control layer; injectable) ──

def _query_vlm(prompt: str, *, image_contents: Optional[str] = None,
               temperature: float = 0.0) -> dict:
    """Default live call to the subject VLM (Ollama qwen2.5vl). Text prompt +
    optional image. Tests never reach this — they inject a scripted query_fn.
    The image rides as context ONLY; the prompt forbids re-perception."""
    import requests

    from main import extract_json_block  # lazy: keeps offline import light
    api_url = os.getenv("QWEN_API_URL",
                        "http://localhost:11434/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("QWEN_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    content: Any = prompt
    if image_contents:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_contents}},
        ]
    payload = {
        "model": _models.SUBJECT_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        # F50: NO response_format. On a thinking model (qwen3-vl) the
        # json_object constraint collided with the reasoning phase and the
        # content came back literally "{}" — two tokens, empty. Without the
        # constraint the thinking lands in a separate field and the content is
        # clean JSON; extract_json_block already tolerates surrounding prose.
        # Same lesson as the judges (F26): format constraints suppress the
        # very output they were meant to shape.
    }
    r = requests.post(api_url, headers=headers, json=payload,
                      timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    return extract_json_block(r.json()["choices"][0]["message"]["content"])


# ── Prompt templates (module-level, prompt-neutral: no scene tokens/ids) ──
#
# The frozen scene is INJECTED as a data block; the template itself carries no
# scene-specific tokens. The model reasons from the given entities + the image
# and returns JSON. It is told, explicitly, not to invent entities.

_EFFECT_LINE = ("may_spread_to, may_harm, blocks_access_to, isolates, "
                "exposes, increases_risk_to, worsens, threatens")

RECOMMEND_PROMPT = """You are an emergency-response analyst. Your
recommendations are FOR the emergency response team handling this scene —
write instructions they will use to do their job.
A prior stage has
already identified every object in this scene, its state, which objects are
hazards (sources of harm), and which are at risk. That list is FINAL and below.
You may look at the image to judge layout and distances, but you must NOT add,
rename, or re-identify any object. Use ONLY the object_ids given.

SCENE (final — do not change it):
{scene_block}

Produce emergency-response recommendations for THIS scene. Return valid JSON
with exactly these keys:

- scene_summary: one short sentence describing the scene as a whole.
- key_observations: array of short strings about what you see. Use only the
  given object_ids.
- assumptions: array of short strings you INFER beyond the visible evidence.
- uncertainty_notes: array of short strings about what you are unsure of and why.
- recommendations: array, one entry per ACTION you would take. Two actions may
  rest on the SAME (threat, state) — protecting two people from one hazard is
  two actions and one hazard. Every entry needs a real quad: if you cannot see
  which declared threat an action responds to, drop the action rather than
  writing "N/A" or leaving a slot empty. Do not pad to a fixed
  count.{empty_clause} Each entry:
    - rank: integer (1 = highest priority)
    - action: one specific responder action (no "and"/"then" compounds).
      Whenever your action acts on an entity that IS in the scene list above,
      name it by its object_id, exactly as listed — never by a prose
      description of it. The list describes the SCENE, not the response: the
      responders, teams, vehicles, equipment and services you would bring in
      are not in it, and you should name those in ordinary words. Every
      object_id your action names must also appear in the quad below: the quad
      explains why THAT action on THOSE entities, not merely some other true
      danger in the scene.
    - reason: plain English of exactly the form "Because {{threat}} is
      {{state}}, it {{effect}} {{affected_objects}}." It must obey the SAME
      four rules the quad obeys, listed under structured_reasoning below: the
      same legal threats, the same legal states, the same effect list, the same
      role bans. Write it in prose; do not relax the rules for prose.
    - related_object_ids: array of object_ids the reason touches
    - structured_reasoning: a JSON OBJECT with exactly the four keys below —
      never a sentence, never an arrow string. It carries the SAME claim the
      reason makes, with each part in its own key: the reason's subject goes in
      `threat`, its state in `state`, its verb in `effect`, and the entities it
      says are harmed in `affected_objects`. If you cannot fill the four keys
      without changing the claim, the reason was wrong: rewrite the reason.
        - threat: object_id of the SOURCE of harm — one of the object_ids on
          the `threats:` line above. The quad always describes the DANGER the
          action responds to, never the action itself, so a rescue still names
          the hazard the victim is being rescued FROM: "rescue X" is
          <hazard> -> may_harm -> X. An entity is never its own threat, and an
          at-risk entity is never the threat — being in danger is not the same
          as causing it.
        - state: that threat's hazard-bearing state. It must be one of the
          state= values listed for that object_id above. 'distress' and
          'proximity' are at_risk_as ROLES, not states; never use them here.
        - effect: exactly one of [{effects}]
        - affected_objects: {affected_clause} Plain ids only — never an id
          with a suffix attached.
      Together, the action, the reason and the quad must all describe ONE
      causal claim: the reason and the quad each explain, on their own, why
      THAT action on THOSE entities is the right response to THAT danger.
    - expected_consequence: the immediate result of THIS action if it succeeds
    - remaining_risk: an (object_id, state) pair this action does NOT address;
      must differ across recommendations. The state here obeys the same rule as
      the quad's: a state= value listed for that object_id, never an at_risk_as
      role such as 'distress' or 'proximity'.
    - possible_follow_up_action: the next step after this action
- assumptions_advisory: array (may be empty) of flags about entities you cannot
  see but suspect are present or at risk (e.g. occupants inside a structure).
  Each entry: {{ "suspected": short description, "anchor_object_id": an existing
  object_id it relates to, "cue": why you suspect it, "suggested_action": what a
  responder might do }}. These are ADVISORY only — do not put suspected unseen
  entities into recommendations or the quads.
"""

# O1 — the empty-recommendations permission (Sunny's paired arm).
#
# F_park_control, a SAFE scene, still produced a recommendation and a causal
# edge (dog_1 exposes swing_1). Before calling that a model defect we have to
# rule out our own prompt: `affected_objects: NON-EMPTY` and the framing
# "Produce emergency-response recommendations" read as an instruction to
# produce, and only `assumptions_advisory` is explicitly allowed to be empty.
# We may be forcing the fabrication we then measure.
#
# The clause is a PERMISSION, never steering. It says the array may be empty;
# it does NOT say "do not invent hazards" or "be conservative" — that would
# teach the answer, break iron rule 5, and Goodhart the measurement: we would
# fix the behaviour and destroy our ability to observe it in the same move.
# The wording is lifted verbatim from the sentence assumptions_advisory
# already carries, so this removes an inconsistency in our own schema rather
# than adding an instruction.
#
# It ships OFF. The experiment is PAIRED — F_park run twice in one session,
# clause on and clause off, everything else fixed — because the clause sits in
# the shared prompt and would otherwise move all six scenes at once, leaving
# the weights calibrated against a shifted baseline.
EMPTY_RECS_CLAUSE = (" MAY BE EMPTY if no entity in the scene is hazardous.")

# The twin permission. `affected_objects: NON-EMPTY` forces the model to name a
# victim even when its hazard threatens nobody in particular — the pressure
# that produced the placeholder self-loops F18 re-files. Relaxing it is the
# other half of the same experiment, so it rides the same flag: both
# permissions on, or both off. Splitting them would leave a paired run unable
# to say which one moved the output.
AFFECTED_REQUIRED = "NON-EMPTY list of object_ids harmed."
AFFECTED_OPTIONAL = ("list of object_ids harmed; MAY BE EMPTY if this hazard "
                     "threatens nothing in particular.")

RECS_MAY_BE_EMPTY = os.getenv("REC_ALLOW_EMPTY", "0") == "1"


LLM_PICK_PROMPT = """You are an emergency-response analyst. Below are your own
recommendations for a scene and the hazards involved. Reason about causal
impact: if a responder could neutralize exactly ONE hazard to reduce the most
downstream harm, which hazard-and-state would it be, and why?

RECOMMENDATIONS AND HAZARDS:
{recs_block}

Return valid JSON with exactly:
- threat: the object_id of the single hazard to neutralize
- state: that hazard's state
- reason: one sentence on why neutralizing it removes the most harm
"""


# ── Boundary-guarded parsing (never crash on garbage) ───────────────────

def _as_list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_advisory(value: Any, notes: list[str]) -> list[dict]:
    """The assumptions advisory: a list of flag dicts. Tolerant — a malformed
    entry is dropped with a note, never a crash."""
    out: list[dict] = []
    if not isinstance(value, list):
        if value:
            notes.append(f"advisory_not_a_list({type(value).__name__})")
        return out
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            notes.append(f"advisory_{i}_malformed({item!r})")
            continue
        out.append({
            "suspected": str(item.get("suspected", "")),
            "anchor_object_id": str(item.get("anchor_object_id", "")),
            "cue": str(item.get("cue", "")),
            "suggested_action": str(item.get("suggested_action", "")),
        })
    return out


# ── The quad arrives in whatever shape the model felt like (F25) ────────
#
# A_fire round 4: the model returned a PERFECTLY CORRECT quad —
#     "house_1 -> burning -> may_harm -> person_1, dog_1"
# — as an arrow STRING instead of an object. The frozen normalizer takes a
# non-dict quad to {}, which becomes the all-"N/A" placeholder, and the entire
# causal claim of all three recommendations was destroyed SILENTLY: parse_notes
# came back empty, Graph A came back empty, and the card checks then charged the
# model for a quad we had deleted ourselves.
#
# So: recover the shape here, before the frozen normalizer sees it, and NOTE
# every recovery. And note the general case too — a non-empty quad that
# normalizes to all-N/A is a parse loss whatever its shape, which is the guard
# that catches the NEXT shape rather than only this one.

_ARROW_EFFECT = re.compile(r"-{1,3}\s*([a-z_]+)\s*-{1,3}>")


def coerce_quad(value: Any, notes: list[str], rank: Any = None) -> Any:
    """Best-effort recovery of a quad written as prose. Returns a dict when it
    can read one, otherwise the value untouched so the frozen normalizer
    handles it as before. Never raises."""
    if isinstance(value, dict) or not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    # "a --may_harm--> b"  ->  "a -> may_harm -> b"
    text = _ARROW_EFFECT.sub(r"-> \1 ->", text)
    text = text.replace("\u2192", "->").replace("\u00b7", "->")
    parts = [p.strip(" .") for p in re.split(r"-+>", text) if p.strip(" .")]
    if len(parts) < 3:
        notes.append(f"quad_rank_{rank}_unreadable({value[:60]!r})")
        return value
    effects = {e.strip() for e in _EFFECT_LINE.split(",")}

    def _ids(chunk: str) -> list[str]:
        return [x.strip() for x in re.split(r"[,;]| and ", chunk) if x.strip()]

    if len(parts) >= 4:
        threat, state, effect, affected = parts[0], parts[1], parts[2], parts[3]
    elif parts[1] in effects:
        # the state was omitted, not the effect
        threat, state, effect, affected = parts[0], "", parts[1], parts[2]
    else:
        threat, state, effect, affected = parts[0], parts[1], "", parts[2]
    notes.append(f"quad_rank_{rank}_recovered_from_string")
    return {"threat": threat, "state": state, "effect": effect,
            "affected_objects": _ids(affected)}


def _quad_is_empty(q: Any) -> bool:
    if not isinstance(q, dict):
        return True
    vals = [str(q.get(k, "")).strip().upper() for k in ("threat", "state",
                                                        "effect")]
    return all(v in ("", "N/A") for v in vals)


def parse_recommend(raw: Any) -> tuple[dict, list[dict], list[dict], list[str]]:
    """Parse a raw recommendation VLM answer into
    (reasoning_frame, recommendations, advisory, notes).

    Reuses Arm A's `normalize_recommendations` for the recommendations list, so
    the quad shape (affected_objects plural, effect defaults) matches Arm A
    exactly. Guarded at every step: non-dict input becomes {} with a note, a
    malformed list becomes []."""
    from main import normalize_recommendations  # lazy

    notes: list[str] = []
    if not isinstance(raw, dict):
        notes.append(f"recommend_raw_not_dict({type(raw).__name__})->{{}}")
        raw = {}

    frame = {
        "scene_summary": str(raw.get("scene_summary", "")),
        "key_observations": _as_list_of_str(raw.get("key_observations")),
        "assumptions": _as_list_of_str(raw.get("assumptions")),
        "uncertainty_notes": _as_list_of_str(raw.get("uncertainty_notes")),
    }

    # F25: recover prose quads BEFORE the frozen normalizer, which takes any
    # non-dict quad to the all-N/A placeholder.
    _recs_in = raw.get("recommendations")
    if isinstance(_recs_in, list):
        _fixed = []
        for _r in _recs_in:
            if isinstance(_r, dict) and "structured_reasoning" in _r:
                _r = {**_r, "structured_reasoning": coerce_quad(
                    _r["structured_reasoning"], notes, _r.get("rank"))}
            _fixed.append(_r)
        _recs_in = _fixed

    try:
        recommendations = normalize_recommendations(_recs_in)
    except Exception as exc:  # normalize should be tolerant, but never crash
        notes.append(f"recommendations_unparseable({exc})->[]")
        recommendations = []
    if not isinstance(recommendations, list):
        notes.append("recommendations_not_a_list->[]")
        recommendations = []

    # F25, the general guard: whatever the shape, a recommendation that ARRIVED
    # with a quad and LEFT without one is a parse loss, and it must never again
    # be silent. This is what was missing — not the arrow-string case
    # specifically, but any total loss of the claim Stage 4 exists to measure.
    _in = _recs_in if isinstance(_recs_in, list) else []
    for _i, _out in enumerate(recommendations):
        _src = _in[_i] if _i < len(_in) and isinstance(_in[_i], dict) else {}
        if _src.get("structured_reasoning") and \
                _quad_is_empty(_out.get("structured_reasoning")):
            notes.append(f"quad_rank_{_out.get('rank')}_LOST_IN_PARSE"
                         f"({str(_src.get('structured_reasoning'))[:60]!r})")

    advisory = _parse_advisory(raw.get("assumptions_advisory"), notes)
    return frame, recommendations, advisory, notes


def parse_pick(raw: Any) -> tuple[dict, list[str]]:
    """Parse the LLM direct-impact pick into {threat, state, reason}."""
    notes: list[str] = []
    if not isinstance(raw, dict):
        notes.append(f"pick_raw_not_dict({type(raw).__name__})->empty")
        raw = {}
    return ({"threat": str(raw.get("threat", "")),
             "state": str(raw.get("state", "")),
             "reason": str(raw.get("reason", ""))}, notes)


# ── Adapters: frozen record + assessment -> the dict shapes Arm A wants ──

def _detected_dicts(record: Any) -> list[dict]:
    return [{"object_id": o.object_id, "label": o.label, "state": o.state}
            for o in record.detected_objects]


def _state_of(record: Any, oid: str) -> str:
    for o in record.detected_objects:
        if o.object_id == oid:
            return o.state
    return "unknown"


def _threat_dicts(record: Any, assessment: Any) -> list[dict]:
    """threats as [{object_id, state}] — state pulled from the frozen record,
    since the assessment's ThreatEntry carries only object_id + reason."""
    return [{"object_id": t.object_id, "state": _state_of(record, t.object_id)}
            for t in assessment.threats]


def _at_risk_dicts(assessment: Any) -> list[dict]:
    return [{"object_id": a.object_id} for a in assessment.at_risk]


def _scene_block(record: Any, assessment: Any) -> str:
    """A compact, readable dump of the frozen scene for the prompt.

    F16: the at-risk ROLE gets its own named key. It used to be rendered as
    'child_2·proximity', which gave '·' a second meaning — everywhere else in
    the system 'id·word' is entity·STATE — and the model generalised from the
    syntax, writing roles into the quad's state slot and role-suffixed ids
    into affected_objects. One separator, one meaning."""
    at_risk_as = {a.object_id: a.kind for a in assessment.at_risk}
    lines = ["entities:"]
    for o in record.detected_objects:
        line = (f"  {o.object_id}: {o.label}, state={o.state} "
                f"({o.state_kind})")
        if o.object_id in at_risk_as:
            line += f", at_risk_as={at_risk_as[o.object_id]}"
        lines.append(line)
    lines.append("threats: " + (", ".join(t.object_id for t in assessment.threats)
                                or "none"))
    lines.append("at_risk: " + (", ".join(a.object_id
                                          for a in assessment.at_risk) or "none"))
    if record.caption:
        lines.append(f"caption: {record.caption}")
    return "\n".join(lines)


def _recs_block(recommendations: list[dict]) -> str:
    lines = []
    for r in recommendations:
        q = r.get("structured_reasoning", {}) or {}
        lines.append(
            f"  rank {r.get('rank')}: {r.get('action')} "
            f"[{q.get('threat')}·{q.get('state')} {q.get('effect')} "
            f"{q.get('affected_objects')}]")
    return "\n".join(lines) or "  (none)"


# ── Event helper ────────────────────────────────────────────────────────

def _emitter(on_event: Any):
    def emit(event_type: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"type": event_type, **data})
    return emit


# ── Node cores: one function per straight-line step ─────────────────────

def run_recommend(record: Any, assessment: Any, *, query_fn: QueryFn,
                  on_event: Any = None) -> dict:
    """Step 1. The model produces the reasoning frame + recommendations (the
    HARD layer) and the assumptions advisory (the ADVISORY layer)."""
    emit = _emitter(on_event)
    emit("stage_started", stage="recommend")
    prompt = RECOMMEND_PROMPT.format(
        scene_block=_scene_block(record, assessment), effects=_EFFECT_LINE,
        empty_clause=EMPTY_RECS_CLAUSE if RECS_MAY_BE_EMPTY else "",
        affected_clause=(AFFECTED_OPTIONAL if RECS_MAY_BE_EMPTY
                         else AFFECTED_REQUIRED))
    raw = query_fn(prompt)
    frame, recommendations, advisory, notes = parse_recommend(raw)
    for n in notes:
        emit("recommend_parse_note", note=n)
    emit("recommendations_ready",
         n_recs=len(recommendations),
         ranks=[r.get("rank") for r in recommendations],
         n_advisory=len(advisory))
    return {"frame": frame, "recommendations": recommendations,
            "advisory": advisory, "recommend_notes": notes, "recommend_raw": raw}


# ── Measured uncertainty over the recommendation step (channel 2) ───────
#
# Same machinery as Stage 2: re-ask the SAME recommend prompt K times at a
# raised temperature and measure how stable the advice is. The canonical
# recommendations (temp 0) are untouched — probes describe their stability,
# they never replace them. Off by default (n_probes=0) so the hermetic spine
# and the equivalence tests stay model-free.

REC_PROBE_TEMPERATURE = float(os.getenv("REC_PROBE_TEMP", "0.7"))
DEFAULT_REC_N_PROBES = int(os.getenv("REC_N_PROBES", "5"))


def _recommend_reading(recommendations: list[dict]) -> dict:
    """Reduce one probe's recommendations to the stability-relevant reading
    measure_recommendations consumes. Ids are stripped of any '·state' suffix
    so 'house_1' and 'house_1·burning' count as the same entity."""
    recs = recommendations or []

    def _rank(r: dict) -> int:
        try:
            return int(r.get("rank"))
        except (TypeError, ValueError):
            return 9999

    _bare = bare_id          # F16: one normaliser, defined at module level

    ordered = sorted(recs, key=_rank)
    top_threat = ""
    if ordered:
        q0 = ordered[0].get("structured_reasoning", {}) or {}
        top_threat = _bare(q0.get("threat"))

    threat_ids: list[str] = []
    affected_ids: list[str] = []
    edges: list[tuple] = []
    effect_by_threat: dict[str, str] = {}
    for r in recs:
        q = r.get("structured_reasoning", {}) or {}
        t = _bare(q.get("threat"))
        eff = str(q.get("effect", ""))
        aff = tuple(sorted(_bare(x) for x in (q.get("affected_objects") or [])
                           if _bare(x)))
        if t:
            threat_ids.append(t)
            effect_by_threat.setdefault(t, eff)
        affected_ids.extend(aff)
        edges.append((t, eff, aff))
    return {"top_threat": top_threat, "n_recs": len(recs),
            "threat_ids": threat_ids, "affected_ids": affected_ids,
            "edges": edges, "effect_by_threat": effect_by_threat}


def _canonical_threats(recommendations: list[dict]) -> set:
    """The threat ids the CANONICAL (temp-0) recommendations actually use —
    the entities whose stability we care about, so probe noise (a victim
    mislabeled a threat once) doesn't clutter the uncertainty panel."""
    out: set = set()
    for r in recommendations or []:
        t = bare_id((r.get("structured_reasoning", {}) or {}).get("threat"))
        if t:
            out.add(t)
    return out


def measure_recommend_uncertainty(prompt: str, n_probes: int, *,
                                  probe_fn: QueryFn | None = None,
                                  explain_fn: Any = None,
                                  canonical_threats: set | None = None,
                                  emit: Any = None):
    """Re-ask the recommend prompt n_probes times at probe temperature and
    measure advice dispersion. A probe whose answer is garbage still counts:
    parse_recommend yields [] and instability from garbage IS instability. If
    EVERY probe dies in transport, the advice reads as unmeasured (score 1.0),
    never as unanimous — the same conservative default Stage 2 uses."""
    from agentic.uncertainty import (Driver, explain,  # lazy
                                     measure_recommendations)
    probe_fn = probe_fn or (lambda p: _query_vlm(p,
                                                 temperature=REC_PROBE_TEMPERATURE))
    readings: list[dict] = []
    probe_recs: list[list] = []     # F49/runoff: each probe's FULL parsed recs
    for i in range(n_probes):
        try:
            _frame, recs, _adv, _notes = parse_recommend(probe_fn(prompt))
        except Exception as exc:               # transport failure ≠ dispersion
            if emit:
                emit("recommend_probe_error", index=i, error=str(exc))
            continue
        reading = _recommend_reading(recs)
        readings.append(reading)
        probe_recs.append(recs)
        if emit:
            # F49 (capture spec 9.9): the probe's FULL parsed output rides in
            # the event, prose included. The reading above keeps only the quad
            # skeleton, so before this every probe's action/reason text was
            # discarded at parse time — and those five answers to the same
            # prompt are the preference-pair corpus (JUDGES.md 9.2). Every run
            # before this change lost them permanently. Graph B probes never
            # had this bug; they are stored whole in graph_b_uncertainty.
            emit("recommend_probe", index=i, n_recs=reading["n_recs"],
                 top_threat=reading["top_threat"], recs=recs)
    mu = measure_recommendations(readings, canonical_threats=canonical_threats)
    if n_probes > 0 and not readings:
        mu.score = 1.0
        mu.drivers.append(Driver(
            kind="probes_failed",
            evidence=f"all {n_probes} recommend probes failed in transport",
            action="fix the model endpoint and re-run; treat this advice as "
                   "unmeasured until then"))
    explain(mu, "recommendation set", explain_fn)
    return mu, probe_recs


def run_recommend_uncertainty(record: Any, assessment: Any, *,
                              probe_fn: QueryFn | None = None,
                              explain_fn: Any = None, n_probes: int = 0,
                              canonical_threats: set | None = None,
                              on_event: Any = None) -> dict:
    """Node core: measured uncertainty over the recommendation step. Off unless
    n_probes>0. Shared by both controls (called in the SAME place), so the twin
    stays byte-identical. Returns {'uncertainty': MeasuredUncertainty dict}."""
    emit = _emitter(on_event)
    if n_probes <= 0:
        return {"uncertainty": {}}
    prompt = RECOMMEND_PROMPT.format(
        scene_block=_scene_block(record, assessment), effects=_EFFECT_LINE,
        empty_clause=EMPTY_RECS_CLAUSE if RECS_MAY_BE_EMPTY else "",
        affected_clause=(AFFECTED_OPTIONAL if RECS_MAY_BE_EMPTY
                         else AFFECTED_REQUIRED))
    mu, probe_recs = measure_recommend_uncertainty(
        prompt, n_probes, probe_fn=probe_fn, explain_fn=explain_fn,
        canonical_threats=canonical_threats, emit=emit)
    emit("recommend_uncertainty_ready", score=mu.score, n_probes=mu.n_probes,
         drivers=len(mu.drivers))
    # probe_recs ride beside the measurement: the runoff judge needs the FULL
    # prose candidates, and the skeleton readings above cannot carry it.
    return {"uncertainty": mu.model_dump(), "probe_recs": probe_recs}


def bare_id(x: Any) -> str:
    """The one id normaliser (F16). '·' means entity·state and only that, so a
    '<id>·<anything>' arriving from the model is reduced to the id. The old
    prompt emitted at-risk as 'ambulance_1·proximity' — a SECOND meaning for
    the same separator — and the model copied it into affected_objects, so
    Graph A targets stopped being ids: they resolved to no node, scored as
    fabrication, and drove A-vs-B alignment to 0.0 on D_aerial. The prompt no
    longer teaches that form; this is the belt to that braces, because a model
    may still emit one and a stray dot must never again manufacture a
    violation."""
    return str(x or "").split("·")[0].strip()


def _sanitize_recs(recommendations: list[dict]) -> list[dict]:
    """Strip '·suffix' from every id a quad carries, immediately before the
    frozen builder sees it. Arm A is untouched; this is a layer in front of
    it. Only ids are normalised — the model's prose is left exactly as
    written, because the reason string is evidence."""
    out: list[dict] = []
    for r in recommendations or []:
        if not isinstance(r, dict):
            continue
        q = r.get("structured_reasoning")
        if not isinstance(q, dict):
            out.append(r)
            continue
        aff = q.get("affected_objects")
        aff = aff if isinstance(aff, (list, tuple)) else []
        rel = r.get("related_object_ids")
        rel = rel if isinstance(rel, (list, tuple)) else None
        out.append({**r,
                    "structured_reasoning": {
                        **q,
                        "threat": bare_id(q.get("threat")),
                        "affected_objects": [bare_id(a) for a in aff
                                             if bare_id(a)]},
                    **({"related_object_ids": [bare_id(x) for x in rel
                                               if bare_id(x)]}
                       if rel is not None else {})})
    return out


def build_graph_a(record: Any, assessment: Any, recommendations: list[dict],
                  *, on_event: Any = None) -> dict:
    """Step 2. Reassemble the recommendation quads into Graph A — CODE, via
    Arm A's build_causal_graph. Nodes = all frozen entities; edges = one per
    quad target. build_causal_graph also computes intervention_candidates.

    Ids are normalised through bare_id() first (F16): this is the single choke
    point where a '·'-suffixed id would otherwise enter the graph."""
    from main import build_causal_graph  # lazy
    emit = _emitter(on_event)
    graph_a = build_causal_graph(
        _detected_dicts(record),
        _threat_dicts(record, assessment),
        _sanitize_recs(recommendations),
        _at_risk_dicts(assessment))
    # F18: re-file placeholder self-loops as node annotations. Runs AFTER the
    # frozen builder — Arm A is untouched, this is a layer on top of its output.
    graph_a = annotate_one_ended(graph_a, record)
    emit("graph_a_built",
         n_nodes=len(graph_a.get("nodes", [])),
         n_edges=len(graph_a.get("edges", [])),
         warnings=graph_a.get("graph_warnings", []))
    return graph_a


# Graph B inverted the causal direction on B_pool twice running:
# `child_1 · drowning --may_harm--> pool_1`, the victim named as the source and
# the hazard as the harmed party — while the same JSON marked child_1
# `hazardous: false`. The threat REASON we hand it reads "The pool is hazardous
# due to the presence of child_1, who is drowning": the sentence explains the
# flag but ends on the victim, and the model took word order for arrow
# direction. So the fix names that trap and shows the arrow once, correctly.
#
# Lives in Arm B — main.py stays byte-identical. Electricity because none of
# the six calibration scenes involve it; `wire`/`worker` without digits so the
# example carries no id-shaped token; only the CORRECT arrow is shown, with the
# wrong reading described in words rather than written out.
GRAPH_B_DIRECTION_EXAMPLE = """\
Direction rule, worked through:

  Suppose the prior analysis says:
    "The wire is hazardous due to the presence of the worker, who is
     being electrocuted."

  That sentence explains WHY the wire was flagged. It is not a causal
  claim, and its word order is not the arrow.

  Read the states instead:
    wire   · live         -> hazard-bearing -> this is the SOURCE
    worker · electrocuted -> at-risk        -> this is the TARGET

  Correct edge:  wire --may_harm--> worker,  via_state: live

  The arrow always runs from the hazard-bearing state to the at-risk
  one, no matter which entity the sentence happens to name first.
"""


def _graph_b_prompt(record: Any, assessment: Any) -> str:
    """Assemble the FULL Graph B prompt the way Arm A does: the neutral
    template, the deny-inferred policy, the caption, and — crucially — the
    frozen detected_objects (WITH their ids) + threats. "Independent" means
    the RECOMMENDATIONS are withheld, NOT the ids: Graph B must speak the same
    entity language as Graph A or the two can't be compared. The template
    itself even says "detected_objects is the single source of truth for
    object_ids" and "do not add nodes beyond the detected_objects supplied"."""
    import json

    from main import GRAPH_B_INFERRED_DENIED, GRAPH_B_PROMPT
    # F54 AMENDED (Sunny, 2026-08-19, section-by-section prompt review). The
    # threats RETURN to the context, deliberately: Graph B's job is causal
    # STRUCTURE, not hazard detection — Stage 2 already ruled on the threats;
    # B wires them. The victims side stays unseeded (the at-risk register is
    # still never sent), so it remains the earned, informative half of every
    # A-vs-B comparison. Cost accepted with eyes open: "same hazards 1.00" is
    # partly the model echoing our list.
    #
    # The OPENING PARAGRAPH is Sunny's, approved verbatim 2026-08-19 — the
    # first prompt text shipped under the inspection rule. It drops two frozen
    # sentences: the recommendations-withheld explanation (the model cannot
    # copy what it never sees — explaining an absence to a reader who would
    # never notice it) and "regardless of which a responder would address
    # first" (carried by "cover every causal pathway"). main.py untouched:
    # the frozen prompt is sliced at its first section header and the new
    # opening is prepended — the same paragraph-swap idea as Arm A's own
    # variant mechanism.
    _OPENING = (
        "You are extracting the causal graph that explains how the hazards "
        "in this scene threaten the safety of the entities in the "
        "environment, from the perspective of an emergency response analyst. "
        "Cover every causal pathway you believe holds — direct harm, cascade "
        "between hazards, exposure, proximity risk. Below are the "
        "detected_objects and threats from a prior analysis of the scene.")
    _body = GRAPH_B_PROMPT[GRAPH_B_PROMPT.index("## State vocabulary"):]
    # Section 2 (Sunny, prompt review): of the whole instancing paragraph,
    # "that's it" — only the last sentence does live work here. The ten-node
    # budgeting and people-counting rules governed Arm A, where the model
    # built the graph straight from the image; in this pipeline Stage 1 fixes
    # the entity list before Graph B ever runs, so the cap is upstream.
    _inst_start = _body.index("Representative instancing:")
    _keep = "Do not add nodes beyond the detected_objects supplied."
    _inst_end = _body.index(_keep) + len(_keep)
    _body = _body[:_inst_start] + _keep + _body[_inst_end:]
    # Sections 3-7 preamble (Sunny, approved verbatim 2026-08-19). Two fixes
    # from his review: the graph's anatomy was never DEFINED before the law
    # referenced it (nodes/edges arrived at the output schema, ~70% in — the
    # rules of a game before the pieces), and the three state lists arrived
    # unannounced, leaving the reader to reverse-engineer that they are one
    # decoder. The direction rule ("FROM the entity doing the harm") is
    # promoted here from the appended worked example, which stays as
    # reinforcement.
    _PREAMBLE = (
        "The causal graph you are extracting has two parts: NODES — one per "
        "detected\nentity — and EDGES, each a single causal claim of the "
        "form\n`source --effect--> target`, read as \"the source's hazardous "
        "state acts on\nthe target in this way.\" An edge always runs FROM "
        "the entity doing the harm\nTO the entity receiving it.\n\n"
        "Every entity in the detected_objects input below carries a STATE "
        "word in its\n`state` field. The lists that follow decode that "
        "word into the entity's\nrole in the causal graph:\n\n"
        "  hazardous  \u2192  the entity may be the SOURCE of edges\n"
        "  at_risk    \u2192  the entity is a TARGET already being harmed; "
        "never a source\n"
        "  otherwise  \u2192  a target only; at risk exactly when a "
        "hazard's edge reaches it\n\n"
        "An entity's role comes ONLY from its `state` field — never from its "
        "label,\nits prominence, or the order it is listed in. Every rule "
        "later in this prompt\nrefers back to these lists.")
    _body = _body.replace(_keep, _keep + "\n\n" + _PREAMBLE, 1)
    # Sections 3-7 (Sunny, approved verbatim 2026-08-19, prompt review).
    # One middle block replaces five frozen sections. His rulings, in order:
    # engulfing is out (perception remaps it away); the collapse tie-breaker
    # and behavioral families were Stage-1 state-choosing guidance; the
    # living-beings carve-out and the proximity clause collapsed into a
    # per-entity CLASSIFIER (the C omission becomes a required decision with
    # two legal outcomes — draw the edge, or deliberately claim safety); the
    # "Normal states" list died with its category ("normal" is not a graph
    # concept — it is everything else); and the categories are NAMED BY THE
    # SCHEMA FLAGS (hazardous / at_risk) so "state" again means only the
    # input field.
    _mid_start = _body.index("Hazard-bearing states")
    _mid_end = _body.index("**Fluid / gaseous hazards")
    _MIDDLE = (
        "Hazardous (entity is a SOURCE of harm): burning, burnt, collapsed,\n"
        "collapsing, fallen, crushed, flooded, leaking, approaching, "
        "charging, aiming,\ncoiled, rabid, armed, striking, rising, "
        "spreading, billowing, seeping,\nescalating, hazardous_in_context. "
        "`hazardous_in_context` is the last-resort\nfallback when no "
        "specific state applies.\n\n"
        "For each entity, read its `state` field and classify it in this "
        "order:\n\n"
        "1. hazardous (list above) — the entity is a SOURCE of harm.\n"
        "2. at_risk (list below) — the entity is ALREADY BEING HARMED. Draw "
        "the\n   edge from the hazard causing that harm to it.\n"
        "3. Decide whether an active hazard can reach the entity. If it can, "
        "the\n   entity is EXPOSED: mark it `at_risk: true` and draw that "
        "hazard's edge to\n   it. If not, leave it false — a deliberate "
        "claim that it is safe.\n\n"
        "At-risk (entity is a TARGET of harm): injured, bleeding, fleeing, "
        "trapped,\ncowering, drowning, suffocating, unconscious. Any entity "
        "— person, animal,\nvehicle, or structure — may carry one of these "
        "when it's not hazardous or\nnot at-risk by proximity.\n\n")
    _body = _body[:_mid_start] + _MIDDLE + _body[_mid_end:]
    # Fluid area (Sunny, approved 2026-08-19): old section 7's emission half
    # was Stage 1's job (objects already arrive); its routing core folds into
    # the effect table. Provenance keeps the counterfactual plumbing, with
    # "removing the source removes the fluid" corrected to what suppression
    # actually buys — the fluid stops being FED; released fluid may persist
    # (stop the leak, the spill is still on the road).
    _fl_start = _body.index("**Fluid / gaseous hazards")
    _fl_end = _body.index("**Independent harm channels.**")
    _FLUID = (
        "**Diffuse hazards (water, smoke, gas, dust, spills) — edges keyed "
        "to the\nTARGET.** The fluid is the source of outward harm: edges to "
        "people and\nexposed entities run FROM the fluid, not from an entity "
        "it has inundated.\nA fluid's outgoing edge uses: `increases_risk_to` "
        "when the target is already\nhazardous (the fluid escalates an "
        "existing hazard); `may_harm` when the\ntarget is a person or "
        "animal; `may_spread_to` when the target is intact and\nin the "
        "trajectory (conversion pending).\n\n"
        "**Fluid provenance — keep the graph connected.** When the fluid's "
        "producing\nsource is visible (smoke from a burning house, dust from "
        "a collapsing\nbuilding, a spill from a leaking tanker), emit "
        "`source \u2192 fluid` with effect\n`increases_risk_to` — the "
        "source feeds the fluid: remove the source and the\nfluid stops "
        "growing, though what has already been released may persist. Do\n"
        "NOT leave a fluid disconnected from its visible producer. If the "
        "producer is\noff-frame or unidentifiable, the fluid may stand "
        "alone.\n\n")
    _body = _body[:_fl_start] + _FLUID + _body[_fl_end:]
    # Harm channels (Sunny, approved 2026-08-19): compressed to its one
    # non-redundant law — a target collects one edge PER reaching hazard.
    # "Producer and fluid are separate hazards" was settled by Stage 1
    # (they arrive as separate objects); "judge independently" is the
    # classifier's step 3 already.
    _hc_start = _body.index("**Independent harm channels.**")
    _hc_end = _body.index("## Effect vocabulary")
    _HC = ("A target collects one edge PER hazard that reaches it — a person "
           "near a\nburning house gets edges from BOTH the house and its "
           "smoke. Each hazard's\nreach is judged independently; they are "
           "independently suppressible.\n\n")
    _body = _body[:_hc_start] + _HC + _body[_hc_end:]
    # NO SELF-LOOPS (Sunny, 2026-08-19): the graphs allow standalone nodes —
    # a lone hazard stands alone and claims nothing — so the self-loop was a
    # workaround for a prohibition we do not have. Three frozen passages
    # permitted or mandated them; all three amended together.
    _w_old = _body[_body.index("- worsens"):_body.index("- threatens")]
    _body = _body.replace(_w_old,
        "- worsens            — escalates a hazard already present on "
        "ANOTHER\n                       hazardous entity whose mechanism "
        "mutually amplifies\n                       this one (see "
        "Mutual-hazard rule; emit both directions)\n", 1)
    # Rule 3 (self-reference) is DELETED, not turned into a ban (Sunny: "just
    # remove it and don't talk about it in the prompt") — a ban still teaches
    # the concept in order to forbid it, and mentioning a thing invites it
    # (F2). The prompt is silent; if the model invents a self-loop anyway,
    # the code side surfaces it like everything else. Rules renumbered.
    _r3_old = _body[_body.index("3. Self-reference"):_body.index("4. Choose")]
    _body = _body.replace(_r3_old, "", 1)
    _body = _body.replace("4. Choose the most specific",
                          "3. Choose the most specific", 1)
    _r5_old = _body[_body.index("5. Hazardous-node edge requirements:")
                    :_body.index("6. Do NOT produce")]
    _body = _body.replace(_r5_old,
        "4. A hazardous node may have outgoing edges (standard threat), only "
        "incoming\n   edges (pure casualty — e.g., a flooded car hit by "
        "water), or no edges at\n   all when nothing in the scene is "
        "affected by it.\n", 1)
    _body = _body.replace("6. Do NOT produce", "5. Do NOT produce", 1)
    context = {
        "detected_objects": [
            {"object_id": o.object_id, "label": o.label,
             "state": o.state, "bbox": o.bbox}
            for o in record.detected_objects],
        "threats": [
            {"object_id": t.object_id, "state": _state_of(record, t.object_id),
             "reason": getattr(t, "reason", "")}
            for t in assessment.threats],
    }
    return (f"{_OPENING}\n\n"
            f"{_body}\n\n"
            f"{GRAPH_B_DIRECTION_EXAMPLE}\n\n"
            f"Inferred-entity policy:\n{GRAPH_B_INFERRED_DENIED}\n\n"
            f"Caption:\n{record.caption or 'N/A'}\n\n"
            f"Prior analysis (detected_objects + threats):"
            f"\n{json.dumps(context, indent=2)}")


def run_graph_b(record: Any, assessment: Any, *, query_fn: QueryFn,
                on_event: Any = None) -> dict:
    """Step 3. The model's INDEPENDENT causal graph — recommendations withheld,
    but the frozen entities (with ids) supplied so it speaks the same id
    language as Graph A. Returns the graph + its own suppression_pick."""
    from main import normalize_graph_b  # lazy
    emit = _emitter(on_event)
    raw = query_fn(_graph_b_prompt(record, assessment))
    detected_ids = {o.object_id for o in record.detected_objects}
    graph_b = normalize_graph_b(raw if isinstance(raw, dict) else {}, detected_ids)
    emit("graph_b_built",
         n_nodes=len(graph_b.get("nodes", [])),
         n_edges=len(graph_b.get("edges", [])),
         suppression_pick=graph_b.get("suppression_pick", {}))
    return graph_b


def annotate_one_ended(graph: dict, record: Any = None) -> dict:
    """F18. A one-ended causal claim is a NODE ANNOTATION, not an edge.

    Two real situations have no legal quad: a hazard that threatens nothing
    specific yet, and a victim whose source the model cannot name. Given no
    way to say either, the model fabricates — it makes the victim its own
    threat (`child_1 -> child_1`) or leaves the hazard with zero edges. Both
    are placeholders standing in for a missing half.

    So: the placeholder self-loop LEAVES the edge set and becomes a flag on
    the node. Because the flags are node-level, the frozen edge comparators
    (`compare_graphs_topological`) never see them — B_pool goes from 4 edges
    to 2, which is what it always meant.

      unattached    hazardous, no outgoing edge
      unattributed  at-risk, no incoming edge

    NO SELF-LOOP SURVIVES — not even `worsens` (Sunny, 2026-07-28). An earlier
    draft kept the `worsens` self-loop as the legal way to say "this hazard is
    bad on its own". Dropped deliberately: two self-loops identical in shape
    with opposite meanings is precisely the ambiguity that produced F16. A
    hazard that threatens nothing is `unattached` and says so in one way only.

    'No incoming edge' is therefore unambiguous too — and it has to be, or the
    flag misses B_pool, the scene it exists for: child_1 and child_2 each had
    an incoming edge, from themselves."""
    g = dict(graph or {})
    nodes = [dict(n) for n in (g.get("nodes") or []) if isinstance(n, dict)]
    edges = [dict(e) for e in (g.get("edges") or []) if isinstance(e, dict)]

    kept, dropped = [], []
    for e in edges:
        src, tgt = bare_id(e.get("source")), bare_id(e.get("target"))
        if src and src == tgt:         # every self-loop, no exception
            dropped.append(e)
        else:
            kept.append(e)

    # No self-loops remain, so a node's in/out degree is unambiguous: an edge
    # always names another entity.
    out_ids = {bare_id(e.get("source")) for e in kept}
    in_ids = {bare_id(e.get("target")) for e in kept}

    kind = {}
    if record is not None:
        kind = {o.object_id: o.state_kind for o in record.detected_objects}

    for n in nodes:
        nid = bare_id(n.get("id") or n.get("object_id"))
        k = kind.get(nid, "")
        hazardous = bool(n.get("hazardous")) or k == "hazard_bearing"
        at_risk = bool(n.get("at_risk")) or k == "at_risk"
        if hazardous and nid not in out_ids:
            n["unattached"] = True
            n["annotation_note"] = "declared hazardous; no target named"
        if at_risk and nid not in in_ids:
            n["unattributed"] = True
            n["annotation_note"] = "declared at risk; no source named"

    g["nodes"], g["edges"] = nodes, kept
    if dropped:
        g.setdefault("graph_warnings", []).append(
            f"F18: {len(dropped)} placeholder self-loop(s) re-filed as node "
            f"annotations: "
            + ", ".join(f"{bare_id(e.get('source'))}"
                        f"·{e.get('effect')}" for e in dropped))
    return g


def build_graph_a_probes(record: Any, assessment: Any, uncertainty: Any, *,
                         on_event: Any = None) -> list[dict]:
    """F19, the free half. Every probe already produced a full recommendation
    set, and `uncertainty.candidates[i].edges` records each as
    [threat, effect, [affected]] triples — then nothing read them. Rebuild one
    Graph A per probe from those triples. ZERO extra model calls."""
    emit = _emitter(on_event)
    u = uncertainty if isinstance(uncertainty, dict) else {}
    cands = u.get("candidates") or []
    out: list[dict] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        edges = []
        for e in (c.get("edges") or []):
            # each triple: [threat, effect, [affected...]]
            if not isinstance(e, (list, tuple)) or len(e) < 3:
                continue
            src, eff, targets = bare_id(e[0]), str(e[1] or ""), e[2]
            if not isinstance(targets, (list, tuple)):
                targets = [targets]
            for t in targets:
                tgt = bare_id(t)
                if src and tgt:
                    edges.append({"source": src, "effect": eff, "target": tgt})
        out.append({"nodes": [{"object_id": o.object_id}
                              for o in record.detected_objects],
                    "edges": edges})
    if out:
        emit("graph_a_probes_built", n=len(out),
             n_edges=[len(g["edges"]) for g in out])
    return out


def run_graph_b_probes(record: Any, assessment: Any, *,
                       probe_fn: QueryFn | None = None, n_probes: int = 0,
                       on_event: Any = None) -> list[dict]:
    """F19. Ask the INDEPENDENT graph n_probes more times at probe temperature.

    Until now Graph B was asked ONCE, at temp 0, and used as the yardstick that
    Graph A is measured against — while Graph A demonstrably wobbles ('5
    distinct sets' on every scene). A single sample cannot referee an unstable
    one. Probing B also answers a question never asked: IS the model's causal
    belief itself stable? If B wobbles as badly as A, the yardstick is elastic
    and every A-vs-B number in the project inherits that.

    A probe that dies in transport is skipped, not counted as disagreement."""
    from main import normalize_graph_b  # lazy
    emit = _emitter(on_event)
    if n_probes <= 0:
        return []
    probe_fn = probe_fn or (
        lambda p: _query_vlm(p, temperature=REC_PROBE_TEMPERATURE))
    prompt = _graph_b_prompt(record, assessment)
    detected_ids = {o.object_id for o in record.detected_objects}
    out: list[dict] = []
    for i in range(n_probes):
        try:
            raw = probe_fn(prompt)
        except Exception as exc:                     # transport != dispersion
            emit("graph_b_probe_error", index=i, error=str(exc)[:200])
            continue
        g = normalize_graph_b(raw if isinstance(raw, dict) else {}, detected_ids)
        # F42: resolve entity names the model made up before measuring whether
        # its BELIEF is stable. One D_aerial run produced four invented ids
        # across five probes — chemical_spill_1, spill_consequence_1,
        # chemical_worker_1, chemical_worker_2 — and every one of them made an
        # edge look new. `edges 0.8` then reported "the mechanism never
        # settles" when a large part of what never settled was the NAMING.
        # Same fix ab_alignment already had; the probe path never got it.
        from agentic.evals4 import resolve_invented_ids
        g, alias = resolve_invented_ids(g, record)
        if alias:
            emit("graph_b_probe_renamed", index=i, aliases=alias)
        out.append(g)
        # Log the EDGES, not just the count. The recommend probes log only
        # n_recs, and that is exactly why the reversed-edge run could not be
        # diagnosed after the fact — a count cannot tell you which way an
        # arrow pointed.
        emit("graph_b_probe", index=i,
             edges=[[bare_id(e.get("source")), str(e.get("effect") or ""),
                     bare_id(e.get("target"))]
                    for e in (g.get("edges") or []) if isinstance(e, dict)],
             pick=str(((g.get("suppression_pick") or {}).get("object_id")
                       or (g.get("suppression_pick") or {}).get("threat")
                       or "")),
             n_edges=len(g.get("edges", [])))
    return out


def measure_graph_b_uncertainty(record: Any, assessment: Any, n_probes: int = 0,
                                *, probe_fn: QueryFn | None = None,
                                on_event: Any = None) -> dict:
    """Is the model's causal BELIEF stable, or is the yardstick elastic?

    Graph B is asked once and then used to judge Graph A. When it reverses an
    edge we cannot tell whether the model believes the reversal or whether it
    was a coin flip — the difference between a finding and a fluke. Three
    dispersion readings, 0 = stable, 1 = scattered:

      edge_set_instability  1 - mean Jaccard of each probe against the modal
                            edge set
      direction_instability for every unordered entity pair any probe connects,
                            the share of probes disagreeing about which end is
                            the SOURCE. This is the one that catches a victim
                            in the hazard slot.
      pick_instability      1 - the modal suppression_pick's share

    Off by default, same convention as the recommend probes, so the hermetic
    suite stays model-free. Transport failure is not dispersion — but if EVERY
    probe dies, nothing was measured, and an unmeasured belief must read as
    maximally unstable rather than as agreement."""
    emit = _emitter(on_event)
    if n_probes <= 0:
        return {}
    graphs = run_graph_b_probes(record, assessment, probe_fn=probe_fn,
                                n_probes=n_probes, on_event=on_event)
    # F42: entity names the probes invented that could NOT be resolved to a
    # scene entity. `chemical_spill_1` maps to `spill_1` and is fixed silently;
    # `spill_consequence_1` and `chemical_worker_1` map to nothing — the model
    # put entities in its own causal graph that do not exist. That is a finding
    # in its own right, and it was buried in a sixteen-line edge dump.
    _known = {o.object_id for o in getattr(record, "detected_objects", []) or []}
    _invented = sorted({str(x) for g in graphs
                        for e in (g.get("edges") or [])
                        for x in (e.get("source"), e.get("target"))
                        if x and str(x) not in _known})
    if not graphs:
        out = {"n_probes": 0, "requested": n_probes, "score": 1.0,
               "edge_set_instability": 1.0, "direction_instability": 1.0,
               "pick_instability": 1.0, "modal_edges": [],
               "flags": [{"kind": "graph_b_probes_failed",
                          "evidence": f"all {n_probes} Graph B probes failed "
                                      f"in transport",
                          "action": "fix the model endpoint and re-run; treat "
                                    "Graph B as unmeasured until then"}]}
        emit("graph_b_uncertainty_ready", **{k: v for k, v in out.items()
                                             if k != "flags"})
        return out

    def _edges(g):
        return {(bare_id(e.get("source")), str(e.get("effect") or ""),
                 bare_id(e.get("target")))
                for e in (g.get("edges") or []) if isinstance(e, dict)
                and bare_id(e.get("source")) and bare_id(e.get("target"))}

    sets = [_edges(g) for g in graphs]
    n = len(sets)
    # modal edge set: the one most probes reproduce exactly
    counts: dict[frozenset, int] = {}
    for s in sets:
        counts[frozenset(s)] = counts.get(frozenset(s), 0) + 1
    modal = set(max(counts, key=lambda k: counts[k])) if counts else set()

    def _jac(a, b):
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    edge_inst = round(1 - sum(_jac(s, modal) for s in sets) / n, 3)

    # direction: per unordered pair, how many probes disagree on the source
    pair_dir: dict[frozenset, list[str]] = {}
    for s in sets:
        for src, _eff, tgt in s:
            pair_dir.setdefault(frozenset((src, tgt)), []).append(src)
    disagreements = []
    for pair, sources in pair_dir.items():
        if len(pair) < 2:
            continue                       # a self-loop has no direction
        top = max(set(sources), key=sources.count)
        minority = len(sources) - sources.count(top)
        disagreements.append(minority / n)
    dir_inst = round(max(disagreements) if disagreements else 0.0, 3)

    picks = [str(((g.get("suppression_pick") or {}).get("object_id")
                  or (g.get("suppression_pick") or {}).get("threat") or ""))
             for g in graphs]
    pick_inst = (round(1 - picks.count(max(set(picks), key=picks.count)) / n, 3)
                 if any(picks) else 0.0)

    # THE EVIDENCE, not just the score. Every probe graph is already stored;
    # printing 'direction 0.6' throws away which edge flipped and which way.
    # A number cannot say "the model wrote both directions in one answer".
    dir_ev: list[dict] = []
    for pair, sources in pair_dir.items():
        if len(pair) < 2:
            continue
        a, b = sorted(pair)
        for src in (a, b):
            tgt = b if src == a else a
            votes = sum(1 for s_ in sets
                        if any(e[0] == src and e[2] == tgt for e in s_))
            if votes:
                dir_ev.append({"source": src, "target": tgt, "votes": votes,
                               "of": n})
    dir_ev.sort(key=lambda r: -r["votes"])

    # probes that carry BOTH directions of the same pair — self-contradiction
    # inside one answer, which dispersion across probes cannot express.
    both_ways = []
    for i, s_ in enumerate(sets):
        for pair in pair_dir:
            if len(pair) < 2:
                continue
            a, b = sorted(pair)
            if any(e[0] == a and e[2] == b for e in s_) and \
                    any(e[0] == b and e[2] == a for e in s_):
                both_ways.append({"probe": i, "a": a, "b": b})

    # effect wobble per ordered pair: which mechanism, how many times
    eff_ev: dict[str, dict[str, int]] = {}
    for s_ in sets:
        for src, eff, tgt in s_:
            eff_ev.setdefault(f"{src}->{tgt}", {})
            eff_ev[f"{src}->{tgt}"][eff] = (
                eff_ev[f"{src}->{tgt}"].get(eff, 0) + 1)

    pick_ev: dict[str, int] = {}
    for pk in picks:
        if pk:
            pick_ev[pk] = pick_ev.get(pk, 0) + 1

    score = round((edge_inst + dir_inst + pick_inst) / 3, 3)
    out = {"n_probes": n, "requested": n_probes, "score": score,
           "direction_evidence": dir_ev,
           "both_directions_in_one_probe": both_ways,
           "effect_evidence": eff_ev,
           "pick_evidence": pick_ev,
           "edge_set_instability": edge_inst,
           "direction_instability": dir_inst,
           "pick_instability": pick_inst,
           "modal_edges": sorted([list(e) for e in modal]),
           "invented_ids": _invented,
           "flags": [],
           # the probe graphs themselves, so nothing re-probes: asking twice
           # would double the model calls AND give the two controls different
           # probe sequences.
           "graphs": graphs}
    emit("graph_b_uncertainty_ready", **{k: v for k, v in out.items()
                                         if k != "flags"})
    return out


def _resolve_pick_id(threat: Any, detected_ids: set[str]) -> str:
    """Normalize a pick's threat to a FROZEN object_id so the three picks are
    comparable. Handles the two ways picks come in dirty: a state jammed onto
    the id ("house_1·burning") and a stray label. Strips the '·state' suffix,
    then validates against the record. Returns '' if it can't resolve."""
    t = str(threat or "").strip()
    if t in detected_ids:
        return t
    head = bare_id(t)                        # tolerate "house_1·burning"
    if head in detected_ids:
        return head
    return head                              # best-effort; may be off-vocab


def pick_targets(record: Any, graph_a: dict, graph_b: dict,
                 recommendations: list[dict], *, query_fn: QueryFn,
                 on_event: Any = None) -> dict:
    """Step 4. Three ways to choose what to intervene on:
      A_pick   = pick_suppression_framework(Graph A)   [code, coupled]
      B_pick   = Graph B's own suppression_pick         [model, decoupled]
      llm_pick = direct impact ask from recommendations [model, coupled]
    Each pick is resolved to a frozen object_id, and agreement is measured on
    those ids (not raw strings) — so "house_1", "house_1·burning" and a stray
    label all collapse to house_1 and agreement reflects real agreement."""
    from main import pick_suppression_framework  # lazy
    emit = _emitter(on_event)
    detected_ids = {o.object_id for o in record.detected_objects}

    ranked = pick_suppression_framework(graph_a, top_n=3)
    a_pick = (ranked[0] if ranked else {"threat": "", "state": ""})
    b_pick = graph_b.get("suppression_pick", {}) or {"threat": "", "state": ""}

    raw = query_fn(LLM_PICK_PROMPT.format(recs_block=_recs_block(recommendations)))
    llm_pick, notes = parse_pick(raw)
    for n in notes:
        emit("pick_parse_note", note=n)

    # resolve each to a frozen id; stamp it back on the pick for the UI
    for pick in (a_pick, b_pick, llm_pick):
        pick["object_id"] = _resolve_pick_id(pick.get("threat"), detected_ids)

    ids = [p["object_id"] for p in (a_pick, b_pick, llm_pick) if p["object_id"]]
    from collections import Counter
    bloc = max(Counter(ids).values()) if ids else 0
    agreement = round(bloc / 3, 3)
    unanimous = bloc == 3

    picks = {"a_pick": a_pick, "b_pick": b_pick, "llm_pick": llm_pick,
             "agreement": agreement, "unanimous": unanimous,
             "a_ranked": ranked}
    emit("targets_picked",
         a_pick=a_pick["object_id"], b_pick=b_pick["object_id"],
         llm_pick=llm_pick["object_id"], agreement=agreement,
         unanimous=unanimous)
    return picks


def run_evals(record: Any, assessment: Any, recommendations: list[dict],
              graph_a: dict, graph_b: dict, *, on_event: Any = None,
              graphs_a: list | None = None,
              graphs_b: list | None = None) -> dict:
    """Step 5 (Phase 1b, deterministic). Three code evals: the corrected
    conformance breakdown, the WITHIN-A internal alignment (recommendation
    coverage), and the A-vs-B (declared vs structured) alignment. No model —
    both controls compute it identically.

    F19: when probe graphs are supplied, the A-vs-B point estimate is joined by
    its distribution over the probe cross-product."""
    from agentic.evals4 import (ab_alignment, ab_alignment_distribution,
                                conformance_breakdown,
                                explanation_alignment,
                                graph_b_internal_alignment, internal_alignment,
                                set_report)
    emit = _emitter(on_event)
    # F24 — the card is checked FIRST, then its findings are routed into the
    # two reports that already exist: broke-a-rule goes to
    # conformance, surface-against-surface goes to internal alignment. No third
    # score, so the trust weights are untouched. The standalone result is kept
    # for the per-card footers, which need the findings grouped by rank.
    explain = explanation_alignment(record, assessment, recommendations)
    conf = conformance_breakdown(graph_a, graph_b,
                                 card_findings=explain["conformance"],
                                 n_cards=len(recommendations))
    internal = internal_alignment(record, assessment, recommendations,
                                  card_findings=explain["internal_alignment"])
    # F29 — pure aggregation of everything tagged level="set", plus the
    # action-mode rollup. Nothing is re-scored, so this panel existing cannot
    # move a single number.
    across = set_report(internal, conf, explain)
    align = ab_alignment(graph_a, graph_b, record)
    gb_internal = graph_b_internal_alignment(graph_b)
    if graphs_a and graphs_b:
        dist = ab_alignment_distribution(graphs_a, graphs_b, canonical=align)
        align = {**align, "distribution": dist}
        emit("alignment_distribution_ready",
             pairs=dist.get("pairs"),
             structural=dist.get("structural"),
             b_stability=dist.get("b_stability"),
             b_distinct_edge_sets=dist.get("b_distinct_edge_sets"))
    emit("conformance_ready", validity=conf["validity"],
         n_issues=conf["n_issues"], breakdown=conf["breakdown"],
         raw_a_validity=conf["raw_a_validity"],
         raw_b_validity=conf["raw_b_validity"])
    emit("internal_alignment_ready", score=internal["score"],
         n_failures=internal["n_failures"], breakdown=internal["breakdown"])
    emit("alignment_ready", a_fidelity=align["a_fidelity"],
         b_coverage=align["b_coverage"], structural=align["structural"],
         a_only=align["a_only"], b_only=align["b_only"])
    # emitted last so the established event order is unchanged — the twin
    # equivalence tests assert it exactly.
    emit("graph_b_internal_ready", score=gb_internal["score"],
         n_failures=gb_internal["n_failures"],
         breakdown=gb_internal["breakdown"], measured=gb_internal["measured"])
    # F24 — appended after graph_b_internal so the established event order is
    # unchanged; the twin equivalence tests assert it exactly.
    emit("explanation_alignment_ready", score=explain["score"],
         n_failures=explain["n_failures"], breakdown=explain["breakdown"],
         modes=explain["modes"],
         n_conformance=len(explain["conformance"]),
         n_internal=len(explain["internal_alignment"]))
    # appended last so the established event order is only ever added to —
    # the twin equivalence tests assert the stream exactly.
    emit("set_report_ready", n_findings=across["n_findings"],
         n_coverage=len(across["coverage"]), n_pairwise=len(across["pairwise"]),
         modes=across["modes"], mode_verdict=across["mode_verdict"])
    return {"conformance": conf, "internal_alignment": internal,
            "graph_b_internal": gb_internal,
            "explanation_alignment": explain,
            "set_report": across,
            "alignment": align}


def run_card_judge(record: Any, assessment: Any, recommendations: list[dict],
                   explanation: dict, *, judge_fn: Any = None,
                   on_event: Any = None) -> dict:
    """F24 step (advisory). Ask an independent judge whether each card's two
    explanations actually EXPLAIN the action, and whether they mean the same
    thing. OFF unless judge_fn is supplied — the hermetic spine and the twin
    equivalence tests must stay model-free.

    Never returns a score and never touches one. It is display-only precisely
    so it can run during calibration without contaminating the numbers the
    trust weights are being fitted to."""
    if judge_fn is None or not recommendations:
        return {"card_judge": {}}
    from agentic.judge_card import judge_cards
    emit = _emitter(on_event)
    out = judge_cards(recommendations, _scene_block(record, assessment),
                      judge_fn=judge_fn)
    emit("card_judge_ready", n_cards=out["n_cards"], flags=out["flags"],
         n_probes=out["n_probes"], advisory=True)
    return {"card_judge": out}


def run_graph_judge(record: Any, assessment: Any, graph_a: dict, graph_b: dict,
                    alignment: dict, *, judge_fn: Any = None,
                    on_event: Any = None) -> dict:
    """F38 step (advisory). Ask an independent judge the two questions the
    A-vs-B arithmetic cannot answer:

      Q1  the graphs agree on the hazard and name different entities harmed —
          which set is more exposed?
      Q2  both assert the same pair with a different effect — would a responder
          DO something different?

    Everything else about comparing two graphs (conflict, omission, reversed
    direction, invented entity) is a set operation and stays in code.

    OFF unless judge_fn is supplied. Never returns a score."""
    if judge_fn is None:
        return {"graph_judge": {}}
    from agentic.judge_graph import judge_graphs
    emit = _emitter(on_event)
    _ar = {str(getattr(x, "object_id", "")) for x in
           (getattr(assessment, "at_risk", None) or [])}
    out = judge_graphs(graph_a, graph_b,
                       (alignment or {}).get("decomposition") or {},
                       _scene_block(record, assessment), judge_fn=judge_fn,
                       at_risk_ids=_ar, on_event=None)
    if out:
        emit("graph_judge_ready", advisory=True,
             asked_victims=out.get("victims") is not None,
             n_mechanisms=len(out.get("mechanisms") or []))
    return {"graph_judge": out}


def run_runoff_judge(record: Any, assessment: Any, probe_recs: list,
                     graph_b_probes: list, image_path: str = "", *,
                     judge_fn: Any = None, judge_image_fn: Any = None,
                     n_probes: int | None = None, on_event: Any = None) -> dict:
    """JUDGES.md step-1 judge (advisory). When the subject's probes disagreed,
    show the two leading candidates to an independent judge and ask which one
    the record supports — once over the recommendation candidates, once over
    the Graph B candidates. Each verdict is produced by TWIN judges (text-only
    official + image-aware witness); only their agreement is displayed.

    OFF unless judge_fn is supplied — the hermetic spine and the twin
    equivalence tests stay model-free. Never returns a score.

    The emitted events carry both candidates VERBATIM plus both twins'
    verdicts, votes and reasoning: each event is a §9.2 preference record in
    waiting, and the F5 majority guard is applied at carry time, not here."""
    if judge_fn is None:
        return {"runoff_judge": {}}
    from agentic.judge_runoff import (default_image_judge,
                                      runoff_graph_b,
                                      runoff_recommendations)
    emit = _emitter(on_event)
    if judge_image_fn is None:
        judge_image_fn = default_image_judge(image_path)
    def _emit_one(app: str, v: dict) -> None:
        # emitted AS EACH application finishes — the live view labels the
        # judging phase from these, and one emit at the end left 20 minutes
        # of judge work rendering as "folding the trust score" (Sunny).
        emit("runoff_judged", application=app, advisory=True,
             prompt_version=v["prompt_version"],
             text_verdict=v["text"]["verdict"], text_votes=v["text"]["votes"],
             image_verdict=(v["image"] or {}).get("verdict"),
             twins_agree=v["twins_agree"],
             candidate_a=v["candidate_a"], candidate_b=v["candidate_b"],
             text_reasoning=v["text"].get("reasoning", ""),
             image_reasoning=(v["image"] or {}).get("reasoning", ""),
             code_facts=v["code_facts"])

    out: dict = {}
    r = runoff_recommendations(probe_recs or [], record, assessment,
                               judge_fn=judge_fn,
                               judge_image_fn=judge_image_fn,
                               n_probes=n_probes)
    if r:
        out["recommendations"] = r
        _emit_one("recommendations", r)
    g = runoff_graph_b(graph_b_probes or [], record, assessment,
                       judge_fn=judge_fn, judge_image_fn=judge_image_fn,
                       n_probes=n_probes)
    if g:
        out["graph_b"] = g
        _emit_one("graph_b", g)
    return {"runoff_judge": out}


def run_trust(recommendations: list[dict], conformance: dict,
              internal_alignment: dict, alignment: dict, uncertainty: dict,
              picks: dict, *, record: Any = None, assessment: Any = None,
              graph_b: Any = None,
              graph_b_internal: Any = None, graph_b_uncertainty: Any = None,
              on_event: Any = None) -> dict:
    """Step 6 (Phase 1b, deterministic). Fold the objective evals + measured
    uncertainty + pick agreement into ONE trust score with a ranked breakdown
    (global) and per-recommendation trust. Also attaches the SEPARATE
    consequence-to-victims axis per rec (life-safety weight, never folded into
    trust). No model — both controls compute it identically."""
    from agentic.evals4 import compute_trust, consequence_scores  # lazy
    emit = _emitter(on_event)
    cons = consequence_scores(recommendations, assessment, record)
    # F17: a safe scene has no comparand for A-vs-B or the pick routes. Keyed
    # on the declared scene, not on a graph looking empty.
    no_hazards = (str(getattr(assessment, "disaster_scenario", "")) == "No"
                  and not any(o.state_kind == "hazard_bearing"
                              for o in record.detected_objects))
    # F48 — the singular error library, detected before trust so trust can
    # price it. Deterministic; no model, no judge, no ground truth.
    from agentic.errors4 import singular_errors
    sing = singular_errors(record, assessment, recommendations, graph_b)
    for _e in sing:
        emit("singular_error", id=_e["id"], detail=_e["detail"],
             entities=_e["entities"], consequence=_e["consequence"],
             deduction=_e["deduction"])
    trust = compute_trust(recommendations, conformance, internal_alignment,
                          alignment, uncertainty, picks, consequence=cons,
                          no_hazards=no_hazards,
                          graph_b_internal=graph_b_internal,
                          graph_b_uncertainty=graph_b_uncertainty,
                          singular_errors=sing)
    emit("trust_ready", score=trust["score"], band=trust["band"],
         top_contributor=(trust["contributors"][0]["signal"]
                          if trust["contributors"] else None))
    return {"trust": trust}


# ── The durable Stage-4 result ──────────────────────────────────────────

class Stage4Result(BaseModel):
    """What the Phase-1a spine produces. Sub-structures stay as dicts (matching
    Arm A's own shapes) — this wraps them with typed top-level fields, the same
    way AssessmentResult keeps raw_answer as a dict."""
    frame: dict = Field(default_factory=dict)
    recommendations: list[dict] = Field(default_factory=list)
    advisory: list[dict] = Field(default_factory=list)
    graph_a: dict = Field(default_factory=dict)
    graph_b: dict = Field(default_factory=dict)
    picks: dict = Field(default_factory=dict)
    conformance: dict = Field(default_factory=dict)         # corrected breakdown (1b)
    internal_alignment: dict = Field(default_factory=dict)  # within-A coverage (1b)
    # F24 — action / reason / quad held to ONE law, then compared. The prose
    # and the structure are two independent explanations of the same action;
    # this is where they are made to answer for each other.
    explanation_alignment: dict = Field(default_factory=dict)
    # F29 — findings about the SET rather than any one card (coverage gaps,
    # one card repeating another), plus what the set as a whole acts on.
    set_report: dict = Field(default_factory=dict)
    # F24 — the advisory judge. Display-only: it never enters a score, so it
    # can run during calibration without moving what the weights are fitted to.
    card_judge: dict = Field(default_factory=dict)
    # JUDGES.md step 1 — the runoff judge, twin verdicts (text official,
    # image witness) over the two leading probe candidates. Advisory.
    runoff_judge: dict = Field(default_factory=dict)
    # F38 — the graph judge. Two questions only, both advisory: which graph's
    # victims are more exposed, and whether a different effect on the same pair
    # would change what a responder does.
    graph_judge: dict = Field(default_factory=dict)
    alignment: dict = Field(default_factory=dict)           # A-vs-B (1b)
    uncertainty: dict = Field(default_factory=dict)         # measured probe U (1b)
    trust: dict = Field(default_factory=dict)               # folded trust score (1b)
    graph_b_uncertainty: dict = Field(default_factory=dict)  # is B's belief stable?
    graph_b_internal: dict = Field(default_factory=dict)     # does B agree with itself?
    parse_notes: list[str] = Field(default_factory=list)
    # the raw model answer, kept verbatim as evidence — Any, because a
    # malformed answer (a bare string, a list) must still be preserved.
    raw_answer: Any = None


# ── Python control: the straight line, in order ─────────────────────────

def run_stage4(record: Any, assessment: Any, image_path: str = "",
               *, query_fn: QueryFn | None = None,
               probe_fn: QueryFn | None = None, explain_fn: Any = None,
               judge_fn: Any = None,
               n_probes: int = 0, on_event: Any = None) -> Stage4Result:
    """The Phase-1a straight line + measured uncertainty:
    recommend -> probe U -> Graph A -> Graph B -> picks -> evals.
    n_probes>0 turns on channel-2 uncertainty (probes re-ask the recommend
    step at raised temperature). The LangGraph twin (graph_s4.py) lifts each
    step 1:1 and must stay byte-identical to this under scripted models."""
    emit = _emitter(on_event)
    query_fn = query_fn or (lambda p: _query_vlm(p, temperature=0.0))

    rec = run_recommend(record, assessment, query_fn=query_fn, on_event=on_event)
    unc = run_recommend_uncertainty(
        record, assessment, probe_fn=probe_fn, explain_fn=explain_fn,
        n_probes=n_probes,
        canonical_threats=_canonical_threats(rec["recommendations"]),
        on_event=on_event)
    graph_a = build_graph_a(record, assessment, rec["recommendations"],
                            on_event=on_event)
    graph_b = run_graph_b(record, assessment, query_fn=query_fn,
                          on_event=on_event)
    # F19: probe BOTH sides. The A side is free — the probe recommendation sets
    # already exist and were being discarded after reduction — so build a
    # Graph A per probe. The B side costs n_probes model calls.
    graphs_a = build_graph_a_probes(record, assessment, unc["uncertainty"],
                                    on_event=on_event)
    gbu = measure_graph_b_uncertainty(record, assessment, n_probes,
                                      probe_fn=probe_fn, on_event=on_event)
    graphs_b = gbu.get("graphs") or []
    picks = pick_targets(record, graph_a, graph_b, rec["recommendations"],
                         query_fn=query_fn, on_event=on_event)
    evals = run_evals(record, assessment, rec["recommendations"],
                      graph_a, graph_b, on_event=on_event,
                      graphs_a=graphs_a, graphs_b=graphs_b)
    judged = run_card_judge(record, assessment, rec["recommendations"],
                            evals["explanation_alignment"], judge_fn=judge_fn,
                            on_event=on_event)
    gjudged = run_graph_judge(record, assessment, graph_a, graph_b,
                              evals["alignment"], judge_fn=judge_fn,
                              on_event=on_event)
    rjudged = run_runoff_judge(record, assessment,
                               unc.get("probe_recs") or [],
                               (gbu or {}).get("graphs") or [],
                               image_path, judge_fn=judge_fn,
                               on_event=on_event)
    trust = run_trust(rec["recommendations"], evals["conformance"],
                      evals["internal_alignment"], evals["alignment"],
                      unc["uncertainty"], picks, record=record,
                      assessment=assessment,
                      graph_b=graph_b,
                      graph_b_internal=evals.get("graph_b_internal"),
                      graph_b_uncertainty=gbu, on_event=on_event)

    emit("stage_done", stage="recommend")
    return Stage4Result(
        frame=rec["frame"], recommendations=rec["recommendations"],
        advisory=rec["advisory"], graph_a=graph_a, graph_b=graph_b,
        picks=picks, conformance=evals["conformance"],
        internal_alignment=evals["internal_alignment"],
        explanation_alignment=evals.get("explanation_alignment", {}),
        set_report=evals.get("set_report", {}),
        card_judge=judged.get("card_judge", {}),
        graph_judge=gjudged.get("graph_judge", {}),
        runoff_judge=rjudged.get("runoff_judge", {}),
        alignment=evals["alignment"], uncertainty=unc["uncertainty"],
        trust=trust["trust"],
        graph_b_uncertainty=gbu,
        graph_b_internal=evals.get("graph_b_internal") or {},
        parse_notes=rec["recommend_notes"],
        raw_answer=rec["recommend_raw"])
