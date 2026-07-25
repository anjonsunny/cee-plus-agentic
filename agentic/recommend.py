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
        "model": os.getenv("QWEN_MODEL_NAME", "qwen2.5vl:7b"),
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
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

RECOMMEND_PROMPT = """You are an emergency-response analyst. A prior stage has
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
- recommendations: array, one entry per distinct (threat, state) causal logic
  you act on. Do not pad to a fixed count. Each entry:
    - rank: integer (1 = highest priority)
    - action: one specific responder action (no "and"/"then" compounds)
    - reason: plain English of the form "Because {{threat}} is {{state}}, it
      {{effect}} {{affected_objects}}." Every object_id in the reason must also
      appear in the quad, and vice versa.
    - related_object_ids: array of object_ids the reason touches
    - structured_reasoning: the causal quad —
        - threat: object_id of the entity whose state drives the hazard
        - state: that threat's hazard-bearing state
        - effect: exactly one of [{effects}]
        - affected_objects: NON-EMPTY list of object_ids harmed
    - expected_consequence: the immediate result of THIS action if it succeeds
    - remaining_risk: an (object_id, state) pair this action does NOT address;
      must differ across recommendations
    - possible_follow_up_action: the next step after this action
- assumptions_advisory: array (may be empty) of flags about entities you cannot
  see but suspect are present or at risk (e.g. occupants inside a structure).
  Each entry: {{ "suspected": short description, "anchor_object_id": an existing
  object_id it relates to, "cue": why you suspect it, "suggested_action": what a
  responder might do }}. These are ADVISORY only — do not put suspected unseen
  entities into recommendations or the quads.
"""

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

    try:
        recommendations = normalize_recommendations(raw.get("recommendations"))
    except Exception as exc:  # normalize should be tolerant, but never crash
        notes.append(f"recommendations_unparseable({exc})->[]")
        recommendations = []
    if not isinstance(recommendations, list):
        notes.append("recommendations_not_a_list->[]")
        recommendations = []

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
    """A compact, readable dump of the frozen scene for the prompt."""
    lines = ["entities:"]
    for o in record.detected_objects:
        lines.append(f"  {o.object_id}: {o.label}, state={o.state} "
                     f"({o.state_kind})")
    lines.append("threats: " + (", ".join(t.object_id for t in assessment.threats)
                                or "none"))
    lines.append("at_risk: " + (", ".join(f"{a.object_id}·{a.kind}"
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
        scene_block=_scene_block(record, assessment), effects=_EFFECT_LINE)
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

    def _bare(x: Any) -> str:
        return str(x or "").split("·")[0].strip()

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
        t = str((r.get("structured_reasoning", {}) or {}).get("threat", ""))
        t = t.split("·")[0].strip()
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
    for i in range(n_probes):
        try:
            _frame, recs, _adv, _notes = parse_recommend(probe_fn(prompt))
        except Exception as exc:               # transport failure ≠ dispersion
            if emit:
                emit("recommend_probe_error", index=i, error=str(exc))
            continue
        reading = _recommend_reading(recs)
        readings.append(reading)
        if emit:
            emit("recommend_probe", index=i, n_recs=reading["n_recs"],
                 top_threat=reading["top_threat"])
    mu = measure_recommendations(readings, canonical_threats=canonical_threats)
    if n_probes > 0 and not readings:
        mu.score = 1.0
        mu.drivers.append(Driver(
            kind="probes_failed",
            evidence=f"all {n_probes} recommend probes failed in transport",
            action="fix the model endpoint and re-run; treat this advice as "
                   "unmeasured until then"))
    explain(mu, "recommendation set", explain_fn)
    return mu


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
        scene_block=_scene_block(record, assessment), effects=_EFFECT_LINE)
    mu = measure_recommend_uncertainty(prompt, n_probes, probe_fn=probe_fn,
                                       explain_fn=explain_fn,
                                       canonical_threats=canonical_threats,
                                       emit=emit)
    emit("recommend_uncertainty_ready", score=mu.score, n_probes=mu.n_probes,
         drivers=len(mu.drivers))
    return {"uncertainty": mu.model_dump()}


def build_graph_a(record: Any, assessment: Any, recommendations: list[dict],
                  *, on_event: Any = None) -> dict:
    """Step 2. Reassemble the recommendation quads into Graph A — CODE, via
    Arm A's build_causal_graph. Nodes = all frozen entities; edges = one per
    quad target. build_causal_graph also computes intervention_candidates."""
    from main import build_causal_graph  # lazy
    emit = _emitter(on_event)
    graph_a = build_causal_graph(
        _detected_dicts(record),
        _threat_dicts(record, assessment),
        recommendations,
        _at_risk_dicts(assessment))
    emit("graph_a_built",
         n_nodes=len(graph_a.get("nodes", [])),
         n_edges=len(graph_a.get("edges", [])),
         warnings=graph_a.get("graph_warnings", []))
    return graph_a


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
    return (f"{GRAPH_B_PROMPT}\n\n"
            f"Inferred-entity policy:\n{GRAPH_B_INFERRED_DENIED}\n\n"
            f"Caption:\n{record.caption or 'N/A'}\n\n"
            f"Prior analysis (detected_objects + threats only — "
            f"recommendations withheld):\n{json.dumps(context, indent=2)}")


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


def _resolve_pick_id(threat: Any, detected_ids: set[str]) -> str:
    """Normalize a pick's threat to a FROZEN object_id so the three picks are
    comparable. Handles the two ways picks come in dirty: a state jammed onto
    the id ("house_1·burning") and a stray label. Strips the '·state' suffix,
    then validates against the record. Returns '' if it can't resolve."""
    t = str(threat or "").strip()
    if t in detected_ids:
        return t
    head = t.split("·")[0].strip()          # tolerate "house_1·burning"
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
              graph_a: dict, graph_b: dict, *, on_event: Any = None) -> dict:
    """Step 5 (Phase 1b, deterministic). Three code evals: the corrected
    conformance breakdown, the WITHIN-A internal alignment (recommendation
    coverage), and the A-vs-B (declared vs structured) alignment. No model —
    both controls compute it identically."""
    from agentic.evals4 import (ab_alignment, conformance_breakdown,
                                internal_alignment)
    emit = _emitter(on_event)
    conf = conformance_breakdown(graph_a, graph_b)
    internal = internal_alignment(record, assessment, recommendations)
    align = ab_alignment(graph_a, graph_b)
    emit("conformance_ready", validity=conf["validity"],
         n_issues=conf["n_issues"], breakdown=conf["breakdown"],
         raw_a_validity=conf["raw_a_validity"],
         raw_b_validity=conf["raw_b_validity"])
    emit("internal_alignment_ready", score=internal["score"],
         n_failures=internal["n_failures"], breakdown=internal["breakdown"])
    emit("alignment_ready", a_fidelity=align["a_fidelity"],
         b_coverage=align["b_coverage"], structural=align["structural"],
         a_only=align["a_only"], b_only=align["b_only"])
    return {"conformance": conf, "internal_alignment": internal,
            "alignment": align}


def run_trust(recommendations: list[dict], conformance: dict,
              internal_alignment: dict, alignment: dict, uncertainty: dict,
              picks: dict, *, record: Any = None, assessment: Any = None,
              on_event: Any = None) -> dict:
    """Step 6 (Phase 1b, deterministic). Fold the objective evals + measured
    uncertainty + pick agreement into ONE trust score with a ranked breakdown
    (global) and per-recommendation trust. Also attaches the SEPARATE
    consequence-to-victims axis per rec (life-safety weight, never folded into
    trust). No model — both controls compute it identically."""
    from agentic.evals4 import compute_trust, consequence_scores  # lazy
    emit = _emitter(on_event)
    cons = consequence_scores(recommendations, assessment, record)
    trust = compute_trust(recommendations, conformance, internal_alignment,
                          alignment, uncertainty, picks, consequence=cons)
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
    alignment: dict = Field(default_factory=dict)           # A-vs-B (1b)
    uncertainty: dict = Field(default_factory=dict)         # measured probe U (1b)
    trust: dict = Field(default_factory=dict)               # folded trust score (1b)
    parse_notes: list[str] = Field(default_factory=list)
    # the raw model answer, kept verbatim as evidence — Any, because a
    # malformed answer (a bare string, a list) must still be preserved.
    raw_answer: Any = None


# ── Python control: the straight line, in order ─────────────────────────

def run_stage4(record: Any, assessment: Any, image_path: str = "",
               *, query_fn: QueryFn | None = None,
               probe_fn: QueryFn | None = None, explain_fn: Any = None,
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
    picks = pick_targets(record, graph_a, graph_b, rec["recommendations"],
                         query_fn=query_fn, on_event=on_event)
    evals = run_evals(record, assessment, rec["recommendations"],
                      graph_a, graph_b, on_event=on_event)
    trust = run_trust(rec["recommendations"], evals["conformance"],
                      evals["internal_alignment"], evals["alignment"],
                      unc["uncertainty"], picks, record=record,
                      assessment=assessment, on_event=on_event)

    emit("stage_done", stage="recommend")
    return Stage4Result(
        frame=rec["frame"], recommendations=rec["recommendations"],
        advisory=rec["advisory"], graph_a=graph_a, graph_b=graph_b,
        picks=picks, conformance=evals["conformance"],
        internal_alignment=evals["internal_alignment"],
        alignment=evals["alignment"], uncertainty=unc["uncertainty"],
        trust=trust["trust"], parse_notes=rec["recommend_notes"],
        raw_answer=rec["recommend_raw"])
