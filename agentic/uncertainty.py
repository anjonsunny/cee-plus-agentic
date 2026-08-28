"""Measured uncertainty (channel 2) with CAUSAL, ACTIONABLE explanations.

TWO CHANNELS, NEVER MERGED (agreed 2026-07-21)
==============================================
- Channel 1, self-reported confidence: the model's own number, recorded
  as SUBJECT DATA (its calibration is something we measure, not trust).
  Lives in the stage contracts, not here.
- Channel 2, measured uncertainty: computed by the instrument, never
  asked. THIS module. For verdict stages: dispersion of K probe re-asks
  (same prompt, temperature up) around the canonical temp-0 verdict.
  For perception: a structural score composed from evidence already in
  the record (box provenance, vocab escapes, STOOD tickets).

EVERY VALUE SHIPS WITH ITS CAUSES (Sunny: "so that it is actionable")
=====================================================================
A bare 0.4 is a shrug. Each uncertainty value carries `drivers`: the
deterministic, code-found causes (WHICH field scattered, HOW it split,
WHICH structural flags fired), each with the evidence and a concrete
suggested action. An optional explainer LLM (the DIALOGUE model, not the
subject — the subject must not narrate its own reliability) composes the
drivers into a short causal narrative; when no LLM is available the
deterministic drivers ARE the explanation. The narrative may only
restate drivers — it introduces no new claims.

CALIBRATION (Sunny: the six frozen scenes act as GT for now)
============================================================
Structural-factor weights below are PRIORS, marked for calibration:
probe runs on the six scenes give real dispersion numbers, and weights
get tuned so the structural score tracks measured dispersion. Until that
run, treat the score as ordinal (higher = less trustworthy), not
calibrated probability.
"""
from __future__ import annotations

import os

from agentic import models as _models
from collections import Counter
from typing import Any, Callable, Optional  # noqa: F401

from pydantic import BaseModel, Field

# ── Dispersion math (deterministic, tested) ─────────────────────────────


def agreement(values: list[Any]) -> float:
    """Modal agreement: fraction of probes voting for the most common
    value. 1.0 = unanimous, 1/n = maximal scatter. Empty list -> 1.0
    (nothing measured is not the same as disagreement; n_probes records
    that nothing was measured)."""
    if not values:
        return 1.0
    _, top = Counter(values).most_common(1)[0]
    return top / len(values)


def spread(levels: list[int]) -> dict[str, float]:
    if not levels:
        return {"min": 0.0, "max": 0.0, "std": 0.0}
    n = len(levels)
    mean = sum(levels) / n
    var = sum((x - mean) ** 2 for x in levels) / n
    return {"min": float(min(levels)), "max": float(max(levels)),
            "std": round(var ** 0.5, 3)}


# ── The uncertainty record ──────────────────────────────────────────────


class Driver(BaseModel):
    """One code-found cause of uncertainty: what, the evidence, and what
    to do about it. This is the causal explanation's skeleton."""
    kind: str
    evidence: str
    action: str


class MeasuredUncertainty(BaseModel):
    n_probes: int
    scenario_agreement: float = 1.0
    type_agreement: float = 1.0
    bucket_agreement: float = 1.0
    level: dict[str, float] = Field(default_factory=dict)   # min/max/std
    score: float = 0.0            # 0 = probes unanimous, 1 = full scatter
    drivers: list[Driver] = Field(default_factory=list)
    explanation: str = ""         # narrative over drivers (LLM or fallback)
    explainer: str = "deterministic"   # "llm" when a narrator composed it
    # GRANULAR uncertainty (Sunny 2026-07-22: per-verdict-piece U so the
    # SOURCE of instability is pinpointable, not just a global number):
    #   fields:   {"disaster_scenario": {"u": 0.0, "evidence": "..."} ...}
    #   threats:  {"house_1": {"u": 0.0, "votes": "5/5"} ...}
    #   at_risk:  {"person_1": {"u": 0.4, "votes": "3/5"} ...}
    granular: dict[str, Any] = Field(default_factory=dict)
    # Distinct probe CANDIDATES ranked by votes (same scenario + bucket +
    # threat set + at-risk set = one candidate). Feeds the runoff.
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    # Top-2 runoff verdict (Sunny 2026-07-22: when U is high, blind-judge
    # the two most-voted readings; winner becomes reflection CONTEXT,
    # never the installed answer).
    runoff: Optional[dict[str, Any]] = None


def probe_candidates(answers: list[dict[str, Any]],
                     coarse: bool = False) -> list[dict[str, Any]]:
    """Group FULL probe answers into distinct candidates, ranked by
    votes. Fine key: scenario + bucket + threat set + at-risk set (reason
    wording never splits a candidate). When EVERY fine candidate is a
    singleton (full scatter — a top-2 of arbitrary singletons would be a
    coin-flip runoff), regroup COARSELY on scenario + bucket so the
    runoff compares genuinely opposed READINGS, not noise."""
    groups: dict[Any, dict[str, Any]] = {}
    for p in answers:
        if coarse:
            key: Any = (p.get("disaster_scenario"), p.get("severity_bucket"))
        else:
            key = (p.get("disaster_scenario"), p.get("severity_bucket"),
                   frozenset(t.get("object_id") for t in p.get("threats") or []),
                   frozenset((r.get("object_id"), r.get("kind"))
                             for r in p.get("at_risk") or []))
        g = groups.setdefault(key, {"votes": 0, "answer": p})
        g["votes"] += 1
    ranked = sorted(groups.values(), key=lambda g: -g["votes"])
    if not coarse and ranked and ranked[0]["votes"] == 1 and len(ranked) > 2:
        return probe_candidates(answers, coarse=True)
    return ranked


# ── Verdict-stage measurement (Stage 2 now; later verdict stages too) ───


# Disaster-type FAMILIES: the six-scene calibration run showed the raw
# type string wobbles cosmetically ("house fire" / "fire" / "structural
# fire" — same event, different words) which overstated measured type-U
# everywhere. Agreement is measured at family level; the raw split stays
# in the driver evidence. Keyword map, first match wins.
TYPE_FAMILIES = (
    ("fire", ("fire", "blaze", "burning")),
    ("water", ("drown", "flood", "water", "pool", "swim")),
    ("hazmat", ("chemical", "spill", "fuel", "leak", "hazmat", "gas",
                "contamin")),
    ("collapse", ("collapse", "structural failure", "cave-in", "rubble")),
    ("storm", ("storm", "tornado", "hurricane", "wind")),
    ("none", ("n/a", "none")),
)


def type_family(t: str) -> str:
    low = " ".join(str(t or "").lower().replace("-", " ").split())
    if not low:
        return "none"
    for fam, keys in TYPE_FAMILIES:
        if any(k in low for k in keys):
            return fam
    return low                      # unknown types stay themselves


def _norm_type(t: str) -> str:
    """Fold to the disaster-type FAMILY before measuring agreement:
    'House Fire' vs 'structural fire' is wording, not disagreement;
    'fire' vs 'flood' is real."""
    return type_family(t)


def measure_verdicts(probes: list[dict[str, Any]]) -> MeasuredUncertainty:
    """Channel-2 for a verdict stage: dispersion across parsed probe
    verdicts (each: scenario, disaster_type, disaster_level, bucket).
    Drivers name the scattered field, show the split, and prescribe."""
    n = len(probes)
    scen = [p["scenario"] for p in probes]
    typ = [_norm_type(p["disaster_type"]) for p in probes]
    buck = [p["bucket"] for p in probes]
    lvl = [int(p["level"]) for p in probes]

    mu = MeasuredUncertainty(
        n_probes=n,
        scenario_agreement=round(agreement(scen), 3),
        type_agreement=round(agreement(typ), 3),
        bucket_agreement=round(agreement(buck), 3),
        level=spread(lvl),
    )
    mu.score = round(1 - (mu.scenario_agreement + mu.type_agreement
                          + mu.bucket_agreement) / 3, 3)

    def split(vals: list[Any]) -> str:
        return ", ".join(f"{v}×{c}" for v, c in Counter(vals).most_common())

    if mu.scenario_agreement < 1.0:
        mu.drivers.append(Driver(
            kind="scenario_flip",
            evidence=f"probes split on disaster_scenario: {split(scen)}",
            action="verdict unstable at its root — run the state-coherence "
                   "checks (S2/S3) against Stage 1 states before trusting "
                   "any downstream use of this verdict"))
    if mu.type_agreement < 1.0:
        raw = split([str(p["disaster_type"]) for p in probes])
        mu.drivers.append(Driver(
            kind="type_split",
            evidence=f"probes split on disaster-type FAMILY: {split(typ)} "
                     f"(raw wording: {raw})",
            action="the declared entity states do not pin down the disaster "
                   "type — candidate for caption clarification or (once "
                   "built) contextual re-perception of the ambiguous region"))
    if mu.bucket_agreement < 1.0:
        mu.drivers.append(Driver(
            kind="bucket_split",
            evidence=f"probes straddle severity buckets: {split(buck)} "
                     f"(levels {mu.level['min']:.0f}-{mu.level['max']:.0f})",
            action="severity sits on a bucket boundary — Stage 12 must "
                   "treat escalation/de-escalation readings on this scene "
                   "as within-noise unless the shift crosses TWO buckets"))
    elif mu.level.get("std", 0) > 1.0:
        mu.drivers.append(Driver(
            kind="level_wobble",
            evidence=f"levels scatter {mu.level['min']:.0f}-"
                     f"{mu.level['max']:.0f} (std {mu.level['std']}) within "
                     f"one bucket",
            action="raw 0-10 number is noisy but the bucket is stable — "
                   "safe to use the bucket, do not lean on the raw level"))
    # Per-field granular block (entity granularity is added by
    # measure_merged when probes carry threat/at-risk lists).
    def _split(vals: list[Any]) -> str:
        return ", ".join(f"{x}×{c}" for x, c in Counter(vals).most_common())

    mu.granular["fields"] = {
        "disaster_scenario": {"u": round(1 - mu.scenario_agreement, 3),
                              "evidence": _split(scen)},
        "disaster_type": {"u": round(1 - mu.type_agreement, 3),
                          "evidence": _split(typ)},
        "disaster_level": {"u": round(1 - mu.bucket_agreement, 3),
                           "evidence": f"levels {_split(lvl)} -> buckets "
                                       f"{_split(buck)}"},
    }
    return mu


def measure_merged(probes: list[dict[str, Any]]) -> MeasuredUncertainty:
    """Granular channel-2 for the MERGED assessment: everything
    measure_verdicts does, PLUS per-entity membership stability for the
    threat and at-risk lists. Each probe dict additionally carries
    threat_ids: [..] and at_risk_ids: [..].

    Membership U for entity e = 1 - (probes citing e / all probes): an
    entity in every probe's list is stable (U=0); one that flickers in
    and out is the pinpointed source of instability. The GLOBAL score
    averages the three field components with every entity component, so
    one flickering entity visibly moves the total."""
    mu = measure_verdicts(probes)
    n = len(probes)
    if n == 0:
        return mu

    def membership(key: str) -> dict[str, dict[str, Any]]:
        seen: Counter = Counter()
        for p in probes:
            for oid in set(p.get(key) or []):
                seen[str(oid)] += 1
        return {oid: {"u": round(1 - c / n, 3), "votes": f"{c}/{n}"}
                for oid, c in sorted(seen.items())}

    threats = membership("threat_ids")
    at_risk = membership("at_risk_ids")
    mu.granular["threats"] = threats
    mu.granular["at_risk"] = at_risk

    for label, table in (("threat", threats), ("at_risk", at_risk)):
        for oid, g in table.items():
            if g["u"] <= 0.0:
                continue
            mu.drivers.append(Driver(
                kind=f"{label}_membership_split",
                evidence=f"{oid} appears in only {g['votes']} probe "
                         f"{label} lists",
                action=f"reflect on {oid} specifically: quote its state, "
                       f"the geometry hint that nominates it, and ask for "
                       f"a decision with cited evidence (rules G1-G4)"))
    components = ([1 - mu.scenario_agreement, 1 - mu.type_agreement,
                   1 - mu.bucket_agreement]
                  + [g["u"] for g in threats.values()]
                  + [g["u"] for g in at_risk.values()])
    mu.score = round(sum(components) / len(components), 3)
    return mu


# ── Recommendation-stage measurement (Stage 4) ──────────────────────────


def _split_counter(vals: list[Any]) -> str:
    return ", ".join(f"{v}×{c}" for v, c in Counter(vals).most_common())


def measure_recommendations(readings: list[dict[str, Any]],
                            canonical_threats: set | None = None
                            ) -> MeasuredUncertainty:
    """Channel-2 for the recommendation step: re-ask the SAME recommend prompt
    K times at probe temperature and measure how stable the ADVICE is. Unlike a
    verdict, recommendations have no fixed identity across re-asks, so — exactly
    like the merged assessment's threat/at-risk membership — stability is
    measured at the entity and claim level, not per-slot.

    Each reading (one probe) carries:
      top_threat        the rank-1 recommendation's threat id ('' if none)
      n_recs            how many recommendations the probe produced
      threat_ids        threat id per recommendation (the quad.threat's)
      affected_ids      every affected object id across the recommendations
      effect_by_threat  {threat: effect} — the causal mechanism it chose
      edges             [(threat, effect, affected_tuple)] — the causal claims

    Global score = mean over: the top-priority-target flip, the count wobble,
    each entity's threat-membership U, each entity's affected-membership U, and
    each threat's effect-choice U — so one flickering entity or one flip-flopped
    mechanism visibly moves the total (Sunny: the source must be pinpointable)."""
    n = len(readings)
    mu = MeasuredUncertainty(n_probes=n)
    if n == 0:
        return mu

    top = [r.get("top_threat", "") for r in readings]
    counts = [int(r.get("n_recs", 0)) for r in readings]
    top_agree = agreement([t for t in top if t]) if any(top) else 1.0
    count_agree = agreement(counts)

    def membership(key: str) -> dict[str, dict[str, Any]]:
        seen: Counter = Counter()
        for r in readings:
            for oid in set(r.get(key) or []):
                seen[str(oid)] += 1
        return {oid: {"u": round(1 - c / n, 3), "votes": f"{c}/{n}"}
                for oid, c in sorted(seen.items())}

    # threat-membership stability. When the CANONICAL threats are known, report
    # only those — every one of them, even those never re-produced (u=1.0) — so
    # the panel isn't cluttered with off-list noise (a probe that once mislabels
    # a victim as a threat). Without them, fall back to every seen threat.
    threat_seen: Counter = Counter()
    for r in readings:
        for oid in set(r.get("threat_ids") or []):
            threat_seen[str(oid)] += 1
    if canonical_threats is not None:
        threats = {t: {"u": round(1 - threat_seen.get(t, 0) / n, 3),
                       "votes": f"{threat_seen.get(t, 0)}/{n}"}
                   for t in sorted(canonical_threats)}
    else:
        threats = membership("threat_ids")
    affected = membership("affected_ids")

    # Sunny (2026-08-28, run ui_3e6c6d2a): the flat mean priced a one-off
    # stray (an entity a single probe mentioned once) the same as a real
    # flip in the advice's core. Weight each claim by how many probes
    # assert it: a majority-backed claim carries full weight — its wobble
    # is a core wobble — while a 1-of-n stray counts at quarter weight.
    def _claim_weight(c: int) -> float:
        return 1.0 if c > n / 2 else max(c / n, 0.25)

    for group in (threats, affected):
        for g in group.values():
            c = int(str(g.get("votes", "0/0")).split("/")[0])
            g["w"] = round(_claim_weight(c), 2)

    # effect-choice stability per threat: over the probes where a threat is
    # named, does the model commit to ONE causal mechanism for it? (This is the
    # reason↔quad claim's run-to-run stability — the effect flip we care about.)
    # Restricted to canonical threats too, for the same declutter reason.
    effect_votes: dict[str, list[str]] = {}
    for r in readings:
        for t, eff in (r.get("effect_by_threat") or {}).items():
            if canonical_threats is not None and str(t) not in canonical_threats:
                continue
            effect_votes.setdefault(str(t), []).append(str(eff))
    effects: dict[str, dict[str, Any]] = {}
    for t, effs in sorted(effect_votes.items()):
        a = agreement(effs)
        effects[t] = {"u": round(1 - a, 3),
                      "votes": f"{Counter(effs).most_common(1)[0][1]}/{len(effs)}",
                      "evidence": _split_counter(effs),
                      "w": round(_claim_weight(len(effs)), 2)}

    mu.granular = {
        "fields": {
            "top_priority_target": {"u": round(1 - top_agree, 3),
                                    "evidence": _split_counter([t or "∅"
                                                                for t in top])},
            "recommendation_count": {"u": round(1 - count_agree, 3),
                                     "evidence": _split_counter(counts)},
        },
        "threats": threats,
        "affected": affected,
        "effects": effects,
    }

    # distinct advice candidates (same causal-claim set = one candidate), ranked
    groups: dict[Any, dict[str, Any]] = {}
    for r in readings:
        key = frozenset((e[0], e[1], tuple(e[2]))
                        for e in (r.get("edges") or []))
        g = groups.setdefault(key, {"votes": 0, "edges": None})
        g["votes"] += 1
        if g["edges"] is None:
            g["edges"] = [[e[0], e[1], list(e[2])] for e in (r.get("edges") or [])]
    mu.candidates = sorted(groups.values(), key=lambda g: -g["votes"])

    # weighted mean: the two core fields at full weight, every claim at
    # its vote weight — so "core stable, fringe noisy" stops reading as
    # flat instability.
    weighted = ([(1 - top_agree, 1.0), (1 - count_agree, 1.0)]
                + [(g["u"], g["w"]) for g in threats.values()]
                + [(g["u"], g["w"]) for g in affected.values()]
                + [(g["u"], g["w"]) for g in effects.values()])
    wsum = sum(w for _, w in weighted)
    mu.score = (round(sum(u * w for u, w in weighted) / wsum, 3)
                if wsum else 0.0)

    # drivers — each unstable piece, with the evidence and what to do
    if top_agree < 1.0:
        mu.drivers.append(Driver(
            kind="top_target_flip",
            evidence=f"the #1-priority target changes across re-asks: "
                     f"{_split_counter([t or '∅' for t in top])}",
            action="the single highest-priority intervention is not stable — "
                   "treat the top pick as low-confidence; the pick-agreement "
                   "check and the trust score should weigh this heavily"))
    if count_agree < 1.0:
        mu.drivers.append(Driver(
            kind="rec_count_wobble",
            evidence=f"number of recommendations varies: {_split_counter(counts)}",
            action="the model does not settle on how many recommendations to "
                   "give — compare the distinct recommendation sets below "
                   "before trusting any single run's set"))
    for oid, g in threats.items():
        if g["u"] > 0.0:
            mu.drivers.append(Driver(
                kind="threat_membership_split",
                evidence=f"{oid} is named a threat in only {g['votes']} probes",
                action=f"reflect on {oid}: is it truly a hazard here? its "
                       f"recommendation is on unstable ground"))
    for t, g in effects.items():
        if g["u"] > 0.0:
            mu.drivers.append(Driver(
                kind="effect_choice_unstable",
                evidence=f"{t}'s causal mechanism splits: {g['evidence']}",
                action=f"the model will not commit to ONE effect for {t} — its "
                       f"reason↔quad claim is unreliable; flag that "
                       f"recommendation's trust and consider a semantic check"))
    return mu


# ── Perception structural score (Stage 1, always-on, cheap) ─────────────

# PRIORS pending calibration against the six-scene probe run.
STRUCTURAL_FACTORS = {
    "fallback_box":   (0.25, "box did not come from the detector "
                             "(vlm/sam fallback)"),
    "low_box_conf":   (0.25, "detector confidence below 0.5"),
    "vocab_extension": (0.20, "label needed the vocabulary escape hatch"),
    "label_note":     (0.10, "label was coerced by the canonicalizer"),
    "unknown_state":  (0.20, "state is outside every known state kind"),
    "stood_ticket":   (0.25, "entity involved in an unresolved (STOOD) "
                             "repair ticket"),
}


def entity_uncertainty(entity: Any,
                       stood_ids: set[str] | None = None) -> dict[str, Any]:
    """Structural uncertainty for one DetectedObject (or its dict form):
    score in [0,1] plus the factors that caused it, each with evidence.
    Purely from recorded provenance — no model call, runs everywhere."""
    e = entity if isinstance(entity, dict) else entity.model_dump()
    stood_ids = stood_ids or set()
    factors: list[dict[str, Any]] = []

    def hit(key: str, evidence: str) -> None:
        weight, generic = STRUCTURAL_FACTORS[key]
        factors.append({"factor": key, "weight": weight,
                        "evidence": evidence or generic})

    if e.get("box_source") not in ("dino_matched",):
        hit("fallback_box", f"box_source={e.get('box_source')!r}")
    conf = e.get("box_confidence")
    if conf is not None and conf < 0.5:
        hit("low_box_conf", f"box_confidence={conf}")
    if e.get("vocab_extension"):
        hit("vocab_extension", f"label={e.get('label')!r} via escape hatch")
    if e.get("label_note"):
        hit("label_note", f"note: {e.get('label_note')}")
    if e.get("state_kind") == "unknown":
        hit("unknown_state", f"state={e.get('state')!r} has no known kind")
    if e.get("object_id") in stood_ids:
        hit("stood_ticket", f"{e.get('object_id')} in a STOOD ticket")

    return {"score": round(min(1.0, sum(f["weight"] for f in factors)), 3),
            "factors": factors}


def stood_entity_ids(repair_trace: dict[str, Any] | None) -> set[str]:
    """object_ids implicated in violations that survived Loop 1 (STOOD).
    Violations reference entities loosely (index or raw text); we collect
    whatever id-shaped evidence the trace carries."""
    out: set[str] = set()
    for rnd in (repair_trace or {}).get("rounds", []) or []:
        for v in rnd.get("stood", []) or []:
            oid = v.get("object_id")
            if oid:
                out.add(str(oid))
    return out


# ── The causal narrative (optional LLM over the drivers) ────────────────

ExplainFn = Callable[[str], str]

_EXPLAIN_PROMPT = """You are explaining a MEASURED uncertainty value to a
researcher. Below are the code-found causes (drivers) with evidence, from
{n} probe re-asks of a scene-assessment verdict.

Canonical verdict: {verdict}
Uncertainty score: {score} (0 = unanimous probes, 1 = full scatter)
Drivers:
{drivers}

Write 2-3 sentences: state WHAT is uncertain, WHY (cite the driver
evidence), and WHAT ACTION would reduce it. Only use the drivers above —
do not invent causes. Phrase claims as measurements ("probes split"),
never as facts about the real scene."""


def _ollama_explain(prompt: str) -> str:
    """Narrator = the DIALOGUE model (instrument side), never the subject:
    the model under evaluation must not narrate its own reliability."""
    import requests

    api_url = os.getenv("QWEN_API_URL",
                        "http://localhost:11434/v1/chat/completions")
    r = requests.post(api_url, json={
        "model": _models.DIALOGUE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }, timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


def explain(mu: MeasuredUncertainty, verdict_line: str,
            explain_fn: Optional[ExplainFn] = None) -> MeasuredUncertainty:
    """Attach the causal narrative. Deterministic fallback ALWAYS works
    (drivers joined verbatim); the LLM only rewords the same causes. Any
    explainer failure falls back silently — an explanation outage must
    never break a pipeline run."""
    if not mu.drivers:
        mu.explanation = (f"probes unanimous across all {mu.n_probes} "
                          f"re-asks; no measured instability")
        return mu
    fallback = " ".join(f"[{d.kind}] {d.evidence} -> {d.action}."
                        for d in mu.drivers)
    if explain_fn is None:
        mu.explanation = fallback
        return mu
    try:
        drivers_txt = "\n".join(f"- {d.kind}: {d.evidence}\n  action: "
                                f"{d.action}" for d in mu.drivers)
        mu.explanation = explain_fn(_EXPLAIN_PROMPT.format(
            n=mu.n_probes, verdict=verdict_line, score=mu.score,
            drivers=drivers_txt)) or fallback
        mu.explainer = "llm"
    except Exception:
        mu.explanation = fallback
    return mu
