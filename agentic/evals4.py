"""Stage 4 — evals (Phase 1b): conformance breakdown + A-vs-B alignment.

WHY THIS EXISTS
===============
Arm A's conformance metric (1 − violations/edges) is misleading on the VLM's
output (audit F15): it SATURATES (5 and 40 violations score the same once past
the edge count), divides node-level violations by EDGE count, DOUBLE-COUNTS (one
error trips 2-3 rules), and is SEVERITY-BLIND. Because Arm A is frozen (iron rule
1, import only), we cannot fix its rules — so this module imports the raw
violation LIST from Arm A's checker and computes a CORRECTED Arm B read on top:

  - dedupe: one issue per (graph, entity, category) — kills the double-count
  - severity-weight: cosmetic "no-effect" rules don't dent trust
  - category breakdown: the violations grouped by pattern, so the panel shows
    WHERE and WHY the model can't be trusted — not one saturated number
  - non-saturating validity: normalized by nodes+edges, degrades gracefully

Arm A's RAW frozen numbers are recorded alongside (raw_a_validity /
raw_b_validity) so the three arms stay comparable.

A-vs-B (declared Graph B vs structured Graph A): the clean precision/recall pair
is kept (a_fidelity, b_coverage), and ONE structural definition is used
(topological multiset on de-duplicated edges — the audit found the three tiers
use three different, non-comparable definitions).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Rule → (category, severity) ─────────────────────────────────────────
#
# Category groups violations into the model's failure PATTERNS (what the
# panel surfaces). Severity: 2 = serious (fabrication / role error),
# 1 = structural, 0 = cosmetic (dropped from the trust penalty). Grounded in
# Arm A's own consequence grades (audit F15): the "no-effect" rules are 0.

RULE_META: dict[str, tuple[str, int]] = {
    # made-up / illegal vocabulary
    "effect_not_in_vocabulary": ("made-up vocabulary", 2),
    "fluid_encoded_as_state": ("made-up vocabulary", 1),
    # hazard-vs-victim role mix-ups
    "hazard_flag_state_mismatch": ("role mix-up", 2),
    "hazardous_and_at_risk": ("role mix-up", 2),
    "distress_state_on_non_living": ("role mix-up", 1),
    "edge_from_non_hazardous": ("role mix-up", 2),
    "may_harm_hazardous_target": ("role mix-up", 1),
    # fabrications — pointing at nothing
    "unresolved_endpoint": ("fabrication", 2),
    # internal inconsistency
    "via_state_mismatch": ("inconsistency", 1),
    "via_state_not_hazard_bearing": ("inconsistency", 1),
    # wrong effect choice
    "fluid_wrong_effect_for_person": ("effect choice", 1),
    "uncoupled_obstruction": ("effect choice", 1),
    "spread_between_hazards": ("effect choice", 1),
    "may_spread_to_person": ("effect choice", 1),
    # structure / edge shape
    # F18: self_loop_not_worsens and redundant_self_loop are RETIRED. Those
    # rules fired on placeholders standing in for a missing half of a causal
    # claim; the placeholders no longer exist as edges (annotate_one_ended
    # re-files them as node flags), so nothing is left for them to fire on.
    # Kept in the map at severity 0 so an Arm A run that still emits them is
    # recorded rather than dropped silently.
    "self_loop_not_worsens": ("structure", 0),
    "one_way_worsens": ("structure", 1),
    "redundant_self_loop": ("structure", 0),
    # F18: the one-ended claims. These are OBSERVATIONS about the shape of the
    # graph — code derives them, the model declares nothing — so they are
    # severity 0 and never touch trust. The escalating rung that once lived
    # here was dropped (Sunny, 2026-07-28): scoring it double-counted with
    # internal_alignment's "at-risk used as a threat", and an accusation the
    # model has no channel to answer needs a disposition ladder and a
    # threat:null field to discharge it. The real defect is charged once, at
    # the recommendation layer, where the model actually wrote it.
    "unattached_hazard": ("coverage gap", 0),
    "unattributed_victim": ("coverage gap", 0),
    # coverage gaps
    "smoke_superset_violation": ("coverage gap", 2),
    "hazardous_node_no_edges": ("coverage gap", 1),
    # cosmetic — no operational effect, must NOT tank trust
    "redundant_instancing": ("cosmetic", 0),
    "node_budget_exceeded": ("cosmetic", 0),
}

# F23: the previous pattern had no left anchor, so on a multi-word id it
# matched only the final segment — 'tanker_truck_1' read as 'truck_1',
# 'lifeguard_chair_1' as 'chair_1'. Every reason naming a multi-word entity
# was then reported as not naming it. C_tanker's whole internal-alignment
# penalty was this. Anchored form, same one dialogue.py already uses;
# 'presumed_<noun>_in_<id>' still matches whole.
_ID_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d+\b")


def _meta(rule: str) -> tuple[str, int]:
    return RULE_META.get(rule, ("other", 1))


def _entity_of(detail: str) -> str:
    """Best-effort: the entity a violation concerns — the first object_id in
    its detail string (the source, for edge rules). '' if none found."""
    m = _ID_RE.search(str(detail or ""))
    return m.group(0) if m else ""


def _size(graph: dict) -> int:
    return len(graph.get("nodes") or []) + len(graph.get("edges") or [])


def _raw_validity(n_violations: int, n_edges: int, factor: float) -> float:
    """Arm A's frozen formula, reproduced for comparability (A uses factor 0.5,
    B uses 1.0)."""
    ratio = min(1.0, n_violations / max(1, n_edges))
    return round(1.0 - factor * ratio, 3)


def _one_ended_issues(graph: dict, graph_name: str) -> list[dict[str, Any]]:
    """F18. Read the node annotations back out as issues.

    Observations about the graph's shape, not accusations: severity 0, no
    score impact. The real defect — a victim the model used as a threat, a
    hazard it never acted through — is charged once, at the recommendation
    layer, by internal_alignment."""
    out: list[dict[str, Any]] = []
    for n in (graph.get("nodes") or []):
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or n.get("object_id") or "")
        if n.get("unattached"):
            out.append({"graph": graph_name, "rule": "unattached_hazard",
                        "detail": f"{nid}: declared hazardous, no target named"})
        if n.get("unattributed"):
            out.append({"graph": graph_name, "rule": "unattributed_victim",
                        "detail": f"{nid}: at risk, no source named"})
    return out


def conformance_breakdown(graph_a: dict, graph_b: dict) -> dict[str, Any]:
    """Corrected Arm B conformance read for both graphs, plus Arm A's raw
    frozen numbers. Deduped by (graph, entity, category), severity-weighted,
    non-saturating, and grouped by failure pattern for the panel.

    F18: the one-ended node annotations are folded in here, so a hazard with
    no target and a victim with no source are reported as themselves rather
    than as whatever rule the placeholder self-loop happened to trip."""
    from main import check_graph_rule_conformance  # lazy; Arm A frozen

    graph_a = graph_a or {}
    graph_b = graph_b or {}
    raw_a = (check_graph_rule_conformance(graph_a, "graph_a")
             + _one_ended_issues(graph_a, "graph_a"))
    raw_b = check_graph_rule_conformance(graph_b, "graph_b")

    # dedupe: one issue per (graph, entity, category). The drowning-entity that
    # trips three role rules becomes ONE "role mix-up" issue.
    seen: set[tuple[str, str, str]] = set()
    issues: list[dict[str, Any]] = []
    for graph_name, raw in (("graph_a", raw_a), ("graph_b", raw_b)):
        for v in raw:
            rule = v.get("rule", "")
            cat, sev = _meta(rule)
            ent = _entity_of(v.get("detail", ""))
            key = (graph_name, ent or rule, cat)
            if key in seen:
                continue
            seen.add(key)
            issues.append({"graph": graph_name, "rule": rule, "category": cat,
                           "severity": sev, "entity": ent,
                           "detail": v.get("detail", "")})

    # category breakdown (the panel's core): count + max severity + examples
    by_cat: dict[str, dict[str, Any]] = {}
    for it in issues:
        c = by_cat.setdefault(it["category"], {"count": 0, "severity": 0,
                                               "examples": []})
        c["count"] += 1
        c["severity"] = max(c["severity"], it["severity"])
        if len(c["examples"]) < 3:
            c["examples"].append(f"{it['rule']}: {it['detail']}"[:120])
    breakdown = sorted(
        ({"category": k, **v} for k, v in by_cat.items()),
        key=lambda r: (-r["severity"], -r["count"]))

    # severity-weighted penalty — cosmetic (sev 0) contributes nothing
    weighted = sum(it["severity"] for it in issues)
    # size on DE-DUPLICATED edges, so padding with duplicate edges can't
    # inflate the denominator and fake a better score.
    size = _size(_dedup_edges(graph_a)) + _size(_dedup_edges(graph_b))
    # non-saturating validity: 1 at zero penalty, →0 as penalty grows, never
    # clamps flat. weighted/(weighted+size) is naturally in [0,1).
    validity = round(1.0 - weighted / (weighted + max(1, size)), 3)

    return {
        "issues": issues,                       # deduped, severity-tagged
        "breakdown": breakdown,                 # by failure pattern (the panel)
        "n_issues": len(issues),                # deduped count
        "weighted_penalty": weighted,
        "validity": validity,                   # corrected, non-saturating
        # Arm A's raw frozen numbers, kept for arm comparability
        "raw_a_violations": len(raw_a),
        "raw_b_violations": len(raw_b),
        "raw_a_validity": _raw_validity(len(raw_a),
                                        len(graph_a.get("edges") or []), 0.5),
        "raw_b_validity": _raw_validity(len(raw_b),
                                        len(graph_b.get("edges") or []), 1.0),
    }


# ── Internal alignment of Graph A (within-recommendation coverage) ──────

def internal_alignment(record: Any, assessment: Any,
                       recommendations: list[dict]) -> dict[str, Any]:
    """Does each recommendation hang together internally? The within-A
    coverage/consistency checks — Arm A's, PLUS the three the prompt states
    but Arm A never enforces (audit F15): Rule-4 strict (reason ids in related
    AND quad), action-collapse (no two recs the same verb+target), and
    threatens-last-resort. This is the id-level 'do the parts line up' check;
    the SEMANTIC 'do the prose and the quad mean the same thing' check is the
    judge/RAG layer (Phase 2)."""
    detected_ids = {o.object_id for o in record.detected_objects}
    state_of = {o.object_id: o.state for o in record.detected_objects}
    at_risk_ids = {a.object_id for a in assessment.at_risk}

    fails: list[dict[str, Any]] = []

    def fail(cat: str, sev: int, detail: str, rank: Any = None) -> None:
        fails.append({"category": cat, "severity": sev, "detail": detail,
                      "rank": rank})

    seen_quad: dict[tuple, Any] = {}
    seen_risk: dict[str, Any] = {}
    seen_action: dict[tuple, Any] = {}
    all_affected: set[str] = set()
    threats_used: set[str] = set()

    for r in recommendations:
        rank = r.get("rank")
        q = r.get("structured_reasoning", {}) or {}
        # F16: ids are compared here, so they are normalised here. A stray
        # '·suffix' must never make an id fail to match itself.
        from agentic.recommend import bare_id
        threat = bare_id(q.get("threat"))
        threats_used.add(threat)
        affected = [bare_id(x) for x in (q.get("affected_objects") or [])
                    if bare_id(x)]
        all_affected.update(affected)
        quad_ids = {i for i in ({threat} | set(affected)) if i}
        reason_ids = set(_ID_RE.findall(str(r.get("reason", ""))))
        # Raw model output: this arrives as anything. A bare string would
        # iterate per character and an int would raise, so coerce first.
        _rel = r.get("related_object_ids") or []
        if not isinstance(_rel, (list, tuple, set)):
            _rel = [_rel] if isinstance(_rel, str) else []
        related = {bare_id(x) for x in _rel if bare_id(x)}

        # coverage: every quad id appears in the reason
        miss = quad_ids - reason_ids
        if miss:
            fail("coverage gap", 1,
                 f"rec {rank}: quad ids not in reason: {sorted(miss)}", rank)
        # F20: the prompt (recommend.py) specifies reason == quad "and vice
        # versa", with related_object_ids as "the object_ids the reason
        # touches" — so all three carry the same set, and a divergence has
        # three distinct shapes worth different severities. Scanning reason
        # ids alone made one shape unobservable: an id declared in related but
        # absent from BOTH quad and reason was never examined (D_aerial's
        # spill_1, which sits in Graph B's chain and no recommendation acts
        # through). Scan the union, then split by case.
        for rid in sorted((reason_ids | related) & detected_ids):
            in_quad, in_related = rid in quad_ids, rid in related
            if in_quad and in_related:
                continue
            if in_quad:
                # case A: the quad carries it, the mirror list does not. A
                # bookkeeping slip against our own spec — the causal claim is
                # intact, so it is recorded, not charged.
                fail("bookkeeping", 0,
                     f"rec {rank}: {rid} is in the quad but missing from "
                     f"related_object_ids", rank)
            elif in_related:
                # case B: declared related, no quad covers it — a dangling
                # declaration. Declared-vs-operative divergence, the thing
                # CEE+ measures.
                fail("coverage gap", 1,
                     f"rec {rank}: {rid} is declared in related_object_ids "
                     f"but no quad covers it", rank)
            else:
                # case C: named in the reason only.
                fail("coverage gap", 1,
                     f"rec {rank}: {rid} is named in the reason but is in "
                     f"neither the quad nor related_object_ids", rank)
        # state match: quad.state == the threat's frozen state
        if threat in state_of and q.get("state") and \
                str(q.get("state")) != state_of[threat]:
            fail("inconsistency", 2,
                 f"rec {rank}: quad state '{q.get('state')}' != {threat}'s "
                 f"state '{state_of[threat]}'", rank)
        # self-loop: threat in its own affected only with 'worsens'
        if threat in affected and q.get("effect") != "worsens":
            # F20 severity ladder: cause-and-effect reversal is not the same
            # class of error as a duplicated field. Role inversion sits ABOVE
            # the old ceiling of 2 — it is the causal-direction error CEE+
            # exists to catch.
            fail("role mix-up", 3,
                 f"rec {rank}: {threat} harms itself with '{q.get('effect')}'",
                 rank)
        # threatens is a last resort (Arm A prompt-only)
        if q.get("effect") == "threatens":
            fail("effect choice", 1, f"rec {rank}: uses 'threatens' (last resort)",
                 rank)
        # duplicate quad
        qk = (threat, str(q.get("state")), str(q.get("effect")),
              tuple(sorted(affected)))
        if qk in seen_quad:
            fail("duplicate", 1, f"rec {rank}: same quad as rec {seen_quad[qk]}",
                 rank)
        else:
            seen_quad[qk] = rank
        # distinct remaining_risk
        rr = str(r.get("remaining_risk", "")).strip().lower()
        if rr and rr in seen_risk:
            fail("duplicate", 0,
                 f"rec {rank}: remaining_risk duplicates rec {seen_risk[rr]}", rank)
        elif rr:
            seen_risk[rr] = rank
        # action-collapse (Arm A prompt-only): same (verb, target)
        words = str(r.get("action", "")).strip().lower().split()
        ak = (words[0] if words else "", threat)
        if ak[0] and ak in seen_action:
            fail("duplicate", 1,
                 f"rec {rank}: action collapses to rec {seen_action[ak]} "
                 f"(same verb+target)", rank)
        else:
            seen_action[ak] = rank

    # every at-risk entity must be acted on, and never used as a threat
    for aid in sorted(at_risk_ids - all_affected):
        fail("coverage gap", 2,
             f"at-risk {aid} is not addressed by any recommendation")
    for aid in sorted(at_risk_ids & threats_used):
        # F20: the victim named as the hazard — severity 3, same ladder rung
        # as the self-loop above and for the same reason.
        fail("role mix-up", 3, f"at-risk {aid} used as a threat")

    by_cat: dict[str, dict[str, Any]] = {}
    for f in fails:
        c = by_cat.setdefault(f["category"], {"count": 0, "severity": 0,
                                              "examples": []})
        c["count"] += 1
        c["severity"] = max(c["severity"], f["severity"])
        if len(c["examples"]) < 3:
            c["examples"].append(f["detail"][:120])
    breakdown = sorted(({"category": k, **v} for k, v in by_cat.items()),
                       key=lambda r: (-r["severity"], -r["count"]))
    weighted = sum(f["severity"] for f in fails)
    size = max(1, len(recommendations) * 4 + len(at_risk_ids))
    score = round(1.0 - weighted / (weighted + size), 3)
    return {"failures": fails, "breakdown": breakdown,
            "n_failures": len(fails), "score": score}


# ── A-vs-B alignment (declared Graph B vs structured Graph A) ────────────

def _dedup_edges(graph: dict) -> dict:
    """Collapse parallel edges (same source/effect/target) so the multiset
    comparison doesn't misbehave on duplicates (audit F15).

    Also normalises endpoints through bare_id (F16): this runs immediately
    before the frozen comparator, and an 'ambulance_1·proximity' target would
    resolve to an off-vocabulary class that can never match Graph B's
    'ambulance_1' — which is what drove D_aerial's alignment to 0.0."""
    from agentic.recommend import bare_id
    seen: set[tuple] = set()
    edges = []
    for e in (graph.get("edges") or []):
        if not isinstance(e, dict):
            continue
        e = {**e, "source": bare_id(e.get("source")),
             "target": bare_id(e.get("target"))}
        k = (e.get("source"), e.get("effect"), e.get("target"))
        if k in seen:
            continue
        seen.add(k)
        edges.append(e)
    return {"nodes": graph.get("nodes") or [], "edges": edges}


# ── Consequence: how bad the outcome is for VICTIMS (a separate axis) ───
#
# Trust asks "can we rely on the advice?" Consequence asks a DIFFERENT question
# (Sunny): "if this hazard isn't dealt with, how bad is it for the victims?"
# It is the recommendation's life-safety weight, not its reliability. Built on
# Arm A's frozen entity categories (WEIGHTED_ENTITY_PATTERNS) + the assessment's
# at-risk kinds. Weights are PRIORS, calibratable on the six scenes.

EFFECT_CONSEQUENCE = {          # how directly the effect threatens life (prior)
    "may_harm": 1.0, "threatens": 1.0,
    "may_spread_to": 0.85, "exposes": 0.85,
    "isolates": 0.6, "blocks_access_to": 0.6,
    "increases_risk_to": 0.5, "worsens": 0.5,
}
_HUMAN_HINTS = ("person", "man", "woman", "child", "kid", "boy", "girl",
                "people", "worker", "driver", "pedestrian", "occupant",
                "victim", "patient", "baby", "infant", "human", "elderly")
_ANIMAL_HINTS = ("dog", "cat", "animal", "pet", "horse", "cow", "bird",
                 "livestock")
# structures that LIKELY hold people — occupancy risk lifts them above property
# and above an animal (Sunny: a nearby house may have people inside).
_STRUCTURE_HINTS = ("house", "home", "building", "apartment", "flat",
                    "residence", "dorm", "hotel", "office", "store", "shop",
                    "mall", "factory", "warehouse", "tower", "cabin", "hut",
                    "tent")


def _victim_weight(vid: str, label: str, kind: str | None,
                   category: str | None) -> float:
    """Vulnerability of one victim (0..1). Ordered by WHAT the victim is and how
    much human life it likely holds — direct human life first, then places full
    of people (a hospital/school of vulnerable occupants ranks with humans),
    then structures that may have occupants, then animals, then plain property —
    with the at-risk KIND as a modifier ('distress' > 'proximity'). Species/role
    ranks BEFORE kind, which fixes 'a dog outranked the person' and 'a nearby
    house sat below a dog'."""
    text = f"{vid} {label}".lower()
    if category == "vulnerable":                   # child/elderly/patient/...
        base = 1.0
    elif category == "institutional":              # hospital/school — vulnerable occupants
        base = 0.95
    elif any(h in text for h in _HUMAN_HINTS):
        base = 0.9
    elif category == "responder":                  # firefighter/ambulance/... (human)
        base = 0.8
    elif any(h in text for h in _STRUCTURE_HINTS):  # likely occupants inside
        base = 0.65
    elif category == "cultural":                   # heritage + possible people
        base = 0.6
    elif any(h in text for h in _ANIMAL_HINTS):
        base = 0.5
    else:
        base = 0.2                                 # plain property (car, road, ...)
    if kind == "distress":                         # actively being harmed
        base = min(1.0, base + 0.1)
    elif kind == "proximity":                      # at risk, not yet harmed
        base = min(1.0, base + 0.03)
    return round(base, 3)


def consequence_scores(recommendations: Any, assessment: Any,
                       record: Any = None) -> dict[Any, dict[str, Any]]:
    """Per-recommendation consequence-to-victims, keyed by rank. Each: score
    (0..1), band, the worst victim that drives it, and the effect. Deterministic;
    guarded against missing pieces."""
    from main import _entity_weight_category  # Arm A, frozen (import only)

    recs = recommendations if isinstance(recommendations, list) else []
    kind_of: dict[str, str] = {}
    if assessment is not None:
        for a in getattr(assessment, "at_risk", []) or []:
            kind_of[str(getattr(a, "object_id", ""))] = str(getattr(a, "kind", ""))
    label_of: dict[str, str] = {}
    if record is not None:
        for o in getattr(record, "detected_objects", []) or []:
            label_of[str(getattr(o, "object_id", ""))] = str(getattr(o, "label", ""))

    out: dict[Any, dict[str, Any]] = {}
    for r in recs:
        q = r.get("structured_reasoning", {}) or {}
        effect = str(q.get("effect", ""))
        eff_w = EFFECT_CONSEQUENCE.get(effect, 0.5)
        victims = [str(v) for v in (q.get("affected_objects") or []) if str(v)]
        best_v, best_w = None, 0.0
        for v in victims:
            label = label_of.get(v, "")
            cat = _entity_weight_category(v) or _entity_weight_category(label)
            w = _victim_weight(v, label, kind_of.get(v), cat)
            if w > best_w:                          # first max wins (stable)
                best_w, best_v = w, {"id": v, "kind": kind_of.get(v),
                                     "category": cat, "weight": round(w, 3)}
        score = round(eff_w * best_w, 3)
        band = "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
        out[r.get("rank")] = {"score": score, "band": band,
                              "worst_victim": best_v, "effect": effect}
    return out


# ── Trust: fold the objective signals into one score + a breakdown ──────
#
# The trust score is a DETERMINISTIC roll-up of the code-computed evals — no
# model, no judge, no ground truth. It answers "why should we trust (or not)
# the VLM's advice so far?" with a ranked breakdown, worst-first. Weights are
# PRIORS, calibratable on the six frozen scenes; the band stays ordinal until
# then. Decided with Sunny (2026-07-24): A-vs-B leads (advice divorced from the
# model's own beliefs is the strongest distrust signal); per-recommendation
# trust folds ONLY that rec's entity-specific signals (a rec is not punished
# for a scene-wide issue it didn't cause).

# SEVERITY = how much a FULL failure of this check should cost trust (0..1).
# This is the check's consequence, and it scales the check's impact directly
# (impact = penalty × severity) — NOT a share of an average. So a full A-vs-B
# failure (severity 1.0) can take trust all the way to 0; a malformed graph
# (0.4) can't sink it past 0.6 on its own. PRIORS — calibrate on more runs.
TRUST_WEIGHTS = {
    "ab_alignment":       0.30,   # advice vs the model's own graph — leads
    "uncertainty":        0.25,   # recommendations not reproducible on re-ask
    "internal_alignment": 0.20,   # recommendations don't hang together
    "pick_agreement":     0.15,   # the three target-choosers disagree
    "conformance":        0.10,   # the causal graph isn't well-formed
}

_TRUST_TEXT = {
    "ab_alignment": "the recommendations diverge from the model's own "
                    "independent causal graph",
    "uncertainty": "the recommendations change when you ask again",
    "internal_alignment": "the recommendations don't fully hang together",
    "pick_agreement": "the three ways of choosing a target disagree",
    "conformance": "the causal graph breaks well-formedness rules",
}
_TRUST_ACTION = {
    "ab_alignment": "reconcile the plan with the model's declared graph — the "
                    "diverging links are the ones to challenge in reflection",
    "uncertainty": "the advice isn't stable; lean on the recommendations the "
                   "re-asks agree on, distrust the ones that appear once",
    "internal_alignment": "fix the coverage gaps / duplicates before trusting "
                          "the set as complete",
    "pick_agreement": "no single target is agreed — treat the top pick as "
                      "provisional until the routes converge",
    "conformance": "clean the malformed edges; cosmetic issues don't count",
}


def _trust_evidence(signal: str, conformance: dict, internal: dict,
                    alignment: dict, uncertainty: dict, picks: dict) -> str:
    if signal == "ab_alignment":
        return (f"agreement {alignment.get('structural')}, "
                f"{len(alignment.get('a_only') or [])} asserted-not-believed, "
                f"{len(alignment.get('b_only') or [])} seen-but-not-acted")
    if signal == "uncertainty":
        return (f"{uncertainty.get('n_probes', 0)} re-asks, score "
                f"{uncertainty.get('score')}, "
                f"{len(uncertainty.get('candidates') or [])} distinct sets")
    if signal == "internal_alignment":
        return f"{internal.get('n_failures', 0)} internal issue(s)"
    if signal == "pick_agreement":
        return f"agreement {picks.get('agreement')}"
    if signal == "conformance":
        return (f"validity {conformance.get('validity')}, "
                f"{conformance.get('n_issues', 0)} issue(s)")
    return ""


TRUST_MATERIAL_MIN = 0.05   # contributions below this are noise, not narrated


def _trust_narrative(score: float, band: str,
                     contributors: list[dict]) -> str:
    # only MATERIAL contributors are narrated; near-zero ones are noise
    hits = [c for c in contributors
            if c["contribution"] >= TRUST_MATERIAL_MIN][:3]
    if not hits:
        return (f"Trust is {band} ({score}). No material signal dents it — the "
                "recommendations are reproducible, hang together, match the "
                "model's own graph, and the graph is well-formed.")
    # Framing MUST match the verdict: if trust is high, the dents didn't change
    # it, so they are minor notes — not "biggest reasons" paraded as if serious
    # (Sunny). Only moderate/low trust leads with the failing reasons.
    if band == "high":
        note = (f" Minor note: {hits[0]['text']} ({hits[0]['evidence']}), but it "
                "didn't change the verdict." if hits else "")
        return f"Trust is high ({score}) — the advice looks reliable.{note}"
    parts = [f"Trust is {band} ({score}).",
             f"Biggest reason: {hits[0]['text']} ({hits[0]['evidence']})."]
    for c in hits[1:]:
        parts.append(f"Also: {c['text']} ({c['evidence']}).")
    return " ".join(parts)


def _per_rec_trust(recommendations: list[dict], uncertainty: dict,
                   internal: dict, consequence: dict) -> list[dict]:
    """Per-recommendation trust — ONLY entity-specific signals: whether this
    rec's threat reproduces across the re-asks, whether its mechanism holds,
    and the internal-alignment failures tagged to its rank. Score = 1 - mean
    of those penalties (clamped). The consequence axis (life-safety weight to
    victims) is a SEPARATE measure, attached per-rec for display, never folded
    into trust."""
    gran = uncertainty.get("granular") or {}
    threats_u = gran.get("threats") or {}
    effects_u = gran.get("effects") or {}
    n_probes = int(uncertainty.get("n_probes") or 0)
    fails = internal.get("failures") or []

    out: list[dict] = []
    for r in recommendations:
        q = r.get("structured_reasoning", {}) or {}
        threat = str(q.get("threat", "")).split("·")[0].strip()
        rank = r.get("rank")
        contribs: list[dict] = []

        if n_probes and threat:
            tu = threats_u.get(threat)
            if tu is None:
                contribs.append({"signal": "uncertainty", "penalty": 1.0,
                                 "text": f"{threat} never reappeared in "
                                         f"{n_probes} re-asks"})
            elif tu.get("u", 0) > 0:
                contribs.append({"signal": "uncertainty", "penalty": tu["u"],
                                 "text": f"{threat} a threat in only "
                                         f"{tu.get('votes')} re-asks"})
            eu = effects_u.get(threat)
            if eu and eu.get("u", 0) > 0:
                contribs.append({"signal": "uncertainty", "penalty": eu["u"],
                                 "text": f"{threat} mechanism splits: "
                                         f"{eu.get('evidence')}"})
        for f in fails:
            if f.get("rank") == rank:
                contribs.append({"signal": "internal_alignment",
                                 "penalty": round(f.get("severity", 1) / 2, 3),
                                 "text": f.get("detail", "")})

        if contribs:
            pen = round(sum(c["penalty"] for c in contribs) / len(contribs), 3)
            score = max(0.0, round(1 - pen, 3))
            worst = max(contribs, key=lambda c: c["penalty"])
        else:
            score, worst = 1.0, None
        cons = (consequence or {}).get(rank, {})
        out.append({"rank": rank, "threat": threat, "score": score,
                    "worst_contributor": worst, "contributors": contribs,
                    "consequence": cons.get("score"),
                    "consequence_band": cons.get("band"),
                    "worst_victim": cons.get("worst_victim")})
    return out


def compute_trust(recommendations: Any, conformance: Any, internal_alignment: Any,
                  alignment: Any, uncertainty: Any, picks: Any,
                  consequence: Any = None,
                  no_hazards: bool = False) -> dict[str, Any]:
    """Fold the objective evals into a global trust score + ranked breakdown,
    plus per-recommendation trust (with the separate consequence axis attached
    for display). Deterministic; guarded against missing or malformed inputs."""
    recommendations = recommendations if isinstance(recommendations, list) else []
    consequence = consequence if isinstance(consequence, dict) else {}
    conformance = conformance if isinstance(conformance, dict) else {}
    internal = internal_alignment if isinstance(internal_alignment, dict) else {}
    alignment = alignment if isinstance(alignment, dict) else {}
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
    picks = picks if isinstance(picks, dict) else {}

    def _f(v, default=1.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    penalty = {
        "ab_alignment": round(1 - _f(alignment.get("structural", 1.0)), 3),
        "uncertainty": round(_f(uncertainty.get("score", 0.0), 0.0), 3),
        "internal_alignment": round(1 - _f(internal.get("score", 1.0)), 3),
        "pick_agreement": round(1 - _f(picks.get("agreement", 1.0)), 3),
        "conformance": round(1 - _f(conformance.get("validity", 1.0)), 3),
    }
    # F17 — the NULL PATH. On a safe scene there is nothing to compare: Graph B
    # is empty, so ab_alignment has no comparand, and the three pick routes all
    # return nothing, which is AGREEMENT and not disagreement. Scoring those as
    # penalty 1.0 punished F_park_control for being safe — it landed at 0.448,
    # BELOW the burning house at 0.577. A signal with no comparand is NOT
    # APPLICABLE: it is dropped and the remaining weights renormalise. Nothing
    # is silent — each dropped signal is listed with its reason.
    not_applicable: dict[str, str] = {}
    if no_hazards:
        # Graph B has nothing to declare, so A-vs-B has no comparand. This is
        # keyed on the SCENE being safe (no disaster, no hazard-bearing
        # entity), never on a graph merely looking empty — an empty graph on a
        # hazard scene is a real signal and must still be scored.
        if not (alignment.get("b_total") or 0):
            not_applicable["ab_alignment"] = (
                "safe scene: no hazard declared, so the model's causal graph "
                "has nothing to agree or disagree with")
        # Three routes returning nothing is AGREEMENT, not disagreement.
        if not any(str((picks.get(k) or {}).get("object_id") or "").strip()
                   not in ("", "None", "none")
                   for k in ("a_pick", "b_pick", "llm_pick")):
            not_applicable["pick_agreement"] = (
                "safe scene: no hazard to suppress — all three routes "
                "returned nothing, which is agreement")

    contributors: list[dict] = []
    applicable = {s: w for s, w in TRUST_WEIGHTS.items()
                  if s not in not_applicable}
    total_w = sum(applicable.values()) or 1.0
    for sig, w in TRUST_WEIGHTS.items():
        if sig in not_applicable:
            continue
        # renormalise so the applicable weights still sum to 1
        w_eff = round(w / total_w, 4)
        p = max(0.0, min(1.0, penalty[sig]))
        contributors.append({
            "signal": sig, "weight": w_eff, "weight_nominal": w, "penalty": p,
            "contribution": round(w_eff * p, 3),
            "text": _TRUST_TEXT[sig], "action": _TRUST_ACTION[sig],
            "evidence": _trust_evidence(sig, conformance, internal, alignment,
                                        uncertainty, picks)})
    contributors.sort(key=lambda c: -c["contribution"])
    global_penalty = round(sum(c["contribution"] for c in contributors), 3)
    score = round(1 - global_penalty, 3)
    band = "high" if score >= 0.7 else "moderate" if score >= 0.4 else "low"
    return {
        "score": score, "band": band, "global_penalty": global_penalty,
        "weights": TRUST_WEIGHTS,
        "not_applicable": [{"signal": s, "reason": r}
                           for s, r in sorted(not_applicable.items())],
        # HOW MUCH was measured, always beside the score. Renormalising the
        # weights makes the arithmetic clean but hides that fewer signals were
        # read — and a high score built on 3 of 5 signals is not the same claim
        # as one built on 5. The number must never travel without this.
        #
        # The case this exists for: if perception misses the hazard AND stage 2
        # says "No", the null path opens on a scene that is actually dangerous
        # and a blind run scores 1.0. Trust cannot tell that apart from a truly
        # safe scene — both look identical from here — so it must at least say
        # how little it had to go on.
        "signals_measured": f"{len(contributors)}/{len(TRUST_WEIGHTS)}",
        "contributors": contributors,
        "explanation": (
            _trust_narrative(score, band, contributors)
            + (f" Measured on {len(contributors)} of {len(TRUST_WEIGHTS)} "
               f"signals — "
               + ", ".join(sorted(not_applicable)) + " had nothing to compare."
               if not_applicable else "")),
        "per_rec": _per_rec_trust(recommendations, uncertainty, internal,
                                  consequence),
    }


def _id_edges_for_keys(graph: dict, keys: list) -> list[dict]:
    """Map topological (class-level) edge keys back to CONCRETE id edges from
    the graph — so the panel shows 'house_1 —may_spread_to→ house_2', not
    'structure —may_spread_to→ structure'. The match is class-level, so a key
    can map to more than one id edge; we consume them in order and fall back to
    the class label only if nothing matches."""
    from collections import Counter, defaultdict

    from main import _topological_edge_key  # Arm A, frozen
    nodes = {n.get("id"): n for n in (graph.get("nodes") or [])
             if isinstance(n, dict)}
    by_key: dict = defaultdict(list)
    for e in _dedup_edges(graph or {}).get("edges", []):
        by_key[tuple(_topological_edge_key(e, nodes))].append(e)
    used: Counter = Counter()
    out: list[dict] = []
    for k in keys:
        k = tuple(k)
        pool = by_key.get(k, [])
        e = pool[used[k]] if used[k] < len(pool) else (pool[0] if pool else None)
        used[k] += 1
        if e:
            out.append({"source": e.get("source"), "effect": e.get("effect"),
                        "target": e.get("target")})
        else:                                       # no id edge — keep the class
            out.append({"source": k[0], "effect": k[1], "target": k[2]})
    return out


def ab_alignment(graph_a: dict, graph_b: dict) -> dict[str, Any]:
    """Declared (Graph B) vs structured (Graph A) agreement — ONE structural
    definition (topological multiset on de-duplicated edges), plus the clean
    precision/recall pair. a_fidelity = of the recommendations' causal edges,
    how many the model's independent graph backs up. b_coverage = of the
    independent graph, how many the recommendations reproduce. B is the
    yardstick. a_only/b_only are resolved to CONCRETE id edges for display."""
    from main import compare_graphs_topological  # lazy
    ga, gb = graph_a or {}, graph_b or {}
    cmp = compare_graphs_topological(_dedup_edges(ga), _dedup_edges(gb))
    return {
        "a_fidelity": round(cmp.get("a_fidelity_topo", 1.0), 3),
        "b_coverage": round(cmp.get("b_coverage_topo", 1.0), 3),
        "structural": round(cmp.get("structural_topo", 1.0), 3),
        "matched": cmp.get("matched", 0),
        "a_total": cmp.get("a_total", 0),
        "b_total": cmp.get("b_total", 0),
        # in recs, not independently declared / declared, not acted on — id-level
        "a_only": _id_edges_for_keys(ga, cmp.get("a_only_keys", [])),
        "b_only": _id_edges_for_keys(gb, cmp.get("b_only_keys", [])),
    }


def _edge_key(e: Any) -> tuple:
    from agentic.recommend import bare_id
    if not isinstance(e, dict):
        return ()
    return (bare_id(e.get("source")), str(e.get("effect") or ""),
            bare_id(e.get("target")))


def ab_alignment_distribution(graphs_a: list, graphs_b: list,
                              canonical: dict | None = None) -> dict[str, Any]:
    """F19. A-vs-B over the FULL probe cross-product instead of one pair.

    The single-pair `ab_alignment` is reused unchanged, called len(A)*len(B)
    times. Reports the canonical point estimate alongside the spread, so a
    number that looked precise now carries its own instability:

        structural 0.40  (spread 0.20-0.55 over 5x5)

    Also reports per-edge BELIEF RATE, which is what gives
    'asserted-not-believed' its strength: an edge the model's own independent
    graph denies 5/5 times is a real declared-vs-operative gap; one it denies
    3/5 is a coin flip. Scoring those identically is F15's severity-blindness
    reappearing in the alignment signal.

    b_stability answers the question probing B was really for: is the model's
    causal BELIEF stable, or is the yardstick itself elastic?"""
    ga = [g for g in (graphs_a or []) if isinstance(g, dict)]
    gb = [g for g in (graphs_b or []) if isinstance(g, dict)]
    if not ga or not gb:
        return {"n_a": len(ga), "n_b": len(gb), "pairs": 0,
                "canonical": canonical or {}, "note": "not enough probes"}

    vals: dict[str, list[float]] = {"structural": [], "a_fidelity": [],
                                    "b_coverage": []}
    for a in ga:
        for b in gb:
            r = ab_alignment(a, b)
            for k in vals:
                vals[k].append(float(r.get(k, 0.0)))

    def _stat(xs: list[float]) -> dict[str, float]:
        s = sorted(xs)
        n = len(s)
        med = s[n // 2] if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 3)
        return {"median": round(med, 3), "min": round(s[0], 3),
                "max": round(s[-1], 3),
                "spread": round(s[-1] - s[0], 3)}

    # belief rate: for each A-edge, the fraction of B-probes that carry it
    a_edges: dict[tuple, int] = {}
    for a in ga:
        for e in (a.get("edges") or []):
            k = _edge_key(e)
            if k:
                a_edges[k] = a_edges.get(k, 0) + 1
    b_sets = [{_edge_key(e) for e in (b.get("edges") or [])} for b in gb]
    asserted_not_believed = []
    for k in sorted(a_edges):
        believed = sum(1 for s in b_sets if k in s)
        asserted_not_believed.append({
            "edge": f"{k[0]} --{k[1]}--> {k[2]}",
            "asserted_in": f"{a_edges[k]}/{len(ga)}",
            "believed_in": f"{believed}/{len(gb)}",
            "belief_rate": round(believed / len(gb), 3)})
    asserted_not_believed.sort(key=lambda r: r["belief_rate"])

    # is the yardstick itself stable? fraction of B-probes carrying each B-edge
    all_b: dict[tuple, int] = {}
    for s in b_sets:
        for k in s:
            all_b[k] = all_b.get(k, 0) + 1
    b_stability = (round(sum(v / len(gb) for v in all_b.values()) / len(all_b), 3)
                   if all_b else 1.0)

    return {
        "n_a": len(ga), "n_b": len(gb), "pairs": len(ga) * len(gb),
        "canonical": canonical or {},
        "structural": _stat(vals["structural"]),
        "a_fidelity": _stat(vals["a_fidelity"]),
        "b_coverage": _stat(vals["b_coverage"]),
        "asserted_not_believed": asserted_not_believed[:12],
        "b_stability": b_stability,
        "b_distinct_edge_sets": len({frozenset(s) for s in b_sets}),
    }


# ── O1: the paired-arm guard ────────────────────────────────────────────

def paired_arm_guard(off_uncertainty: Any, on_uncertainty: Any,
                     tolerance: float | None = None) -> dict[str, Any]:
    """O1. Did the empty-recommendations clause do more than grant permission?

    The obvious guard — same recommendation SET clause-on as clause-off — cannot
    work: B_pool's five probes produce five distinct sets with one vote each, so
    two runs differ with no clause present at all. Set identity compares a single
    sample against a single sample, which is the very objection F19 exists to
    raise, turned on the guard itself.

    So compare DISTRIBUTIONS. For each (threat, effect, affected-set) triple,
    its belief rate is the fraction of probes carrying it. The clause ships only
    if no triple's rate moves further than the spread already present inside the
    clause-off arm — i.e. further than the noise we can measure without it.

    Costs nothing: `uncertainty.candidates[i].edges` already stores the triples.
    """
    def _rates(u: Any) -> tuple[dict[tuple, float], int]:
        """Returns (belief rates, probe count). The two are separate on
        purpose: an arm whose probes all returned NO edges has rates {} but a
        real probe count, and that is the single most important case here —
        it is what "the clause silenced the model" looks like. Treating it as
        missing data would blind the guard to the thing it exists to catch."""
        u = u if isinstance(u, dict) else {}
        cands = [c for c in (u.get("candidates") or []) if isinstance(c, dict)]
        n = len(cands)
        if not n:
            return {}, 0
        counts: dict[tuple, int] = {}
        for c in cands:
            seen = set()
            for e in (c.get("edges") or []):
                if not isinstance(e, (list, tuple)) or len(e) < 3:
                    continue
                tgts = e[2] if isinstance(e[2], (list, tuple)) else [e[2]]
                seen.add((str(e[0]), str(e[1]),
                          tuple(sorted(str(t) for t in tgts))))
            for k in seen:
                counts[k] = counts.get(k, 0) + 1
        return {k: v / n for k, v in counts.items()}, n

    off, n_off = _rates(off_uncertainty)
    on, n_on = _rates(on_uncertainty)
    if not n_off or not n_on:
        return {"verdict": "insufficient",
                "reason": "a probe arm has no probes", "moves": []}
    # the noise floor: how far a single triple's rate can sit from certainty
    # inside the clause-off arm alone
    noise = max((min(r, 1 - r) for r in off.values()), default=0.0)
    tol = noise if tolerance is None else float(tolerance)

    moves = []
    for k in sorted(set(off) | set(on)):
        a, b = off.get(k, 0.0), on.get(k, 0.0)
        d = round(b - a, 3)
        if abs(d) > tol:
            moves.append({"triple": f"{k[0]} --{k[1]}--> {list(k[2])}",
                          "off_rate": round(a, 3), "on_rate": round(b, 3),
                          "delta": d})
    moves.sort(key=lambda m: -abs(m["delta"]))
    return {"verdict": "ship" if not moves else "hold",
            "noise_floor": round(tol, 3),
            "reason": ("no triple moved beyond the clause-off arm's own spread"
                       if not moves else
                       f"{len(moves)} triple(s) moved beyond the noise floor — "
                       f"the clause is reshaping output, not only permitting "
                       f"silence"),
            "moves": moves[:12]}
