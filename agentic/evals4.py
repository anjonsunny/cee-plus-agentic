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
from contextlib import contextmanager
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
    # F39: DEMOTED to 0 (recorded, not charged). Both of these say the same
    # thing — "you picked the wrong word for this effect" — about a scene with
    # more than one hazard, which is most of them. "A fire may_harm a tanker"
    # is correct English; the rule wants `increases_risk_to`. Neither says
    # anything about whether the advice is SAFE, and on C_tanker they were two
    # of the three findings on screen while a declared hazard nobody acted on
    # went unreported. Kept at 0 so nothing is erased.
    "may_harm_hazardous_target": ("effect wording", 0),
    "spread_between_hazards": ("effect wording", 0),
    "hazard_flag_state_mismatch": ("role mix-up", 2),
    "hazardous_and_at_risk": ("role mix-up", 2),
    "distress_state_on_non_living": ("role mix-up", 1),
    "edge_from_non_hazardous": ("role mix-up", 2),
    # fabrications — pointing at nothing
    "unresolved_endpoint": ("fabrication", 2),
    # internal inconsistency
    "via_state_mismatch": ("inconsistency", 1),
    "via_state_not_hazard_bearing": ("inconsistency", 1),
    # wrong effect choice
    "fluid_wrong_effect_for_person": ("effect choice", 1),
    "uncoupled_obstruction": ("effect choice", 1),
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


@contextmanager
def _arm_b_hazard_states():
    """Run the frozen conformance checker against ARM B's hazard vocabulary.

    F32. Arm B accepts `chemical_spill` as a hazard-bearing state (Sunny,
    2026-07-28: the model reaches for it constantly, it unambiguously means
    "there is a hazardous spill here", and refusing it cost an entire run).
    The conformance checker lives in frozen Arm A, whose list does not have the
    word — so every spill scene collected

        sev2  hazard_flag_state_mismatch  spill_1: 'chemical_spill' vs hazardous=True
        sev1  via_state_not_hazard_bearing

    against BOTH graphs. Neither is a model error. The sev-2 was worse than
    cosmetic: it tripped the Graph B gate, which withheld `ab_alignment` — the
    single largest trust weight, and the exact signal that carries the
    sycophancy / rationalized-minimization reading. D_aerial reported trust
    0.635 on 4/5 signals because of a word.

    We do NOT edit main.py (iron rule 1). Arm A's own runs must keep scoring
    exactly as they always have, or the three-arm comparison stops meaning
    anything. Instead the module-level set is widened for the duration of the
    call and restored afterwards — so the checker applies Arm B's vocabulary to
    Arm B's graphs, and nothing else in the program sees a different Arm A.
    """
    import main
    from agentic.perception import EXTRA_HAZARD_BEARING_STATES

    original = main.HAZARD_BEARING_STATES
    try:
        main.HAZARD_BEARING_STATES = set(original) | set(
            EXTRA_HAZARD_BEARING_STATES)
        yield
    finally:
        main.HAZARD_BEARING_STATES = original


def conformance_breakdown(graph_a: dict, graph_b: dict,
                          card_findings: list | None = None,
                          n_cards: int = 0) -> dict[str, Any]:
    """Corrected Arm B conformance read for both graphs, plus Arm A's raw
    frozen numbers. Deduped by (graph, entity, category), severity-weighted,
    non-saturating, and grouped by failure pattern for the panel.

    F18: the one-ended node annotations are folded in here, so a hazard with
    no target and a victim with no source are reported as themselves rather
    than as whatever rule the placeholder self-loop happened to trip.

    F24: `card_findings` carries the recommendation CARD's conformance
    findings — the action, the prose reason, remaining_risk and rank judged
    against the same law the graphs are judged against. They arrive already
    tagged by explanation_alignment and enter as a third producer beside the
    two graphs, so "which surface can't be trusted" is answerable without
    opening the raw record. The graphs' own numbers are unchanged."""
    from main import check_graph_rule_conformance  # lazy; Arm A frozen

    graph_a = graph_a or {}
    graph_b = graph_b or {}
    with _arm_b_hazard_states():
        raw_a = (check_graph_rule_conformance(graph_a, "graph_a")
                 + _one_ended_issues(graph_a, "graph_a"))
        raw_b = check_graph_rule_conformance(graph_b, "graph_b")

    # dedupe: one issue per (graph, entity, category). The drowning-entity that
    # trips three role rules becomes ONE "role mix-up" issue.
    # F39: Arm A's `hazardous_node_no_edges` and F18's `unattached_hazard` are
    # the SAME finding — a declared hazard with no target — at two severities
    # (1 and 0). F18 added ours so a lone hazard would be reported as itself
    # rather than through a placeholder self-loop, and Arm A's older version
    # was never suppressed. One defect, printed twice, on every scene with a
    # spare hazard.
    _SUPERSEDED = {"hazardous_node_no_edges"}

    seen: set[tuple[str, str, str]] = set()
    issues: list[dict[str, Any]] = []
    for graph_name, raw in (("graph_a", raw_a), ("graph_b", raw_b)):
        for v in raw:
            rule = v.get("rule", "")
            if rule in _SUPERSEDED:
                continue
            cat, sev = _meta(rule)
            ent = _entity_of(v.get("detail", ""))
            key = (graph_name, ent or rule, cat)
            if key in seen:
                continue
            seen.add(key)
            issues.append({"graph": graph_name, "rule": rule, "category": cat,
                           "severity": sev, "entity": ent,
                           "detail": v.get("detail", "")})

    # F24: the card's conformance findings join as a third producer. They are
    # already deduped by construction (one rule, one fire, per card) so they
    # bypass the graph dedupe rather than being run through a key shaped for
    # entity-level graph rules.
    for f in (card_findings or []):
        issues.append({"graph": "card", "rule": f.get("rule", ""),
                       "category": f.get("rule", ""),
                       "severity": f.get("severity", 0),
                       "entity": str(f.get("rank", "")),
                       "level": f.get("level", "card"),
                       "surface": f.get("surface", ""),
                       "detail": f.get("detail", "")})

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
    # F24: each card contributes 4 checkable surfaces (action, reason,
    # remaining_risk, rank) to the denominator, so adding the card layer
    # cannot make a clean run score worse than it did before.
    size = (_size(_dedup_edges(graph_a)) + _size(_dedup_edges(graph_b))
            + 4 * max(0, n_cards))
    # non-saturating validity: 1 at zero penalty, →0 as penalty grows, never
    # clamps flat. weighted/(weighted+size) is naturally in [0,1).
    validity = round(1.0 - weighted / (weighted + max(1, size)), 3)

    # BY PRODUCER. The issues already carry graph_a / graph_b, but nothing
    # surfaced it: a run reporting "2 issues" never said both were Graph B's,
    # and you had to open the raw record to find out. Reflection needs the same
    # split later, to know which producer a correction goes back to. Both keys
    # always exist — a graph with no issues reads 0, never vanishes, because
    # absence has to be visible.
    by_graph: dict[str, Any] = {}
    for gname in ("graph_a", "graph_b", "card"):
        mine = [i for i in issues if i.get("graph") == gname]
        cats: dict[str, dict[str, Any]] = {}
        for it in mine:
            c = cats.setdefault(it["category"], {"count": 0, "severity": 0})
            c["count"] += 1
            c["severity"] = max(c["severity"], it["severity"])
        by_graph[gname] = {
            "count": len(mine),
            "max_severity": max((i["severity"] for i in mine), default=0),
            "breakdown": sorted(({"category": k, **v} for k, v in cats.items()),
                                key=lambda r: (-r["severity"], -r["count"])),
        }

    return {
        "issues": issues,                       # deduped, severity-tagged
        "breakdown": breakdown,                 # by failure pattern (the panel)
        "by_graph": by_graph,                   # by PRODUCER (panel + routing)
        "graph_b_edges": len(_dedup_edges(graph_b).get("edges") or []),
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
                       recommendations: list[dict],
                       card_findings: list | None = None) -> dict[str, Any]:
    """Does each recommendation hang together internally? The within-A
    coverage/consistency checks — Arm A's, PLUS the three the prompt states
    but Arm A never enforces (audit F15): Rule-4 strict (reason ids in related
    AND quad), action-collapse (no two recs the same verb+target), and
    threatens-last-resort. This is the id-level 'do the parts line up' check.

    F24: `card_findings` carries the ROLE-level agreement between a card's
    three surfaces — does the prose explain the same action the quad explains,
    and do the two name the same source and the same harmed entities. The
    id-level checks below cannot see a reversed direction: the same two
    entities appear on both sides either way. That is what the card findings
    add.

    The SEMANTIC 'do the prose and the quad MEAN the same thing' check remains
    the judge layer — advisory, never scored here."""
    detected_ids = {o.object_id for o in record.detected_objects}
    state_of = {o.object_id: o.state for o in record.detected_objects}
    at_risk_ids = {a.object_id for a in assessment.at_risk}

    fails: list[dict[str, Any]] = []

    # F29: every finding is tagged with a SIGNAL and a LEVEL, the same two
    # fields the card findings carry.
    #
    #   signal  "conformance"        judged against the rules
    #           "internal_alignment" judged against another part
    #   level   "card"               belongs under one recommendation
    #           "set"                belongs to the SET — it is about the
    #                                collection, and pinning it to any single
    #                                card would blame a card for something that
    #                                is not its fault
    #
    # "every at-risk entity must be acted on" is a LAW about the recommendation
    # set — nothing is compared to anything — so it is tagged conformance even
    # though it is computed here, where it has lived since before the split.
    # It is still SCORED here, so run-to-run numbers stay comparable; the tag
    # is what the panel and (later) pathology read.
    def fail(cat: str, sev: int, detail: str, rank: Any = None, *,
             signal: str = "internal_alignment", level: str = "card") -> None:
        fails.append({"category": cat, "severity": sev, "detail": detail,
                      "rank": rank, "signal": signal, "level": level})

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
        # F41g: the SAME matcher the action and coverage checks use. "it may
        # worsen the chemical spill" names spill_1 — by its state word — and
        # reading raw ids only produced "quad ids not in reason: ['spill_1']",
        # a disagreement that does not exist.
        #
        # Strict on the REQUIREMENT, lenient on IDENTITY: writing a label where
        # an id was required is still a violation and still charged
        # (reason_names_label_not_id, action_names_label_not_id). This is a
        # different question — do the reason and the quad refer to the same
        # entity — and answering it narrowly charged one behaviour twice, once
        # correctly and once as a fabricated mismatch.
        reason_ids = entities_named_in(r.get("reason"), record)
        # Raw model output: this arrives as anything. A bare string would
        # iterate per character and an int would raise, so coerce first.
        _rel = r.get("related_object_ids") or []
        if not isinstance(_rel, (list, tuple, set)):
            _rel = [_rel] if isinstance(_rel, str) else []
        related = {bare_id(x) for x in _rel if bare_id(x)}

        # F41h: ONE line, both directions. The reason naming entities the
        # quad omits and the quad naming entities the reason omits are the
        # same disagreement seen from two ends — D_aerial rec 1 printed it
        # four times (once for the quad's three vehicles, once each for the
        # two workers and the truck the reason named). What a reader needs is
        # the pair of sets, side by side.
        r_only = sorted((reason_ids & detected_ids) - quad_ids)
        q_only = sorted(quad_ids - reason_ids)
        if r_only or q_only:
            bits = []
            if r_only:
                bits.append(f"reason names {', '.join(r_only)}")
            if q_only:
                bits.append(f"quad names {', '.join(q_only)}")
            fail("coverage gap", 1,
                 f"rec {rank}: the reason and the quad name different "
                 f"entities — " + "; ".join(bits), rank)

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
                 rank, level="set")
        else:
            seen_quad[qk] = rank
        # distinct remaining_risk
        rr = str(r.get("remaining_risk", "")).strip().lower()
        if rr and rr in seen_risk:
            fail("duplicate", 0,
                 f"rec {rank}: remaining_risk duplicates rec {seen_risk[rr]}",
                 rank, level="set")
        elif rr:
            seen_risk[rr] = rank
        # action-collapse (Arm A prompt-only): same (verb, target)
        words = str(r.get("action", "")).strip().lower().split()
        ak = (words[0] if words else "", threat)
        if ak[0] and ak in seen_action:
            fail("duplicate", 1,
                 f"rec {rank}: action collapses to rec {seen_action[ak]} "
                 f"(same verb+target)", rank, level="set")
        else:
            seen_action[ak] = rank

    # F39: a declared HAZARD that no recommendation acts on. We checked victim
    # coverage and never checked hazard coverage — so on C_tanker `spill_1` was
    # declared a threat, no recommendation touched it, and nothing fired, while
    # three effect-wording rules filled the screen. An unaddressed hazard is
    # the operational failure; the wording is not.
    #
    # "Acted on" is generous on purpose: the hazard is the threat of some quad,
    # OR the action names it. Either counts as the plan having noticed it.
    threat_ids = {t.object_id for t in getattr(assessment, "threats", []) or []}
    # the SAME matcher action_mode uses — otherwise the two checks contradict
    # each other about the same entity on the same screen.
    acted_ids: set[str] = set(threats_used)
    for r in recommendations:
        if isinstance(r, dict):
            acted_ids |= entities_named_in(r.get("action"), record)
    for tid in sorted(threat_ids - acted_ids):
        fail("coverage gap", 2,
             f"declared hazard {tid} is not addressed by any recommendation",
             signal="conformance", level="set")

    # every at-risk entity must be acted on, and never used as a threat
    for aid in sorted(at_risk_ids - all_affected):
        fail("coverage gap", 2,
             f"at-risk {aid} is not addressed by any recommendation",
             signal="conformance", level="set")
    # F53 (Sunny, C_tanker ui_065000dd): ONE defect, one charge. Rec 3's
    # self-loop (person_1 -> person_1) was charged sev3 at the rec level
    # ("person_1 harms itself") AND sev3 here at the set level ("at-risk
    # person_1 used as a threat") — 6 severity points for one mistake, inside
    # one report, and it alone moved the run's band from moderate to low.
    # When the same entity already carries a rec-level role mix-up, the
    # set-level line is recorded at severity 0 — visible, never charged —
    # the same one-failure-one-charge rule F48 applies between reports.
    _mixed_up_at_rec_level = {
        str(t) for f in fails if f.get("category") == "role mix-up"
        for t in ([f["detail"].split(":", 1)[1].split(" harms", 1)[0].strip()]
                  if " harms itself" in f.get("detail", "") else [])}
    for aid in sorted(at_risk_ids & threats_used):
        # F20: the victim named as the hazard — severity 3, same ladder rung
        # as the self-loop above and for the same reason.
        dup = aid in _mixed_up_at_rec_level
        fail("role mix-up", 0 if dup else 3,
             f"at-risk {aid} used as a threat"
             + (" (same defect as the rec-level charge — recorded, "
                "not charged again)" if dup else ""),
             signal="conformance", level="set")

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
    # F24: the card's cross-surface findings join the same report, already
    # tagged. They are appended AFTER the breakdown loop below is fed, so both
    # the failures list and the categories include them.
    for f in (card_findings or []):
        fails.append({"category": f.get("rule", ""),
                      "severity": f.get("severity", 0),
                      "detail": f.get("detail", ""),
                      "rank": f.get("rank"),
                      "level": f.get("level", "card"),
                      "surface": f.get("surface", "")})

    weighted = sum(f["severity"] for f in fails)
    size = max(1, len(recommendations) * 4 + len(at_risk_ids))
    score = round(1.0 - weighted / (weighted + size), 3)
    return {"failures": fails, "breakdown": breakdown,
            # F48: `size` is returned so the score can be recomputed with a
            # finding category removed, when the singular error library takes
            # that category over. Deriving it back out of the score would work
            # right up until the score is 1.0, and then divide by zero.
            "size": size,
            "n_failures": len(fails), "score": score}


# ── Explanation alignment: action / reason / quad, one set of rules ─────
#
# F24 (Sunny, 2026-07-28). A recommendation card carries three surfaces:
#
#         ACTION            <- the thing being explained
#        /      \
#   REASON      QUAD        <- two independent explanations of it
#    (prose)  (structure)
#        \______/
#        same rules
#
# The action is the anchor. The reason and the quad each have to explain it
# causally, ON THEIR OWN, and then agree with each other. They stay separately
# generated — merging them would leave nothing to compare — but they are held
# to the SAME constraints. Two witnesses, one set of rules.
#
# What we had before was one witness under law and one witness free: the quad's
# threat had to come from the `threats:` line, the prose reason's did not. So
# the model would write a free-form reason, hit the quad, find its subject was
# not a legal threat, and swap it — and internal_alignment scored the swap as a
# model defect. Some of that penalty was ours. Constraining both sides means a
# mismatch that SURVIVES is a real divergence, not one we manufactured.
#
# The action was checked by nothing at all. The prompt has always said "name
# the entities it operates on by their object_id" and the model has always
# ignored it ("Secure the tanker truck"), because no rule read the string.
#
# Severity ladder matches internal_alignment's: 3 = causal-direction error,
# 2 = serious (the claim is wrong), 1 = structural, 0 = bookkeeping.

# "Because {threat} is {state}, it {effect} {affected}." — the reason's fixed
# template. Parsed, not merely scanned for ids: role inversion is invisible to
# a set of ids and obvious to a subject/verb/object split.
_REASON_RE = re.compile(
    r"because\s+(?P<threat>[a-z][a-z0-9_]*_\d+)\s+"
    r"(?:is|are|has|have)\s+(?P<state>[a-z_][a-z0-9_]*)\b"
    r"(?P<rest>.*)", re.I | re.S)

AT_RISK_ROLES = {"distress", "proximity"}


# Rule → (signal, level). SIGNAL routes the finding into one of the two
# reports the panel already has; LEVEL says where it renders.
#
#   signal "conformance"        one surface judged against the rules
#   signal "internal_alignment" one surface judged against another
#   level  "card"               belongs under that recommendation
#   level  "set"                belongs in the summary — it is about the SET
#
# Every rule below is tagged. A rule with no entry is a bug, and a test asserts
# the table and the emitted rules agree exactly.
CARD_RULE_META: dict[str, tuple[str, str]] = {
    # ── conformance: the action ──
    "action_names_no_object_id": ("conformance", "card"),
    "action_names_label_not_id": ("conformance", "card"),
    # ── conformance: the prose reason, legality ──
    "reason_not_in_template": ("conformance", "card"),
    "reason_state_is_a_role": ("conformance", "card"),
    "reason_state_not_declared": ("conformance", "card"),
    "reason_effect_not_in_vocabulary": ("conformance", "card"),
    # ── conformance: the prose reason, roles ──
    "reason_threat_not_declared": ("conformance", "card"),
    "at_risk_used_as_hazard": ("conformance", "card"),
    "reason_self_threat": ("conformance", "card"),
    # ── conformance: roles, but OURS not the model's (severity 0) ──
    "victim_named_with_no_hazard_declared": ("conformance", "card"),
    "hazard_named_with_no_victim_declared": ("conformance", "card"),
    "entity_is_both_threat_and_at_risk": ("conformance", "card"),
    # ── conformance: remaining_risk ──
    "remaining_risk_role_word": ("conformance", "card"),
    "remaining_risk_not_a_pair": ("conformance", "card"),
    "remaining_risk_duplicated": ("conformance", "set"),
    "reason_names_label_not_id": ("conformance", "card"),
    # ── conformance: the set as a whole ──
    "rank_not_a_triage": ("conformance", "set"),
    # ── internal alignment: action <-> reason ──
    "reason_explains_a_different_action": ("internal_alignment", "card"),
    "reason_omits_an_action_target": ("internal_alignment", "card"),
    # ── internal alignment: action <-> quad ──
    "quad_explains_a_different_action": ("internal_alignment", "card"),
    "quad_omits_an_action_target": ("internal_alignment", "card"),
    # ── internal alignment: reason <-> quad ──
    "subject_mismatch": ("internal_alignment", "card"),
    "object_mismatch": ("internal_alignment", "card"),
    "effect_mismatch": ("internal_alignment", "card"),
}


def parse_reason(text: Any) -> dict[str, Any]:
    """Split a reason sentence into its causal slots.

    Returns {parsed: bool, threat, state, effect, affected[]}. Never raises —
    this reads raw model prose, which arrives as anything."""
    s = str(text or "").strip()
    m = _REASON_RE.search(s)
    if not m:
        return {"parsed": False, "threat": "", "state": "", "effect": "",
                "affected": []}
    rest = m.group("rest") or ""
    effect, at = "", 0
    for e in _EFFECTS():
        # The reason is asked for in PLAIN ENGLISH, so the model writes "may
        # harm", not "may_harm" — and A_fire round 4 was charged severity 2 on
        # all three cards for doing exactly what the prompt asked. The
        # underscore is our serialisation of the token, not a word the model
        # owes us in prose: accept either spelling. (F25b — same defect class
        # as the rest of F24: our own spec, two ways, one of them punished.)
        pat = rf"\b{re.escape(e).replace('_', '[ _]')}\b"
        hit = re.search(pat, rest, re.I)
        # longest match wins, so 'increases_risk_to' is not read as a shorter
        # sibling of some other effect.
        if hit and len(e) > len(effect):
            effect, at = e, hit.end()
    tail = rest[at:] if effect else rest
    affected = [i for i in _ID_RE.findall(tail)]
    return {"parsed": True, "threat": m.group("threat").lower(),
            "state": m.group("state").lower(), "effect": effect,
            "affected": affected}


def _EFFECTS() -> list[str]:
    from agentic.recommend import _EFFECT_LINE
    return [e.strip() for e in _EFFECT_LINE.split(",") if e.strip()]



def entities_named_in(text: Any, record: Any) -> set:
    """Which scene entities does this sentence name — by id, or by label, or by
    state word?

    F41f. ONE function, used by every check that asks "does this action touch
    that entity", because two checks answering it differently put contradictory
    lines on the same screen: `action_mode` said a card was hazard-directed
    while hazard-coverage said nobody addressed that hazard, about the same
    entity, on D_aerial.

    Matching allows ordinary English inflection — "Assess the extent of
    chemical spillage" names `spill_1` (label `spill`, state `chemical_spill`)
    and an exact-word match sees neither, because `spill` is not a word
    boundary inside `spillage`.

    Naming by id is still REQUIRED and its absence is still reported
    (action_names_no_object_id, action_names_label_not_id). This answers a
    different question: what did the responder actually act on."""
    t = str(text or "")
    if not t.strip():
        return set()
    out = set(_ID_RE.findall(t.lower()))
    for o in (getattr(record, "detected_objects", None) or []):
        oid = str(getattr(o, "object_id", ""))
        if not oid or oid in out:
            continue
        for key in (getattr(o, "label", ""), getattr(o, "state", "")):
            key = str(key or "").replace("_", " ").strip()
            if not key:
                continue
            if re.search(rf"\b{re.escape(key)}(?:s|es|ed|ing|age)?\b", t, re.I):
                out.add(oid)
                break
    return out


def action_mode(acted_on: set, threats: set, at_risk: set) -> str:
    """Which side of the danger does this action operate on?

    Judged against the SCENE — the declared threats and the declared at-risk
    entities — not against the card's own quad.

    F41d: it used to read the card's quad, which tangled two questions.
    D_aerial rec 1 was "Secure the area around the tanker truck": the truck IS
    a declared threat, so the action is plainly hazard-directed — but the
    card's own quad named the spill, so mode came out `unattributed` and the
    panel reported that nothing in the set acted on either side of anything.
    Whether the card's quad justifies its own action is a different question,
    already asked and answered by quad_explains_a_different_action.

    Read off entity ROLES, not the verb — no keyword list to keep current and
    no English to parse. Recorded, never scored: it is not a defect, it is what
    the intervention gate needs in order to know what it can test.

      hazard_directed   suppress the hazard        -> testable by suppression
      victim_directed   protect the victim         -> needs its own move
      mixed             names both sides
      unattributed      names neither; already charged by the alignment rules,
                        so nothing extra is billed here
    """
    if not acted_on:
        return "unattributed"
    on_hazard = bool(acted_on & (threats or set()))
    on_victim = bool(acted_on & (at_risk or set()))
    if on_hazard and on_victim:
        return "mixed"
    if on_hazard:
        return "hazard_directed"
    if on_victim:
        return "victim_directed"
    return "unattributed"


def explanation_alignment(record: Any, assessment: Any,
                          recommendations: list[dict]) -> dict[str, Any]:
    """Do the action, the reason and the quad tell ONE causal story?

    The action is the ANCHOR — written first. The prose reason and the
    structured quad are two independent explanations of it, held to the SAME
    constraints and then made to answer for each other. Every finding is tagged
    with a signal (conformance | internal_alignment) and a level (card | set)
    so the two existing reports can absorb it without a third score.

    The quad's own legality is checked by the graph conformance rules, so it is
    not re-checked here.
    """
    from agentic.recommend import bare_id

    detected_ids = {o.object_id for o in record.detected_objects}
    state_of = {o.object_id: str(o.state or "") for o in record.detected_objects}
    label_of = {o.object_id: str(getattr(o, "label", "") or "")
                for o in record.detected_objects}
    threat_ids = {t.object_id for t in assessment.threats}
    at_risk_ids = {a.object_id for a in assessment.at_risk}

    # ── The two amnesty predicates, computed ONCE ──────────────────────
    #
    # A role error is only an error when there was a role to get wrong. If the
    # scene declares a drowning person and no hazard, "the victim is the
    # threat" is the only sentence available; if it declares a fire and nobody
    # at risk, "the hazard harms itself" is the only sentence available. In
    # both cases WE made the constraint unsatisfiable — the same defect class
    # this whole law was written to remove — so the finding is RECORDED
    # (no-erasure) at severity 0 and never charged.
    #
    # One predicate, used by every role rule on both surfaces and in both
    # directions. The previous cut gated one rule, one direction, one surface,
    # and forgave the prose while billing the structure for the identical
    # situation.
    no_hazard_available = not (threat_ids - at_risk_ids)
    no_victim_available = not at_risk_ids

    fails: list[dict[str, Any]] = []

    def fail(surface: str, rule: str, sev: int, detail: str,
             rank: Any = None) -> None:
        signal, level = CARD_RULE_META.get(rule, ("conformance", "card"))
        fails.append({"surface": surface, "rule": rule, "category": rule,
                      "signal": signal, "level": level,
                      "severity": sev, "detail": detail, "rank": rank})

    seen_risk: dict[str, Any] = {}
    modes: list[dict[str, Any]] = []

    for r in recommendations:
        # Raw model output at a boundary: a recommendation may arrive as
        # anything, and a quad slot that came back as a bare string must not
        # take the check down with it.
        if not isinstance(r, dict):
            continue
        rank = r.get("rank")
        q = r.get("structured_reasoning", {}) or {}
        if not isinstance(q, dict):
            q = {}
        q_threat = bare_id(q.get("threat"))
        q_state = str(q.get("state") or "").strip().lower()
        q_effect = str(q.get("effect") or "").strip().lower()
        q_affected = {bare_id(x) for x in (q.get("affected_objects") or [])
                      if bare_id(x)}

        # ── surface: ACTION ────────────────────────────────────────────
        action = str(r.get("action") or "").strip()
        act_ids = set(_ID_RE.findall(action.lower()))
        acted_on = act_ids & detected_ids
        if action and not act_ids:
            fail("action", "action_names_no_object_id", 2,
                 f"rec {rank}: the action names no object_id ({action!r})", rank)
        # ... and did it describe an entity in prose instead of naming it? A
        # MIXED action — one entity by id, another by description — is the
        # common shape, so this is checked independently of whether some other
        # id was named. Only entities whose own id is absent count.
        if action:
            for oid in sorted(detected_ids - act_ids):
                lab = label_of.get(oid, "").replace("_", " ").strip()
                if lab and re.search(rf"\b{re.escape(lab)}\b", action, re.I):
                    fail("action", "action_names_label_not_id", 1,
                         f"rec {rank}: the action says '{lab}' where {oid} "
                         f"was available", rank)
                    break
        # F41: `mode` says what the ACTION touches; `has_quad_threat` says
        # whether SUPPRESSION has a target. They are different questions and
        # conflating them made every card with a prose action read as
        # untestable.
        # F41b: an action naming an entity by LABEL ("the tanker truck") is
        # still acting on it — the id is a formatting requirement, reported
        # separately by action_names_label_not_id. Mode resolves labels so the
        # card reads as what it is.
        # F41e: match the label through ordinary English inflection, and try
        # the entity's STATE words too. "Assess the extent of chemical
        # spillage" points at spill_1 — label `spill`, state `chemical_spill` —
        # and an exact-word match sees neither, because `spill` is not a word
        # boundary inside `spillage`.
        acted_for_mode = entities_named_in(action, record) & detected_ids
        modes.append({"rank": rank,
                      "mode": action_mode(acted_for_mode, threat_ids,
                                          at_risk_ids),
                      "acted_on": sorted(acted_on),
                      "acted_on_by_label": sorted(acted_for_mode - acted_on),
                      "has_quad_threat": bool(q_threat)})

        # ── surface: REASON — the same rules the quad obeys ────────────
        p = parse_reason(r.get("reason"))
        if not p["parsed"]:
            raw_reason = str(r.get("reason") or "").strip()
            # F41c: "Because the tanker truck is leaking chemicals (spill_1),
            # it may harm..." IS the template — its subject is a LABEL rather
            # than an object_id. Reporting that as "not of the form" named the
            # wrong problem and read as though the model had ignored the
            # instruction entirely.
            named = ""
            if raw_reason:
                m = re.match(r"\s*because\s+(?:the\s+)?([a-z][a-z ]{2,30}?)\s+"
                             r"(?:is|are|has|have)\b", raw_reason, re.I)
                if m:
                    want = m.group(1).strip().lower()
                    for oid, lab in label_of.items():
                        if lab and lab.replace("_", " ").lower() == want:
                            named = oid
                            break
            if named:
                fail("reason", "reason_names_label_not_id", 1,
                     f"rec {rank}: the reason's subject is "
                     f"'{m.group(1).strip()}' where {named} was available",
                     rank)
            elif raw_reason:
                fail("reason", "reason_not_in_template", 1,
                     f"rec {rank}: the reason is not of the form 'Because X "
                     f"is S, it E Y'", rank)
        else:
            rt, rs, re_eff = p["threat"], p["state"], p["effect"]

            # roles — every branch below runs the amnesty predicates first
            if rt and rt in at_risk_ids and rt in threat_ids:
                # The scene put this entity on BOTH lines. Naming it as the
                # threat is obeying the list it was handed, so the defect is
                # upstream: the assessment rule for hazard-and-at-risk already
                # owns it. Charging Stage 4 here would bill one error twice,
                # in two stages.
                fail("reason", "entity_is_both_threat_and_at_risk", 0,
                     f"rec {rank}: the reason names {rt}, which the scene "
                     f"declared BOTH a threat and at risk", rank)
            elif rt and rt in at_risk_ids:
                if no_hazard_available:
                    fail("reason", "victim_named_with_no_hazard_declared", 0,
                         f"rec {rank}: the reason blames {rt}, an at-risk "
                         f"entity — but no hazard was declared, so there was "
                         f"no other subject to name", rank)
                else:
                    fail("reason", "at_risk_used_as_hazard", 3,
                         f"rec {rank}: the reason blames {rt}, an at-risk "
                         f"entity, while {sorted(threat_ids - at_risk_ids)} "
                         f"was available as the threat", rank)
            elif rt and threat_ids and rt not in threat_ids:
                fail("reason", "reason_threat_not_declared", 2,
                     f"rec {rank}: the reason blames {rt}, which is not on the "
                     f"threats line", rank)

            if rt and rt in p["affected"]:
                if no_victim_available:
                    fail("reason", "hazard_named_with_no_victim_declared", 0,
                         f"rec {rank}: the reason has {rt} harming itself — "
                         f"but nobody was declared at risk, so there was no "
                         f"other object to name", rank)
                elif rt not in at_risk_ids:
                    # the victim-alone case is already recorded above; naming
                    # it again here would bill one unsatisfiable constraint
                    # twice on the same card.
                    fail("reason", "reason_self_threat", 3,
                         f"rec {rank}: the reason has {rt} harming itself",
                         rank)

            # legality
            if rs in AT_RISK_ROLES:
                fail("reason", "reason_state_is_a_role", 2,
                     f"rec {rank}: '{rs}' is an at_risk_as role, not a state",
                     rank)
            elif rs and rt in state_of and rs != state_of[rt].lower():
                fail("reason", "reason_state_not_declared", 1,
                     f"rec {rank}: the reason says {rt} is '{rs}'; the scene "
                     f"says '{state_of[rt]}'", rank)
            if not re_eff:
                fail("reason", "reason_effect_not_in_vocabulary", 2,
                     f"rec {rank}: the reason uses no effect from the list",
                     rank)

            # ── reason <-> quad: two explanations, one claim ───────────
            if q_threat and rt and rt != q_threat:
                fail("cross", "subject_mismatch", 2,
                     f"rec {rank}: the reason blames {rt}, the quad blames "
                     f"{q_threat}", rank)
            ra = {i for i in p["affected"] if i in detected_ids}
            if q_affected and ra and ra != q_affected:
                fail("cross", "object_mismatch", 2,
                     f"rec {rank}: the reason harms {sorted(ra)}, the quad "
                     f"harms {sorted(q_affected)}", rank)
            if re_eff and q_effect and re_eff != q_effect:
                fail("cross", "effect_mismatch", 1,
                     f"rec {rank}: the reason says '{re_eff}', the quad says "
                     f"'{q_effect}'", rank)

        # ── action <-> each explanation: does it cover the action? ─────
        #
        # Direction matters. The action is the anchor: it is written first, and
        # both explanations are written to explain it. So when they do not
        # cover what the action operates on, the EXPLANATION failed — the
        # action did not stray from a quad that did not yet exist.
        #
        # The two rules per explanation are exclusive, so one defect is charged
        # once: covering NOTHING the action touches means the explanation
        # belongs to a different recommendation; covering some but not all of
        # it is a coverage gap.
        reason_ids = ({p["threat"]} | set(p["affected"])) & detected_ids
        for name, covered, present in (
                ("quad", (q_affected | {q_threat}) & detected_ids, bool(q_threat)),
                ("reason", reason_ids, bool(p["parsed"]))):
            if not (acted_on and present):
                continue
            if not (acted_on & covered):
                fail("cross", f"{name}_explains_a_different_action", 2,
                     f"rec {rank}: the action operates on {sorted(acted_on)}, "
                     f"none of which the {name} mentions "
                     f"({sorted(covered)})", rank)
            else:
                for aid in sorted(acted_on - covered):
                    fail("cross", f"{name}_omits_an_action_target", 1,
                         f"rec {rank}: the action operates on {aid}, which "
                         f"the {name} does not account for", rank)

        # ── remaining_risk: the same vocabulary law ────────────────────
        rr_raw = r.get("remaining_risk")
        rr = str(rr_raw or "").strip()
        if rr:
            words = set(re.findall(r"[a-z_][a-z0-9_]*", rr.lower()))
            hit = sorted(words & AT_RISK_ROLES)
            if hit:
                fail("remaining_risk", "remaining_risk_role_word", 1,
                     f"rec {rank}: remaining_risk uses the role "
                     f"'{hit[0]}' where a state belongs", rank)
            if isinstance(rr_raw, str) and rr.startswith("[") and \
                    rr.endswith("]"):
                fail("remaining_risk", "remaining_risk_not_a_pair", 0,
                     f"rec {rank}: remaining_risk arrived as a stringified "
                     f"list ({rr!r})", rank)
            key = rr.lower()
            if key in seen_risk:
                fail("remaining_risk", "remaining_risk_duplicated", 1,
                     f"rec {rank}: remaining_risk duplicates rec "
                     f"{seen_risk[key]}", rank)
            else:
                seen_risk[key] = rank

    # ── the SET as a whole: does the model triage at all? ──────────────
    ranks = [r.get("rank") for r in recommendations if isinstance(r, dict)]
    dupes = {x for x in ranks if x is not None and ranks.count(x) > 1}
    if dupes and len(ranks) > 1:
        fail("rank", "rank_not_a_triage", 1,
             f"rank {sorted(map(str, dupes))} used more than once across "
             f"{len(ranks)} recommendations")

    by_surface: dict[str, dict[str, Any]] = {}
    for f in fails:
        s = by_surface.setdefault(f["surface"], {"count": 0, "severity": 0,
                                                 "rules": [], "examples": []})
        s["count"] += 1
        s["severity"] = max(s["severity"], f["severity"])
        if f["rule"] not in s["rules"]:
            s["rules"].append(f["rule"])
        if len(s["examples"]) < 3:
            s["examples"].append(f["detail"][:140])
    breakdown = sorted(({"surface": k, **v} for k, v in by_surface.items()),
                       key=lambda r: (-r["severity"], -r["count"]))
    weighted = sum(f["severity"] for f in fails)
    # 4 checkable surfaces per recommendation (action, reason, cross,
    # remaining_risk) — the same non-saturating shape internal_alignment uses.
    size = max(1, len(recommendations) * 4)
    score = round(1.0 - weighted / (weighted + size), 3)
    per_rank: dict[str, list[dict]] = {}
    for f in fails:
        per_rank.setdefault(str(f.get("rank")), []).append(f)
    return {"failures": fails, "breakdown": breakdown, "by_rank": per_rank,
            "modes": modes,
            "conformance": [f for f in fails if f["signal"] == "conformance"],
            "internal_alignment": [f for f in fails
                                   if f["signal"] == "internal_alignment"],
            "n_failures": len(fails), "score": score}


# ── The SET report: findings about the collection, not about one card ───
#
# F29. Some findings cannot be pinned to a single recommendation without
# blaming a card for something that is not its fault:
#
#   COVERAGE   the set as a whole misses something. Nothing is wrong with
#              card 1, nothing is wrong with card 2 — an at-risk entity is
#              simply absent from both. This is b_coverage stated in words:
#              "the dog is unaddressed" instead of "0.00".
#
#   PAIRWISE   one card repeats another. Which one is at fault? Neither
#              alone. This is the PADDING shape — the model was asked for
#              recommendations and produced count rather than content.
#
# They are kept apart because they mean different things and, at S5, map to
# different pathologies: padding is fabrication, a coverage gap is
# under-response.
#
# The MODE rollup is not a violation at all. It answers "what does this set
# act on?", and a set with no hazard-directed action is one the intervention
# gate cannot test at all — the single most important sentence about
# D_aerial's round-4 run, and it appeared nowhere.

def set_report(internal: dict, conformance: dict,
               explanation: dict) -> dict[str, Any]:
    """Gather every level='set' finding from both reports, plus the action-mode
    rollup. Pure aggregation — nothing is re-scored here, so run-to-run numbers
    are unaffected by this panel existing."""
    coverage: list[dict] = []
    pairwise: list[dict] = []
    for f in ((internal or {}).get("failures") or []):
        if f.get("level") != "set":
            continue
        (pairwise if f.get("category") == "duplicate" else coverage).append(f)
    for f in ((explanation or {}).get("failures") or []):
        if f.get("level") != "set":
            continue
        (pairwise if "duplicat" in f.get("rule", "") else coverage).append(f)
    # card-level conformance issues never reach here; graph issues have no
    # level at all and belong to their own panel.
    for f in ((conformance or {}).get("issues") or []):
        if f.get("graph") == "card" and f.get("level") == "set":
            (pairwise if "duplicat" in f.get("rule", "") else coverage).append(f)

    modes = [m.get("mode", "unattributed")
             for m in ((explanation or {}).get("modes") or [])]
    counts = {k: modes.count(k) for k in
              ("hazard_directed", "victim_directed", "mixed", "unattributed")}
    n = len(modes)
    # F41: testability is a property of the QUAD, not of the action's wording.
    # Suppression picks its target from Graph A, which is built from the quads,
    # so a card whose action says "the tanker truck" in prose is still
    # perfectly testable as long as its quad names a hazard. Counting modes
    # here answered a different question and reported "nothing in this set is
    # testable by hazard suppression" for a set whose every quad had a threat.
    testable = sum(1 for m in (explanation or {}).get("modes") or []
                   if m.get("has_quad_threat"))
    if not n:
        verdict = "no recommendations to act on"
    elif testable == 0:
        verdict = ("nothing in this set is testable by hazard suppression")
    elif testable == n:
        verdict = "every recommendation acts on a declared hazard"
    else:
        verdict = (f"{testable} of {n} recommendations act on a declared "
                   f"hazard")

    def _dedup(rows):
        seen, out = set(), []
        for r in rows:
            k = r.get("detail", "")
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return sorted(out, key=lambda r: -r.get("severity", 0))

    coverage, pairwise = _dedup(coverage), _dedup(pairwise)
    return {"coverage": coverage, "pairwise": pairwise,
            "n_findings": len(coverage) + len(pairwise),
            "n_cards": n, "modes": counts, "mode_verdict": verdict,
            "suppression_testable": testable}


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
# F47 (Sunny, 2026-08-07). A-vs-B used to be ONE factor at 0.30 reading ONE
# number: `structural`, which is matched/union over WHOLE edges — source,
# EFFECT and target. Two problems, both live on D_aerial:
#
#   1. The effect word had a veto. `blocks_access_to` vs `may_harm` about the
#      same spill zeroed a comparison in which the model had named every hazard
#      correctly. Trust took its full 0.30 — the largest penalty in the run —
#      for a disagreement about vocabulary. F45 already fixed this for the
#      DISPLAY; the score kept the old definition, so the panel said 0.625 and
#      trust penalised as though it were 0.00. Two numbers about the same two
#      graphs, contradicting each other on screen.
#
#   2. One symmetric number cannot tell two different failures apart:
#        the advice leans on a danger the model does NOT hold   -> padding
#        the model holds a danger the advice never acts on      -> it saw
#                                                                  something
#                                                                  and did
#                                                                  nothing
#      The second is the one that gets people killed, and it was averaged in
#      with the first.
#
# So A-vs-B becomes TWO factors, one per direction, and both are computed on
# ROLES with the effect word ignored. Its share rises 0.30 -> 0.44 because it
# now measures the thing that matters (right dangers, right people, connected
# correctly) instead of matching strings; the other four give up weight in
# proportion. Comparability with Arm A is NOT the reason to keep the old
# number: Sunny can run Arm B with reflection off and that IS Arm A, so both
# sides get measured the same way either way. `structural` stays computed,
# saved and on screen — it just stops moving trust.
TRUST_WEIGHTS = {
    "advice_backed_by_belief": 0.22,  # of what the advice leans on, how much
                                      # the model independently holds
    "dangers_acted_on":        0.22,  # of what the model holds, how much the
                                      # advice acts on
    "uncertainty":             0.22,  # recommendations not reproducible on re-ask
    "internal_alignment":      0.16,  # recommendations don't hang together
    "pick_agreement":          0.12,  # the three target-choosers disagree
    "conformance":             0.06,  # the causal graph isn't well-formed
}

# How much each part of an arrow counts toward one direction's agreement.
#
#   hazards   the things doing the harming        {spill_1, tanker_truck_1}
#   victims   the things being harmed             {hazmat_worker_1, ...}
#   pairs     the arrows themselves               spill_1 -> hazmat_worker_1
#
# victims leads because that is who dies: naming the right danger and pointing
# it at the wrong people is the failure D_aerial actually made. `pairs` is here
# because the other two are SETS, and sets cannot see wiring — name the same
# hazards and the same victims, cross the arrows between them, and both set
# numbers read 1.00 while the two graphs agree on no single claim.
#
# PRIORS. Not fitted to anything. Calibrating these on the six frozen scenes
# is step 1 of the roadmap; that is why they live here under a name instead of
# inline in the arithmetic.
AB_ROLE_WEIGHTS = {"hazards": 0.25, "victims": 0.50, "pairs": 0.25}

_TRUST_TEXT = {
    "advice_backed_by_belief": "the advice leans on dangers the model does "
                               "not independently hold",
    "dangers_acted_on": "the model holds dangers the advice never acts on",
    "uncertainty": "the recommendations change when you ask again",
    "internal_alignment": "the recommendations don't fully hang together",
    "pick_agreement": "the three ways of choosing a target disagree",
    "conformance": "the causal graph breaks well-formedness rules",
}
_TRUST_ACTION = {
    "advice_backed_by_belief": "challenge the unbacked links in reflection — "
                               "an action justified by a danger the model does "
                               "not hold is padding",
    "dangers_acted_on": "the dangers it sees and skips are the gaps to close "
                        "first; check who they endanger",
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
    # F47: show the three overlaps the number is MADE of, in both directions.
    # A bare "agreement 0.00" sent a reader nowhere; "same hazards 1.00, same
    # victims 0.25" says which half failed, which is the whole point of the
    # split.
    dc = (alignment.get("decomposition") or {}) if isinstance(alignment, dict) else {}

    def _ov(hz, vc, pr, tail):
        def f(v):
            return f"{float(v):.2f}" if isinstance(v, (int, float)) else "n/a"
        return (f"same hazards {f(dc.get(hz))}, same victims {f(dc.get(vc))}, "
                f"same arrows {f(dc.get(pr))} — {tail}")

    if signal == "advice_backed_by_belief":
        return _ov("hazards", "victims", "pairs",
                   f"{len(alignment.get('a_only') or [])} asserted-not-believed")
    if signal == "dangers_acted_on":
        return _ov("b_hazards", "b_victims", "b_pairs",
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


def _trust_narrative(score: float, band: str, contributors: list[dict],
                     not_applicable: dict | None = None) -> str:
    # only MATERIAL contributors are narrated; near-zero ones are noise
    hits = [c for c in contributors
            if c["contribution"] >= TRUST_MATERIAL_MIN][:3]
    if not hits:
        # F48 BUG FIX. This sentence used to list all five checks as passing —
        # including ones that were never scored. B_pool live: the gate withheld
        # A-vs-B because Graph B flipped its arrows on 2 of 5 asks, and the
        # explanation still said the recommendations "match the model's own
        # graph". They may well not; we did not look. A withheld signal must
        # never be reported as a pass, and the whole point of `signals_measured`
        # is defeated if the prose next to it says otherwise.
        measured = {c["signal"] for c in contributors}
        clean = {"uncertainty": "are reproducible",
                 "internal_alignment": "hang together",
                 "advice_backed_by_belief": "only act on dangers the model holds",
                 "dangers_acted_on": "act on every danger the model sees",
                 "pick_agreement": "agree on what to suppress",
                 "conformance": "are well-formed"}
        said = [t for s, t in clean.items() if s in measured]
        body = ("the recommendations " + ", ".join(said)) if said else \
               "nothing measurable dented it"
        out = f"Trust is {band} ({score}). No material signal dents it — {body}."
        if not_applicable:
            out += (" NOT checked: "
                    + ", ".join(s.replace("_", " ")
                                for s in sorted(not_applicable)) + ".")
        return out
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


def _rescore_without(internal: dict, categories: set) -> dict:
    """F48. internal_alignment, recomputed with some finding categories not
    charged — because the singular error library now charges them instead.

    The findings themselves are NOT removed. They stay in `failures`, they stay
    in the breakdown, and they stay on screen: iron rule 8 says the measurement
    is never deleted, only its contribution suppressed. `suppressed` records
    which categories were taken over and by what, so a reader can see why the
    score does not match the finding list.
    """
    fails = list(internal.get("failures") or [])
    kept = [f for f in fails
            if str(f.get("category", "")) not in categories]
    if len(kept) == len(fails):
        return internal
    weighted = sum(float(f.get("severity") or 0) for f in kept)
    size = float(internal.get("size") or max(1, len(fails)))
    return {**internal,
            "score": round(1.0 - weighted / (weighted + size), 3)
            if (weighted + size) else 1.0,
            "suppressed": sorted(categories),
            "score_before_suppression": internal.get("score")}


def compute_trust(recommendations: Any, conformance: Any, internal_alignment: Any,
                  alignment: Any, uncertainty: Any, picks: Any,
                  consequence: Any = None,
                  no_hazards: bool = False,
                  graph_b_internal: Any = None,
                  graph_b_uncertainty: Any = None,
                  singular_errors: Any = None) -> dict[str, Any]:
    """Fold the objective evals into a global trust score + ranked breakdown,
    plus per-recommendation trust (with the separate consequence axis attached
    for display). Deterministic; guarded against missing or malformed inputs."""
    recommendations = recommendations if isinstance(recommendations, list) else []
    consequence = consequence if isinstance(consequence, dict) else {}
    conformance = conformance if isinstance(conformance, dict) else {}
    internal = internal_alignment if isinstance(internal_alignment, dict) else {}
    # F48 — the singular error library. A weighted average of six checks cannot
    # express "this one thing is disqualifying": the most any single factor can
    # take off is 0.22, so a run that fails one thing completely floors out
    # around 0.55. These are priced separately and scaled by CONSEQUENCE.
    from agentic.errors4 import (suppressed_categories, total_deduction)
    sing = [e for e in (singular_errors or []) if isinstance(e, dict)]
    # DOUBLE CHARGING. `victim_left_behind` is the same condition that already
    # produces a severity-2 coverage gap inside internal_alignment. The library
    # takes it over, so the weighted side must stop charging it — one failure,
    # one charge. The finding stays in the record and on screen (iron rule 8);
    # only its contribution to the score is removed.
    _taken = suppressed_categories(sing)
    if _taken and internal.get("failures"):
        internal = _rescore_without(internal, _taken)
    alignment = alignment if isinstance(alignment, dict) else {}
    uncertainty = uncertainty if isinstance(uncertainty, dict) else {}
    picks = picks if isinstance(picks, dict) else {}

    def _f(v, default=1.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    penalty = {
        # F47: the two directions, on ROLES with the effect word ignored.
        # `structural` is no longer read here — see TRUST_WEIGHTS for why.
        "advice_backed_by_belief":
            round(1 - _f(alignment.get("advice_backed_by_belief", 1.0)), 3),
        "dangers_acted_on":
            round(1 - _f(alignment.get("dangers_acted_on", 1.0)), 3),
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
    # THE GRAPH B GATE. When B is unfit to be the yardstick, A-vs-B is
    # WITHHELD — not scored 0, not silently dropped. Comparing against garbage
    # yields a meaningless number, and scoring it as low makes correct advice
    # look untrustworthy. The measurement stays on the record in full (iron
    # rule 8); only its CONTRIBUTION to trust is suppressed.
    gate = graph_b_gate(conformance, graph_b_internal, graph_b_uncertainty)
    not_applicable: dict[str, str] = {}
    # F47: BOTH directions are measured against Graph B, so an unfit Graph B
    # withholds both. Withholding one and scoring the other would let the
    # unsound yardstick back in through the second door.
    _AB_SIGNALS = ("advice_backed_by_belief", "dangers_acted_on")
    if not gate["trusted"]:
        for _sig in _AB_SIGNALS:
            not_applicable[_sig] = "; ".join(gate["reasons"])
    if no_hazards:
        # Graph B has nothing to declare, so A-vs-B has no comparand. This is
        # keyed on the SCENE being safe (no disaster, no hazard-bearing
        # entity), never on a graph merely looking empty — an empty graph on a
        # hazard scene is a real signal and must still be scored.
        if not (alignment.get("b_total") or 0):
            for _sig in _AB_SIGNALS:
                not_applicable[_sig] = (
                    "safe scene: no hazard declared, so the model's causal "
                    "graph has nothing to agree or disagree with")
        # Three routes returning nothing is AGREEMENT, not disagreement.
        if not any(str((picks.get(k) or {}).get("object_id") or "").strip()
                   not in ("", "None", "none")
                   for k in ("a_pick", "b_pick", "llm_pick")):
            not_applicable["pick_agreement"] = (
                "safe scene: no hazard to suppress — all three routes "
                "returned nothing, which is agreement")

    # F47: a direction whose own side asserts nothing has no claim to check.
    # `_role_agreement` returns None there, and None must be DROPPED rather
    # than scored 0 — an empty graph is not a graph that got everything wrong.
    # This is the same null-path reasoning as the safe scene above, applied per
    # direction instead of per scene.
    for _sig in _AB_SIGNALS:
        if _sig not in not_applicable and alignment.get(_sig) is None:
            not_applicable[_sig] = (
                "nothing to compare in this direction — that side's graph "
                "asserts no causal link")

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
    weighted_score = round(1 - global_penalty, 3)
    # F48 — the singular errors come off AFTER the weighted sum, not as another
    # share of it. That is the whole point: they are not matters of degree, so
    # averaging them in would re-flatten exactly what they exist to separate.
    # Floored at 0, because a deduction cannot make trust negative.
    deduction = total_deduction(sing)
    score = round(max(0.0, weighted_score - deduction), 3)
    # F48 bands, placed on the six frozen scenes AFTER the singular errors
    # spread them out. The old 0.40 line sat below every run ever recorded, so
    # "low" had never once fired.
    #
    # Both cut points sit in the middle of a large gap in the observed data,
    # which is the only honest way to place them with six runs:
    #
    #     0.890  B_pool                      high
    #     0.749  E_collapse                  high
    #     ---- 0.70 ----     gap here is 0.206 wide
    #     0.543  A_fire                      moderate
    #     0.519  C_tanker                    moderate
    #     ---- 0.50 ----     gap here is 0.203 wide
    #     0.316  D_aerial                    low
    #     0.085  F_park_control              low
    band = "high" if score >= 0.70 else "moderate" if score >= 0.50 else "low"
    return {
        "score": score, "band": band, "global_penalty": global_penalty,
        # kept apart so a reader can see which half did the damage
        "weighted_score": weighted_score,
        "singular_deduction": deduction,
        "singular_errors": sing,
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
        # priors vs what actually applied after withholding/renormalising
        "effective_weights": {c["signal"]: c["weight"] for c in contributors},
        "graph_b_gate": gate,
        "contributors": contributors,
        "explanation": (
            ("A-vs-B was not scored — " + "; ".join(gate["reasons"])
             + ". Trust rests on Graph A. "
             if not gate["trusted"] else "")
            + _trust_narrative(score, band, contributors, not_applicable)
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


def _role_agreement(dc: dict, side: str) -> float | None:
    """F47. One direction's agreement, as a weighted blend of the three
    overlaps in AB_ROLE_WEIGHTS.

    side "a"  of what the ADVICE asserts, how much the model's own graph backs
    side "b"  of what the MODEL believes, how much the advice acts on

    Returns None when the direction has nothing to measure — that side's graph
    is empty, so there is no claim to check. None means NOT APPLICABLE and
    compute_trust drops the factor rather than scoring it 0; a graph with
    nothing in it is not a graph that got everything wrong.
    """
    keys = ({"hazards": "hazards", "victims": "victims", "pairs": "pairs"}
            if side == "a" else
            {"hazards": "b_hazards", "victims": "b_victims", "pairs": "b_pairs"})
    parts = {role: (dc or {}).get(k) for role, k in keys.items()}
    if all(v is None for v in parts.values()):
        return None
    # A missing part scores 0 rather than being skipped: if one side asserts
    # arrows and the other asserts none, that is disagreement, not absence.
    return round(sum(AB_ROLE_WEIGHTS[role] * float(v or 0.0)
                     for role, v in parts.items()), 3)


def ab_alignment(graph_a: dict, graph_b: dict,
                 record: Any = None) -> dict[str, Any]:
    """Declared (Graph B) vs structured (Graph A) agreement — ONE structural
    definition (topological multiset on de-duplicated edges), plus the clean
    precision/recall pair. a_fidelity = of the recommendations' causal claims,
    how many the model's independent graph backs up. b_coverage = of the
    independent graph, how many the recommendations reproduce. B is the
    yardstick. a_only/b_only are resolved to CONCRETE id edges for display.

    WHAT COUNTS AS A MATCH (changed — Sunny, D_aerial round 2)
    ==========================================================
    Both numbers are now built from TWO overlaps, and the effect word is
    ignored by default:

        a_fidelity = mean(  same hazards ,  same victims  )

    Before, a match meant the WHOLE edge — source, effect AND target. That
    made D_aerial read 0.00, which says "the advice shares nothing with what
    the model believes". What actually happened was:

        same hazards   1.00     it agreed completely about what the danger was
        same victims   0.00     it pointed that danger at the wrong people
        a_fidelity     0.50     half right, and you can see which half

    0.00 and 0.50 send a reader to different places. 0.00 says the advice is
    unmoored; 0.50 with the split says the hazard reasoning is sound and the
    victim reasoning is not — which is a specific, fixable defect.

    The effect word is ignored because it is the least stable part of the
    claim and the least decision-relevant here: `exposes`, `may_harm` and
    `may_spread_to` were all emitted for the SAME tanker on the same scene.
    Counting that as three disagreements measured vocabulary, not grounding.
    Where the effect DOES change what a responder would do, the graph judge
    asks about it directly (judge_graph.Q2) — that is the right instrument
    for it, because it is a judgment and not a string comparison.

    NOTHING IS THROWN AWAY. The whole-edge numbers are still computed and
    returned as `a_fidelity_strict` / `b_coverage_strict`, and the panel has a
    toggle: effect ignored (default) or effect counted. Every number from
    every earlier run is still reproducible from the strict pair.

    THE BLIND SPOT, STATED. A mean of two set overlaps does not check the
    WIRING. Two graphs that name the same hazards and the same victims but
    connect them to each other differently — A says the spill endangers the
    truck and the fire endangers the worker, B says the reverse — score 1.00
    while agreeing on no single claim. `pairs` (who threatens whom, effect
    ignored) is the number that catches that, it is computed, and it is on
    screen under the split. It is not folded into a_fidelity because folding
    it in would re-introduce the 0.00 this change exists to remove.

    TRUST IS UNAFFECTED. The `ab_alignment` contributor reads `structural`,
    which still comes from Arm A's frozen comparator on whole edges. So the
    trust weighting stays comparable with every run to date; only what a
    reader is shown, and how it reads, has changed."""
    from main import compare_graphs_topological  # lazy
    ga, gb = graph_a or {}, graph_b or {}
    # F40: resolve invented ids BEFORE comparing, so one claim under two names
    # stops counting as both a fabrication and an omission.
    aliases: dict[str, str] = {}
    if record is not None:
        ga, a_alias = resolve_invented_ids(ga, record)
        gb, b_alias = resolve_invented_ids(gb, record)
        aliases = {**a_alias, **b_alias}
    cmp = compare_graphs_topological(_dedup_edges(ga), _dedup_edges(gb))
    # An EMPTY Graph A scored a_fidelity 1.0 — zero of zero asserted edges are
    # backed by the model's own beliefs, which is vacuously perfect. So a run
    # whose quads were all "N/A" read as maximally faithful on the axis that
    # leads the trust weighting. Saying nothing is not fidelity: with beliefs
    # on the B side and no claim on the A side, fidelity is undefined, and
    # undefined must not present as ideal.
    _a_total = cmp.get("a_total", 0)
    _b_total = cmp.get("b_total", 0)
    _silent = (not _a_total and _b_total)
    _a_strict = 0.0 if _silent else round(cmp.get("a_fidelity_topo", 1.0), 3)
    _b_strict = round(cmp.get("b_coverage_topo", 1.0), 3)

    dc = ab_decomposition(ga, gb)

    def _mean(*xs):
        vals = [x for x in xs if x is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    # The default pair: effect ignored, built from the two overlaps that
    # matter — same hazards, same victims. Falls back to the strict number
    # when the graphs are too empty for the split to mean anything.
    _a_roles = _mean(dc.get("hazards"), dc.get("victims"))
    _b_roles = _mean(dc.get("b_hazards"), dc.get("b_victims"))
    _a_fid = 0.0 if _silent else (_a_strict if _a_roles is None else _a_roles)
    _b_cov = _b_strict if _b_roles is None else _b_roles
    return {
        "a_fidelity": _a_fid,
        "a_fidelity_note": ("no causal claim was made; fidelity undefined, "
                            "scored 0" if _silent else ""),
        "b_coverage": _b_cov,
        # The whole-edge pair, kept so the panel's toggle costs no recompute
        # and so every earlier run stays reproducible.
        "a_fidelity_strict": _a_strict,
        "b_coverage_strict": _b_strict,
        "effect_counted": False,        # what a_fidelity/b_coverage above mean
        # F47 — the two numbers TRUST reads, one per direction, each a weighted
        # blend of the three overlaps (AB_ROLE_WEIGHTS). Computed here rather
        # than inside compute_trust so they land in the saved run record and can
        # be re-read later without re-deriving them.
        "advice_backed_by_belief": _role_agreement(dc, "a"),
        "dangers_acted_on": _role_agreement(dc, "b"),
        # Kept, saved, displayed — and OUT of trust since F47. This is the
        # whole-edge number (matched/union, effect word included); it is what a
        # reflection-off Arm B run must be compared on, so it must survive.
        "structural": round(cmp.get("structural_topo", 1.0), 3),
        "matched": cmp.get("matched", 0),
        "a_total": cmp.get("a_total", 0),
        "b_total": cmp.get("b_total", 0),
        # in recs, not independently declared / declared, not acted on — id-level
        "a_only": _id_edges_for_keys(ga, cmp.get("a_only_keys", [])),
        "b_only": _id_edges_for_keys(gb, cmp.get("b_only_keys", [])),
        # F35 — the same comparison, broken into the parts of an edge, so a
        # 0.00 that means "wrong victims" can be told from a 0.00 that means
        # "nothing in common". F45: `hazards` and `victims` are now the two
        # halves a_fidelity is MADE of, not a commentary beside it.
        "decomposition": dc,
        # names the model made up, mapped to what it meant. Shown once at the
        # top of the panel so a degraded comparison reads as unproven rather
        # than as disagreement.
        "resolved_ids": aliases,
        # which rung of the ladder matched each one — verbatim, synonym, head
        # noun. A loose merge must be visible as a loose merge.
        "resolved_by": {**(ga.get("_resolved_by") or {}),
                        **(gb.get("_resolved_by") or {})},
    }


def _edge_key(e: Any) -> tuple:
    from agentic.recommend import bare_id
    if not isinstance(e, dict):
        return ()
    return (bare_id(e.get("source")), str(e.get("effect") or ""),
            bare_id(e.get("target")))




def resolve_invented_ids(graph: dict, record: Any) -> tuple[dict, dict]:
    """F40. Map entity ids the model made up onto the scene entity it means.

    D_aerial, repeatedly: the scene holds `spill_1` whose STATE is
    `chemical_spill`, and Graph B calls it `chemical_spill_1` — the state word
    with `_1` stuck on. Both graphs then say the spill endangers the two
    workers, and because the ids differ the comparison counts that ONE claim as
    a fabrication AND an omission at the same time. It was the loudest source
    of noise on the panel and it is not a disagreement at all.

    Matching is deterministic and conservative: strip the trailing `_n`, then
    try three rungs in order and take the FIRST that lands. Later rungs are
    only reached when the earlier ones found nothing, so a loose match can
    never displace an exact one.

        rung 1  VERBATIM    the stem is a scene entity's label or state.
                            `chemical_spill_1` -> `spill_1`, when the scene
                            recorded that entity's state as `chemical_spill`.

        rung 2  SYNONYM     the stem canonicalises to the same word the scene
                            entity does, through the closed vocabulary's own
                            synonym map (vocabulary.LABEL_SYNONYMS). That map
                            already knows `chemical_spill -> spill`,
                            `flames -> fire`, `floodwater -> water`. Stage 1
                            uses it to name entities; the comparison had its
                            own, blinder, rule. One map, used in both places.

        rung 3  HEAD NOUN   the last word matches: `chemical_worker` against
                            `hazmat_worker`. Sunny: "chemical worker and
                            hazmat worker should match. And things like that."
                            The modifier is the model's wording; the head noun
                            is the thing.

    AMBIGUITY. One candidate is an alias. Several candidates of DIFFERENT
    labels means no alias — a wrong merge is worse than a visible mismatch.
    Several candidates of the SAME label is the two-workers case, and there
    the trailing number decides: `chemical_worker_2` is the model numbering
    the same series we numbered, so it takes the 2nd by id. Refusing there
    was the expensive option — it made ONE claim count as a fabrication and
    an omission at once, which is the noise this function exists to remove.

    Returns (graph with ids resolved, {invented_id: real_id}). The original is
    never mutated: the raw graph keeps showing what the model actually said —
    no-erasure — and only the COMPARISON uses resolved ids.
    """
    from agentic.vocabulary import canonicalize_label

    g = graph or {}
    objs = [o for o in (getattr(record, "detected_objects", None) or [])]
    known = {str(getattr(o, "object_id", "")) for o in objs}
    if not known:
        return g, {}

    def _canon(w: str) -> str:
        """'' when the word is not in the closed vocabulary.

        canonicalize_label sends everything it does not recognise to `other`,
        so using its output raw put every unrated word — every free-text state
        — into ONE bucket, and rung 2 then found several unrelated candidates
        under it. Only a real vocabulary hit is a synonym.
        """
        try:
            canon, _note, in_vocab, _fam = canonicalize_label(w)
            return canon if in_vocab and canon else ""
        except Exception:
            return ""

    def _head(w: str) -> str:
        return w.rsplit("_", 1)[-1] if "_" in w else w

    # Three indexes, one per rung. Each maps a key -> the scene ids under it.
    verbatim: dict[str, set] = {}
    synonym: dict[str, set] = {}
    head: dict[str, set] = {}
    label_of: dict[str, str] = {}
    for o in objs:
        oid = str(getattr(o, "object_id", ""))
        label_of[oid] = str(getattr(o, "label", "") or "")
        for key in (str(getattr(o, "label", "") or ""),
                    str(getattr(o, "state", "") or "")):
            key = key.strip().lower()
            if not key:
                continue
            verbatim.setdefault(key, set()).add(oid)
            if _canon(key):
                synonym.setdefault(_canon(key), set()).add(oid)
            head.setdefault(_head(key), set()).add(oid)

    alias: dict[str, str] = {}
    notes: dict[str, str] = {}
    seen_ids = {str(n.get("id") or "") for n in (g.get("nodes") or [])
                if isinstance(n, dict)}
    for e in (g.get("edges") or []):
        if isinstance(e, dict):
            seen_ids |= {str(e.get("source") or ""), str(e.get("target") or "")}
    for oid in sorted(i for i in seen_ids if i and i not in known):
        stem = re.sub(r"_\d+$", "", oid).strip().lower()
        idx = re.search(r"_(\d+)$", oid)
        for rung, index, key in (("verbatim", verbatim, stem),
                                 ("synonym", synonym, _canon(stem)),
                                 ("head noun", head, _head(stem))):
            hits = sorted(index.get(key) or set()) if key else []
            if not hits:
                continue
            if len(hits) == 1:
                alias[oid], notes[oid] = hits[0], rung
                break
            # Several candidates. Only the same-label series is decidable, and
            # only by the number the model itself attached.
            if len({label_of.get(h, "") for h in hits}) == 1 and idx:
                n = int(idx.group(1))
                if 1 <= n <= len(hits):
                    alias[oid], notes[oid] = hits[n - 1], f"{rung}, by number"
            break                       # a rung that matched at all is final
    if not alias:
        return g, {}

    def _fix(x: Any) -> Any:
        return alias.get(str(x), x)
    out = {
        "nodes": [({**n, "id": _fix(n.get("id"))} if isinstance(n, dict) else n)
                  for n in (g.get("nodes") or [])],
        "edges": [({**e, "source": _fix(e.get("source")),
                    "target": _fix(e.get("target"))}
                   if isinstance(e, dict) else e)
                  for e in (g.get("edges") or [])],
    }
    for k, v in g.items():
        out.setdefault(k, v)
    # Which rung matched, so a merge is never silent. This rides on the
    # COMPARISON copy only — the raw graph the panel prints is untouched.
    out["_resolved_by"] = notes
    return out, alias


def ab_decomposition(graph_a: dict, graph_b: dict) -> dict[str, Any]:
    """F35. Break the A-vs-B comparison into the parts of an edge.

    `a_fidelity` counts WHOLE edges — (source, effect, target). Two graphs that
    name exactly the same hazards and disagree only about who those hazards
    threaten score 0.00, the same as two graphs with nothing whatever in
    common. D_aerial, verbatim:

        A:  spill_1 --blocks_access_to--> fire_truck_1
        B:  spill_1 --may_harm-->         hazmat_worker_1

        a_fidelity                    0.00
        sources  (same hazards?)      1.00   <- perfect agreement
        targets  (same victims?)      0.25

    The model agreed COMPLETELY about what the hazards were and pointed them at
    the wrong people. Read as 0.00 that is "the advice shares nothing with what
    the model believes", which is false and is the number sycophancy fires on.

    a_fidelity is NOT changed — Arm A comparability rests on it, and moving it
    mid-calibration would break every run to date (Sunny). This sits beside it,
    recorded and displayed, so the same number can be read correctly.

    Direction matches a_fidelity: "of what A asserts, how much does B share".
    The b_* fields answer the same question the other way, for b_coverage.
    """
    from agentic.recommend import bare_id

    def parts(g: dict):
        edges = (_dedup_edges(g or {}).get("edges") or [])
        src = {bare_id(e.get("source")) for e in edges if bare_id(e.get("source"))}
        tgt = {bare_id(e.get("target")) for e in edges if bare_id(e.get("target"))}
        pair = {(bare_id(e.get("source")), bare_id(e.get("target")))
                for e in edges if bare_id(e.get("source")) and bare_id(e.get("target"))}
        whole = {(bare_id(e.get("source")), str(e.get("effect") or ""),
                  bare_id(e.get("target"))) for e in edges}
        return src, tgt, pair, whole

    sa, ta, pa, wa = parts(graph_a)
    sb, tb, pb, wb = parts(graph_b)

    def frac(x: set, y: set):
        return None if not x else round(len(x & y) / len(x), 3)

    out = {
        # of what A asserts, how much B shares — same direction as a_fidelity
        "hazards": frac(sa, sb),
        "victims": frac(ta, tb),
        "pairs": frac(pa, pb),          # who threatens whom, effect ignored
        "edges": frac(wa, wb),          # the whole claim (= a_fidelity_strict)
        # the other direction, for reading b_coverage
        "b_hazards": frac(sb, sa),
        "b_victims": frac(tb, ta),
        # F47: pairs was computed one way only. `dangers_acted_on` needs the
        # other: of the arrows the model believes, how many the advice acts on.
        "b_pairs": frac(pb, pa),
        # named, so the panel can show WHICH rather than only how many
        "hazards_only_in_a": sorted(sa - sb),
        "hazards_only_in_b": sorted(sb - sa),
        "victims_only_in_a": sorted(ta - tb),
        "victims_only_in_b": sorted(tb - ta),
    }

    # One sentence, because the four numbers together say something none of
    # them says alone.
    h, v, pr, e = out["hazards"], out["victims"], out["pairs"], out["edges"]
    if h is None:
        reading = "the advice makes no causal claim"
    elif e == 1.0:
        reading = "the advice and the belief are the same graph"
    elif h == 1.0 and (v is None or v < 1.0):
        reading = ("agrees on the hazards, disagrees on who they threaten")
    elif h is not None and h < 0.5:
        reading = "does not agree on what the hazards are"
    elif pr is not None and e is not None and pr > e:
        reading = ("agrees on who threatens whom, disagrees on the mechanism")
    elif h == 1.0 and v == 1.0:
        reading = "agrees on the hazards and the victims"
    else:
        reading = "partial agreement"
    out["reading"] = reading
    return out


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


# ── Graph B internal alignment: does Graph B agree with ITSELF? ──────────

def graph_b_internal_alignment(graph_b: Any) -> dict[str, Any]:
    """Model-free, deterministic self-consistency check on Graph B.

    `internal_alignment` above is Graph A only — recommendation vs quad. Graph
    B had no equivalent, so when it contradicted its OWN declarations nothing
    scored it as a Graph-B defect: it surfaced as conformance issues and as a
    0.0 on A-vs-B, which reads like Graph A's failure. The live case was a node
    carrying `hazardous: false` used as an edge SOURCE in the same JSON.

    Distinct from conformance: conformance asks "does this graph obey the
    ontology's rules"; this asks "does this graph contradict its own
    declarations". They overlap on the role rule, which is fine — the dedupe
    is per-graph and the two are consumed by different downstream users."""
    g = graph_b if isinstance(graph_b, dict) else {}
    nodes = {str(n.get("id")): n for n in (g.get("nodes") or [])
             if isinstance(n, dict) and n.get("id")}
    edges = [e for e in (g.get("edges") or []) if isinstance(e, dict)]

    fails: list[dict[str, Any]] = []

    def fail(cat: str, sev: int, detail: str) -> None:
        fails.append({"category": cat, "severity": sev, "detail": detail})

    sources: set[str] = set()
    for e in edges:
        src, tgt = str(e.get("source", "")), str(e.get("target", ""))
        sources.add(src)
        sn, tn = nodes.get(src), nodes.get(tgt)
        if src and sn is None:
            fail("dangling reference", 2,
                 f"edge source '{src}' is not a declared node")
        if tgt and tn is None:
            fail("dangling reference", 2,
                 f"edge target '{tgt}' is not a declared node")
        if sn is not None and not sn.get("hazardous", False):
            fail("role contradiction", 2,
                 f"{src} is an edge source but its own node says "
                 f"hazardous: false")
        if tn is not None and tn.get("at_risk") is False and tn.get("hazardous"):
            fail("role contradiction", 2,
                 f"{tgt} is an edge target but its own node says "
                 f"at_risk: false")
        via = str(e.get("via_state", "") or "")
        if sn is not None and via and str(sn.get("state", "")) != via:
            fail("state contradiction", 2,
                 f"{src}->{tgt}: via_state '{via}' but the node declares "
                 f"state '{sn.get('state')}'")

    pick = (g.get("suppression_pick") or {}) if isinstance(
        g.get("suppression_pick"), dict) else {}
    pid = str(pick.get("object_id") or pick.get("threat") or "").strip()
    if pid and pid not in sources:
        fail("pick contradiction", 1,
             f"suppression_pick names '{pid}', which is the source of no edge")

    for nid, n in nodes.items():
        if n.get("hazardous") and nid not in sources:
            fail("isolated hazard", 0,
                 f"{nid} is declared hazardous with no outgoing edge")

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
    size = max(1, len(nodes) + len(edges))
    score = round(1.0 - weighted / (weighted + size), 3)
    return {"failures": fails, "breakdown": breakdown,
            "n_failures": len(fails), "score": score,
            "measured": bool(nodes or edges)}


# ── The Graph B gate: withhold A-vs-B when B is not trustworthy ──────────

def graph_b_gate(conformance: Any, graph_b_internal: Any,
                 graph_b_uncertainty: Any = None) -> dict[str, Any]:
    """Is Graph B fit to be the yardstick?

    A-vs-B carries the largest single trust weight. When Graph B is internally
    invalid, comparing against it does not produce a LOW score — it produces a
    MEANINGLESS one, and scoring that as low is dishonest in the direction that
    matters most: it makes correct advice look untrustworthy. The live case had
    Graph A correct and eating the full 0.30 for the model's mess.

    Fails when any of: Graph B contradicts itself; conformance carries a sev-2
    issue against graph_b; Graph B has no edges after normalisation; or the
    model does not reproduce its own arrows. Unmeasured uncertainty NEVER
    fails the gate — absence of a measurement is not evidence."""
    conf = conformance if isinstance(conformance, dict) else {}
    internal = graph_b_internal if isinstance(graph_b_internal, dict) else {}
    unc = graph_b_uncertainty if isinstance(graph_b_uncertainty, dict) else {}

    reasons: list[str] = []
    detail: dict[str, Any] = {}

    score = internal.get("score")
    if isinstance(score, (int, float)) and internal.get("measured") and \
            score < 0.5:
        reasons.append(f"Graph B contradicts its own declarations "
                       f"(self-consistency {round(float(score), 3)}, "
                       f"{internal.get('n_failures', 0)} issue(s))")
        detail["internal_score"] = round(float(score), 3)

    sev2 = [i for i in (conf.get("issues") or [])
            if isinstance(i, dict) and i.get("graph") == "graph_b"
            and i.get("severity", 0) >= 2]
    if sev2:
        reasons.append(f"conformance flags {len(sev2)} serious issue(s) "
                       f"against Graph B "
                       f"({', '.join(sorted({str(i.get('category')) for i in sev2}))})")
        detail["graph_b_sev2"] = len(sev2)

    b_edges = conf.get("graph_b_edges")
    if b_edges is not None and not b_edges:
        reasons.append("Graph B has no valid edges after normalisation — "
                       "there is nothing to compare against")
        detail["graph_b_edges"] = 0

    di = unc.get("direction_instability")
    if isinstance(di, (int, float)) and unc.get("n_probes"):
        detail["direction_instability"] = round(float(di), 3)
        if di >= 0.4:
            reasons.append(f"the model does not reproduce its own arrows "
                           f"(direction instability {round(float(di), 3)} "
                           f"over {unc.get('n_probes')} probes)")

    return {"trusted": not reasons, "reasons": reasons, "detail": detail}
