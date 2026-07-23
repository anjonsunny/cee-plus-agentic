"""Retrieval mode switch + RAG shadow (Sunny, 2026-07-23).

THREE MODES, set by AGENTIC_RETRIEVAL:
  rulebook  (default) — exact-key lookup. The check's name IS the key, so
                        we fetch exactly the right rule. Deterministic.
                        The pipeline default; equivalence relies on it.
  rag                 — RAG top-1 (LlamaIndex + Chroma) by a natural-language
                        query. Opt-in experiment; may differ from exact.
  both                — run both; exact-key stays authoritative (so pipeline
                        behavior is unchanged), the RAG pick is RECORDED for
                        comparison. This is the shadow.

WHY EXACT-KEY IS THE DEFAULT, NOT RAG
=====================================
When you already hold the exact key, embedding search can only match it or
get it WRONG (a near-neighbour). On a small rulebook, opposite rules sit
close in meaning (S2 "missed a disaster" vs S3 "false alarm"). So RAG is
measured here, never the decider — its value lands in Stage 4, where the
checks are semantic and there is no key.

THE AGREEMENT REPORT
====================
`agreement_report()` asks, for every check kind: does RAG's top-1 (from a
neutral natural-language query that does NOT contain the kind's own name)
match the rule the exact key returns? It reports the match rate and every
mismatch, with the engine that answered (vector vs keyword fallback), so a
degraded run can never masquerade as a semantic one.

    python -m agentic.retrieval          # print the report
"""
from __future__ import annotations

import os
from typing import Any, Optional

from agentic import rulebook
from agentic.rulebook import RULES, RuleChunk


# ── Neutral natural-language queries, one per rule ──────────────────────
#
# A realistic "analyst question" for each rule. Deliberately does NOT
# contain the kind's underscore-token, so the test measures semantic
# separability, not a keyword match on the kind name.

QUERIES: dict[str, str] = {
    "family_name_as_label":
        "the model gave a broad category word instead of a specific object noun",
    "label_out_of_vocab":
        "the object name is not in our allowed list of nouns",
    "state_out_of_vocab":
        "the condition word is made up and not a recognized state",
    "missing_anchor_bbox":
        "an entity has no rough box to say which instance is meant",
    "caption_entity_missing":
        "the caption mentions something that is absent from the entity list",
    "duplicate_entity":
        "the same person appears twice with two nearly identical boxes",
    "threat_reason_victim_shaped":
        "a source of harm is justified by describing it as endangered",
    "scenario_level_incoherent":
        "the yes or no verdict disagrees with its own severity number",
    "missed_disaster_incoherence":
        "the answer says nothing is wrong but dangerous conditions are present",
    "false_alarm_incoherence":
        "the answer claims an emergency while nothing harmful is declared",
    "id_not_in_perception":
        "the answer cites an entity that does not exist in the record",
    "threat_state_not_hazardous":
        "something named a source of danger is not in any harmful condition",
    "hazard_not_in_threats":
        "a harmful entity was left out of the list of dangers",
    "hazard_as_at_risk":
        "an object is put both as a source of harm and as a victim",
    "proximity_without_hazard":
        "someone is marked in danger from nearness but nothing harmful is active",
    "at_risk_kind_mismatch":
        "the victim category does not match what the entity's condition implies",
    "geometry_adjacency":
        "an entity sits next to or overlaps an active source of harm",
    "geometry_is_a_hint":
        "closeness in the flat image is only a hint, not proof of real nearness",
}


# A UI toggle can override the env var for the current process.
_MODE_OVERRIDE: Optional[str] = None


def set_retrieval(mode: str | None) -> None:
    """Set (or clear, with None) the in-process retrieval choice."""
    global _MODE_OVERRIDE
    _MODE_OVERRIDE = (mode or "").strip().lower() or None


def retrieval_mode() -> str:
    """rulebook (default) | rag | both — UI override, else AGENTIC_RETRIEVAL."""
    if _MODE_OVERRIDE:
        return _MODE_OVERRIDE
    return os.getenv("AGENTIC_RETRIEVAL", "rulebook").strip().lower()


def query_for(kind: str) -> str:
    return QUERIES.get(kind, kind.replace("_", " "))


def _rag_top1(kind: str, embed_model: Any = None) -> Optional[dict[str, Any]]:
    """RAG's single best rule for this kind's neutral query. Lazy-imports
    the RAG stack so 'rulebook' mode never pays for llama_index."""
    from agentic import rulebook_rag
    hits = rulebook_rag.search(query_for(kind), k=1, embed_model=embed_model)
    return hits[0] if hits else None


# Per-run shadow log: every rag/both lookup records exact-key vs RAG top-1
# so the UI can show the comparison for the rules a run actually used.
_SHADOW_LOG: list[dict[str, Any]] = []


def drain_shadow_log() -> list[dict[str, Any]]:
    """Return and clear the shadow comparisons collected so far."""
    global _SHADOW_LOG
    out = _SHADOW_LOG[:]
    _SHADOW_LOG.clear()
    return out


def retrieve_rule(kind: str, *, mode: str | None = None,
                  embed_model: Any = None) -> tuple[Optional[RuleChunk],
                                                    dict[str, Any]]:
    """Return (rule_chunk, meta). In 'rulebook' mode this is byte-identical
    to rulebook.retrieve(kind) — which keeps the pipeline (and the
    LangGraph equivalence) unchanged. 'rag' returns RAG's pick. 'both'
    returns the EXACT rule (authoritative) but records RAG's pick in meta."""
    mode = (mode or retrieval_mode())
    exact = rulebook.retrieve(kind)
    if mode == "rulebook":
        return exact, {"mode": "rulebook", "source": "exact_key"}

    top1 = _rag_top1(kind, embed_model)
    rag_kind = top1["kind"] if top1 else None
    agree = bool(top1 and rag_kind == kind)
    meta = {"mode": mode, "rag_kind": rag_kind, "agree": agree,
            "backend": top1["backend"] if top1 else None,
            "score": top1["score"] if top1 else None}

    # record the comparison for the UI's per-run "RAG shadow" panel
    _SHADOW_LOG.append({
        "kind": kind,
        "exact_rule_id": exact.rule_id if exact else None,
        "rag_kind": rag_kind,
        "rag_rule_id": top1["rule_id"] if top1 else None,
        "agree": agree,
        "score": top1["score"] if top1 else None,
        "backend": top1["backend"] if top1 else None,
    })

    if mode == "rag":
        meta["source"] = "rag_top1"
        return (RULES.get(rag_kind) if rag_kind else exact), meta
    # both: exact stays authoritative; RAG is shadow-recorded
    meta["source"] = "exact_key"
    return exact, meta


# ── The agreement report ────────────────────────────────────────────────

def agreement_report(embed_model: Any = None) -> dict[str, Any]:
    """Exact-key vs RAG top-1 across every check kind."""
    rows: list[dict[str, Any]] = []
    for kind, chunk in RULES.items():
        top1 = _rag_top1(kind, embed_model)
        rows.append({
            "kind": kind,
            "rule_id": chunk.rule_id,
            "query": query_for(kind),
            "rag_kind": top1["kind"] if top1 else None,
            "rag_rule_id": top1["rule_id"] if top1 else None,
            "score": top1["score"] if top1 else None,
            "backend": top1["backend"] if top1 else None,
            "agree": bool(top1 and top1["kind"] == kind),
        })
    n = len(rows)
    agree = sum(1 for r in rows if r["agree"])
    return {
        "n": n, "agree": agree,
        "rate": round(agree / n, 3) if n else None,
        "backend": rows[0]["backend"] if rows else None,
        "mismatches": [r for r in rows if not r["agree"]],
        "rows": rows,
    }


def format_report(rep: dict[str, Any]) -> str:
    lines = []
    lines.append("EXACT-KEY vs RAG top-1 — retrieval agreement")
    lines.append(f"  engine: {rep['backend']}")
    lines.append(f"  agreement: {rep['agree']}/{rep['n']} "
                 f"= {rep['rate']:.0%}" if rep["rate"] is not None else "  n/a")
    lines.append("")
    lines.append(f"  {'rule':<6}{'kind':<30}{'RAG top-1':<30}{'':>6}")
    for r in rep["rows"]:
        mark = "  ok" if r["agree"] else "  MISS"
        rag = f"{r['rag_rule_id'] or '-'} {r['rag_kind'] or '-'}"
        lines.append(f"  {r['rule_id']:<6}{r['kind']:<30}{rag:<30}{mark}")
    if rep["mismatches"]:
        lines.append("")
        lines.append("  mismatches (where RAG picked a different rule):")
        for r in rep["mismatches"]:
            lines.append(f"    {r['rule_id']} {r['kind']}")
            lines.append(f"      query : \"{r['query']}\"")
            lines.append(f"      RAG   : {r['rag_rule_id']} {r['rag_kind']} "
                         f"(score {r['score']})")
    return "\n".join(lines)


# ── B · does RAG's rule choice change the ANSWER? ───────────────────────

def compare_retrieval_modes(record: Any, *, query_fn: Any = None,
                            n_probes: int = 0, explain_fn: Any = None
                            ) -> dict[str, Any]:
    """Run Stage 2 on the SAME record twice — once quoting exact-key rules,
    once quoting RAG's picks — and report whether the final verdict changed.
    Probes off by default (n_probes=0) so the diff isolates the rule choice
    from probe randomness. Only the rules that DISAGREE can move the answer;
    where every fired rule agrees, the two runs are identical by
    construction."""
    from agentic.assessment import run_assessment
    from agentic.evals import substantive_key

    saved = _MODE_OVERRIDE
    try:
        set_retrieval("rulebook")
        rb = run_assessment(record, query_fn=query_fn, n_probes=n_probes,
                            reflect=True, explain_fn=explain_fn)
        set_retrieval("rag")
        rg = run_assessment(record, query_fn=query_fn, n_probes=n_probes,
                            reflect=True, explain_fn=explain_fn)
    finally:
        set_retrieval(saved)

    def _line(a) -> str:
        th = ",".join(t.object_id for t in a.threats) or "-"
        ar = ",".join(f"{r.object_id}·{r.kind}" for r in a.at_risk) or "-"
        return (f"{a.disaster_scenario} · {a.disaster_type} · "
                f"L{a.disaster_level} · threats[{th}] · at_risk[{ar}]")

    a, b = rb.assessment, rg.assessment
    changed = substantive_key(a.model_dump()) != substantive_key(b.model_dump())
    return {"changed": changed,
            "rulebook_line": _line(a), "rag_line": _line(b),
            "rulebook_result": rb, "rag_result": rg}


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--diff":
        from pathlib import Path

        from agentic.perception import PerceptionResult
        rec = PerceptionResult.model_validate_json(
            Path(args[1]).read_text())
        d = compare_retrieval_modes(rec)     # live model (Ollama)
        print("RULEBOOK vs RAG — result diff (probes off)")
        print(f"  rulebook : {d['rulebook_line']}")
        print(f"  rag      : {d['rag_line']}")
        print(f"  changed  : {d['changed']}"
              + ("  <-- RAG's rule choice moved the answer"
                 if d["changed"] else "  (identical)"))
    else:
        print(format_report(agreement_report()))
