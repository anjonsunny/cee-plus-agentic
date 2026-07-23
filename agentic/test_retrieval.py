"""Retrieval mode switch + RAG shadow + agreement report.

Hermetic: the vector path uses the same deterministic hash embedding as
test_rulebook_rag (no downloads); the rulebook path needs nothing. The
critical guarantee: mode 'rulebook' is byte-identical to exact-key lookup,
so the LangGraph equivalence and pipeline behavior are unchanged.

Run:  pytest agentic/test_retrieval.py -q
"""
from __future__ import annotations

import hashlib

import pytest

from agentic import retrieval, rulebook, rulebook_rag
from agentic.rulebook import RULES

# Vector-path tests need llama_index/chromadb; skip cleanly without them
# (the RAG stack is optional — the pipeline never depends on it).
needs_rag = pytest.mark.skipif(
    not rulebook_rag.HAVE_RAG,
    reason="llama_index/chromadb not installed")


# ── mode switch ─────────────────────────────────────────────────────────

def test_mode_default_is_rulebook(monkeypatch):
    monkeypatch.delenv("AGENTIC_RETRIEVAL", raising=False)
    assert retrieval.retrieval_mode() == "rulebook"
    monkeypatch.setenv("AGENTIC_RETRIEVAL", "RAG")
    assert retrieval.retrieval_mode() == "rag"


def test_rulebook_mode_is_exact_key(monkeypatch):
    """The default must equal rulebook.retrieve — this is what keeps the
    pipeline and the LangGraph equivalence unchanged."""
    monkeypatch.delenv("AGENTIC_RETRIEVAL", raising=False)
    for kind in RULES:
        chunk, meta = retrieval.retrieve_rule(kind)
        assert chunk is rulebook.retrieve(kind)
        assert meta["source"] == "exact_key" and meta["mode"] == "rulebook"
    # unknown kind -> None, never crashes (same as exact-key)
    assert retrieval.retrieve_rule("no_such_kind")[0] is None


def test_queries_do_not_leak_the_kind_name():
    """A fair agreement test: the query must not echo the kind's own
    phrase (that would be keyword matching, not semantics). Shared domain
    words (state, threat, hazard) are fine; the whole kind name is not."""
    for kind, q in retrieval.QUERIES.items():
        phrase = kind.replace("_", " ")
        assert phrase not in q.lower(), (kind, phrase)
        assert RULES[kind].rule_id.lower() not in q.lower()


# ── agreement report ────────────────────────────────────────────────────

def _hash_embedding():
    from llama_index.core.embeddings import BaseEmbedding

    class HashEmbedding(BaseEmbedding):
        def _embed(self, text: str) -> list[float]:
            v = [0.0] * 64
            for tok in text.lower().split():
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            return [x / n for x in v]

        def _get_text_embedding(self, t): return self._embed(t)
        def _get_query_embedding(self, q): return self._embed(q)
        async def _aget_query_embedding(self, q): return self._embed(q)

    return HashEmbedding()


def test_agreement_report_shape_keyword_fallback():
    """Works with no embedding stack via the keyword fallback: the report
    is well-formed and every kind is covered."""
    from agentic import rulebook_rag
    # force the fallback path deterministically
    rep_rows = []
    for kind in RULES:
        hits = rulebook_rag._keyword_search(retrieval.query_for(kind), 1)
        rep_rows.append(hits[0]["kind"] if hits else None)
    assert len(rep_rows) == len(RULES)


@needs_rag
def test_agreement_report_vector_path_is_wellformed():
    rep = retrieval.agreement_report(embed_model=_hash_embedding())
    assert rep["n"] == len(RULES)
    assert 0 <= rep["agree"] <= rep["n"]
    assert rep["backend"] == "llamaindex_chroma"
    # every row present and typed
    kinds = {r["kind"] for r in rep["rows"]}
    assert kinds == set(RULES.keys())
    for r in rep["rows"]:
        assert r["rag_kind"] in RULES or r["rag_kind"] is None
        assert isinstance(r["agree"], bool)
    # formatting never crashes
    assert "agreement" in retrieval.format_report(rep)


@needs_rag
def test_rag_mode_returns_rag_pick(monkeypatch):
    """In 'rag' mode retrieve_rule returns RAG's chunk and records agreement."""
    monkeypatch.setenv("AGENTIC_RETRIEVAL", "rag")
    chunk, meta = retrieval.retrieve_rule(
        "state_out_of_vocab", embed_model=_hash_embedding())
    assert meta["mode"] == "rag" and meta["source"] == "rag_top1"
    assert "agree" in meta and meta["rag_kind"] in RULES


@needs_rag
def test_both_mode_keeps_exact_but_records_rag(monkeypatch):
    """'both' is the shadow: exact-key stays authoritative, RAG recorded."""
    monkeypatch.setenv("AGENTIC_RETRIEVAL", "both")
    chunk, meta = retrieval.retrieve_rule(
        "threat_state_not_hazardous", embed_model=_hash_embedding())
    assert chunk is rulebook.retrieve("threat_state_not_hazardous")  # exact wins
    assert meta["source"] == "exact_key" and "rag_kind" in meta


def test_ui_override_beats_env(monkeypatch):
    from agentic.retrieval import set_retrieval, retrieval_mode
    monkeypatch.setenv("AGENTIC_RETRIEVAL", "rulebook")
    set_retrieval("both")
    assert retrieval_mode() == "both"
    set_retrieval(None)
    assert retrieval_mode() == "rulebook"


# ── B · rulebook-vs-RAG result diff ─────────────────────────────────────

def test_compare_modes_detects_rule_driven_change(monkeypatch):
    """When RAG picks a different rule, the model is quoted different rule
    text during reflection — which can change the final answer. This proves
    compare_retrieval_modes catches that. Keyword engine (deterministic)."""
    from agentic import rulebook_rag
    monkeypatch.setattr(rulebook_rag, "HAVE_RAG", False)  # keyword, no download
    monkeypatch.delenv("AGENTIC_RETRIEVAL", raising=False)

    from agentic.perception import DetectedObject, PerceptionResult

    def _o(oid, label, fam, state, kind):
        return DetectedObject(object_id=oid, label=label, family=fam,
                              state=state, state_kind=kind, bbox=[0, 0, 9, 9],
                              box_source="dino_matched")
    # building is a hazard; the verdict OMITS it from threats -> S6 fires.
    rec = PerceptionResult(
        image_path="/x.jpg", image_size=[100, 80], caption="collapse",
        entity_source="vlm",
        detected_objects=[_o("building_1", "building", "structure",
                             "collapsed", "hazard_bearing")])

    verdict = {"disaster_scenario": "Yes", "disaster_type": "collapse",
               "disaster_level": 8, "confidence": 0.9, "threats": [],
               "at_risk": []}
    add_building = {"disaster_scenario": "Yes", "disaster_type": "collapse",
                    "disaster_level": 8, "confidence": 0.9,
                    "threats": [{"object_id": "building_1",
                                 "reason": "collapsed structure"}],
                    "at_risk": []}

    def q(prompt):
        # hazard_not_in_threats -> exact S6, RAG(keyword) S8.
        if "RULE S6:" in prompt:      # quoted the RIGHT rule -> model fixes it
            return add_building
        if "RULE S8:" in prompt:      # quoted the WRONG rule -> model stands
            return verdict
        return verdict                # first pass

    d = retrieval.compare_retrieval_modes(rec, query_fn=q, n_probes=0)
    assert d["changed"] is True
    assert "building_1" in d["rulebook_line"]      # exact quoted S6 -> fixed
    assert "threats[-]" in d["rag_line"]           # rag quoted S8 -> not fixed
