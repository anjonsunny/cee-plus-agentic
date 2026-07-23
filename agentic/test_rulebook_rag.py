"""Tests for semantic rulebook retrieval (LlamaIndex + Chroma) and its
keyword fallback. Hermetic: the vector path uses a deterministic hash
embedding (no downloads, no network); the fallback path needs nothing.

Run:  pytest agentic/test_rulebook_rag.py -q
"""
from __future__ import annotations

import hashlib

import pytest

from agentic import rulebook_rag
from agentic.rulebook import RULES
from agentic.rulebook_rag import _keyword_search, chunk_text, search

# ── Chunking: rule = chunk, from the single source of truth ─────────────


def test_chunks_carry_statement_rationale_example():
    text = chunk_text("scenario_level_incoherent",
                      RULES["scenario_level_incoherent"])
    assert "S1" in text and "Why:" in text and "Example:" in text
    assert "{" not in text.split("Example:")[0].replace("{\"", "")  # no raw
    # template placeholders leak into retrievable text
    assert "Re-issue" not in text          # template excluded


# ── Keyword fallback (always available) ─────────────────────────────────


def test_keyword_fallback_ranks_relevant_rule_first():
    hits = _keyword_search("severity level contradicts the scenario verdict", 2)
    assert hits and hits[0]["rule_id"] == "S1"
    assert all(h["backend"] == "keyword_fallback" for h in hits)


def test_keyword_fallback_empty_on_nonsense():
    assert _keyword_search("zzqx qwerty", 3) == []


# ── Vector path (skipped cleanly when the stack is not installed) ───────

pytestmark_rag = pytest.mark.skipif(not rulebook_rag.HAVE_RAG,
                                    reason="llama_index/chromadb not installed")


def _hash_embedding():
    """Deterministic per-text embedding: hashed bag of words. Unlike a
    constant MockEmbedding it differentiates texts, so ranking is real."""
    from llama_index.core.embeddings import BaseEmbedding

    class HashEmbedding(BaseEmbedding):
        def _embed(self, text: str) -> list[float]:
            v = [0.0] * 64
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % 64] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            return [x / norm for x in v]

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._embed(text)

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._embed(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._embed(query)

    return HashEmbedding()


@pytestmark_rag
def test_vector_search_covers_all_rules_and_reports_backend():
    hits = search("severity level contradicts the scenario", k=3,
                  embed_model=_hash_embedding())
    assert len(hits) == 3
    assert all(h["backend"] == "llamaindex_chroma" for h in hits)
    assert {h["rule_id"] for h in hits} <= {c.rule_id for c in RULES.values()}


@pytestmark_rag
def test_vector_search_finds_s_family_for_verdict_query():
    hits = search("disaster verdict severity scenario contradiction "
                  "internally incoherent level", k=2,
                  embed_model=_hash_embedding())
    assert any(h["rule_id"].startswith("S") for h in hits)


@pytestmark_rag
def test_index_builds_from_rulebook_without_duplication():
    idx = rulebook_rag.build_index(embed_model=_hash_embedding())
    retriever = idx.as_retriever(similarity_top_k=len(RULES))
    kinds = {h.metadata["kind"] for h in retriever.retrieve("rule")}
    assert kinds == set(RULES.keys())      # every rule indexed exactly once
