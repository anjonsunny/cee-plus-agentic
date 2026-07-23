"""Semantic retrieval over the rulebook: LlamaIndex + Chroma (Sunny's
call, 2026-07-21 — build the real thing now that the corpus is growing).

TWO RETRIEVAL SHAPES, TWO TOOLS
===============================
- KIND-shaped lookup ("a state_out_of_vocab violation fired; fetch its
  rule"): a violation kind is a KEY. rulebook.retrieve() stays exact —
  running an embedding search to resolve a dictionary key would be
  ceremony, not retrieval.
- QUERY-shaped lookup ("which rules bear on severity contradictions?",
  "what governs invented entities?"): natural-language questions with no
  key. THIS module. Used by the dialogue agent's rule tool (so the
  analyst can cite the rulebook) and by repair prompts that need
  neighboring rules, not just the fired one.

DESIGN
======
- Chunking is RULE = CHUNK, inherited from rulebook.py: statement,
  rationale, example, and template travel together, because a rule split
  across chunks is a broken rule. The index is built FROM rulebook.RULES
  — one source of truth; this module never defines rules.
- Chroma runs in-process (EphemeralClient) and rebuilds at startup: at
  this corpus size, index build is milliseconds and a stale persisted
  index is a real bug risk for zero gain. Swap to PersistentClient when
  the corpus makes rebuilds noticeable.
- Embeddings: HuggingFace BAAI/bge-small-en-v1.5 by default (local, no
  API). Tests inject a deterministic hash embedding: hermetic, no
  downloads, and it still differentiates texts (unlike MockEmbedding's
  constant vector, which cannot rank).
- GRACEFUL DEGRADATION: if llama_index/chromadb are not installed,
  search() falls back to keyword scoring over the same chunks. The
  pipeline must never fail because a retrieval library is missing;
  `backend` on each result says which path answered.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.rulebook import RULES, RuleChunk  # noqa: E402

try:  # optional heavy deps; fallback works without them
    import chromadb  # noqa: F401
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    HAVE_RAG = True
except ImportError:
    HAVE_RAG = False
    BaseEmbedding = object  # type: ignore[assignment,misc]


def chunk_text(kind: str, chunk: RuleChunk) -> str:
    """The retrievable text of one rule: everything but the fill-in
    template (placeholders are noise to an embedder)."""
    return (f"Rule {chunk.rule_id} ({kind}). {chunk.rule}\n"
            f"Why: {chunk.rationale}\nExample: {chunk.example}")


# ── The index (built lazily, once per process) ──────────────────────────

_STATE: dict[str, Any] = {"index": None, "embed": None}


def build_index(embed_model: Any = None) -> Any:
    """Build the Chroma-backed vector index over rulebook.RULES. Pass
    embed_model to override (tests inject a hash embedding); default is
    a local HuggingFace model, downloaded once, no API."""
    if not HAVE_RAG:
        raise RuntimeError("llama_index/chromadb not installed")
    if embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")
    client = chromadb.EphemeralClient()
    # EphemeralClient shares one in-process store: reusing a collection
    # name across builds stacks duplicate documents. Rebuild clean.
    try:
        client.delete_collection("rulebook")
    except Exception:
        pass                                   # first build: nothing to drop
    store = ChromaVectorStore(
        chroma_collection=client.get_or_create_collection("rulebook"))
    docs = [Document(text=chunk_text(kind, c),
                     metadata={"kind": kind, "rule_id": c.rule_id})
            for kind, c in RULES.items()]
    from llama_index.core import StorageContext
    return VectorStoreIndex.from_documents(
        docs, embed_model=embed_model,
        storage_context=StorageContext.from_defaults(vector_store=store))


def _get_index(embed_model: Any = None) -> Any:
    if _STATE["index"] is None or embed_model is not None:
        idx = build_index(embed_model)
        if embed_model is None:
            _STATE["index"] = idx
        return idx
    return _STATE["index"]


# ── Keyword fallback (no deps; same interface, honest about itself) ─────


def _keyword_search(query: str, k: int) -> list[dict[str, Any]]:
    q_terms = {t for t in query.lower().split() if len(t) > 2}
    scored = []
    for kind, c in RULES.items():
        text = chunk_text(kind, c).lower()
        score = sum(1 for t in q_terms if t in text)
        if score:
            scored.append((score, kind, c))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [{"kind": kind, "rule_id": c.rule_id, "rule": c.rule,
             "score": float(score), "backend": "keyword_fallback"}
            for score, kind, c in scored[:k]]


# ── Public search ───────────────────────────────────────────────────────


def search(query: str, k: int = 2,
           embed_model: Any = None) -> list[dict[str, Any]]:
    """Query-shaped retrieval: top-k rules for a natural-language
    question. Vector search when the stack is installed, keyword scoring
    when not; `backend` reports which one answered so a degraded result
    can never masquerade as a semantic one."""
    if not HAVE_RAG:
        return _keyword_search(query, k)
    try:
        retriever = _get_index(embed_model).as_retriever(similarity_top_k=k)
        hits = retriever.retrieve(query)
    except Exception:
        return _keyword_search(query, k)
    out = []
    for h in hits:
        kind = h.metadata.get("kind", "")
        c = RULES.get(kind)
        out.append({"kind": kind,
                    "rule_id": h.metadata.get("rule_id", ""),
                    "rule": c.rule if c else "",
                    "score": round(float(h.score or 0.0), 4),
                    "backend": "llamaindex_chroma"})
    return out


if __name__ == "__main__":
    for q in ("severity contradicts the scenario",
              "the model invented an entity not in the vocabulary",
              "disaster verdict with no supporting hazard"):
        print(f"\nQ: {q}")
        for hit in search(q, k=2):
            print(f"  {hit['rule_id']} ({hit['kind']}) "
                  f"score={hit['score']} [{hit['backend']}]")
