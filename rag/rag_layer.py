"""
RAGLayer
========
Sits between the Orchestrator and the Agents.
Enriches every query with retrieved financial context before it reaches an agent.

RAG status is always explicit — the orchestrator receives a detailed status
so it can log, warn, adjust confidence, or react differently per failure mode:

    RAGStatus.OK              → context injected, proceed normally
    RAGStatus.EMPTY_INDEX     → no documents indexed yet, answer without context
    RAGStatus.NO_MATCH        → index has docs but nothing relevant was found
    RAGStatus.LOW_CONFIDENCE  → candidates found but all exceeded distance threshold
    RAGStatus.ERROR           → retrieval exception, proceed without context
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from .retriever import FinancialRetriever, RetrievalStatus


# ─────────────────────────────────────────────────────────────────────────────
# RAG status — distinct from RetrievalStatus, this is the orchestrator-facing view
# ─────────────────────────────────────────────────────────────────────────────

class RAGStatus(Enum):
    OK             = "ok"             # context retrieved and injected
    EMPTY_INDEX    = "empty_index"    # no documents in index
    NO_MATCH       = "no_match"       # docs exist but nothing relevant found
    LOW_CONFIDENCE = "low_confidence" # candidates found but all too dissimilar
    ERROR          = "error"          # exception during retrieval


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM = (
    "You are an expert financial analyst. "
    "Reason carefully, cite your logic, and provide structured, professional analysis."
)

_CONTEXT_TEMPLATE = """{base_system}

--- RELEVANT FINANCIAL CONTEXT ---
The following excerpts were retrieved from indexed financial documents.
Use them to inform your analysis where relevant. If they contradict each other,
note the discrepancy. If they are irrelevant to the query, ignore them.

{context_block}
--- END CONTEXT ---"""

# Distance threshold — L2 distance above this means the chunk is probably
# not relevant to the query. Calibrated for finance-embeddings-investopedia.
# Lower = stricter. Increase if you're getting too many rejections.
MAX_L2_DISTANCE = 1.2


# ─────────────────────────────────────────────────────────────────────────────
# RAGLayer
# ─────────────────────────────────────────────────────────────────────────────

class RAGLayer:
    """
    Wraps a FinancialRetriever and enriches queries with retrieved context.
    Always returns a detailed RAGStatus so the Orchestrator can react
    to every failure mode explicitly rather than silently falling back.

    Args:
        retriever:    A FinancialRetriever instance. Created with defaults if None.
        top_k:        Number of chunks to retrieve per query.
        max_distance: L2 distance ceiling for relevance filtering.
    """

    def __init__(
        self,
        retriever:    Optional[FinancialRetriever] = None,
        top_k:        int   = 5,
        max_distance: float = MAX_L2_DISTANCE,
    ):
        self.retriever    = retriever or FinancialRetriever()
        self.top_k        = top_k
        self.max_distance = max_distance

    # ── Public API ───────────────────────────────────────────────────────────

    def enrich(self, query: str, task_hint: str = "") -> dict:
        """
        Retrieve relevant context and build an enriched prompt.

        Returns:
            {
                "system_prompt":    str,        enriched system prompt (or base if no context)
                "user_prompt":      str,        the original query unchanged
                "retrieved_chunks": list,       raw retrieval results
                "rag_used":         bool,       True only if context was injected
                "rag_status":       RAGStatus,  exact status for orchestrator logic
                "rag_status_msg":   str,        human-readable explanation
            }
        """
        # Step 1: raw retrieval — always get a status back
        raw_chunks, retrieval_status = self.retriever.retrieve(query, top_k=self.top_k)

        # Step 2: map retrieval status to RAG-layer failure modes
        if retrieval_status == RetrievalStatus.EMPTY_INDEX:
            return self._no_context(
                query, [], RAGStatus.EMPTY_INDEX,
                "RAG index is empty. Index documents first with rag.add_documents()."
            )

        if retrieval_status == RetrievalStatus.RETRIEVAL_ERROR:
            return self._no_context(
                query, [], RAGStatus.ERROR,
                "RAG retrieval raised an exception. Check logs for details."
            )

        if retrieval_status == RetrievalStatus.NO_RESULTS:
            return self._no_context(
                query, [], RAGStatus.NO_MATCH,
                "RAG found no candidate chunks in the index."
            )

        # Step 3: apply distance threshold to filter low-relevance chunks
        passing  = [c for c in raw_chunks if c["score"] <= self.max_distance]
        rejected = [c for c in raw_chunks if c["score"] >  self.max_distance]

        if not passing:
            best_score = min(c["score"] for c in raw_chunks)
            return self._no_context(
                query, raw_chunks, RAGStatus.LOW_CONFIDENCE,
                f"All {len(raw_chunks)} candidate chunk(s) exceeded the distance "
                f"threshold ({self.max_distance}). Best score: {best_score:.3f}. "
                f"Consider lowering max_distance or adding more relevant documents."
            )

        # Step 4: inject passing chunks into the system prompt
        context_block = self._format_context(passing)
        system_prompt = _CONTEXT_TEMPLATE.format(
            base_system=_BASE_SYSTEM,
            context_block=context_block,
        )

        msg = (
            f"Injected {len(passing)} chunk(s) "
            f"(rejected {len(rejected)} below threshold)."
        )

        return {
            "system_prompt":    system_prompt,
            "user_prompt":      query,
            "retrieved_chunks": passing,
            "rag_used":         True,
            "rag_status":       RAGStatus.OK,
            "rag_status_msg":   msg,
        }

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None):
        self.retriever.add_texts(texts, metadatas=metadatas)
        self.retriever.save()

    def add_file(self, filepath: str, metadata: dict | None = None):
        self.retriever.add_file(filepath, metadata=metadata)
        self.retriever.save()

    @property
    def index_size(self) -> int:
        return self.retriever.n_docs

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _no_context(query: str, raw_chunks: list, status: RAGStatus, msg: str) -> dict:
        """Build a no-context result with an explicit status and message."""
        return {
            "system_prompt":    _BASE_SYSTEM,
            "user_prompt":      query,
            "retrieved_chunks": raw_chunks,
            "rag_used":         False,
            "rag_status":       status,
            "rag_status_msg":   msg,
        }

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        lines = []
        for i, chunk in enumerate(chunks, 1):
            meta   = chunk["metadata"]
            source = meta.get("source", "unknown")
            ticker = meta.get("ticker", "")
            date   = meta.get("date",   "")

            parts = [f"[{i}]", f"source={source}"]
            if ticker: parts.append(f"ticker={ticker}")
            if date:   parts.append(f"date={date}")

            lines.append(" ".join(parts) + "\n" + chunk["text"])

        return "\n\n".join(lines)
