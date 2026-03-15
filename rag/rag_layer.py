"""
RAGLayer
========
Sits between the Orchestrator and the Agents.
Enriches every query with retrieved financial context before it reaches an agent.

RAG status is always explicit — the orchestrator receives a detailed status
so it can log, warn, or react differently per failure mode:

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


class RAGStatus(Enum):
    OK             = "ok"
    EMPTY_INDEX    = "empty_index"
    NO_MATCH       = "no_match"
    LOW_CONFIDENCE = "low_confidence"
    ERROR          = "error"


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

MAX_L2_DISTANCE = 1.2


class RAGLayer:
    """
    Wraps a FinancialRetriever and enriches queries with retrieved context.
    Always returns a detailed RAGStatus so the Orchestrator can react
    to every failure mode explicitly.
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

    def enrich(self, query: str, task_hint: str = "") -> dict:
        raw_chunks, retrieval_status = self.retriever.retrieve(query, top_k=self.top_k)

        if retrieval_status == RetrievalStatus.EMPTY_INDEX:
            return self._no_context(query, [], RAGStatus.EMPTY_INDEX,
                "RAG index is empty. Index documents first.")

        if retrieval_status == RetrievalStatus.RETRIEVAL_ERROR:
            return self._no_context(query, [], RAGStatus.ERROR,
                "RAG retrieval raised an exception. Check logs.")

        if retrieval_status == RetrievalStatus.NO_RESULTS:
            return self._no_context(query, [], RAGStatus.NO_MATCH,
                "RAG found no candidate chunks in the index.")

        passing  = [c for c in raw_chunks if c["score"] <= self.max_distance]
        rejected = [c for c in raw_chunks if c["score"] >  self.max_distance]

        if not passing:
            best = min(c["score"] for c in raw_chunks)
            return self._no_context(query, raw_chunks, RAGStatus.LOW_CONFIDENCE,
                f"All {len(raw_chunks)} candidates exceeded threshold "
                f"({self.max_distance}). Best score: {best:.3f}.")

        context_block = self._format_context(passing)
        system_prompt = _CONTEXT_TEMPLATE.format(
            base_system=_BASE_SYSTEM, context_block=context_block)

        return {
            "system_prompt":    system_prompt,
            "user_prompt":      query,
            "retrieved_chunks": passing,
            "rag_used":         True,
            "rag_status":       RAGStatus.OK,
            "rag_status_msg":   f"Injected {len(passing)} chunk(s) (rejected {len(rejected)}).",
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

    @staticmethod
    def _no_context(query: str, raw_chunks: list, status: RAGStatus, msg: str) -> dict:
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
            parts  = [f"[{i}]", f"source={meta.get('source','unknown')}"]
            if meta.get("ticker"): parts.append(f"ticker={meta['ticker']}")
            if meta.get("date"):   parts.append(f"date={meta['date']}")
            lines.append(" ".join(parts) + "\n" + chunk["text"])
        return "\n\n".join(lines)
