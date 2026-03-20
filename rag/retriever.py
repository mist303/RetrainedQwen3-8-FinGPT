"""
FinancialRetriever
==================
Local FAISS vector store for financial document retrieval.

Embedding model: FinLang/finance-embeddings-investopedia
  - Built on MiniLM-L6-v2, fine-tuned on Investopedia financial text
  - 384-dim vectors (same as MiniLM — index format compatible)
  - Understands financial jargon far better than vanilla MiniLM:
    EBITDA, yield curve, basis points, P/E, WACC etc. land correctly
  - Fast enough to run on CPU between inference calls
  - Falls back to all-MiniLM-L6-v2 if the finance model is unavailable

Install dependencies:
    pip install faiss-cpu sentence-transformers

Usage
-----
    retriever = FinancialRetriever()
    retriever.add_texts(["Apple reported record Q4 revenue..."],
                        metadatas=[{"source": "earnings", "ticker": "AAPL"}])
    chunks = retriever.retrieve("Apple revenue", top_k=3)
"""

from __future__ import annotations  # makes all type hints lazy strings — fixes
                                     # NameError when SentenceTransformer isn't
                                     # imported (DEPS_AVAILABLE = False)
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from enum import Enum

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INDEX_DIR = Path(__file__).parent / "index"

# Primary: finance-domain fine-tune of MiniLM — same 384-dim, better financial vocab
# Fallback: vanilla MiniLM if the finance model isn't cached
EMBEDDING_MODEL          = "FinLang/finance-embeddings-investopedia"
EMBEDDING_MODEL_FALLBACK = "all-MiniLM-L6-v2"
EMBEDDING_DIM            = 384

CHUNK_SIZE    = 512   # characters per chunk
CHUNK_OVERLAP = 64    # overlap to avoid losing context at boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Status enum — returned with every retrieve() call
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalStatus(Enum):
    OK               = "ok"                # chunks found and returned
    EMPTY_INDEX      = "empty_index"       # no documents have been indexed yet
    NO_RESULTS       = "no_results"        # index has docs but none were relevant
    SCORE_TOO_HIGH   = "score_too_high"    # candidates found but all exceeded distance threshold
    RETRIEVAL_ERROR  = "retrieval_error"   # exception during FAISS search


# ─────────────────────────────────────────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# ─────────────────────────────────────────────────────────────────────────────
# FinancialRetriever
# ─────────────────────────────────────────────────────────────────────────────

class FinancialRetriever:
    """
    FAISS flat L2 index backed by finance-domain sentence embeddings.

    retrieve() now returns a (chunks, status) tuple so callers always know
    exactly why they got zero results — empty index, no match, or exception.
    """

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        if not DEPS_AVAILABLE:
            raise ImportError(
                "RAG dependencies not installed.\n"
                "Run: pip install faiss-cpu sentence-transformers"
            )

        self.index_dir       = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path    = self.index_dir / "index.faiss"
        self._texts_path    = self.index_dir / "texts.pkl"
        self._metadata_path = self.index_dir / "metadata.pkl"
        self._model_tag_path = self.index_dir / "embedding_model.txt"

        self.embedder       = self._load_embedder()
        self.embedding_model_name = getattr(self.embedder, '_model_card_vars', {}).get('model_name', EMBEDDING_MODEL)

        if self._index_path.exists():
            self._load()
        else:
            self._init_empty()

    # ── Embedder loading ─────────────────────────────────────────────────────

    def _load_embedder(self) -> SentenceTransformer:
        """
        Try to load the finance-domain model first.
        Fall back to vanilla MiniLM if it's not cached and we're offline.
        """
        for model_name in [EMBEDDING_MODEL, EMBEDDING_MODEL_FALLBACK]:
            try:
                print(f"Loading embedding model: {model_name}")
                embedder = SentenceTransformer(model_name)
                if model_name == EMBEDDING_MODEL_FALLBACK:
                    print(
                        f"  ⚠ Using fallback embedder ({EMBEDDING_MODEL_FALLBACK}).\n"
                        f"  For better financial retrieval, run:\n"
                        f"  pip install sentence-transformers\n"
                        f"  python -c \"from sentence_transformers import SentenceTransformer; "
                        f"SentenceTransformer('{EMBEDDING_MODEL}')\""
                    )
                return embedder
            except Exception as e:
                print(f"  Could not load {model_name}: {e}")
        raise RuntimeError("Could not load any embedding model.")

    # ── Index management ─────────────────────────────────────────────────────

    def _init_empty(self):
        self.index    = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.texts    = []
        self.metadata = []
        print("Created new empty FAISS index.")

    def _load(self):
        self.index = faiss.read_index(str(self._index_path))
        with open(self._texts_path,    "rb") as f: self.texts    = pickle.load(f)
        with open(self._metadata_path, "rb") as f: self.metadata = pickle.load(f)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors from {self.index_dir}")

    def save(self):
        faiss.write_index(self.index, str(self._index_path))
        with open(self._texts_path,    "wb") as f: pickle.dump(self.texts,    f)
        with open(self._metadata_path, "wb") as f: pickle.dump(self.metadata, f)
        print(f"Saved FAISS index: {self.index.ntotal} vectors")

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def n_docs(self) -> int:
        return self.index.ntotal

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None, auto_chunk: bool = True):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        assert len(texts) == len(metadatas)

        chunks, chunk_metas = [], []
        for text, meta in zip(texts, metadatas):
            text_chunks = chunk_text(text) if auto_chunk else [text]
            chunks.extend(text_chunks)
            chunk_metas.extend([meta] * len(text_chunks))

        if not chunks:
            return

        vecs = self._embed(chunks)
        self.index.add(vecs)
        self.texts.extend(chunks)
        self.metadata.extend(chunk_metas)
        print(f"Indexed {len(chunks)} chunks. Total: {self.index.ntotal}")

    def add_file(self, filepath: str | Path, metadata: Optional[dict] = None):
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        self.add_texts([text], metadatas=[metadata or {"source": str(filepath)}])

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[dict], RetrievalStatus]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Returns:
            (chunks, status)

            chunks: list of {"text", "score", "metadata"} sorted by L2 distance.
                    Empty list on any non-OK status.
            status: RetrievalStatus enum — tells the caller exactly why
                    chunks may be empty:
                      OK             → results returned
                      EMPTY_INDEX    → no documents indexed yet
                      NO_RESULTS     → FAISS returned no valid indices
                      SCORE_TOO_HIGH → candidates exist but all exceed threshold
                                       (this is intentionally NOT filtered here —
                                        the caller decides what to do with scores)
                      RETRIEVAL_ERROR → exception during search
        """
        if self.index.ntotal == 0:
            return [], RetrievalStatus.EMPTY_INDEX

        try:
            q_vec = self._embed([query])
            k     = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(q_vec, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                results.append({
                    "text":     self.texts[idx],
                    "score":    float(dist),
                    "metadata": self.metadata[idx],
                })

            if not results:
                return [], RetrievalStatus.NO_RESULTS

            return results, RetrievalStatus.OK

        except Exception as e:
            print(f"[FinancialRetriever] FAISS search error: {e}")
            return [], RetrievalStatus.RETRIEVAL_ERROR

    def clear(self):
        self._init_empty()
        for p in [self._index_path, self._texts_path, self._metadata_path]:
            if p.exists():
                p.unlink()
        print("Index cleared.")
