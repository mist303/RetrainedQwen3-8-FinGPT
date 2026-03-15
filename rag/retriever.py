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

DEFAULT_INDEX_DIR        = Path(__file__).parent / "index"
EMBEDDING_MODEL          = "FinLang/finance-embeddings-investopedia"
EMBEDDING_MODEL_FALLBACK = "all-MiniLM-L6-v2"
EMBEDDING_DIM            = 384
CHUNK_SIZE               = 512
CHUNK_OVERLAP            = 64


class RetrievalStatus(Enum):
    OK              = "ok"
    EMPTY_INDEX     = "empty_index"
    NO_RESULTS      = "no_results"
    SCORE_TOO_HIGH  = "score_too_high"
    RETRIEVAL_ERROR = "retrieval_error"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


class FinancialRetriever:
    """
    FAISS flat L2 index backed by finance-domain sentence embeddings.
    retrieve() returns a (chunks, status) tuple so callers always know
    exactly why they got zero results.
    """

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        if not DEPS_AVAILABLE:
            raise ImportError("Run: pip install faiss-cpu sentence-transformers")

        self.index_dir       = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path     = self.index_dir / "index.faiss"
        self._texts_path     = self.index_dir / "texts.pkl"
        self._metadata_path  = self.index_dir / "metadata.pkl"

        self.embedder = self._load_embedder()
        self._load() if self._index_path.exists() else self._init_empty()

    def _load_embedder(self) -> "SentenceTransformer":
        for name in [EMBEDDING_MODEL, EMBEDDING_MODEL_FALLBACK]:
            try:
                print(f"Loading embedding model: {name}")
                m = SentenceTransformer(name)
                if name == EMBEDDING_MODEL_FALLBACK:
                    print(f"  ⚠ Using fallback embedder. Download {EMBEDDING_MODEL} for better financial retrieval.")
                return m
            except Exception as e:
                print(f"  Could not load {name}: {e}")
        raise RuntimeError("Could not load any embedding model.")

    def _init_empty(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.texts, self.metadata = [], []
        print("Created new empty FAISS index.")

    def _load(self):
        self.index = faiss.read_index(str(self._index_path))
        with open(self._texts_path,    "rb") as f: self.texts    = pickle.load(f)
        with open(self._metadata_path, "rb") as f: self.metadata = pickle.load(f)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def save(self):
        faiss.write_index(self.index, str(self._index_path))
        with open(self._texts_path,    "wb") as f: pickle.dump(self.texts,    f)
        with open(self._metadata_path, "wb") as f: pickle.dump(self.metadata, f)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    @property
    def n_docs(self) -> int:
        return self.index.ntotal

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None, auto_chunk: bool = True):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        chunks, chunk_metas = [], []
        for text, meta in zip(texts, metadatas):
            parts = chunk_text(text) if auto_chunk else [text]
            chunks.extend(parts)
            chunk_metas.extend([meta] * len(parts))
        if not chunks:
            return
        self.index.add(self._embed(chunks))
        self.texts.extend(chunks)
        self.metadata.extend(chunk_metas)
        print(f"Indexed {len(chunks)} chunks. Total: {self.index.ntotal}")

    def add_file(self, filepath: str | Path, metadata: Optional[dict] = None):
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        self.add_texts([text], metadatas=[metadata or {"source": str(filepath)}])

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[dict], RetrievalStatus]:
        if self.index.ntotal == 0:
            return [], RetrievalStatus.EMPTY_INDEX
        try:
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(self._embed([query]), k)
            results = [
                {"text": self.texts[i], "score": float(d), "metadata": self.metadata[i]}
                for d, i in zip(distances[0], indices[0]) if i != -1
            ]
            return (results, RetrievalStatus.OK) if results else ([], RetrievalStatus.NO_RESULTS)
        except Exception as e:
            print(f"[FinancialRetriever] FAISS search error: {e}")
            return [], RetrievalStatus.RETRIEVAL_ERROR

    def clear(self):
        self._init_empty()
        for p in [self._index_path, self._texts_path, self._metadata_path]:
            if p.exists():
                p.unlink()
        print("Index cleared.")
