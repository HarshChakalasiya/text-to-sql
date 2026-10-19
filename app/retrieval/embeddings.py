"""Dense vector index: Postgres pgvector persistence with in-memory search matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from app.db.session import SessionLocal, engine
from app.llm.provider import LLMProvider, get_llm_provider
from app.retrieval.chunker import Chunk, load_chunks
from app.retrieval.indexing import (
    load_embeddings_from_db,
    pgvector_enabled,
    sync_document_embeddings,
)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


@dataclass
class VectorHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        self.chunks = chunks
        self.matrix = matrix  # shape (n, d), L2-normalized

    @classmethod
    def _build_in_memory(cls, provider: LLMProvider) -> "VectorStore":
        chunks = load_chunks()
        if not chunks:
            return cls([], np.zeros((0, 1), dtype=np.float32))
        vectors = provider.embed([c.text for c in chunks])
        matrix = _normalize(np.asarray(vectors, dtype=np.float32))
        return cls(chunks, matrix)

    @classmethod
    def _build_pgvector(cls, provider: LLMProvider) -> "VectorStore":
        db = SessionLocal()
        try:
            sync_document_embeddings(db, provider)
            chunks, matrix = load_embeddings_from_db(db)
            if matrix.size == 0:
                return cls([], np.zeros((0, 1), dtype=np.float32))
            return cls(chunks, _normalize(matrix))
        finally:
            db.close()

    @classmethod
    def build(cls, provider: LLMProvider | None = None) -> "VectorStore":
        llm = provider or get_llm_provider()
        if pgvector_enabled(engine):
            try:
                return cls._build_pgvector(llm)
            except (OperationalError, ProgrammingError, DBAPIError, OSError):
                return cls._build_in_memory(llm)
        return cls._build_in_memory(llm)

    def search(self, query: str, top_k: int = 5, provider: LLMProvider | None = None) -> list[VectorHit]:
        if self.matrix.size == 0:
            return []
        llm = provider or get_llm_provider()
        q = np.asarray(llm.embed([query])[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.matrix @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [
            VectorHit(chunk=self.chunks[int(i)], score=float(scores[int(i)]))
            for i in top_idx
            if scores[int(i)] > 0
        ]


_STORE: VectorStore | None = None


def get_vector_store(provider: LLMProvider | None = None, *, rebuild: bool = False) -> VectorStore:
    global _STORE
    if _STORE is None or rebuild:
        _STORE = VectorStore.build(provider)
    return _STORE


def reset_vector_store() -> None:
    global _STORE
    _STORE = None
