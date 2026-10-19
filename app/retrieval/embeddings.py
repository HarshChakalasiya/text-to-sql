"""In-memory dense vector index using the active LLM embedding model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.llm.provider import LLMProvider, get_llm_provider
from app.retrieval.chunker import Chunk, load_chunks


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
    def build(cls, provider: LLMProvider | None = None) -> "VectorStore":
        llm = provider or get_llm_provider()
        chunks = load_chunks()
        if not chunks:
            return cls([], np.zeros((0, 1), dtype=np.float32))
        vectors = llm.embed([c.text for c in chunks])
        matrix = _normalize(np.asarray(vectors, dtype=np.float32))
        return cls(chunks, matrix)

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