"""BM25 lexical retrieval over document chunks."""

from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.retrieval.chunker import Chunk, load_chunks

_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus) if chunks else None

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self._bm25 or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [(self.chunks[i], float(score)) for i, score in ranked if score > 0]


@lru_cache
def get_bm25_index() -> BM25Index:
    return BM25Index(load_chunks())