"""Hybrid search: BM25 + dense vectors fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.config import get_settings
from app.llm.provider import LLMProvider
from app.retrieval.bm25 import get_bm25_index
from app.retrieval.chunker import Chunk
from app.retrieval.embeddings import get_vector_store


@dataclass
class HybridResult:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    vector_rank: int | None


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    top_k: int | None = None,
    provider: LLMProvider | None = None,
) -> list[HybridResult]:
    settings = get_settings()
    top_k = top_k or settings.hybrid_top_k
    fetch_k = max(top_k * 3, 10)

    bm25_hits = get_bm25_index().search(query, top_k=fetch_k)
    vector_hits = get_vector_store(provider).search(query, top_k=fetch_k, provider=provider)

    bm25_ids = [c.id for c, _ in bm25_hits]
    vector_ids = [h.chunk.id for h in vector_hits]
    fused = reciprocal_rank_fusion([bm25_ids, vector_ids])

    by_id = {c.id: c for c, _ in bm25_hits}
    for hit in vector_hits:
        by_id[hit.chunk.id] = hit.chunk

    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}
    vector_rank = {cid: i + 1 for i, cid in enumerate(vector_ids)}

    results: list[HybridResult] = []
    for doc_id, score in fused[:top_k]:
        chunk = by_id.get(doc_id)
        if not chunk:
            continue
        results.append(
            HybridResult(
                chunk=chunk,
                score=score,
                bm25_rank=bm25_rank.get(doc_id),
                vector_rank=vector_rank.get(doc_id),
            )
        )
    return results


def format_sources(results: list[HybridResult]) -> list[dict[str, str]]:
    return [
        {
            "id": r.chunk.id,
            "title": r.chunk.title,
            "source": r.chunk.source,
            "text": r.chunk.text,
            "score": f"{r.score:.4f}",
        }
        for r in results
    ]