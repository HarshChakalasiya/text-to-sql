from app.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_prefers_agreement():
    fused = reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ]
    )
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] in {"a", "b"}
    assert "b" in ids[:2]
    assert "a" in ids[:2]