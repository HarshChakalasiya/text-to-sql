from app.retrieval.indexing import content_hash


def test_content_hash_stable():
    assert content_hash("invoice discounting") == content_hash("invoice discounting")
    assert content_hash("a") != content_hash("b")
