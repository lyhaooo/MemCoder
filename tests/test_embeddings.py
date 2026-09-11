from memcoder.memory.embeddings import HashingEmbedder, cosine_similarity


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(64)
    first = embedder.embed("Dijkstra heap shortest path")
    second = embedder.embed("Dijkstra heap shortest path")

    assert first == second
    assert abs(cosine_similarity(first, first) - 1.0) < 1e-9
    assert cosine_similarity(first, embedder.embed("heap shortest path")) > 0

