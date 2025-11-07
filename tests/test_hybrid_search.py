# tests/test_hybrid_search.py

import numpy as np
import faiss

from src.baselines.hybrid.search import hybrid_search
from src.baselines.hybrid.scheduler import linear_nprobe_scheduler

def build_tiny_ivf_index(n = 10, d=4, nlist=2):
    """
    build tiny INF index in-memory
    n: number of vectors in dataset
    d: dimensionality of each vector
    nlist: number of clusters (inverted lists)
    """
    np.random.seed(0)
    rand_vec = np.random.rand(n, d).astype("float32")
    rand_vec /= np.linalg.norm(rand_vec, axis=1, keepdims=True)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(rand_vec)
    ids = np.arange(n, dtype=np.int64)
    index.add_with_ids(rand_vec, ids)
    return index, rand_vec

def test_hybrid_search_runs_basic():
    """
    Smoke test: hybrid_search should run and return (ids, stats) tuple.
    """
    # TODO: build tiny IVF index in-memory, tiny allow_ids, and call hybrid_search
    index, rand_vec = build_tiny_ivf_index()
    qvec = rand_vec[0]
    allow_ids = np.arange(len(rand_vec))
    nprobe_iter = linear_nprobe_scheduler(start=1, step=1, max_nprobe=2)

    ids, stats = hybrid_search(
        qvec=qvec,
        index=index,
        allow_ids=allow_ids,
        K=3,
        nprobe_iter=nprobe_iter
    )

    assert isinstance(ids, list)
    assert isinstance(stats, dict)
    assert len(ids) <= 3
    assert all(isinstance(i, (int, np.integer)) for i in ids)


def test_hybrid_stats_fields_present():
    """
    hybrid_search should return stats containing latency_ms and scored_vectors at minimum.
    """
    # TODO: call hybrid_search and assert required keys in stats
    index, rand_vec = build_tiny_ivf_index()
    qvec = rand_vec[0]
    allow_ids = np.arange(len(rand_vec))
    nprobe_iter = linear_nprobe_scheduler(start=1, step=1, max_nprobe=2)

    ids, stats = hybrid_search(
        qvec=qvec,
        index=index,
        allow_ids=allow_ids,
        K=3,
        nprobe_iter=nprobe_iter
    )

    required_keys = {"latency_ms", "scored_vectors", "nprobe", "retries"}
    for key in required_keys:
        assert key in stats
    
    assert isinstance(stats["latency_ms"], (int, float))
    assert stats["latency_ms"] >= 0
    assert stats["scored_vectors"] > 0