# src/baselines/hybrid/search.py

from __future__ import annotations
from typing import Tuple, List, Dict, Any
import time
import heapq
import numpy as np
import faiss


# -------------------------------------------------------------
# Compute similarity or negative L2
# -------------------------------------------------------------
def compute_score(q, x, metric):
    if metric == faiss.METRIC_INNER_PRODUCT:
        return float(np.dot(q, x))
    diff = q - x
    return -float(diff @ diff)


# -------------------------------------------------------------
# HYBRID HNSW SEARCH (Windows FAISS: flat neighbors buffer)
# -------------------------------------------------------------
def hnsw_hybrid_search(
    qvec: np.ndarray,
    index: faiss.IndexHNSWFlat,
    allow_ids: np.ndarray,
    K: int,
    ef_search: int = 128,
) -> Tuple[List[int], Dict[str, Any]]:

    # --------------------------
    # Handle empty allow-list
    # --------------------------
    if allow_ids is None or len(allow_ids) == 0:
        return [], {
            "latency_ms": 0.0,
            "scored_vectors": 0,
            "nodes_expanded": 0,
            "ef_search": ef_search,
            "kth_at_stop": None,
            "filter_selectivity": 0.0,
        }

    allow_set = set(int(x) for x in allow_ids)

    metric = index.metric_type
    qvec = np.asarray(qvec, dtype=np.float32).ravel()

    hnsw = index.hnsw

    # -------------------------------------------------------------
    # SAFE nb_neighbors across FAISS versions
    # -------------------------------------------------------------
    try:
        # Newer FAISS: requires a layer argument
        M = hnsw.nb_neighbors(0)
    except TypeError:
        # Older FAISS: attribute or no-arg method
        if callable(hnsw.nb_neighbors):
            M = hnsw.nb_neighbors()
        else:
            M = hnsw.nb_neighbors

    # -------------------------------------------------------------
    # Convert SWIG Int32Vector → NumPy (Windows FAISS)
    # -------------------------------------------------------------
    try:
        # Windows FAISS CPU wheels usually provide this:
        neighbors_flat = faiss.rev_int32_swig_ptr(hnsw.neighbors, index.ntotal * M)
    except AttributeError:
        # Fallback: generic rev_swig_ptr
        neighbors_flat = faiss.rev_swig_ptr(hnsw.neighbors, index.ntotal * M)

    frontier = []      # min-heap of (bound, node)
    visited = set()
    top_k = []         # max-heap of (-score, id)

    # --------------------------
    # Start from entry point
    # --------------------------
    entry = hnsw.entry_point
    entry_vec = index.reconstruct(entry)
    entry_s = compute_score(qvec, entry_vec, metric)
    entry_b = -entry_s if metric == faiss.METRIC_INNER_PRODUCT else entry_s
    heapq.heappush(frontier, (entry_b, entry))

    start = time.perf_counter()
    expansions = 0

    # --------------------------
    # Main HNSW search
    # --------------------------
    while frontier and expansions < ef_search:
        bound, node = heapq.heappop(frontier)
        expansions += 1

        if node in visited:
            continue
        visited.add(node)

        # score this node
        vec = index.reconstruct(node)
        s = compute_score(qvec, vec, metric)

        # allow-list filtering
        if node in allow_set:
            if len(top_k) < K:
                heapq.heappush(top_k, (-s, node))
            else:
                if s > -top_k[0][0]:   # better than current worst
                    heapq.heapreplace(top_k, (-s, node))

        # --------------------------
        # Early-stop condition
        # --------------------------
        if len(top_k) == K and frontier:
            worst = -top_k[0][0]  # lowest score in top-k

            if metric == faiss.METRIC_INNER_PRODUCT:
                if frontier[0][0] > -worst:
                    break
            else:
                if frontier[0][0] > worst:
                    break

        # ---------------------------------------------------------
        # Expand neighbors from flat buffer
        # neighbors_flat is a single Int32 array of length ntotal * M
        # neighbors for node i are at:
        #   neighbors_flat[i*M : i*M + M]
        # ---------------------------------------------------------
        start_i = node * M
        end_i = start_i + M
        node_neighbors = neighbors_flat[start_i:end_i]

        for nb in node_neighbors:
            if nb < 0:
                continue
            if nb in visited:
                continue

            nb_vec = index.reconstruct(nb)
            nb_s = compute_score(qvec, nb_vec, metric)

            if metric == faiss.METRIC_INNER_PRODUCT:
                nb_b = -nb_s
            else:
                nb_b = nb_s

            heapq.heappush(frontier, (nb_b, nb))

    # --------------------------
    # Build top-K output
    # --------------------------
    results = [(nid, -s) for (s, nid) in top_k]
    results.sort(key=lambda x: x[1], reverse=True)

    top_ids = [nid for nid, _ in results[:K]]
    kth = results[K-1][1] if len(results) >= K else None

    stats = {
        "latency_ms": (time.perf_counter() - start) * 1000.0,
        "scored_vectors": len(results),
        "nodes_expanded": expansions,
        "ef_search": ef_search,
        "kth_at_stop": kth,
        "filter_selectivity": len(allow_ids) / index.ntotal,
    }

    return top_ids, stats


# -------------------------------------------------------------
# Compatibility wrapper
# -------------------------------------------------------------
def hybrid_search(qvec, index, allow_ids, K, **kwargs):
    return hnsw_hybrid_search(qvec, index, allow_ids, K, ef_search=128)
