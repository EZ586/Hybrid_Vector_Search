# src/baselines/hybrid/search.py

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import time
import heapq
import faiss


# ---------------------------------------------------------------------
# Utility: fast L2 or IP distance
# ---------------------------------------------------------------------
def compute_score(qvec, xvec, metric_type):
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(np.dot(qvec, xvec))
    else:
        diff = qvec - xvec
        return -float(diff @ diff)   # negative L2


# ---------------------------------------------------------------------
# Hybrid HNSW search with:
#  - metadata allow-list filtering
#  - early-stop monotonicity (VBase-style)
#  - top-k heap
#  - low Python overhead
# ---------------------------------------------------------------------
def hnsw_hybrid_search(
    qvec: np.ndarray,
    index: faiss.IndexHNSWFlat,
    allow_ids: np.ndarray,
    K: int,
    ef_search: int = 128,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Custom HNSW search that supports:
      - metadata allow-list filtering (prefilter)
      - safe monotonic early-stop
      - postfiltering
      - top-K heap
      - latency + stats

    Args:
        qvec: (D,) query vector, already normalized for IP if needed
        index: FAISS HNSW index
        allow_ids: 1D array of valid IDs
        K: number of results to return
        ef_search: maximum frontier expansions (default 128)

    Returns:
        (top_ids, stats)
    """

    # ------------------------------------------------------------
    # Handle empty allow-list
    # ------------------------------------------------------------
    if allow_ids is None or len(allow_ids) == 0:
        return [], {
            "latency_ms": 0.0,
            "scored_vectors": 0,
            "nodes_expanded": 0,
            "ef_search": None,
            "kth_at_stop": None,
            "bound_at_stop": None,
            "filter_selectivity": 0.0,
            "notes": "empty allow_ids",
        }

    allow_set = set(int(x) for x in allow_ids)
    metric_type = index.metric_type

    qvec = np.asarray(qvec, dtype=np.float32).ravel()

    hnsw = index.hnsw
    entry = hnsw.entry_point

    # ------------------------------------------------------------
    # Priority queues
    # frontier = min-heap of (est_dist, node)
    # top_k = max-heap of (-score, node) (so worst item is top)
    # ------------------------------------------------------------
    frontier = []
    top_k = []  # store (-score, id)

    visited = set()
    expansions = 0

    # ------------------------------------------------------------
    # Push entry point
    # ------------------------------------------------------------
    entry_vec = index.reconstruct(entry)
    entry_score = compute_score(qvec, entry_vec, metric_type)
    heapq.heappush(frontier, ( -entry_score if metric_type == faiss.METRIC_INNER_PRODUCT else entry_score, entry))

    # ------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------
    start_time = time.perf_counter()

    while frontier and expansions < ef_search:

        est_bound, node = heapq.heappop(frontier)
        expansions += 1

        # Early-stop condition:
        #  We can stop when the best remaining frontier bound
        #  cannot beat the current worst top-k score.
        if len(top_k) >= K:
            worst_top_k = -top_k[0][0]

            # est_bound is stored as:
            #   IP: -score (because smaller is better when negated)
            #   L2: actual distance (smaller is better)
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                # est_bound = -(possible_score), so est_bound > -worst_top_k → stop
                if est_bound > -worst_top_k:
                    break
            else:
                # L2: est_bound = real dist, early-stop when est_bound > worst_top_k
                if est_bound > worst_top_k:
                    break

        # Skip duplicates
        if node in visited:
            continue
        visited.add(node)

        # Retrieve real vector
        vec = index.reconstruct(node)
        score = compute_score(qvec, vec, metric_type)

        # Metadata prefilter: only score if allowed
        if node in allow_set:
            if len(top_k) < K:
                heapq.heappush(top_k, (-score, node))
            else:
                # if this score is better than worst-top-k, replace it
                if score > -top_k[0][0]:   # because stored as (-score)
                    heapq.heapreplace(top_k, (-score, node))

        # Expand neighbors
        neighbors = hnsw.neighbors(node)
        for nb in neighbors:
            if nb == -1:
                continue
            if nb not in visited:
                # frontier bound uses real distance or negative score
                nb_vec = index.reconstruct(nb)
                nb_score = compute_score(qvec, nb_vec, metric_type)

                if metric_type == faiss.METRIC_INNER_PRODUCT:
                    bound = -nb_score
                else:
                    bound = nb_score

                heapq.heappush(frontier, (bound, nb))

    # ------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Extract top K
    results = []
    while top_k:
        neg_score, idx = heapq.heappop(top_k)
        results.append((idx, -neg_score))
    results.sort(key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in results[:K]]
    kth_at_stop = results[K-1][1] if len(results) >= K else None

    stats = {
        "latency_ms": latency_ms,
        "scored_vectors": len(results),
        "nodes_expanded": expansions,
        "ef_search": ef_search,
        "kth_at_stop": kth_at_stop,
        "bound_at_stop": None,  # frontier bound not tracked separately
        "filter_selectivity": len(allow_ids) / index.ntotal if index.ntotal > 0 else None,
        "notes": None,
    }

    return top_ids, stats


# ---------------------------------------------------------------------
# Backward-compatible API
# ---------------------------------------------------------------------
def hybrid_search(
    qvec: np.ndarray,
    index: faiss.IndexHNSWFlat,
    allow_ids: np.ndarray,
    K: int,
    nprobe_iter=None,   # ignored for HNSW, kept for compatibility
    **kwargs
):
    """
    For compatibility with your earlier IVF API.

    Delegates to the new HNSW search.
    """
    return hnsw_hybrid_search(qvec, index, allow_ids, K, ef_search=128)
