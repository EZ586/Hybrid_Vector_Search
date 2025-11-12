# src/baselines/hybrid/search.py

from typing import Iterable, Tuple, List, Dict, Any
import time
import numpy as np
import faiss

from src.baselines.hybrid.list_ordering import build_probe_order


def hybrid_search(
    qvec: np.ndarray,
    index: faiss.IndexIVFFlat,
    allow_ids: np.ndarray,
    K: int,
    nprobe_iter: Iterable[int],
    centroids: np.ndarray | None = None,
    allowed_counts_per_list: np.ndarray | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Predicate-aware ANN loop using FAISS IVF + IDSelectorBatch.

    - enforces the allow-list inside FAISS (not post-filter)
    - progressively increases nprobe using the provided iterator
    - accumulates candidates across probes (doesn't overwrite each round)
    - returns manual-compatible stats fields where possible

    Args:
        qvec: (D,) query vector, already L2-normalized for IP if needed.
        index: trained FAISS IVF index (IP or L2), built over canonical vectors.
        allow_ids: 1D array of IDs allowed by metadata filters.
        K: final number of valid results to return.
        nprobe_iter: iterable of nprobe values to try in order.

    Returns:
        ids: list of up to K valid ids (subset of allow_ids) in similarity order.
        stats: dict with latency_ms, scored_vectors, lists_probed, nprobe, etc.
    """
    # handle empty allow-list early
    if allow_ids is None or len(allow_ids) == 0:
        return [], {
            "latency_ms": 0.0,
            "scored_vectors": 0,
            "lists_probed": 0,
            "nprobe": None,
            "kth_at_stop": None,
            "bound_at_stop": None,
            "filter_selectivity": 0.0,
            "notes": "empty allow_ids",
        }

    # FAISS wants int64 for IDSelectorBatch
    allow_ids = np.asarray(allow_ids, dtype=np.int64)

    selector = faiss.IDSelectorBatch(allow_ids)
    params = faiss.SearchParametersIVF()
    params.sel = selector  # enforce allow-list inside FAISS

    # normalize query shape
    qvec = np.asarray(qvec, dtype=np.float32).reshape(1, -1)

    start_time = time.perf_counter()

    # accumulate candidates across probes: id -> score
    candidates: Dict[int, float] = {}
    scored_vectors = 0
    lists_probed = 0
    last_nprobe = None

    # simple oversample factor; can be tuned
    oversample_factor = 10
    search_k = max(K * oversample_factor, K)

    probe_order = None
    if centroids is not None:
        try:
            probe_order = build_probe_order(
                qvec.squeeze(0), centroids, allowed_counts_per_list
            )
        except Exception as e:
            print(f"[WARN] Failed to compute custom probe order: {e}")

    for nprobe in nprobe_iter:
        lists_probed += 1
        last_nprobe = nprobe
        index.nprobe = nprobe
        params.nprobe = nprobe

        # optionally reorder probe lists in FAISS
        if probe_order is not None and hasattr(index, "set_probe_order"):
            try:
                index.set_probe_order(np.array(probe_order, dtype=np.int64))
            except Exception:
                pass  # FAISS CPU bindings don’t always expose set_probe_order
        
        # search with selector enforced
        D, I = index.search(qvec, search_k, params=params)

        # count how many vectors FAISS actually returned (non -1)
        returned_ids = I[0]
        returned_dists = D[0]
        valid_mask = returned_ids != -1
        scored_vectors += int(valid_mask.sum())

        # merge into candidates (keep best score per id)
        for dist, idx in zip(returned_dists, returned_ids):
            if idx == -1:
                continue
            # if we see the same id in a later probe, keep the better score
            prev = candidates.get(idx)
            if prev is None or dist > prev:
                candidates[idx] = dist

        # stop if we already have enough
        # if len(candidates) >= K:
        #     break

    # sort candidates by score desc (FAISS IP = larger is better)
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:K]
    top_ids = [item[0] for item in top_items]

    # compute kth_at_stop if we actually have K
    if len(top_items) == K:
        kth_at_stop = top_items[-1][1]
    else:
        kth_at_stop = None

    # we don't currently maintain a FAISS bound; leave as None
    bound_at_stop = None

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # we can derive selectivity from index.ntotal
    try:
        total = index.ntotal
        filter_selectivity = float(len(allow_ids)) / float(total) if total > 0 else None
    except AttributeError:
        filter_selectivity = None

    stats: Dict[str, Any] = {
        "latency_ms": latency_ms,
        "scored_vectors": scored_vectors,
        "lists_probed": lists_probed,
        "nprobe": last_nprobe,
        "kth_at_stop": kth_at_stop,
        "bound_at_stop": bound_at_stop,
        "filter_selectivity": filter_selectivity,
        "notes": None,
    }

    return top_ids, stats
