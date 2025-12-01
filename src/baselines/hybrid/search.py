# src/baselines/hybrid/search.py

from typing import Iterable, Tuple, List, Dict, Any, Optional, Callable
import time

import numpy as np
import faiss

from src.baselines.hybrid.early_stop import stop_when_k_and_stable
from src.baselines.hybrid.selector import build_idselector

# Type alias for clarity
SearchState = Dict[str, Any]


def hybrid_search(
    qvec: np.ndarray,
    index: faiss.IndexIVFFlat,
    allow_ids: np.ndarray,
    K: int,
    nprobe_iter: Iterable[int],
    *,
    early_stop_policy: Optional[Callable[[SearchState], Tuple[bool, Optional[str]]]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Minimal stable-only variant of hybrid IVF search.

    - Enforces metadata allow-list inside FAISS via IDSelectorBatch
    - Increases nprobe iteratively (nprobe_iter)
    - Accumulates candidates across probes
    - Early-stop uses ONLY stop_when_k_and_stable
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
            "early_stop_used": False,
            "early_stop_reason": None,
            "probes_run": 0,
        }

    # FAISS wants int64 for IDSelectorBatch
    allow_ids = np.asarray(allow_ids, dtype=np.int64)

    selector = build_idselector(allow_ids)
    params = faiss.SearchParametersIVF()
    params.sel = selector

    # normalize query shape
    qvec = np.asarray(qvec, dtype=np.float32).reshape(1, -1)

    start_time = time.perf_counter()

    # accumulate candidates: id -> score
    candidates: Dict[int, float] = {}
    scored_vectors = 0
    lists_probed = 0
    last_nprobe: Optional[int] = None

    # stable-only early stop
    if early_stop_policy is None:
        policy = stop_when_k_and_stable
    else:
        policy = early_stop_policy

    kth_history: List[float] = []
    early_stop_used = False
    early_stop_reason: Optional[str] = None
    probe_index = 0

    # oversample for FAISS IVF
    oversample_factor = 20
    search_k = max(K * oversample_factor, K)

    for nprobe in nprobe_iter:
        lists_probed += 1
        probe_index += 1
        last_nprobe = nprobe
        index.nprobe = nprobe
        params.nprobe = nprobe

        # search with selector
        D, I = index.search(qvec, search_k, params=params)
        returned_ids = I[0]
        returned_dists = D[0]

        valid_mask = returned_ids != -1
        scored_vectors += int(valid_mask.sum())

        # merge candidates
        for dist, idx in zip(returned_dists, returned_ids):
            if idx == -1:
                continue
            prev = candidates.get(idx)
            if prev is None or dist > prev:
                candidates[idx] = dist

        # --- kth score ---
        current_kth_score: Optional[float] = None
        if candidates:
            scores = np.fromiter(candidates.values(), dtype=np.float32)
            sorted_scores = np.sort(scores)[::-1]
            if len(sorted_scores) >= K:
                current_kth_score = float(sorted_scores[K - 1])
            else:
                current_kth_score = float(sorted_scores[-1])

        # track kth history
        if current_kth_score is not None and len(candidates) >= K:
            kth_history.append(current_kth_score)

        # --- stable-only search state ---
        state: SearchState = {
            "K": K,
            "num_candidates": len(candidates),
            "current_kth_score": current_kth_score,
            "probe_index": probe_index,
            "kth_history": kth_history,
            # stability params (feel free to expose as backend config)
            "window": 3,
            "epsilon": 1e-3,
            "min_probes": 3,
        }

        # apply early stop
        should_stop, reason = policy(state)
        if should_stop:
            early_stop_used = True
            early_stop_reason = reason or "unspecified"
            break

    # sort top-K
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:K]
    top_ids = [item[0] for item in top_items]

    # kth at stop
    if len(top_items) == K:
        kth_at_stop = top_items[-1][1]
    else:
        kth_at_stop = None

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # filter selectivity
    try:
        total = index.ntotal
        filter_selectivity = float(len(allow_ids)) / float(total) if total > 0 else None
    except AttributeError:
        filter_selectivity = None

    # final stats
    stats: Dict[str, Any] = {
        "latency_ms": latency_ms,
        "scored_vectors": scored_vectors,
        "lists_probed": lists_probed,
        "nprobe": last_nprobe,
        "kth_at_stop": kth_at_stop,
        "bound_at_stop": None,       # unused in stable-only
        "filter_selectivity": filter_selectivity,
        "notes": None,
        "early_stop_used": early_stop_used,
        "early_stop_reason": early_stop_reason,
        "probes_run": probe_index,
    }

    return top_ids, stats
