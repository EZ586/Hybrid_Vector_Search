# src/baselines/hybrid/search.py

from typing import Iterable, Tuple, List, Dict, Any, Optional, Callable
import time

import numpy as np
import faiss

from src.baselines.hybrid.early_stop import stop_when_k_reached

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
    global_bound: Optional[float] = None,
    probe_order: Optional[List[int]] = None,
    allowed_counts_per_list: Optional[np.ndarray] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Predicate-aware ANN loop using FAISS IVF + IDSelectorBatch.

    - enforces the allow-list inside FAISS (not post-filter)
    - progressively increases nprobe using the provided iterator
    - accumulates candidates across probes (doesn't overwrite each round)
    - optional early-stop policy can inspect the current state
    - optional probe_order / allowed_counts_per_list are accepted but
      currently used only for logging (FAISS still controls list traversal)

    Args:
        qvec: (D,) query vector, already L2-normalized for IP if needed.
        index: trained FAISS IVF index (IP or L2), built over canonical vectors.
        allow_ids: 1D array of IDs allowed by metadata filters.
        K: final number of valid results to return.
        nprobe_iter: iterable of nprobe values to try in order.
        early_stop_policy: optional callable(state) -> (bool, reason).
            If None, defaults to stop_when_k_reached (baseline behavior).
        global_bound: optional global score bound (passed to policy via state).
        probe_order: optional ordering of IVF lists (currently for logging only).
        allowed_counts_per_list: optional (L,) counts of allowed ids per list.

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
            "early_stop_used": False,
            "early_stop_reason": None,
            "probes_run": 0,
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
    last_nprobe: Optional[int] = None

    # early-stop bookkeeping
    if early_stop_policy is None:
        policy = stop_when_k_reached
        policy_name = "k_only"
    else:
        policy = early_stop_policy
        # backend is responsible for remembering the name; we just report used/triggered
        policy_name = None

    kth_history: List[float] = []
    early_stop_used = False
    early_stop_reason: Optional[str] = None
    probe_index = 0

    # simple oversample factor; can be tuned
    oversample_factor = 10
    search_k = max(K * oversample_factor, K)

    for nprobe in nprobe_iter:
        lists_probed += 1
        probe_index += 1
        last_nprobe = nprobe
        index.nprobe = nprobe

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
            prev = candidates.get(idx)
            if prev is None or dist > prev:
                candidates[idx] = dist

        # compute current kth score if we have at least one candidate
        current_kth_score: Optional[float] = None
        if candidates:
            # get scores sorted desc just enough to find kth
            scores = np.fromiter(candidates.values(), dtype=np.float32)
            if len(scores) >= K:
                # kth best = K-th element of sorted-desc list
                sorted_scores = np.sort(scores)[::-1]
                current_kth_score = float(sorted_scores[K - 1])
            else:
                sorted_scores = np.sort(scores)[::-1]
                current_kth_score = float(sorted_scores[-1])

        # maintain kth history if we have K or more candidates
        if current_kth_score is not None and len(candidates) >= K:
            kth_history.append(current_kth_score)

        # build search state for early-stop policy
        state: SearchState = {
            "K": K,
            "num_candidates": len(candidates),
            "current_kth_score": current_kth_score,
            "probe_index": probe_index,
            "global_bound": global_bound,
            "kth_history": kth_history,
            # window/epsilon are optional; policies can pick defaults
        }

        # apply early-stop policy (if we have at least K or policy chooses otherwise)
        should_stop, reason = policy(state)
        if should_stop:
            early_stop_used = True
            early_stop_reason = reason or "unspecified"
            break

    # sort candidates by score desc (FAISS IP = larger is better)
    sorted_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:K]
    top_ids = [item[0] for item in top_items]

    # compute kth_at_stop if we actually have K
    if len(top_items) == K:
        kth_at_stop = top_items[-1][1]
    else:
        kth_at_stop = None

    # we don't currently maintain a FAISS bound; leave as passed-in or None
    bound_at_stop = global_bound

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
        "early_stop_used": early_stop_used,
        "early_stop_reason": early_stop_reason,
        "probes_run": probe_index,
    }

    # NOTE: we do NOT stuff hybrid-specific extras here; the backend will
    # build a stats["extras"] dict based on its own knobs.
    return top_ids, stats