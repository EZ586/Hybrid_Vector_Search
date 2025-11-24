# src/baselines/hybrid/search.py

from typing import Iterable, Tuple, List, Dict, Any, Optional, Callable
import time

import numpy as np
import faiss

from src.baselines.hybrid.early_stop import stop_when_k_reached
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
    global_bound: Optional[float] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Predicate-aware ANN loop using FAISS IVF + IDSelectorBatch.

    - enforces the allow-list inside FAISS (not post-filter)
    - progressively increases nprobe using the provided iterator
    - accumulates candidates across probes (doesn't overwrite each round)
    - optional early-stop policy can inspect the current state

    Args:
        qvec: (D,) query vector, already L2-normalized for IP if needed.
        index: trained FAISS IVF index (IP or L2), built over canonical vectors.
        allow_ids: 1D array of IDs allowed by metadata filters.
        K: final number of valid results to return.
        nprobe_iter: iterable of nprobe values to try in order.
        early_stop_policy: optional callable(state) -> (bool, reason).
            If None, defaults to stop_when_k_reached (baseline behavior).
        global_bound: optional global score bound (passed to policy via state).

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

    # FAISS wants int64 for IDSelectorBatch (via selector helper)
    allow_ids = np.asarray(allow_ids, dtype=np.int64)

    selector = build_idselector(allow_ids)
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
    else:
        policy = early_stop_policy

    kth_history: List[float] = []
    early_stop_used = False
    early_stop_reason: Optional[str] = None
    probe_index = 0

    # neighbor_radius_history: how the E-th best score evolves over probes
    neighbor_radius_history: List[float] = []
    # window_median_history: median score of *this probe's* returned vectors
    window_median_history: List[float] = []
    # small default window size for rolling median over probe medians
    rm_window_size: int = 3

    # simple oversample factor; can be tuned
    oversample_factor = 20
    search_k = max(K * oversample_factor, K)

    for nprobe in nprobe_iter:
        lists_probed += 1
        probe_index += 1
        last_nprobe = nprobe
        index.nprobe = nprobe
        params.nprobe = nprobe

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

        # --- compute current kth score + RM-style helper metrics ---
        current_kth_score: Optional[float] = None
        neighbor_radius: Optional[float] = None
        probe_median: Optional[float] = None
        rm_window_median: Optional[float] = None

        if candidates:
            # get scores sorted desc just enough to find kth and E-th (for radius)
            scores = np.fromiter(candidates.values(), dtype=np.float32)
            sorted_scores = np.sort(scores)[::-1]

            # kth best = K-th element of sorted-desc list (or worst if < K)
            if len(sorted_scores) >= K:
                current_kth_score = float(sorted_scores[K - 1])
            else:
                current_kth_score = float(sorted_scores[-1])

            # E-neighborhood radius (E >= K, capped by #candidates)
            # here we pick E = min(len(sorted_scores), max(K, 2*K)) as a simple heuristic
            E = min(len(sorted_scores), max(K, 2 * K))
            if E > 0:
                neighbor_radius = float(sorted_scores[E - 1])
                # track how the neighborhood radius evolves over probes
                neighbor_radius_history.append(neighbor_radius)

        # maintain kth history if we have K or more candidates
        if current_kth_score is not None and len(candidates) >= K:
            kth_history.append(current_kth_score)

        # median score of *this probe's* returned vectors (RM helper)
        probe_scores = returned_dists[valid_mask]
        if probe_scores.size > 0:
            probe_median = float(np.median(probe_scores))
            window_median_history.append(probe_median)

        # rolling median over the last `rm_window_size` probe medians
        if window_median_history:
            recent_medians = window_median_history[-rm_window_size:]
            rm_window_median = float(
                np.median(np.asarray(recent_medians, dtype=np.float32))
            )

        # build search state for early-stop policy
        state: SearchState = {
            "K": K,
            "num_candidates": len(candidates),
            "current_kth_score": current_kth_score,
            "probe_index": probe_index,
            "global_bound": global_bound,
            "kth_history": kth_history,
            # RM-style helpers (optional; policies can ignore them)
            "neighbor_radius": neighbor_radius,
            "rm_window_median": rm_window_median,
            "rm_window_size": rm_window_size,
            # window/epsilon are still available for legacy policies
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

    # RM-style summary at stop (optional, for logging/analysis)
    neighbor_radius_at_stop: Optional[float] = None
    rm_window_median_at_stop: Optional[float] = None

    if neighbor_radius_history:
        neighbor_radius_at_stop = neighbor_radius_history[-1]

    if window_median_history:
        recent_medians = window_median_history[-rm_window_size:]
        rm_window_median_at_stop = float(
            np.median(np.asarray(recent_medians, dtype=np.float32))
        )

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
        # RM-style extras (for analysis; current policies may ignore them)
        "neighbor_radius_at_stop": neighbor_radius_at_stop,
        "rm_window_median_at_stop": rm_window_median_at_stop,
    }

    # NOTE: we do NOT stuff hybrid-specific extras here; the backend will
    # build a stats["extras"] dict based on its own knobs.
    return top_ids, stats