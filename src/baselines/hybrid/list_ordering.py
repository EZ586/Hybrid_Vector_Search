# src/baselines/hybrid/list_ordering.py
"""
Utilities for ordering IVF lists for hybrid search.

This module is intentionally PURE:
- it does not load FAISS
- it does not read artifacts
- it only operates on numpy arrays passed in by the caller

Typical usage:
    ordered_list_ids = build_probe_order(qvec, centroids, allowed_counts)
and then the search loop can probe lists in that order.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np


def _ensure_1d_float(vec: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {v.shape}")
    return v


def _ensure_2d_float(mat: np.ndarray, name: str) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {m.shape}")
    return m


def _ensure_1d_int(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    return a


def score_centroids_ip(qvec: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute similarity scores between a query vector and IVF centroids
    using inner product / cosine-style scoring.

    Args:
        qvec: (D,) float32. Assumed to be already normalized.
        centroids: (L, D) float32. One centroid per IVF list.

    Returns:
        scores: (L,) float32. Higher = better / closer list to probe.
    """
    q = _ensure_1d_float(qvec, "qvec")
    C = _ensure_2d_float(centroids, "centroids")

    D = C.shape[1]
    if q.shape[0] != D:
        raise ValueError(
            f"qvec dim {q.shape[0]} must match centroids dim {D} "
            "(centroids.shape == (L, D))"
        )

    # (L, D) @ (D,) -> (L,)
    scores = C @ q
    return scores.astype(np.float32, copy=False)


def order_lists_by_score(
    scores: np.ndarray,
    allowed_counts: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Order IVF list ids by score, optionally pushing empty/forbidden lists to the end.

    Args:
        scores: (L,) float32 scores for each list.
        allowed_counts: optional (L,) int array. If provided, lists with
            count == 0 will be placed at the end.

    Returns:
        list_ids: List[int] of length L. list_ids[0] is the best list to probe.
    """
    s = np.asarray(scores, dtype=np.float32)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {s.shape}")
    L = s.shape[0]
    list_ids = np.arange(L, dtype=np.int64)

    if allowed_counts is None:
        # sort by score desc
        order = np.argsort(-s)
        return list_ids[order].tolist()

    counts = _ensure_1d_int(allowed_counts, "allowed_counts")
    if counts.shape[0] != L:
        raise ValueError(
            f"allowed_counts length {counts.shape[0]} "
            f"must match scores length {L}"
        )

    # two-phase ordering:
    # 1) lists with allowed>0, sorted by score desc
    # 2) lists with allowed==0, sorted by score desc (or just appended)
    nonempty_mask = counts > 0
    nonempty_ids = list_ids[nonempty_mask]
    empty_ids = list_ids[~nonempty_mask]

    nonempty_order = nonempty_ids[np.argsort(-s[nonempty_mask])]
    empty_order = empty_ids[np.argsort(-s[~nonempty_mask])]

    ordered = np.concatenate([nonempty_order, empty_order])
    return ordered.tolist()


def build_probe_order(
    qvec: np.ndarray,
    centroids: np.ndarray,
    allowed_counts: Optional[np.ndarray] = None,
) -> List[int]:
    """
    High-level helper: given a query and IVF metadata, produce a probe order.

    This is the ONLY function that other modules should import.

    Args:
        qvec: (D,) float32 query vector.
        centroids: (L, D) float32 IVF centroids.
        allowed_counts: optional (L,) ints: how many ids in each list
            are allowed by the query's metadata filters.

    Returns:
        ordered_list_ids: List[int] of length L.
    """
    scores = score_centroids_ip(qvec, centroids)
    ordered_list_ids = order_lists_by_score(scores, allowed_counts)
    return ordered_list_ids