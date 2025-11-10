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
    # TODO: validate shapes (len(qvec) == centroids.shape[1])
    # TODO: consider supporting L2 later
    scores = centroids @ qvec
    return scores


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
    # TODO: add shape checks
    L = scores.shape[0]
    list_ids = np.arange(L)

    if allowed_counts is None:
        # sort by score desc
        order = np.argsort(-scores)
        return list_ids[order].tolist()

    # two-phase ordering:
    # 1) lists with allowed>0, sorted by score desc
    # 2) lists with allowed==0, sorted by score desc (or just appended)
    nonempty_mask = allowed_counts > 0
    nonempty_ids = list_ids[nonempty_mask]
    empty_ids = list_ids[~nonempty_mask]

    nonempty_order = nonempty_ids[np.argsort(-scores[nonempty_mask])]
    empty_order = empty_ids[np.argsort(-scores[~nonempty_mask])]

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