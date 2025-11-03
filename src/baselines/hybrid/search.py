# baselines/hybrid/search.py

from typing import Iterable, Tuple, List, Dict, Any

import numpy as np
import faiss


def hybrid_search(
    qvec: np.ndarray,
    index: faiss.IndexIVFFlat,
    allow_ids: np.ndarray,
    K: int,
    nprobe_iter: Iterable[int],
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Predicate-aware ANN loop using FAISS IVF + IDSelectorBatch.

    Args:
        qvec: (D,) query vector, already L2-normalized for IP if needed.
        index: trained FAISS IVF index (IP).
        allow_ids: 1D array of int64/int32 IDs allowed by metadata filters.
        K: final number of valid results to return.
        nprobe_iter: iterable of nprobe values to try in order.

    Returns:
        ids: list of up to K valid ids (subset of allow_ids) in similarity order.
        stats: dict with latency_ms, scored_vectors, nprobe, retries, etc.
    """
    # TODO: implement loop: for nprobe in nprobe_iter: set index.nprobe, search, filter
    raise NotImplementedError
