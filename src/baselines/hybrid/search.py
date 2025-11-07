# baselines/hybrid/search.py

from typing import Iterable, Tuple, List, Dict, Any

import numpy as np
import faiss
import time


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
    selector = faiss.IDSelectorBatch(np.array(allow_ids, dtype=np.int64))
    params = faiss.SearchParametersIVF()
    params.sel = selector

    start_time = time.perf_counter()
    collected_ids, scored_vectors, retries = [], 0, 0

    for nprobe in nprobe_iter:
        index.nprobe = nprobe
        # distances and IDs (oversampled to ensure enough candidates remain after metadata filtering)
        D, I = index.search(qvec.reshape(1, -1), K * 10, params = params)
        valid = [int(i) for i in I[0] if i != -1]
        collected_ids = valid[:K]
        scored_vectors = len(I[0])
        retries += 1
        if len(collected_ids ) >= K:
            break
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    stats = {
        "latency_ms": latency_ms,
        "scored_vectors": scored_vectors,
        "nprobe": nprobe,
        "retries": retries
    }

    return collected_ids, stats