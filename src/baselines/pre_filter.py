import time
import numpy as np
import pandas as pd
from src.selectivity import mask


def pre_filter_search(
    qvec: np.ndarray,
    filters: dict,
    df_metadata: pd.DataFrame,
    vectors: np.ndarray,
    K: int = 10,
):
    """
    Execute the pre-filter baseline and return (ids, stats).

    Parameters
    ----------
    qvec : np.ndarray
        Query vector (will be cast to float32 and L2-normalized if not ~1).
    filters : Mapping[str, Mapping[str, Any]] | None
        Filter dictionary matching the manual's JSON schema.
    df_metadata : pd.DataFrame
        Business metadata. Must contain an integer id 0..N-1 column or be indexed by id.
    K : int
        Number of final ids to return (<=K returned if not enough pass filters).
    """
    start = time.time()

    # Get allowed IDs
    allowed_mask = mask(filters, df_metadata)
    # get indices where value is true
    allowed_ids = np.where(allowed_mask)[0]

    if len(allowed_ids) == 0:
        return {
            "ids": [],
            "stats": {
                "latency_ms": 0.0,
                "scored_vectors": 0,
                "lists_probed": None,
                "nprobe": None,
                "kth_at_stop": None,
                "bound_at_stop": None,
                "notes": "no candidates after filter",
            },
        }

    sub_vectors = vectors[allowed_ids]

    # Conduct Top-K inner product search
    similarity = np.dot(sub_vectors, qvec)
    top_idx = np.argpartition(-similarity, K)[:K]
    sorted_idx = top_idx[np.argsort(-similarity[top_idx])]
    results = allowed_ids[sorted_idx]

    latency_ms = (time.time() - start) * 1000
    stats = {
        "latency_ms": float(latency_ms),
        "scored_vectors": int(len(allowed_ids)),  # number of candidates examined
        "lists_probed": None,
        "nprobe": None,
        "kth_at_stop": None,
        "bound_at_stop": None,
        "notes": "pre-filter top-K inner product",
    }
    results = results.tolist()
    return results, stats
