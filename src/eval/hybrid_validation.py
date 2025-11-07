# eval/hybrid_validation.py

from typing import List, Dict, Any, Tuple
import numpy as np


def validate_hybrid_results(
    hybrid_ids: List[int],
    allow_ids: np.ndarray,
    oracle_ids: List[int],
    K: int,
    total_N: int,
) -> Dict[str, Any]:
    """
    Check that:
      1) all hybrid_ids are subset of allow_ids
      2) compute recall@K vs oracle
      3) compute filter_selectivity = len(allow_ids) / total_N

    Args:
        hybrid_ids: IDs returned by the hybrid backend.
        allow_ids: numpy array of allowed IDs from selector.
        oracle_ids: IDs from brute-force oracle on the same query.
        K: target top-K.
        total_N: total number of rows in the dataset (for selectivity).

    Returns:
        dict with:
          - is_subset: bool
          - recall_at_k: float
          - filter_selectivity: float
          - num_hybrid: int
          - num_allow: int
    """
    # TODO: implement subset check and metrics
    # Defensive conversion to NumPy for fast set operations
    hybrid_ids = np.asarray(hybrid_ids, dtype=np.int64)
    oracle_ids = np.asarray(oracle_ids, dtype=np.int64)
    allow_ids = np.asarray(allow_ids, dtype=np.int64)

    # check all hybrid_ids are subset of allow_ids
    allow_set = set(allow_ids.tolist())
    is_subset = all(hid in allow_set for hid in hybrid_ids)

    # compute recall@K vs oracle
    topk_oracle = oracle_ids[:K] if len(oracle_ids) >= K else oracle_ids
    hits = len(set(hybrid_ids) & set(topk_oracle))
    recall_at_k = hits / len(topk_oracle) if len(topk_oracle) > 0 else 0.0

    # compute filter_selectivity = len(allow_ids) / total_N
    filter_selectivity = (
        len(allow_ids) / total_N if total_N > 0 else float("nan")
    )

    return {
        "is_subset": is_subset,
        "recall_at_k": recall_at_k,
        "filter_selectivity": filter_selectivity,
        "num_hybrid": len(hybrid_ids),
        "num_allow": len(allow_ids),
    }
