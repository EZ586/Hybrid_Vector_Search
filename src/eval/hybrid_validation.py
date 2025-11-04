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
    raise NotImplementedError
