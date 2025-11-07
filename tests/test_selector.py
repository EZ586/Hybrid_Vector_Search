# tests/test_selector.py
# Run: PYTHONPATH=. pytest -q tests/test_selector.py

import numpy as np
import pandas as pd

from src.baselines.hybrid.selector import make_allowlist, build_idselector
from src.eval.hybrid_validation import validate_hybrid_results
import faiss


def test_make_allowlist_eq():
    """
    Should select rows matching a simple eq filter, e.g. state == 'FL'.
    """
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "state": ["FL", "CA", "FL"],
        "stars": [4.5, 3.0, 5.0],
    })
    filters = {"state": {"eq": "FL"}}
    allow_ids = make_allowlist(df, filters)

    # Expect IDs 1 and 3 (both FL)
    assert set(allow_ids.tolist()) == {1, 3}
    assert allow_ids.dtype == np.int64


def test_make_allowlist_between():
    """
    Should select rows within a numeric range, e.g. review_count between [10, 100].
    """
    # TODO: create tiny df and assert IDs
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "review_count": [5, 20, 80, 120],
    })
    filters = {"review_count": {"between": [10, 100]}}
    allow_ids = make_allowlist(df, filters)

    # Expect IDs 2 and 3 (between 10 and 100 inclusive)
    assert set(allow_ids.tolist()) == {2, 3}


def test_build_idselector_wraps_ids():
    """
    Should return a faiss.IDSelectorBatch for the given allow_ids.
    """
    # TODO: call build_idselector and assert type
    allow_ids = np.array([10, 11, 12], dtype=np.int64)
    selector = build_idselector(allow_ids)

    assert isinstance(selector, faiss.IDSelectorBatch)
    # Check it accepts int64 arrays even if empty
    empty_selector = build_idselector(np.array([], dtype=np.int64))
    assert isinstance(empty_selector, faiss.IDSelectorBatch)


def test_validate_hybrid_results_subset_and_metrics():
    """
    Validation should detect subset violation and compute selectivity.
    """
    # TODO: construct small arrays and call validate_hybrid_results
    hybrid_ids = [1, 3]
    allow_ids = np.array([1, 2, 3, 4])
    oracle_ids = [1, 2, 3, 4, 5]
    total_N = 10
    K = 5

    result = validate_hybrid_results(hybrid_ids, allow_ids, oracle_ids, K, total_N)

    # 1) All hybrid IDs ⊆ allow_ids → True
    assert result["is_subset"] is True
    # 2) Recall@K = intersection {1,3} ∩ {1,2,3,4,5} = 2/5
    assert abs(result["recall_at_k"] - 0.4) < 1e-6
    # 3) Selectivity = 4/10 = 0.4
    assert abs(result["filter_selectivity"] - 0.4) < 1e-6
    # 4) Counts
    assert result["num_hybrid"] == 2
    assert result["num_allow"] == 4

    # Now violate subset (include ID not in allow_ids)
    hybrid_bad = [1, 99]
    result_bad = validate_hybrid_results(hybrid_bad, allow_ids, oracle_ids, K, total_N)
    assert result_bad["is_subset"] is False
