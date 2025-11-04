# tests/test_selector.py

import numpy as np
import pandas as pd

from baselines.hybrid.selector import make_allowlist, build_idselector
from eval.hybrid_validation import validate_hybrid_results


def test_make_allowlist_eq():
    """
    Should select rows matching a simple eq filter, e.g. state == 'FL'.
    """
    # TODO: create tiny df and assert IDs
    raise NotImplementedError


def test_make_allowlist_between():
    """
    Should select rows within a numeric range, e.g. review_count between [10, 100].
    """
    # TODO: create tiny df and assert IDs
    raise NotImplementedError


def test_build_idselector_wraps_ids():
    """
    Should return a faiss.IDSelectorBatch for the given allow_ids.
    """
    # TODO: call build_idselector and assert type
    raise NotImplementedError


def test_validate_hybrid_results_subset_and_metrics():
    """
    Validation should detect subset violation and compute selectivity.
    """
    # TODO: construct small arrays and call validate_hybrid_results
    raise NotImplementedError
