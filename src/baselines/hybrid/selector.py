# baselines/hybrid/selector.py

from typing import Dict, Any
import numpy as np
import pandas as pd
import faiss


def make_allowlist(metadata_df: pd.DataFrame, filters: Dict[str, Any]) -> np.ndarray:
    """
    Vectorize the JSON-style filter schema over metadata to produce
    an array of allowed row IDs (int32/int64), aligned with Week-1 schema.

    Args:
        metadata_df: dataframe containing at least the standardized columns
                     (id, state, city, stars, review_count, etc.).
        filters: JSON-like dict using eq / between / in.

    Returns:
        allow_ids: 1D numpy array of integer IDs (e.g. int64) allowed by filters.
    """
    # TODO: apply filters over metadata_df
    raise NotImplementedError


def build_idselector(allow_ids: np.ndarray) -> faiss.IDSelectorBatch:
    """
    Wrap the allow-list into a FAISS IDSelectorBatch.

    Args:
        allow_ids: 1D array of allowed IDs.

    Returns:
        faiss.IDSelectorBatch object ready to be passed to SearchParametersIVF.
    """
    # TODO: construct and return faiss.IDSelectorBatch
    raise NotImplementedError
