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
    mask = np.ones(len(metadata_df), dtype=bool)
    for key, condition in filters.items():
        col = metadata_df[key]

        for op, val in condition.items():
            if op == "eq":
                mask &= col == val
            elif op == "ge":
                mask &= col >= val
            elif op == "le":
                mask &= col <= val
            elif op == "between":
                low, high = val
                mask &= col.between(low, high, inclusive="both")
            elif op == "in":
                mask &= col.isin(val)
            elif op == "like":
                mask &= col.astype(str).str.contains(val, case=False, na=False)
    return metadata_df.loc[mask, "id"].astype(np.int64).to_numpy()
        


def build_idselector(allow_ids: np.ndarray) -> faiss.IDSelectorBatch:
    """
    Wrap the allow-list into a FAISS IDSelectorBatch.

    Args:
        allow_ids: 1D array of allowed IDs.

    Returns:
        faiss.IDSelectorBatch object ready to be passed to SearchParametersIVF.
    """
    # TODO: construct and return faiss.IDSelectorBatch
    return faiss.IDSelectorBatch(np.array(allow_ids, dtype=np.int64))
