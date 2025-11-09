# src/baselines/hybrid/selector.py
"""
Hybrid selector helpers.

- make_allowlist(metadata_df, filters) -> np.ndarray[int64]
- build_idselector(allow_ids) -> faiss.IDSelectorBatch
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import faiss

from src.dataio.validators import (
    parse_filters,
    validate_filters_schema,
    build_allowed_ids,
)


def make_allowlist(metadata_df, filters: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Produce an allow-list of IDs from metadata + JSON-style filters using the
    canonical validator path.

    Args:
        metadata_df: pandas DataFrame already loaded/validated from artifacts
                     (typically via dataio.loaders.load_metadata(...)), must
                     contain 'id' and be indexed/sorted 0..N-1.
        filters: dict or JSON string describing predicates; may be {} or None.

    Returns:
        1D np.ndarray[int64] of allowed ids (sorted).
    """
    # No filters → all rows allowed
    if not filters:
        return metadata_df["id"].to_numpy(dtype=np.int64)

    # Normalize/filter JSON first (handles string vs dict)
    parsed = parse_filters(filters)

    # Check against actual metadata columns and supported ops
    validate_filters_schema(metadata_df, parsed)

    # Canonical vectorized evaluation → ids:int64
    allow_ids = build_allowed_ids(metadata_df, parsed)

    # Ensure sorted, int64
    allow_ids = np.asarray(allow_ids, dtype=np.int64)
    if allow_ids.size > 1 and not np.all(allow_ids[:-1] <= allow_ids[1:]):
        allow_ids = np.sort(allow_ids)

    return allow_ids


def build_idselector(allow_ids: np.ndarray) -> faiss.IDSelectorBatch:
    """
    Wrap allowed ids into a FAISS IDSelectorBatch for IVF searches.

    Args:
        allow_ids: 1D array-like of ids.

    Returns:
        faiss.IDSelectorBatch
    """
    allow_ids = np.asarray(allow_ids, dtype=np.int64)
    return faiss.IDSelectorBatch(allow_ids)