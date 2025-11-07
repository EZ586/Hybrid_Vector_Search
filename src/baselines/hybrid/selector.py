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
    if not filters:
        return metadata_df["id"].to_numpy(dtype=np.int64)

    mask = pd.Series(True, index=metadata_df.index)

    geo_lat, geo_lon = None, None

    for col, ops in filters.items():
        if col in {"lat_between", "lon_between"}:
            if col == "lat_between":
                geo_lat = tuple(ops)
            else:
                geo_lon = tuple(ops)
            continue

        s = metadata_df[col]
        cur = pd.Series(True, index=s.index)

        for op, val in ops.items():
            if op == "eq":
                m = s == val
            elif op == "ne":
                m = s != val
            elif op == "ge":
                m = s >= val
            elif op == "le":
                m = s <= val
            elif op == "gt":
                m = s > val
            elif op == "lt":
                m = s < val
            elif op == "between":
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    raise ValueError(f"'between' expects [lo, hi] for '{col}'")
                lo, hi = val
                m = s.between(lo, hi, inclusive="both")
            elif op == "in":
                if not isinstance(val, (list, tuple, set)):
                    raise ValueError(f"'in' expects list/tuple/set for '{col}'")
                m = s.isin(val)
            else:
                raise ValueError(f"Unsupported operator '{op}' in filter for column '{col}'")

            # Missing values fail predicate
            m &= s.notna()
            cur &= m

        mask &= cur

    # Apply combined geo mask if both lat/lon present
    if (geo_lat is not None) or (geo_lon is not None):
        if geo_lat is None or geo_lon is None:
            raise ValueError("Geo filters require both 'lat_between' and 'lon_between'")
        if "latitude" not in metadata_df.columns or "longitude" not in metadata_df.columns:
            raise ValueError("Geo filters require 'latitude' and 'longitude' columns")

        s_lat = metadata_df["latitude"]
        s_lon = metadata_df["longitude"]

        lat_mask = s_lat.between(geo_lat[0], geo_lat[1], inclusive="both")
        lon_mask = s_lon.between(geo_lon[0], geo_lon[1], inclusive="both")
        geo_mask = lat_mask & lon_mask & s_lat.notna() & s_lon.notna()
        mask &= geo_mask

    allow_ids = metadata_df.loc[mask, "id"].to_numpy(dtype=np.int64)
    return allow_ids



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
