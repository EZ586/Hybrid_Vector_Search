"""
Post-filter baseline:
  • Run ANN over the full vector set with a K′ ladder (e.g., 200→500→1000).
  • Apply metadata filters to candidates.
  • Return the top-K valid ids by similarity score.

Stats contract (returned in `stats`):
  - latency_ms (float): end-to-end wall time in milliseconds
  - scored_vectors (int): total K′ issued/summed across ladder steps
  - retries (int): number of ladder bumps actually executed
  - notes (str): human-readable details (ladder, kept, need, etc.)
  - lists_probed (optional): None for post-filter baseline
  - nprobe (optional): None for post-filter baseline
  - kth_at_stop (Optional[float]): score of the kth kept item when stopping; None if <K kept
  - bound_at_stop (optional): None unless the ANN surface provides a bound

Filter semantics (hard-error on unknowns):
  Supported operators per field spec: eq, ne, in, between, ge, le, gt, lt, like,
  lat_between, lon_between. Missing/NaN values FAIL the predicate. Unknown
  fields/operators cause a ValueError.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Dict, List, Any, Mapping, Optional
import numpy as np
import pandas as pd
import time
import math

_ALLOWED_OPS = {
    "eq", "ne", "in", "between", "ge", "le", "gt", "lt", "like",
    "lat_between", "lon_between", "min", "max"  # aliases for ge/le support
}

class FilterSpecError(ValueError):
    pass


def post_filter_search(
    qvec: np.ndarray,
    ann_index: Any,
    metadata_df: pd.DataFrame,
    filters: Optional[Mapping[str, Mapping[str, Any]]],
    K: int,
    k_ladder: Iterable[int] = (200, 500, 1000),
    max_ladder_steps: Optional[int] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Execute the post-filter baseline and return (ids, stats).

    Parameters
    ----------
    qvec : np.ndarray
        Query vector (will be cast to float32 and L2-normalized if not ~1).
    ann_index : Any
        Object exposing `search(qvec: np.ndarray, k: int) -> (ids, scores)`.
    metadata_df : pd.DataFrame
        Business metadata. Must contain an integer id 0..N-1 column or be indexed by id.
    filters : Mapping[str, Mapping[str, Any]] | None
        Filter dictionary matching the manual's JSON schema.
    K : int
        Number of final ids to return (<=K returned if not enough pass filters).
    k_ladder : Iterable[int]
        Candidate-size ladder for oversampling.
    max_ladder_steps : Optional[int]
        Hard cap on ladder steps. Defaults to len(k_ladder).
    """
    if K <= 0:
        raise ValueError("K must be positive")

    if max_ladder_steps is None:
        max_ladder_steps = len(tuple(k_ladder))

    # Ensure metadata is indexed by id for stable lookups
    if not isinstance(metadata_df.index, pd.Index) or metadata_df.index.name != "id":
        if "id" not in metadata_df.columns:
            raise ValueError("metadata_df must have an 'id' column or be indexed by 'id'.")
        metadata_df = metadata_df.set_index("id", drop=False).sort_index()

    _validate_filters_schema(metadata_df, filters)

    # Normalize query vector
    qvec = np.asarray(qvec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(qvec)
    if norm > 0 and not math.isclose(float(norm), 1.0, rel_tol=1e-3, abs_tol=1e-3):
        qvec = qvec / norm

    t0 = time.time()
    scored_vectors = 0
    kept: Dict[int, float] = {}
    retries = 0

    # Ladder search
    stop_due_to_enough = False
    for step_idx, kprime in enumerate(k_ladder):
        if retries >= max_ladder_steps:
            break

        ids, scores = ann_index.search(qvec, int(kprime))
        ids = np.asarray(list(ids)).astype(int)
        scores = np.asarray(list(scores)).astype(float)
        scored_vectors += int(len(ids))

        # Filter and keep best score per id
        for _id, s in zip(ids, scores):
            if _passes_filters_row(metadata_df, int(_id), filters):
                prev = kept.get(int(_id))
                if prev is None or s > prev:
                    kept[int(_id)] = float(s)

        if len(kept) >= K:
            stop_due_to_enough = True
            break

        retries += 1  

    # Final sort and top-K slice
    if kept:
        sorted_pairs = sorted(kept.items(), key=lambda t: t[1], reverse=True)
        top_ids = [i for i, _ in sorted_pairs[:K]]
        kth_at_stop = float(sorted_pairs[K-1][1]) if stop_due_to_enough and len(sorted_pairs) >= K else None
    else:
        top_ids = []
        kth_at_stop = None

    latency_ms = (time.time() - t0) * 1000.0
    stats: Dict[str, Any] = {
        "latency_ms": float(latency_ms),
        "scored_vectors": int(scored_vectors),
        "retries": int(retries),
        "notes": f"k_ladder={list(k_ladder)}; kept={len(kept)}; need={K}",
        "lists_probed": None,
        "nprobe": None,
        "kth_at_stop": kth_at_stop,
        "bound_at_stop": None,
    }
    return top_ids, stats


def _validate_filters_schema(metadata_df: pd.DataFrame, filters: Optional[Mapping[str, Mapping[str, Any]]]) -> None:
    """Hard-error on unknown fields/operators; allow None/empty as 'no filters'."""
    if not filters:
        return
    # Known fields are any columns in metadata_df
    known_fields = set(metadata_df.columns)
    for field, spec in filters.items():
        if field not in known_fields and field not in {"lat_between", "lon_between"}:  
            raise FilterSpecError(f"Unknown field in filters: {field}")
        if not isinstance(spec, Mapping):
            raise FilterSpecError(f"Filter for field {field} must be a mapping of operators → values")
        for op in spec.keys():
            if op not in _ALLOWED_OPS:
                raise FilterSpecError(f"Unknown operator for field {field}: {op}")


def _is_missing(val: Any) -> bool:
    try:
        import pandas as pd
        return pd.isna(val)
    except Exception:
        return val is None or (isinstance(val, float) and math.isnan(val))


def _passes_filters_row(metadata_df: pd.DataFrame, _id: int, filters: Optional[Mapping[str, Mapping[str, Any]]]) -> bool:
    if not filters:
        return True
    try:
        row = metadata_df.loc[_id]
    except KeyError:
        return False

    for field, spec in filters.items():
        # Geo shorthand: lat_between / lon_between can target latitude/longitude columns
        if field == "lat_between":
            lo, hi = _pair_or_none(spec.get("between"))
            if lo is None or hi is None:
                raise FilterSpecError("lat_between requires a 'between': [lo, hi]")
            val = row.get("latitude")
            if _is_missing(val) or not (lo <= float(val) <= hi):
                return False
            continue
        if field == "lon_between":
            lo, hi = _pair_or_none(spec.get("between"))
            if lo is None or hi is None:
                raise FilterSpecError("lon_between requires a 'between': [lo, hi]")
            val = row.get("longitude")
            if _is_missing(val) or not (lo <= float(val) <= hi):
                return False
            continue

        if field not in row:
            # Unknown or missing column: fail predicate
            return False
        val = row[field]

        # Treat missing as fail
        if _is_missing(val):
            return False

        # Numeric coercion helper
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        # between
        if "between" in spec and spec["between"] is not None:
            lo, hi = _pair_or_none(spec["between"])
            if lo is None or hi is None:
                raise FilterSpecError(f"Field {field}: 'between' must be a pair [lo, hi]")
            v = _num(val)
            if v is None or not (lo <= v <= hi):
                return False

        # ge / le with min / max aliases
        lo = spec.get("ge", spec.get("min"))
        hi = spec.get("le", spec.get("max"))
        if lo is not None:
            v = _num(val)
            if v is None or v < float(lo):
                return False
        if hi is not None:
            v = _num(val)
            if v is None or v > float(hi):
                return False

        # gt / lt
        if "gt" in spec:
            v = _num(val)
            if v is None or v <= float(spec["gt"]):
                return False
        if "lt" in spec:
            v = _num(val)
            if v is None or v >= float(spec["lt"]):
                return False

        # eq / ne
        if "eq" in spec and val != spec["eq"]:
            return False
        if "ne" in spec and val == spec["ne"]:
            return False

        # in
        if "in" in spec and spec["in"] is not None:
            s = set(spec["in"])
            if val not in s:
                return False

        # like (case-insensitive substring). If the field is list-like, join with pipe.
        if "like" in spec and spec["like"] is not None:
            needle = str(spec["like"]).lower()
            if isinstance(val, (list, tuple, set)):
                hay = "|".join(map(str, val)).lower()
            else:
                hay = str(val).lower()
            if needle not in hay:
                return False

    return True


def _pair_or_none(obj: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(obj, (list, tuple)) or len(obj) != 2:
        return (None, None)
    try:
        return (float(obj[0]), float(obj[1]))
    except Exception:
        return (None, None)