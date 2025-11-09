# src/dataio/validators.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Iterable, Tuple, List
import json, math
import numpy as np
import pandas as pd

# Fixed operator set per manual 
SUPPORTED_OPS = {
    "eq", "ne", "in", "between", "ge", "le", "gt", "lt", "like",
    "lat_between", "lon_between"
}

class FilterSpecError(ValueError): ...
class ValidationError(ValueError): ...

# ---------- Basic helpers ----------
def parse_filters(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Parse JSON string/object → dict; hard-error on malformed per spec."""
    if raw is None or raw == {}:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            # Malformed JSON must abort the run (hard error)
            raise ValidationError(f"Malformed filter JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValidationError("Filters must be a JSON object (dict).")
    return raw  # spec: filter object maps column -> {op: value}  :contentReference[oaicite:0]{index=0}

def validate_K(K: int, N: int) -> None:
    # Spec: K must be > 0 and ≤ N
    if not (isinstance(K, int) and 1 <= K <= N):
        raise ValidationError(f"K must be an integer in [1, {N}]")  # :contentReference[oaicite:1]{index=1}

def ensure_unit_l2(qvec: np.ndarray, *, tol: float = 1e-3) -> np.ndarray:
    """
    Validate query vector is L2-normalized within tolerance.
    Spec requires '≈ 1 within 1e-3'; do NOT auto-normalize here.  :contentReference[oaicite:2]{index=2}
    """
    v = np.asarray(qvec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValidationError("Embedding norm is zero; cannot validate")
    if not math.isclose(n, 1.0, rel_tol=tol, abs_tol=tol):
        raise ValidationError(f"Embedding L2 norm {n:.6f} not within tolerance {tol}")
    return v

# ---------- Metadata validation ----------
REQUIRED_META_DTYPES = {
    "id": "int64",
    "state": "string",
    "city": "string",
    "stars": "float32",
    "review_count": "int32",
    "RestaurantsPriceRange2": "int8",
}

GEO_COLS = ("latitude", "longitude")

def _dtype_name(series: pd.Series) -> str:
    # Normalize pandas dtype names for comparison
    dt = series.dtype
    # For pandas StringDtype return "string"
    if pd.api.types.is_string_dtype(dt):
        return "string"
    return str(dt)

def validate_metadata_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Spec checks:
      - Required columns present with exact dtypes (no coercion here)
      - id:int64, unique, contiguous 0..N-1, non-null
    Returns the same DataFrame indexed by 'id' (stable lookups).  :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    """
    # Required columns & dtypes
    for col, want in REQUIRED_META_DTYPES.items():
        if col not in metadata.columns:
            raise ValidationError(f"metadata missing required column '{col}'")
        got = _dtype_name(metadata[col])
        if got != want:
            raise ValidationError(f"metadata[{col}] dtype must be {want}, got {got}")

    # id column rules: non-null, unique, contiguous 0..N-1
    id_series = metadata["id"]
    if id_series.isna().any():
        raise ValidationError("metadata 'id' contains nulls (not allowed)")
    if not id_series.is_unique:
        raise ValidationError("metadata 'id' contains duplicates (not allowed)")
    N = len(metadata)
    # Contiguity: set(ids) == {0..N-1}
    expected = np.arange(N, dtype=np.int64)
    ids_sorted = np.sort(id_series.to_numpy(dtype=np.int64, copy=False))
    if not np.array_equal(ids_sorted, expected):
        raise ValidationError("metadata 'id' must be contiguous 0..N-1")  # :contentReference[oaicite:6]{index=6}
    
    # Required non-null columns (spec)
    for col in ("id", "state", "stars", "review_count"):
        if metadata[col].isna().any():
            raise ValidationError(f"metadata[{col}] contains nulls (not allowed)")

    return metadata.set_index("id", drop=False).sort_index()

# Backward-compat shim for old name in your codebase
validate_metadata_index = validate_metadata_table

# ---------- Filter schema validation ----------
def validate_filters_schema(metadata: pd.DataFrame,
                            filters: Optional[Mapping[str, Mapping[str, Any]]]) -> None:
    """
    Enforce:
      - Unknown fields/operators → hard error
      - Geo may appear as top-level shorthands OR as column-attached operators
      - If any geo op is used → both latitude & longitude ops must be present
    """
    if not filters:
        return

    known_cols = set(metadata.columns)
    geo_seen = {"lat_between": False, "lon_between": False}

    for field, ops in filters.items():
        # accept top-level geo shorthands
        if field in {"lat_between", "lon_between"}:
            val = ops
            if not (isinstance(val, (list, tuple)) and len(val) == 2):
                raise FilterSpecError(f"Geo range for {field} must be [lo, hi]")
            geo_seen[field] = True
            continue
        # ----------------------------------------

        if field not in known_cols:
            raise FilterSpecError(f"Unknown field in filters: '{field}'")
        if not isinstance(ops, Mapping):
            raise FilterSpecError(f"Filter for field '{field}' must be an operator→value mapping")

        for op, val in ops.items():
            if op not in SUPPORTED_OPS:
                raise FilterSpecError(f"Unknown operator for field '{field}': {op}")
            if op == "between":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise FilterSpecError(f"'between' expects [lo, hi] for '{field}'")
            if op == "in":
                if not isinstance(val, (list, tuple, set)):
                    raise FilterSpecError(f"'in' expects list/tuple/set for '{field}'")

            if op in {"lat_between", "lon_between"}:
                # Must be on the matching geo column
                if field not in GEO_COLS:
                    raise FilterSpecError(f"{op} must be applied to its matching geo column ('latitude'/'longitude')")
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise FilterSpecError(f"Geo range for {op} must be [lo, hi]")
                geo_seen[op] = True

    # If any geo op is present, require both lat & lon ops and the columns exist
    if any(geo_seen.values()):
        for c in GEO_COLS:
            if c not in known_cols:
                raise FilterSpecError("Geo filters require 'latitude' and 'longitude' columns")
        if not (geo_seen["lat_between"] and geo_seen["lon_between"]):
            raise FilterSpecError("Geo filters require both 'lat_between' and 'lon_between'")

# ---------- Casting helpers (per spec) ----------
def _cast_scalar_to_dtype(val: Any, series: pd.Series) -> Tuple[bool, Any]:
    """
    Cast filter scalar to the column's dtype.
    On cast failure → return (False, None) and the predicate will exclude all rows (spec: cast failure = excluded). :contentReference[oaicite:10]{index=10}
    """
    dt = series.dtype
    try:
        if pd.api.types.is_bool_dtype(dt):
            # Accept 0/1/true/false (case-insensitive)
            if isinstance(val, str):
                v = val.strip().lower()
                if v in {"true", "t", "1"}: return True, True
                if v in {"false", "f", "0"}: return True, False
                return False, None
            if isinstance(val, (int, np.integer)):  # 0/1
                return True, bool(int(val) != 0)
            if isinstance(val, (bool, np.bool_)):
                return True, bool(val)
            return False, None

        if pd.api.types.is_integer_dtype(dt):
            return True, int(val)
        if pd.api.types.is_float_dtype(dt):
            return True, float(val)
        if pd.api.types.is_string_dtype(dt):
            return True, str(val)
        # Fallback: leave as-is
        return True, val
    except Exception:
        return False, None

def _cast_iterable_to_dtype(vals: Iterable[Any], series: pd.Series) -> Tuple[bool, List[Any]]:
    out: List[Any] = []
    for v in vals:
        ok, vv = _cast_scalar_to_dtype(v, series)
        if not ok:
            # Spec says cast failure = excluded; best interpretation for an IN-list:
            # drop the uncastable value (so it can’t match anything).
            # If EVERYTHING fails, the mask will be all False anyway.
            continue
        out.append(vv)
    return (len(out) > 0), out

def _series_for_like(series: pd.Series) -> pd.Series:
    """
    like: case-insensitive substring on strings; arrays must be pipe-joined (per spec).  :contentReference[oaicite:11]{index=11}
    """
    s = series
    # If elements are list/tuple/sets, join with '|'
    if s.apply(lambda x: isinstance(x, (list, tuple, set))).any():
        s = s.apply(lambda x: "|".join(map(str, x)) if isinstance(x, (list, tuple, set)) else x)
    return s.astype("string")

def _like_mask(series: pd.Series, needle: str) -> pd.Series:
    s = _series_for_like(series)
    return s.str.lower().str.contains(str(needle).lower(), na=False)

# ---------- Filter evaluation ----------
def build_allowed_ids(metadata: pd.DataFrame,
                      filters: Optional[Mapping[str, Mapping[str, Any]]]) -> np.ndarray:
    """
    Vectorized AND of all predicates; missing values FAIL; ranges inclusive; geo requires both ops.
    Returns int64 ndarray of allowed ids.
    """
    if not filters:
        return metadata["id"].to_numpy(dtype=np.int64)

    # Schema checks (unknown fields/operators, geo structure)
    validate_filters_schema(metadata, filters)

    mask = pd.Series(True, index=metadata.index)

    # ollect geo ranges from both styles
    geo_lat = None
    geo_lon = None
    if "lat_between" in filters:
        geo_lat = tuple(filters["lat_between"])
    if "lon_between" in filters:
        geo_lon = tuple(filters["lon_between"])
    # -------------------------------------------

    for col, ops in filters.items():
        # Skip top-level geo keys we handled above
        if col in {"lat_between", "lon_between"}:
            continue

        s = metadata[col]
        cur = pd.Series(True, index=s.index)

        for op, val in ops.items():
            # Handle per-op type casting per spec
            if op in {"eq", "ne", "ge", "le", "gt", "lt"}:
                ok, v = _cast_scalar_to_dtype(val, s)
                if not ok:
                    m = pd.Series(False, index=s.index)
                else:
                    if op == "eq": m = (s == v)
                    elif op == "ne": m = (s != v)
                    elif op == "ge": m = (s >= v)
                    elif op == "le": m = (s <= v)
                    elif op == "gt": m = (s >  v)
                    elif op == "lt": m = (s <  v)

            elif op == "between":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise FilterSpecError(f"'between' expects [lo, hi] for '{col}'")
                ok_lo, lo = _cast_scalar_to_dtype(val[0], s)
                ok_hi, hi = _cast_scalar_to_dtype(val[1], s)
                m = s.between(lo, hi, inclusive="both") if (ok_lo and ok_hi) else pd.Series(False, index=s.index)

            elif op == "in":
                if not isinstance(val, (list, tuple, set)):
                    raise FilterSpecError(f"'in' expects list/tuple/set for '{col}'")
                ok, vals_cast = _cast_iterable_to_dtype(val, s)
                m = s.isin(vals_cast) if ok else pd.Series(False, index=s.index)

            elif op == "like":
                m = _like_mask(s, val)

            elif op in {"lat_between", "lon_between"}:
                # Column-attached geo; record ranges and defer to combined check
                if op == "lat_between": geo_lat = tuple(val)
                else:                   geo_lon = tuple(val)
                # No per-row mask yet
                continue

            else:
                raise FilterSpecError(f"Unknown operator '{op}' for '{col}'")

            # Missing values FAIL the predicate
            m = m & s.notna()
            cur &= m

        mask &= cur

    # apply geo mask if either style was used
    if (geo_lat is not None) or (geo_lon is not None):
        if geo_lat is None or geo_lon is None:
            raise FilterSpecError("Geo filters require both 'lat_between' and 'lon_between'")
        if "latitude" not in metadata.columns or "longitude" not in metadata.columns:
            raise FilterSpecError("Geo filters require 'latitude' and 'longitude' columns")

        s_lat = metadata["latitude"]
        s_lon = metadata["longitude"]

        ok_lo, lat_lo = _cast_scalar_to_dtype(geo_lat[0], s_lat)
        ok_hi, lat_hi = _cast_scalar_to_dtype(geo_lat[1], s_lat)
        ok_lo2, lon_lo = _cast_scalar_to_dtype(geo_lon[0], s_lon)
        ok_hi2, lon_hi = _cast_scalar_to_dtype(geo_lon[1], s_lon)

        if not (ok_lo and ok_hi and ok_lo2 and ok_hi2):
            geo_mask = pd.Series(False, index=metadata.index)
        else:
            lat_ok = s_lat.between(lat_lo, lat_hi, inclusive="both")
            lon_ok = s_lon.between(lon_lo, lon_hi, inclusive="both")
            geo_mask = (lat_ok & lon_ok)

        # Missing values FAIL
        geo_mask = geo_mask & s_lat.notna() & s_lon.notna()
        mask &= geo_mask
    # --------------------------------------------

    return metadata.loc[mask, "id"].to_numpy(dtype=np.int64)

# ---------- Fast membership helper for post-filter ----------
def make_allowed_membership(allowed_ids: np.ndarray) -> set[int]:
    """Build a fast membership set for candidate-id filtering in post-filter."""
    return set(map(int, allowed_ids))