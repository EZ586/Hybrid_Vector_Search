# artifacts/make_queries.py
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

# Use your exact-on-table selectivity
from src.eval.selectivity import compute_selectivity


# ----------------------------
# Labeling (your thresholds)
# ----------------------------
def _label(sel: float) -> str:
    return "strict" if sel < 0.30 else ("medium" if sel < 0.70 else "broad")


# ----------------------------
# Simple samplers (one op per column)
# ----------------------------
def _cat_value(df: pd.DataFrame, col: str, rng: random.Random) -> Optional[str]:
    ser = df[col].astype("string").fillna("__NA__")
    if ser.empty:
        return None
    vc = ser.value_counts(normalize=True)
    keys, probs = vc.index.tolist(), vc.values.tolist()
    v = rng.choices(keys, weights=probs, k=1)[0]
    return None if v == "__NA__" else str(v)

def _cat_values(df: pd.DataFrame, col: str, rng: random.Random, k_choices=(2,3,4)) -> Optional[List[str]]:
    ser = df[col].astype("string").fillna("__NA__")
    if ser.empty:
        return None
    vc = ser.value_counts(normalize=True)
    keys, probs = vc.index.tolist(), vc.values.tolist()
    kk = rng.choice(k_choices)
    vals = sorted(set(x for x in rng.choices(keys, weights=probs, k=kk) if x != "__NA__"))
    return [str(v) for v in vals] if vals else None

def _num_sample(df: pd.DataFrame, col: str, rng: random.Random) -> Optional[float]:
    x = df[col].dropna().to_numpy()
    if x.size == 0:
        return None
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, rng.random()))

def _num_between(df: pd.DataFrame, col: str, rng: random.Random) -> Optional[List[float]]:
    x = df[col].dropna().to_numpy()
    if x.size == 0:
        return None
    x = np.asarray(x, dtype=float)
    q1, q2 = sorted([rng.random(), rng.random()])
    lo, hi = float(np.quantile(x, q1)), float(np.quantile(x, q2))
    if lo >= hi:
        return None
    return [lo, hi]

def _sample_filters(meta: pd.DataFrame, rng: random.Random, spec: Dict[str, List[str]]) -> Dict[str, Any]:
    """Pick 1–3 columns; for each, choose an op and sample a concrete value."""
    cols = [c for c in spec.keys() if c in meta.columns]
    rng.shuffle(cols)
    take = rng.choice([1, 2, 3])
    filters: Dict[str, Any] = {}

    for col in cols[:take]:
        op = rng.choice(spec[col])
        if op == "eq":
            v = _cat_value(meta, col, rng)
            if v is not None:
                filters[col] = {"eq": v}
        elif op == "in":
            vals = _cat_values(meta, col, rng)
            if vals:
                filters[col] = {"in": vals}
        elif op == "like":
            v = _cat_value(meta, col, rng)
            if v:
                token = str(v).replace("|", " ").split(",")[0].split(" ")[0][:6]
                if token:
                    filters[col] = {"like": token}
        elif op == "ge":
            v = _num_sample(meta, col, rng)
            if v is not None:
                filters[col] = {"ge": v}
        elif op == "between":
            pair = _num_between(meta, col, rng)
            if pair:
                filters[col] = {"between": pair}
    return filters


# ----------------------------
# Final sanitizer: remove None, normalize shapes, one-op-per-col
# ----------------------------
def _prune_nones(filters: dict) -> dict:
    """Remove ops with None values and drop columns that end up empty.
       Ensure 'between' is a 2-item list of floats; keep only one op per column."""
    cleaned: Dict[str, Dict[str, Any]] = {}
    for col, cond in filters.items():
        if not isinstance(cond, dict):
            continue
        new_cond: Dict[str, Any] = {}
        for op, val in cond.items():
            if val is None:
                continue
            if op == "between":
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 2:
                    lo, hi = val
                    if lo is not None and hi is not None:
                        lo, hi = float(lo), float(hi)
                        if lo < hi:
                            new_cond["between"] = [lo, hi]
            elif op == "in":
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    keep = [str(v) for v in val if v is not None and str(v).strip()]
                    if keep:
                        new_cond["in"] = keep
            elif op in ("eq", "like"):
                s = str(val).strip()
                if s:
                    new_cond[op] = s
            elif op in ("ge", "le", "gt", "lt"):
                try:
                    new_cond[op] = float(val)
                except Exception:
                    pass
        if new_cond:
            for p in ["between", "eq", "in", "like", "ge", "le", "gt", "lt"]:
                if p in new_cond:
                    cleaned[col] = {p: new_cond[p]}
                    break
    return cleaned


# ----------------------------
# Sanity helpers (optional)
# ----------------------------
def _has_none_or_multiops(d: Any) -> bool:
    if not isinstance(d, dict):
        return True
    for _, cond in d.items():
        if cond is None or not isinstance(cond, dict) or len(cond) != 1:
            return True
        for _, val in cond.items():
            if val is None:
                return True
    return False


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket_dir", default="artifacts/v3",
                    help="Directory containing metadata.parquet; queries.parquet will be written here.")
    ap.add_argument("--out", default=None, help="Override output path for queries.parquet")
    ap.add_argument("--per_bucket", type=int, default=12,
                    help="# queries to keep per selectivity bucket")
    ap.add_argument("--buckets", default="0.70,0.75,0.80,0.85,0.90,0.95,1.00",
                    help="Comma-separated selectivity targets in (0,1]")
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="Relative tolerance around each bucket target (e.g., 0.15 = ±15%)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--spec", default=json.dumps({
        "state": ["eq"],
        "city": ["eq"],
        "categories": ["like","in"],
        "stars": ["ge","between"],
        "review_count": ["ge","between"],
        "RestaurantsPriceRange2": ["in","eq"],
        "is_open": ["eq"]
    }))
    ap.add_argument("--debug_mutations", type=int, default=0,
                    help="If 1, detect and raise if compute_selectivity mutates filters.")
    ap.add_argument("--sanity", type=int, default=0,
                    help="If 1, run minimal in-memory and post-save checks.")
    ap.add_argument("--debug_examples", type=int, default=0,
                    help="(No-op) kept for CLI compatibility.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bucket = Path(args.bucket_dir)
    meta_path = bucket / "metadata.parquet"
    out_path = Path(args.out) if args.out else (bucket / "queries.parquet")

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found at {meta_path}")

    meta = pd.read_parquet(meta_path)
    targets = [float(x) for x in args.buckets.split(",") if x.strip()]
    spec = json.loads(args.spec)

    rows: List[Dict[str, Any]] = []

    for target in tqdm(targets, desc="Buckets", leave=True):
        kept = 0
        attempts = 0
        max_attempts = max(1000, args.per_bucket * 300)

        for _ in range(max_attempts):
            attempts += 1

            # 1) sample & prune
            filters = _sample_filters(meta, rng, spec)
            filters = _prune_nones(filters)
            if not filters:
                continue

            if args.sanity and _has_none_or_multiops(filters):
                continue

            # 2) compute selectivity on a deep copy
            before = deepcopy(filters) if args.debug_mutations else None
            true_sel = compute_selectivity(deepcopy(filters), meta)
            if args.debug_mutations and filters != before:
                raise RuntimeError("compute_selectivity mutated filters in-place")

            # 3) final authoritative prune and guard
            filters = _prune_nones(filters)
            if not filters or _has_none_or_multiops(filters):
                continue

            # 4) accept if within tolerance band
            ok_small = (target < 1e-3 and true_sel <= target * (1 + args.tolerance))
            ok_band  = (abs(true_sel - target) <= args.tolerance * target)

            if true_sel > 0 and (ok_small or ok_band):
                lab = _label(true_sel)
                rows.append({
                    "qtext": f"auto-{lab}",
                    "filters": deepcopy(filters),                        # dict (for in-memory sanity only)
                    "filters_json": json.dumps(filters, sort_keys=True), # -> the only thing we persist
                    "K": 10,
                    "label": lab
                })
                kept += 1
                if kept >= args.per_bucket:
                    break

        # quiet mode: rely on tqdm; no per-bucket print

    if not rows:
        raise RuntimeError("No queries were accepted. Loosen tolerance or adjust spec/buckets.")

    # Assign contiguous qid: 0..Q-1
    for i, r in enumerate(rows):
        r["qid"] = i

    # ----------------------------
    # Build DF, sanity, and SAVE (JSON-only)
    # ----------------------------
    qdf = pd.DataFrame(rows, columns=["qid", "qtext", "filters", "filters_json", "K", "label"])

    if args.sanity:
        bad_in_mem_idx = [i for i, f in enumerate(qdf["filters"]) if _has_none_or_multiops(f)]
        print(f"[sanity] in-memory invalid rows: {len(bad_in_mem_idx)}")

    # 🔒 Persist only JSON column to Parquet (drop dicts to avoid schema expansion/None injection)
    qdf_to_save = qdf.drop(columns=["filters"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qdf_to_save.to_parquet(out_path, index=False)

    # Reload sanity — validate by parsing JSON back to dicts
    if args.sanity:
        q2 = pd.read_parquet(out_path)
        bad_after_idx = [i for i, s in enumerate(q2["filters_json"])
                         if _has_none_or_multiops(json.loads(s))]
        print(f"[sanity] post-save invalid rows: {len(bad_after_idx)}")

    print(f"[OK] wrote {len(qdf)} queries across {len(targets)} buckets → {out_path}")


if __name__ == "__main__":
    main()