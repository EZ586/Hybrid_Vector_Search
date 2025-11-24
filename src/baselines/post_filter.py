
# src/backends/post_filter.py
from __future__ import annotations
from typing import Iterable, Tuple, Dict, List, Any, Mapping, Optional
import numpy as np
import pandas as pd
import time

# Single source of truth for all validation & filter semantics
from src.dataio.validators import (
    ensure_unit_l2,
    validate_K,
    validate_filters_schema,
    build_allowed_ids,
    make_allowed_membership,
)

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
    Post-filter baseline:
      - ANN across full vector set with a candidate K′ ladder
      - Apply manual-compliant filters to retain candidates
      - Return the top-K kept ids by similarity score

    Stats contract (returned in `stats`):
      - latency_ms (float)
      - scored_vectors (int): sum of candidates surfaced across ladder steps
      - retries (int): number of ladder bumps executed (excluding the first)
      - notes (str)
      - lists_probed (None), nprobe (None)
      - kth_at_stop (Optional[float])
      - bound_at_stop (None)
    """
    t0 = time.time()
    # Basic safety/shape constraints
    if not isinstance(metadata_df.index, pd.Index) or metadata_df.index.name != "id":
        if "id" not in metadata_df.columns:
            raise ValueError("metadata_df must have an 'id' column or be indexed by 'id'.")
        metadata_df = metadata_df.set_index("id", drop=False).sort_index()

    validate_K(K, len(metadata_df))
    ensure_unit_l2(qvec)  # assert L2≈1 within tolerance; do not normalize here

    # Schema check & precompute allowed universe (vectorized, NaN=fail, geo, casting, etc.)
    validate_filters_schema(metadata_df, filters)
    allowed_ids = build_allowed_ids(metadata_df, filters)
    allowed = make_allowed_membership(allowed_ids)

    # Derive filter selectivity to adapt the candidate ladder:
    #   selectivity = |allowed_ids| / |all_ids|
    total_ids = len(metadata_df)
    if total_ids > 0:
        selectivity = float(allowed_ids.size) / float(total_ids)
    else:
        # degenerate case; treat as unfiltered
        selectivity = 1.0

    # Start from the caller-provided ladder
    base_ladder = tuple(int(k) for k in k_ladder)

    # Scale candidate budget more aggressively for loose filters:
    #   - tight / moderately selective (< 25%): keep base ladder
    #   - loose (25%–50%): 2x ladder
    #   - very loose / unfiltered (>= 50%): 3x ladder
    if selectivity >= 0.5:
        scale = 3.0
    elif selectivity >= 0.25:
        scale = 2.0
    else:
        scale = 1.0

    if scale != 1.0:
        effective_ladder = tuple(
            int(min(max(int(k * scale), K), total_ids)) for k in base_ladder
        )
    else:
        effective_ladder = base_ladder

    # For loose filters, require more ladder steps before we allow early stop
    # (len(kept) >= K). This protects recall when allowed_ids is huge.
    min_ladder_steps = 1
    if selectivity >= 0.25:
        min_ladder_steps = 2
    if selectivity >= 0.5:
        min_ladder_steps = 3

    if max_ladder_steps is None:
        max_ladder_steps = len(effective_ladder)
    max_ladder_steps = min(max_ladder_steps, len(effective_ladder))

    # Track total scoring work across ladder steps
    scored_total = 0
    kept: Dict[int, float] = {}
    retries = 0
    stop_due_to_enough = False
    kth_at_stop: Optional[float] = None

    # Ladder: progressively widen candidate set until we have ≥K valid
    for step_idx, kprime in enumerate(effective_ladder):
        if retries >= max_ladder_steps:
            break

        ids, scores = ann_index.search(np.asarray(qvec, dtype=np.float32), int(kprime))
        ids = np.asarray(ids, dtype=int)
        scores = np.asarray(scores, dtype=float)
        scored_total += int(kprime)

        # Keep best score per id, but only if id is allowed
        for _id, s in zip(ids, scores):
            i = int(_id)
            if i in allowed:
                ps = kept.get(i)
                if ps is None or s > ps:
                    kept[i] = float(s)

        # Allow early stop only after we've taken enough ladder steps
        # (min_ladder_steps depends on selectivity).
        if len(kept) >= K and (step_idx + 1) >= min_ladder_steps:
            stop_due_to_enough = True
            break

        retries += 1

    # Sort kept by score desc and slice top-K
    if kept:
        sorted_pairs = sorted(kept.items(), key=lambda t: t[1], reverse=True)
        top_ids = [i for i, _ in sorted_pairs[:K]]
        if stop_due_to_enough and len(sorted_pairs) >= K:
            kth_at_stop = float(sorted_pairs[K - 1][1])
    else:
        top_ids = []

    latency_ms = (time.time() - t0) * 1000.0
    stats: Dict[str, Any] = {
        "latency_ms": float(latency_ms),
        # Total candidate vectors requested across ladder steps
        "scored_vectors": int(scored_total),
        "retries": int(retries),
        "notes": f"k_ladder={list(effective_ladder)}; kept={len(kept)}; need={K}",
        "lists_probed": None,
        "nprobe": None,
        "kth_at_stop": kth_at_stop,
        "bound_at_stop": None,
    }
    return top_ids, stats