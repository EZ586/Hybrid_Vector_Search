# src/backends/exact_backend.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np

from src.backend_interface import SearchBackend
from src.utils.timing import time_ms
from src.eval import oracle as ORACLE

# ✅ single source of truth for validation & filters
from src.dataio.validators import (
    validate_K,
    ensure_unit_l2,
    build_allowed_ids,
)

class ExactBackend(SearchBackend):
    """
    Ground-truth exact search backend.

    search(qvec, filters, K) -> (ids: list[int], stats: dict)
      - ids: exact Top-K within the allowed subset (descending inner-product)
      - stats keys: latency_ms, scored_vectors, lists_probed, nprobe,
                    kth_at_stop, bound_at_stop, notes (null if N/A)
    """

    name = "exact"

    def search(self, qvec: np.ndarray, filters: Dict[str, Any], K: int) -> Tuple[List[int], Dict[str, Any]]:
        # Basic contract checks (cheap and robust even if caller already checked)
        N = self.vectors.shape[0]
        validate_K(int(K), N)
        qvec = ensure_unit_l2(qvec)  # assert L2≈1 within tolerance

        # Allowed universe per manual semantics (NaN=fail, geo both-ops, casting, etc.)
        if filters:
            allowed = build_allowed_ids(self.metadata, filters)
        else:
            allowed = np.arange(N, dtype=np.int64)

        # Run oracle brute-force within allowed IDs
        def _do():
            return ORACLE.brute_force(qvec, allowed, int(K))
        ids, latency_ms = time_ms(_do)

        # Ensure types match the interface and fill stats per contract
        ids = list(map(int, ids))
        stats = {
            "latency_ms": float(latency_ms),
            "scored_vectors": int(allowed.size),
            "lists_probed": None,
            "nprobe": None,
            "kth_at_stop": None,
            "bound_at_stop": None,
            "notes": "oracle/ground-truth exact",
        }
        return ids, stats