from __future__ import annotations
from typing import Dict, Any
import numpy as np
from src.backend_interface import SearchBackend
from src.utils.timing import time_ms
from src.eval.oracle import brute_force

class ExactBackend(SearchBackend):
    name = "exact"

    def search(self, qvec: np.ndarray, filters: Dict[str, Any], K: int) -> Dict[str, Any]:
        allowed = np.arange(self.vectors.shape[0], dtype=np.int64)
        def _do():
            return brute_force(qvec, allowed, K)
        ids, latency_ms = time_ms(_do)
        return {
            "ids": ids,
            "stats": {
                "latency_ms": float(latency_ms),
                "scored_vectors": int(allowed.size),  # full scan
                "lists_probed": None,
                "nprobe": None,
                "kth_at_stop": None,
                "bound_at_stop": None,
                "notes": "oracle/ground-truth exact",
            },
        }
