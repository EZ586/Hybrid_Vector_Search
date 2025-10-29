# src/backends/random.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from src.backend_interface import SearchBackend
from src.utils.timing import time_ms


class RandomBackend(SearchBackend):
    name = "random"

    def search(
        self, qvec: np.ndarray, filters: Dict[str, Any], K: int
    ) -> Dict[str, Any]:
        def _do():
            N = self.vectors.shape[0]
            K_eff = min(K, N)
            rng = np.random.default_rng(0)
            return rng.choice(N, size=K_eff, replace=False).tolist()

        ids, latency_ms = time_ms(_do)
        stats = {
            "latency_ms": float(latency_ms),
            "scored_vectors": 0,
            "lists_probed": None,
            "nprobe": None,
            "kth_at_stop": None,
            "bound_at_stop": None,
            "notes": "random baseline",
        }
        return ids, stats
