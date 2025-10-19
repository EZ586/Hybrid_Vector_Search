from __future__ import annotations
import time
from typing import Dict, Any
import numpy as np

from src.backend_interface import SearchBackend


class RandomBackend(SearchBackend):
    name = "random"

    def search(self, qvec: np.ndarray, filters: Dict[str, Any], K: int) -> Dict[str, Any]:
        t0 = time.perf_counter()

        N = self.vectors.shape[0]
        K_eff = min(K, N)

        # No filtering; sample without replacement
        ids = np.random.default_rng(0).choice(N, size=K_eff, replace=False).tolist()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        out = {
            "ids": ids,
            "stats": {
                "latency_ms": float(latency_ms),
                "scored_vectors": int(0),   # random baseline doesn't score; keep 0
            },
        }
        return out
