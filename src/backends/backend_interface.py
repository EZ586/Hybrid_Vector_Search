# src/backends/backend_interface.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


class SearchBackend(ABC):
    """
    Unified backend interface.

    - Backends MUST implement `search(qvec, filters, K)` and return
      `(ids: List[int], stats: Dict[str, Any])`.
    - `ids` MUST be length ≤ K; callers (harness) may pad to K.
    - `stats` SHOULD contain the fields the manual/logging expect:
        {
          "latency_ms": float,
          "scored_vectors": int,
          "lists_probed": int | None,
          "nprobe": int | None,
          "kth_at_stop": float | None,
          "bound_at_stop": float | None,
          "notes": str | None,
          "backend": str,           # optional but helpful
        }
    """

    def __init__(
        self,
        vectors: Optional[np.ndarray] = None,
        metadata: Optional[pd.DataFrame] = None,
        name: str = "backend",
    ) -> None:
        # subclasses may ignore vectors/metadata (e.g. hybrid)
        self.vectors = vectors
        self.metadata = metadata
        self.name = name

    @abstractmethod
    def search(
        self,
        qvec: np.ndarray,
        filters: Dict[str, Any],
        K: int,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Execute a search for a single query.

        Args:
            qvec: (D,) float32 L2-normalized query vector
            filters: JSON-like dict of predicates (may be empty)
            K: desired number of results

        Returns:
            ids: list[int] of results (len ≤ K)
            stats: dict with timing/probing/selectivity fields
        """
        ...