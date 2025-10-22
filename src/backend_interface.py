from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
import pandas as pd

class SearchBackend(ABC):
    """
    All backends must implement .search(qvec, filters, K) and return:
      {
        "ids": List[int],  # Top-K ids, desc similarity
        "stats": {
          "latency_ms": float,
          "scored_vectors": int,
          "lists_probed": int | None,
          "nprobe": int | None,
          "kth_at_stop": float | None,
          "bound_at_stop": float | None,
          "notes": str | None,
        }
      }
    """

    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame, name: str):
        self.vectors = vectors  # N x D float32
        self.metadata = metadata
        self.name = name

    @abstractmethod
    def search(self, qvec: np.ndarray, filters: Dict[str, Any], K: int) -> Dict[str, Any]:
        ...
