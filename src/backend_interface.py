from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class SearchBackend(ABC):
    """
    Contract for all search backends used by the harness.

    .name (str): registry key for this backend (e.g., "random")
    search(qvec, filters, K) -> Dict with:
      - ids: List[int]  (length <= K)
      - stats: Dict[str, Any] with at least:
          * latency_ms (float)
          * scored_vectors (int)  # how many vectors actually scored
    """
    name: str = "base"

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors  # N x D float32

    @abstractmethod
    def search(self, qvec: np.ndarray, filters: Dict[str, Any], K: int) -> Dict[str, Any]:
        ...
