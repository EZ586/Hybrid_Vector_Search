import pandas as pd
import numpy as np
from src.eval.oracle import load_vectors
from src.baselines.pre_filter import pre_filter_search
from src.backend_interface import SearchBackend


"""
Pre-filter backend

Exposes `search(qvec, filters, K) -> (ids, stats)` for the harness.
Sequence: filter → inner product search → select top-K.

Stats contract (returned via `stats`):
  - latency_ms (float): end-to-end wall time in ms
  - scored_vectors (int): total candidate vectors retrieved from metadata filtering
  - lists_probed (optional): None for pre-filter baseline
  - nprobe (optional): None for pre-filter baseline
  - kth_at_stop (Optional[float]): None for pre-filter baseline
  - bound_at_stop (optional): None for pre-filter baseline
  - notes (str): pre-filter top-K inner product
"""
class PreFilterBackend(SearchBackend):

    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame, name: str):
        self.df_metadata = metadata
        self.vectors = vectors
        self.name = name

    def search(self, qvec: np.ndarray, filters: dict, K: int) -> tuple[list[int], dict]:
        ids, stats = pre_filter_search(qvec, filters, self.df_metadata, self.vectors, K)
        return ids, stats
