import pandas as pd
import numpy as np
from src.eval.oracle import load_vectors
from src.baselines.pre_filter import pre_filter_search
from src.backend_interface import SearchBackend

class PreFilterBackend(SearchBackend):

    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame, name: str):
        self.df_metadata = metadata
        self.vectors = vectors
        self.name = name

    def search(self, qvec, filters, K):
        ids, stats = pre_filter_search(qvec, filters, self.df_metadata, self.vectors, K)
        return {"ids": ids, "stats": stats, "method": "pre_filter"}