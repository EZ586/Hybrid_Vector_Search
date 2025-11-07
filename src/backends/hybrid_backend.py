# backends/hybrid_backend.py

from typing import Tuple, List, Dict, Any
import numpy as np
import faiss

from baselines.hybrid.search import hybrid_search
from baselines.hybrid.selector import build_idselector, make_allowlist  # maybe needed indirectly
from baselines.hybrid.scheduler import linear_nprobe_scheduler
from src.backend_interface import SearchBackend  # adjust import to your repo layout


class HybridBackend(SearchBackend):
    """
    SearchBackend wrapper around the hybrid FAISS IVF + allow-list search.
    """

    def __init__(self, index: faiss.IndexIVFFlat, metadata_df):
        self.index = index
        self.metadata_df = metadata_df
        self.name = "hybrid"

    def search(
        self,
        qvec: np.ndarray,
        filters: Dict[str, Any],
        K: int,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Run hybrid search and return ids + stats in harness-compatible format.
        """
        # TODO:
        # 1) build allow_ids from filters (Person B’s code)
        # 2) create scheduler
        # 3) call hybrid_search(...)
        allow_ids = make_allowlist(self.metadata_df, filters)
        nprobe_iter = linear_nprobe_scheduler(start=4, step=4, max_nprobe=64)

        ids, stats = hybrid_search(
            qvec=qvec,
            index=self.index,
            allow_ids=allow_ids,
            K=K,
            nprobe_iter=nprobe_iter
        )
        return ids, stats
        
