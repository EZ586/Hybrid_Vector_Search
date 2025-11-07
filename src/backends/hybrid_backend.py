# backends/hybrid_backend.py

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import os

import numpy as np
import pandas as pd
import faiss

from baselines.hybrid.index import load_ivf_index, DEFAULT_FULL_INDEX_PATH
from baselines.hybrid.search import hybrid_search
from baselines.hybrid.selector import make_allowlist
from baselines.hybrid.scheduler import linear_nprobe_scheduler

from src.backend_interface import SearchBackend  # keep this as in your repo


DEFAULT_METADATA_PATH = "/artifacts/full/v1/metadata.parquet"


class HybridBackend(SearchBackend):
    """
    SearchBackend wrapper around the hybrid FAISS IVF + metadata allow-list search.

    This backend:
    - loads the IVF index built from /artifacts/full/v1/vectors.npy
    - loads the corresponding metadata so we can materialize allow-lists
    - runs the predicate-aware ANN loop from baselines.hybrid.search
    - returns (ids, stats) in a harness-friendly shape
    """

    name = "hybrid"

    def __init__(
        self,
        index_path: str = DEFAULT_FULL_INDEX_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        nprobe_start: int = 4,
        nprobe_step: int = 4,
        nprobe_max: int = 64,
    ) -> None:
        # load FAISS IVF index
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"HybridBackend: index not found at {index_path}. "
                "Build it first with baselines.hybrid.index.build_from_bucket(...)."
            )
        self.index: faiss.IndexIVFFlat = load_ivf_index(index_path)

        # load metadata (for filters → allow_ids)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"HybridBackend: metadata not found at {metadata_path}. "
                "Run yelp_pipeline.py --stage meta/all first."
            )
        self.metadata_df: pd.DataFrame = pd.read_parquet(metadata_path)

        # precompute all ids for the “no filter” case
        self._all_ids: np.ndarray = (
            self.metadata_df["id"].astype("int64").to_numpy()
        )

        # scheduler config
        self._nprobe_start = nprobe_start
        self._nprobe_step = nprobe_step
        self._nprobe_max = nprobe_max

    # ------------------------------------------------------------------ #
    # required by SearchBackend
    # ------------------------------------------------------------------ #
    def search(
        self,
        qvec: np.ndarray,
        filters: Optional[Dict[str, Any]],
        K: int,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Run hybrid search over IVF with an allow-list derived from filters.

        Args:
            qvec: (D,) float32 normalized query vector
            filters: JSON-like dict as produced by your queries.parquet (can be {})
            K: desired number of final results

        Returns:
            (ids, stats)
            ids: list of up to K ints (padded to K with -1 if needed)
            stats: dict with latency_ms, scored_vectors, nprobe, retries, backend
        """
        # 1) build allow_ids from filters; if no filters, allow all
        if filters is None or len(filters) == 0:
            allow_ids = self._all_ids
        else:
            allow_ids = make_allowlist(self.metadata_df, filters)
            # safety: if filter is too tight, fall back to empty array, hybrid loop will try
            if allow_ids.size == 0:
                # still run, but this will return empty quickly
                allow_ids = np.empty((0,), dtype=np.int64)

        # 2) build scheduler
        nprobe_iter = linear_nprobe_scheduler(
            start=self._nprobe_start,
            step=self._nprobe_step,
            max_nprobe=self._nprobe_max,
        )

        # 3) run the predicate-aware ANN loop
        ids, stats = hybrid_search(
            qvec=qvec,
            index=self.index,
            allow_ids=allow_ids,
            K=K,
            nprobe_iter=nprobe_iter,
        )

        # 4) make sure harness can rely on length K
        if len(ids) < K:
            ids = ids + [-1] * (K - len(ids))

        # 5) tag backend name for logging
        stats["backend"] = self.name

        return ids, stats
