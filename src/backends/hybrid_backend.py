# src/backends/hybrid_backend.py

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import os

import numpy as np
import faiss

from src.baselines.hybrid.index import (
    load_hnsw_index,
    DEFAULT_INDEX_PATH,          # HNSW default
    DEFAULT_FULL_INDEX_PATH,
)
from src.baselines.hybrid.search import hybrid_search    # now HNSW search
from src.baselines.hybrid.selector import make_allowlist

from src.dataio.validators import build_allowed_ids
from src.dataio.loaders import load_metadata

from src.backends.backend_interface import SearchBackend

DEFAULT_METADATA_ROOT = "/artifacts"
DEFAULT_METADATA_BUCKET = "v2"


class HybridBackend(SearchBackend):
    """
    SearchBackend wrapper around the hybrid FAISS HNSW + metadata allow-list search.

    This backend:
    - loads a persisted HNSW index
    - loads canonical metadata from /artifacts/v2/
    - builds allow-lists from filters
    - runs the HNSW–based predicate-aware ANN search
    - returns (ids, stats) in a harness-friendly shape
    """

    name = "hybrid"

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        metadata_dir: str = os.path.join(DEFAULT_METADATA_ROOT, DEFAULT_METADATA_BUCKET),
        ef_search: int = 128,        # HNSW search depth parameter
    ) -> None:

        # ------------------------------------------------------------
        # 1) Load HNSW index
        # ------------------------------------------------------------
        if not os.path.exists(index_path) and os.path.exists(DEFAULT_FULL_INDEX_PATH):
            index_path = DEFAULT_FULL_INDEX_PATH

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"HybridBackend: index not found at {index_path}. "
                "Build it first with baselines.hybrid.index.build_hnsw_index_from_artifacts(...)."
            )

        self.index: faiss.IndexHNSWFlat = load_hnsw_index(index_path)

        # user-configurable search depth
        self.ef_search = ef_search
        self.index.hnsw.efSearch = ef_search

        # ------------------------------------------------------------
        # 2) Load metadata
        # ------------------------------------------------------------
        if not os.path.exists(metadata_dir):
            raise FileNotFoundError(
                f"HybridBackend: metadata dir not found at {metadata_dir}. "
                "Make sure /artifacts/v2/ exists."
            )

        self.metadata_df = load_metadata(metadata_dir)
        self._all_ids = self.metadata_df["id"].to_numpy(dtype=np.int64)

    # ------------------------------------------------------------------
    # required by SearchBackend
    # ------------------------------------------------------------------
    def search(
        self,
        qvec: np.ndarray,
        filters: Optional[Dict[str, Any]],
        K: int,
    ) -> Tuple[List[int], Dict[str, Any]]:

        # ------------------------------------------------------------
        # 1) Metadata → allow-list
        # ------------------------------------------------------------
        if not filters:
            allow_ids = self._all_ids
        else:
            allow_ids = build_allowed_ids(self.metadata_df, filters)
            if allow_ids.size == 0:
                allow_ids = np.empty((0,), dtype=np.int64)

        # ------------------------------------------------------------
        # 2) Run hybrid HNSW search
        # (nprobe_iter, centroids, invlists unused in HNSW)
        # ------------------------------------------------------------
        ids, stats = hybrid_search(
            qvec=qvec,
            index=self.index,
            allow_ids=allow_ids,
            K=K,
        )

        # ------------------------------------------------------------
        # 3) Pad to length K (required by harness)
        # ------------------------------------------------------------
        if len(ids) < K:
            ids = ids + [-1] * (K - len(ids))

        # ------------------------------------------------------------
        # 4) Tag backend name
        # ------------------------------------------------------------
        stats["backend"] = self.name
        stats["ef_search"] = self.ef_search

        return ids, stats
