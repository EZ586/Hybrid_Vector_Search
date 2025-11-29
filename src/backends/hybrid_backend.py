from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import os

import numpy as np
import faiss

from src.baselines.hybrid.search import hybrid_search
from src.baselines.hybrid.selector import make_allowlist
from src.baselines.hybrid.scheduler import linear_nprobe_scheduler
from src.baselines.hybrid.early_stop import get_early_stop_policy

from src.dataio.loaders import load_metadata

from src.backends.backend_interface import SearchBackend
import time

DEFAULT_METADATA_ROOT = "/artifacts"
DEFAULT_METADATA_BUCKET = "v2"


class HybridBackend(SearchBackend):
    """
    SearchBackend wrapper around the hybrid FAISS IVF + metadata allow-list search.

    This backend:
    - loads a persisted IVF index (by default from /results/indexes/faiss_ivf.index)
    - loads canonical metadata from /artifacts/v2/ via dataio.loaders
    - materializes allow-lists from JSON-style filters (using baselines.hybrid.selector)
    - computes per-list allowed counts (if possible)
    - optionally applies an early-stop policy over probes
    - runs the predicate-aware ANN loop from baselines.hybrid.search
    - returns (ids, stats) in a harness-friendly shape
    """

    name = "hybrid"

    def __init__(
        self,
        index_path: str,
        metadata_dir: str,
        nprobe_start: int = 4,
        nprobe_step: int = 4,
        nprobe_max: int = 1024,
        hybrid_early_stop: Optional[str] = None,
    ) -> None:
        # 1) load FAISS IVF index
        self.index = faiss.read_index(index_path)

        # 2) load canonical metadata (for filters → allow_ids) using the official loader
        self.metadata_df = load_metadata(metadata_dir)

        # precompute all ids for the “no filter” case
        self._all_ids: np.ndarray = self.metadata_df["id"].to_numpy(dtype=np.int64)

        # 3) scheduler config (nprobe_max may be None and inferred later)
        self._nprobe_start = nprobe_start
        self._nprobe_step = nprobe_step
        self._nprobe_max: Optional[int] = nprobe_max

        # store both the name and the callable for logging + execution
        self._early_stop_name: Optional[str] = hybrid_early_stop
        self._early_stop_policy = get_early_stop_policy(hybrid_early_stop)

        self.n_lists = self.index.nlist

        # If caller did not specify nprobe_max, default to "all lists"
        if self._nprobe_max is None:
            self._nprobe_max = int(self.n_lists)
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
            filters: JSON-like dict as produced by queries.parquet (can be {})
            K: desired number of final results

        Returns:
            (ids, stats)
            ids: list of up to K ints (padded to K with -1 if needed)
            stats: dict with latency_ms, scored_vectors, lists_probed, nprobe, backend, ...
        """
        # 1) build allow_ids from filters; if no filters, allow all
        start=time.time()
        if not filters:
            allow_ids = self._all_ids
        else:
            allow_ids = make_allowlist(self.metadata_df, filters)
            # safety: if filter is too tight, run anyway and let search report zero
            if allow_ids.size == 0:
                allow_ids = np.empty((0,), dtype=np.int64)

        # calculate selectivity
        total_ids = self._all_ids.size if self._all_ids is not None else 0
        selectivity = 0
        if total_ids > 0:
            selectivity = float(allow_ids.size) / float(total_ids)

        # backend defaults (set in __init__)
        base_start = self._nprobe_start
        base_step = self._nprobe_step
        base_max = self._nprobe_max

        # start from the backend defaults
        nprobe_start = base_start
        nprobe_step = base_step
        nprobe_max = base_max

        # Scale nprobe ladder based on selectivity:
        scale = 1.0
        if selectivity < 0.01:
            scale = 2.0
        elif selectivity >= 0.30 and selectivity <= 0.70:
            scale = 3.0

        if scale != 1.0:
            nprobe_start = max(1, int(base_start * scale))
            nprobe_step = max(1, int(base_step * scale))
            nprobe_max = max(nprobe_start, int(base_max * scale))

        # Never exceed the number of IVF lists in the index
        nprobe_start = min(nprobe_start, int(self.n_lists))
        nprobe_max = min(nprobe_max, int(self.n_lists))


        # Finally, build the iterator
        nprobe_iter = linear_nprobe_scheduler(
            start=nprobe_start,
            step=nprobe_step,
            max_nprobe=nprobe_max,
        )

        # 5) run the predicate-aware ANN loop
        ids, stats = hybrid_search(
            qvec=qvec,
            index=self.index,
            allow_ids=allow_ids,
            K=K,
            nprobe_iter=nprobe_iter,
            early_stop_policy=self._early_stop_policy,
        )
        latency_ms = (time.time() - start) * 1000.0
        # 6) make sure harness can rely on length K
        if len(ids) < K:
            ids = ids + [-1] * (K - len(ids))

        # 7) tag backend name + hybrid-specific flags for logging
        stats["backend"] = self.name
        stats["latency_ms"]=latency_ms

        # hybrid-specific extras expected by run.py
        stats["extras"] = {
            "early_stop_policy": self._early_stop_name,
        }

        return ids, stats