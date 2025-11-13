# src/backends/hybrid_backend.py

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import os

import numpy as np
import faiss

from src.baselines.hybrid.index import (
    load_ivf_index,
    DEFAULT_INDEX_PATH,  # new default
    DEFAULT_FULL_INDEX_PATH,  # backward compat
)
from src.baselines.hybrid.search import hybrid_search
from src.baselines.hybrid.selector import make_allowlist
from src.baselines.hybrid.scheduler import linear_nprobe_scheduler

from src.dataio.validators import (
    build_allowed_ids,
)


from src.dataio.loaders import load_metadata

from src.backends.backend_interface import SearchBackend

DEFAULT_METADATA_ROOT = "/artifacts"
DEFAULT_METADATA_BUCKET = "v2"


class HybridBackend(SearchBackend):
    """
    SearchBackend wrapper around the hybrid FAISS IVF + metadata allow-list search.

    This backend:
    - loads a persisted IVF index (by default from /results/indexes/faiss_ivf.index)
    - loads canonical metadata from /artifacts/v2/ via dataio.loaders
    - materializes allow-lists from JSON-style filters (using baselines.hybrid.selector)
    - runs the predicate-aware ANN loop from baselines.hybrid.search
    - returns (ids, stats) in a harness-friendly shape
    """

    name = "hybrid"

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        metadata_dir: str = os.path.join(
            DEFAULT_METADATA_ROOT, DEFAULT_METADATA_BUCKET
        ),
        nprobe_start: int = 4,
        nprobe_step: int = 4,
        nprobe_max: int = 64,
    ) -> None:
        # 1) load FAISS IVF index
        # allow older code that still passes DEFAULT_FULL_INDEX_PATH
        if not os.path.exists(index_path) and os.path.exists(DEFAULT_FULL_INDEX_PATH):
            index_path = DEFAULT_FULL_INDEX_PATH

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"HybridBackend: index not found at {index_path}. "
                "Build it first with baselines.hybrid.index.build_ivf_index_from_artifacts(...)."
            )
        self.index: faiss.IndexIVFFlat = load_ivf_index(index_path)

        # 2) load canonical metadata (for filters → allow_ids) using the official loader
        if not os.path.exists(metadata_dir):
            raise FileNotFoundError(
                f"HybridBackend: metadata dir not found at {metadata_dir}. "
                "Make sure /artifacts/v2/ exists and was generated."
            )
        self.metadata_df = load_metadata(metadata_dir)

        # precompute all ids for the “no filter” case
        self._all_ids: np.ndarray = self.metadata_df["id"].to_numpy(dtype=np.int64)

        # 3) scheduler config
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
            filters: JSON-like dict as produced by queries.parquet (can be {})
            K: desired number of final results

        Returns:
            (ids, stats)
            ids: list of up to K ints (padded to K with -1 if needed)
            stats: dict with latency_ms, scored_vectors, lists_probed, nprobe, backend, ...
        """
        # 1) build allow_ids from filters; if no filters, allow all
        if not filters:
            allow_ids = self._all_ids
        else:
            allow_ids = build_allowed_ids(self.metadata_df, filters)
            # safety: if filter is too tight, run anyway and let search report zero
            if allow_ids.size == 0:
                allow_ids = np.empty((0,), dtype=np.int64)

        # 2) build scheduler
        nprobe_iter = linear_nprobe_scheduler(
            start=self._nprobe_start,
            step=self._nprobe_step,
            max_nprobe=self._nprobe_max,
        )

        # compute centroids if available
        centroids = None
        if hasattr(self.index, "quantizer"):
            try:
                q = self.index.quantizer
                q = faiss.downcast_index(q)
                # print(f"[INFO] Quantizer type: {type(q)}")

                centroids = None

                # ✅ Try extracting from the IVF index itself
                if hasattr(self.index, "reconstruct_n"):
                    try:
                        arr = self.index.reconstruct_n(0, self.index.nlist)
                        centroids = arr.reshape(self.index.nlist, self.index.d)
                        # print(f"[INFO] Extracted centroids via IVF.reconstruct_n: shape={centroids.shape}")
                    except Exception as e:
                        print(f"[WARN] IVF.reconstruct_n failed ({type(e).__name__}): {e}")


                if centroids is None:
                    print("[WARN] No centroids could be extracted.")

            except Exception as e:
                print(f"[WARN] Could not extract centroids ({type(e).__name__}): {e}")
                centroids = None



        if hasattr(self.index, "invlists"):
            try:
                nlist = self.index.nlist
                allowed_counts = np.zeros(nlist, dtype=np.int32)

                for lid in range(nlist):
                    size = self.index.invlists.list_size(lid)
                    if size == 0:
                        continue

                    # ✅ New API for FAISS >= 1.8.0
                    ids_ptr = self.index.invlists.get_ids(lid)
                    ids = faiss.rev_swig_ptr(ids_ptr, size)
                    allowed_counts[lid] = np.isin(ids, allow_ids).sum()

                # print(f"[INFO] Computed allowed_counts per list (nonzero={np.count_nonzero(allowed_counts)})")

            except AttributeError:
                print("[WARN] This FAISS build does not expose invlists.get_ids; skipping allowed_counts.")
                allowed_counts = None

            except Exception as e:
                print(f"[WARN] Could not compute allowed_counts ({type(e).__name__}): {e}")
                allowed_counts = None



        # 3) run the predicate-aware ANN loop
        ids, stats = hybrid_search(
            qvec=qvec,
            index=self.index,
            allow_ids=allow_ids,
            K=K,
            nprobe_iter=nprobe_iter,
            centroids=centroids,
            allowed_counts_per_list=allowed_counts,
        )

        # 4) make sure harness can rely on length K
        if len(ids) < K:
            ids = ids + [-1] * (K - len(ids))

        # 5) tag backend name for logging
        stats["backend"] = self.name

        return ids, stats
