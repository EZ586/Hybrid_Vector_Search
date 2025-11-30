from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import os

import numpy as np
import faiss

from src.baselines.hybrid.index import (
    load_ivf_index,
    DEFAULT_INDEX_PATH,       # new default
    DEFAULT_FULL_INDEX_PATH,  # backward compat
    get_ivf_centroids,
    get_ivf_id_to_list_map,
)
from src.baselines.hybrid.search import hybrid_search
from src.baselines.hybrid.selector import make_allowlist
from src.baselines.hybrid.scheduler import linear_nprobe_scheduler, geometric_nprobe_scheduler
from src.baselines.hybrid.list_ordering import build_probe_order
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
    - optionally builds a list probe order (using list_ordering)
    - optionally applies an early-stop policy over probes
    - runs the predicate-aware ANN loop from baselines.hybrid.search
    - returns (ids, stats) in a harness-friendly shape
    """

    name = "hybrid"

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        metadata_dir: str = os.path.join(DEFAULT_METADATA_ROOT, DEFAULT_METADATA_BUCKET),
        nprobe_start: int = 4,
        nprobe_step: int = 4,
        nprobe_max: Optional[int] = None,
        *,
        hybrid_use_ordering: bool = False,
        hybrid_early_stop: Optional[str] = None,
        hybrid_global_bound: Optional[float] = None,
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

        # 3) scheduler config (nprobe_max may be None and inferred later)
        self._nprobe_start = nprobe_start
        self._nprobe_step = nprobe_step
        self._nprobe_max: Optional[int] = nprobe_max

        # 4) hybrid-specific knobs
        self._use_ordering: bool = bool(hybrid_use_ordering)
        # store both the name and the callable for logging + execution
        self._early_stop_name: Optional[str] = hybrid_early_stop
        self._early_stop_policy = get_early_stop_policy(hybrid_early_stop)
        self._global_bound: Optional[float] = hybrid_global_bound

        # 5) IVF internals (Person A: Task A1)
        # These helpers are expected to return:
        #   centroids: (L, D)
        #   id_to_list: (N,) mapping each id -> list_id or -1
        try:
            self.ivf_centroids: Optional[np.ndarray] = get_ivf_centroids(self.index)
        except Exception:
            self.ivf_centroids = None

        try:
            self.id_to_list: Optional[np.ndarray] = get_ivf_id_to_list_map(self.index)
        except Exception:
            self.id_to_list = None

        self.n_lists: Optional[int] = None
        if self.ivf_centroids is not None:
            self.n_lists = int(self.ivf_centroids.shape[0])

        # If caller did not specify nprobe_max, default to a soft cap
        # so a "no-arg" run doesn't immediately probe the entire index.
        if self._nprobe_max is None:
            if self.n_lists is not None:
                # Allow probing up to all lists, but cap by default
                self._nprobe_max = min(int(self.n_lists), 256)
            else:
                # Fallback if IVF internals are unavailable for some reason
                self._nprobe_max = 64

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _compute_allowed_counts_per_list(self, allow_ids: np.ndarray) -> Optional[np.ndarray]:
        """
        Map allow_ids to IVF lists and count how many allowed ids fall in each list.

        Returns:
            allowed_counts_per_list: (L,) int64 or None if we can't compute it.
        """
        if (
            self.id_to_list is None
            or self.n_lists is None
            or allow_ids.size == 0
        ):
            return None

        # Ensure int64 for safe indexing
        allow_ids = np.asarray(allow_ids, dtype=np.int64)

        # Guard against ids out of range, though this shouldn't happen
        max_id = self.id_to_list.shape[0] - 1
        safe_mask = (allow_ids >= 0) & (allow_ids <= max_id)
        if not np.any(safe_mask):
            return np.zeros(self.n_lists, dtype=np.int64)

        list_ids = self.id_to_list[allow_ids[safe_mask]]
        valid_mask = list_ids >= 0
        if not np.any(valid_mask):
            return np.zeros(self.n_lists, dtype=np.int64)

        counts = np.bincount(
            list_ids[valid_mask],
            minlength=self.n_lists,
        ).astype(np.int64, copy=False)
        return counts

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

        # 2) per-query allowed counts per list (Person A: Task A2)
        allowed_counts_per_list: Optional[np.ndarray] = self._compute_allowed_counts_per_list(
            allow_ids
        )

        # 3) optional probe-order (Person B)
        probe_order = None
        if (
            self._use_ordering
            and self.ivf_centroids is not None
            and self.n_lists is not None
            and self.n_lists > 0
        ):
            try:
                probe_order = build_probe_order(
                    qvec=qvec,
                    centroids=self.ivf_centroids,
                    allowed_counts=allowed_counts_per_list,
                )
            except Exception:
                # If anything goes wrong, just fall back silently
                probe_order = None

        # 4) build a per-query adaptive nprobe scheduler
        #
        # We adjust the ladder based on filter selectivity, but only by
        # scaling the backend's configured (start, step, max) values.
        #
        #   selectivity = |allow_ids| / |all_ids|
        total_ids = self._all_ids.size if self._all_ids is not None else 0
        if total_ids > 0:
            selectivity = float(allow_ids.size) / float(total_ids)
        else:
            # degenerate case; treat as unfiltered
            selectivity = 1.0

        # For mid-selectivity (roughly 1%–40%), use a simpler early-stop policy:
        # just stop once we have K candidates, like the post-filter baseline.
        is_mid_band = 0.01 <= selectivity <= 0.40

        # Default to the backend-wide policy, but override in the mid band.
        effective_early_stop_policy = self._early_stop_policy
        if is_mid_band:
            effective_early_stop_policy = get_early_stop_policy("k_only")

        # Backend defaults (set in __init__), sanitized to be consistent.
        base_start = max(1, int(self._nprobe_start))
        base_step = max(1, int(self._nprobe_step))
        base_max = int(self._nprobe_max) if self._nprobe_max is not None else base_start
        base_max = max(base_start, base_max)

        # Scale the nprobe ladder based on selectivity:
        #
        # - VERY selective (< 1% allowed): scale strongly (2.5x) to chase recall.
        # - Lower mid-band (20%–50%): modest boost (1.75x) to keep latency a bit lower.
        # - Upper mid-band (50%–80%): stronger boost (2x) where recall can be tricky.
        # - High-selectivity tail (80%–100%): also boosted (2.25x) to raise tail recall.
        # - Everything else: keep the base ladder.
        scale = 1.0
        if selectivity < 0.01:
            scale = 2.5
        elif 0.20 <= selectivity <= 0.50:
            scale = 1.75
        elif 0.50 < selectivity < 0.80:
            scale = 2.0
        elif 0.80 <= selectivity <= 1.0:
            scale = 2.25

        nprobe_start = max(1, int(round(base_start * scale)))
        nprobe_step = max(1, int(round(base_step * scale)))
        nprobe_max = max(nprobe_start, int(round(base_max * scale)))

        # Never exceed the number of IVF lists in the index
        if self.n_lists is not None:
            nprobe_start = min(nprobe_start, int(self.n_lists))
            nprobe_max = min(nprobe_max, int(self.n_lists))

        # If we know which lists contain at least one allowed id, we can also
        # clamp nprobe_max so we don't waste probes on predicate-empty lists.
        if allowed_counts_per_list is not None:
            useful_lists = int(np.count_nonzero(allowed_counts_per_list > 0))
            if useful_lists > 0:
                nprobe_max = min(nprobe_max, useful_lists)

        # For almost-unfiltered queries, avoid over-probing: let early-stop
        # and the global max handle the tail.
        if selectivity >= 0.95:
            nprobe_max = min(nprobe_max, 192)

        # Band-specific caps to fine-tune latency vs recall:
        # - 0.20–0.50: slightly lower cap to keep latency down.
        # - 0.50–0.80: larger cap where recall needs more help.
        # - 0.80–1.00: largest cap (within reason) to lift tail recall.
        if 0.20 <= selectivity <= 0.50:
            nprobe_max = min(nprobe_max, 112)
        elif 0.50 < selectivity < 0.80:
            nprobe_max = min(nprobe_max, 128)
        elif 0.80 <= selectivity <= 1.0:
            nprobe_max = min(nprobe_max, 192)

        # Ensure ladder valid
        nprobe_max = max(nprobe_max, nprobe_start)

        # Mid-band: single strong probe (post-filter style)
        if 0.01 <= selectivity <= 0.40:
            target = max(48, min(nprobe_max, 64))
            nprobe_iter = iter([int(target)])

        # High selectivity: geometric ramp
        elif selectivity >= 0.80:
            nprobe_iter = geometric_nprobe_scheduler(
                start=nprobe_start,
                factor=2.0,
                max_nprobe=nprobe_max,
            )

        # Default: linear growth
        else:
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
            early_stop_policy=effective_early_stop_policy,
            global_bound=self._global_bound,
        )
        latency_ms = (time.time() - start) * 1000.0
        # 6) make sure harness can rely on length K
        if len(ids) < K:
            ids = ids + [-1] * (K - len(ids))

        # 7) tag backend name + hybrid-specific flags for logging
        stats["backend"] = self.name
        stats["latency_ms"]=latency_ms

        # hybrid-specific extras expected by run.py
        extras = {
            "has_probe_order": bool(probe_order is not None),
            "has_allowed_counts_per_list": bool(allowed_counts_per_list is not None),
            "early_stop_policy": self._early_stop_name,
            "global_bound": self._global_bound,
        }
        stats["extras"] = extras

        return ids, stats