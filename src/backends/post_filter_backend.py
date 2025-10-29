"""
Post-filter backend

Exposes `search(qvec, filters, K) -> (ids, stats)` for the harness.
Sequence: ANN (K′ ladder) → filter → select top-K.

Stats contract (returned via `stats`):
  - latency_ms (float): end-to-end wall time in ms
  - scored_vectors (int): total candidate vectors retrieved across ladder steps
  - notes (str): details (ladder used, kept, need, etc.)
  - kth_at_stop (Optional[float]): score of the kth kept item when stopping; None if <K kept
  - lists_probed (optional): None for post-filter baseline
  - nprobe (optional): None for post-filter baseline
  - bound_at_stop (optional): None for post-filter baseline
  - extras (object): backend-specific fields (e.g., retries, k_ladder)

Logging note: Only schema-approved fields should be top-level in the run log.
Backend-specific fields are nested under `extras`.
"""
from __future__ import annotations

from typing import Tuple, List, Dict, Any, Iterable, Optional
import numpy as np
import pandas as pd

from src.backend_interface import SearchBackend
from src.baselines.post_filter import post_filter_search
from src.dataio.loaders import load_metadata, load_vectors_index


class PostFilterBackend(SearchBackend):
    name = "post_filter"

    def __init__(
        self,
        artifacts_root: str,
        k_ladder: Iterable[int] = (200, 500, 1000),
        max_ladder_steps: Optional[int] = None,
    ) -> None:
        self.artifacts_root = artifacts_root
        self.k_ladder = tuple(int(k) for k in k_ladder)
        self.max_ladder_steps = max_ladder_steps

        # Load artifacts. Ensure metadata is indexed by `id` for stable lookups.
        md = load_metadata(artifacts_root)  
        if not isinstance(md, pd.DataFrame):
            raise TypeError("load_metadata() must return a pandas DataFrame")
        if getattr(md.index, "name", None) != "id":
            if "id" in md.columns:
                md = md.set_index("id", drop=False).sort_index()
            else:
                raise ValueError("metadata must have an 'id' column or be indexed by 'id'.")
        self.metadata = md

        self.ann_index = load_vectors_index(artifacts_root)  # must expose .search(qvec, k)
        if not hasattr(self.ann_index, "search"):
            raise AttributeError("ann_index must provide a .search(qvec, k) method")

    def search(self, qvec: np.ndarray, filters: Dict[str, Any], K: int) -> Tuple[List[int], Dict[str, Any]]:
        """Run the post-filter baseline and adapt stats for the harness."""
        ids, stats = post_filter_search(
            qvec=qvec,
            ann_index=self.ann_index,
            metadata_df=self.metadata,
            filters=filters,
            K=int(K),
            k_ladder=self.k_ladder,
            max_ladder_steps=self.max_ladder_steps,
        )

        # Ensure required/expected fields exist and normalize optional ones
        stats.setdefault("latency_ms", None)
        stats.setdefault("scored_vectors", None)
        stats.setdefault("notes", "")
        stats.setdefault("kth_at_stop", None)
        stats.setdefault("lists_probed", None)
        stats.setdefault("nprobe", None)
        stats.setdefault("bound_at_stop", None)

        # Move backend-specific fields to `extras` to keep the log schema clean
        extras: Dict[str, Any] = stats.pop("extras", {}) if isinstance(stats.get("extras"), dict) else {}
        retries = stats.pop("retries", 0)
        extras.setdefault("retries", int(retries))
        extras.setdefault("k_ladder", list(self.k_ladder))
        if self.max_ladder_steps is not None:
            extras.setdefault("max_ladder_steps", int(self.max_ladder_steps))
        stats["extras"] = extras

        return ids, stats