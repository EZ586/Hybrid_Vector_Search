# tests/test_hybrid_search.py

import numpy as np
import faiss

from baselines.hybrid.search import hybrid_search
from baselines.hybrid.scheduler import linear_nprobe_scheduler


def test_hybrid_search_runs_basic():
    """
    Smoke test: hybrid_search should run and return (ids, stats) tuple.
    """
    # TODO: build tiny IVF index in-memory, tiny allow_ids, and call hybrid_search
    raise NotImplementedError


def test_hybrid_stats_fields_present():
    """
    hybrid_search should return stats containing latency_ms and scored_vectors at minimum.
    """
    # TODO: call hybrid_search and assert required keys in stats
    raise NotImplementedError
