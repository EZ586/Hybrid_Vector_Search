from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
import uuid

import numpy as np
import pandas as pd

from src.backends.random import RandomBackend
from src.backends.exact import ExactBackend
from src.backends.prefilter_backend import PreFilterBackend
from src.backend_interface import SearchBackend
from src.logger import append_jsonl
from src.eval.oracle import brute_force, load_vectors
from src.eval import metrics as eval_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results" / "week1"

def _coerce_vectors(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return np.ascontiguousarray(arr.astype(np.float32))

def load_vectors_npy(version: str) -> np.ndarray:
    vec_path = ARTIFACTS / version / "v1" / "vectors.npy"
    arr = np.load(vec_path)  # real file via git-lfs
    return _coerce_vectors(arr)

def load_queries(version: str) -> pd.DataFrame:
    qpath = ARTIFACTS / version / "v1" / "queries.parquet"
    return pd.read_parquet(qpath)

def load_metadata(version: str) -> pd.DataFrame:
    mpath = ARTIFACTS / version / "v1" / "metadata.parquet"
    return pd.read_parquet(mpath)

def pick_vector_from_row(row: pd.Series) -> np.ndarray:
    for c in ("vector", "embedding", "qvec", "repr"):
        if c in row and row[c] is not None:
            return np.array(row[c], dtype=np.float32).reshape(-1)
    # fallback: use id as index into vectors (dev-only smoke)
    return None

def compute_recall_at_k(pred_ids, oracle_ids, k: int) -> float:
    try:
        return float(eval_metrics.recall_at_k(oracle_ids, pred_ids, k))
    except Exception:
        return float(len(set(pred_ids[:k]).intersection(oracle_ids[:k])) / float(k))

def get_backend(name: str, vectors: np.ndarray, metadata: pd.DataFrame) -> SearchBackend:
    registry = {
        "random": RandomBackend,
        "exact": ExactBackend,
        "pre_filter": PreFilterBackend
        # later: "pre_filter": PreFilterBackend, "post_filter": PostFilterBackend, "hybrid": ...
    }
    if name not in registry:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(registry)}")
    return registry[name](vectors, metadata, name)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="dev", choices=["dev", "full"])
    ap.add_argument("--backend", default="exact", choices=["exact", "random","pre_filter"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max_queries", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "results_dev.jsonl"))
    args = ap.parse_args()

    vectors = load_vectors_npy(args.version)
    metadata = load_metadata(args.version)
    load_vectors(args.version)  # init oracle globals

    qdf = load_queries(args.version)
    # take first N queries
    rows = qdf.head(args.max_queries)
    print(type(metadata))
    backend = get_backend(args.backend, vectors, metadata)

    # (dev) no filters; selectivity=1.0
    filters: Dict[str, Any] = {}
    selectivity = 1.0

    for _, row in rows.iterrows():
        qid = int(row.get("qid", 0))
        qvec = pick_vector_from_row(row)
        if qvec is None:
            qvec = vectors[qid].astype(np.float32)

        # Run backend
        result = backend.search(qvec, filters=filters, K=args.K)

        # Oracle for recall
        allowed = np.arange(vectors.shape[0], dtype=np.int64)
        oracle_ids = brute_force(qvec, allowed, args.K)
        recall_at_k = compute_recall_at_k(result["ids"], oracle_ids, args.K)

        stats = result["stats"]
        row_out = {
            "run_id": str(uuid.uuid4()),
            "qid": qid,
            "method": backend.name,
            "K": int(args.K),
            "latency_ms": float(stats.get("latency_ms", 0.0)),
            "recall@K": float(recall_at_k),
            "filter_selectivity": float(selectivity),
            "scored_vectors": int(stats.get("scored_vectors", 0)),
            "lists_probed": stats.get("lists_probed"),
            "nprobe": stats.get("nprobe"),
            "kth_at_stop": stats.get("kth_at_stop"),
            "bound_at_stop": stats.get("bound_at_stop"),
            "notes": stats.get("notes"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(row_out, Path(args.out))

    print(f"Wrote {len(rows)} runs to {args.out}")

if __name__ == "__main__":
    main()

# Example run: python -m src.harness.run --version dev --backend random --K 10 \
# --out results/week1_dev/results_dev.jsonl

