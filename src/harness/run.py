from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from src.backends.random import RandomBackend
from src.backend_interface import SearchBackend
from src.logger import append_jsonl
from src.eval.oracle import brute_force, load_vectors
from src.eval import metrics as eval_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results" / "week1_dev"

# ----- Helpers -----

def load_vectors_npy(version: str) -> np.ndarray:
    vec_path = ARTIFACTS / version / "v1" / "vectors.npy"
    arr = np.load(vec_path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

def load_one_query(version: str) -> Tuple[int, np.ndarray]:
    qpath = ARTIFACTS / version / "v1" / "queries.parquet"
    df = pd.read_parquet(qpath)

    # Try a few common column names for the vector payload:
    vec_col: Optional[str] = None
    for c in ("vector", "embedding", "qvec", "repr"):
        if c in df.columns:
            vec_col = c
            break

    if vec_col is None:
        # Fallback: if queries are just integer ids, use the corresponding vector row as a query
        # (good enough for a smoke test)
        row0 = df.iloc[0]
        qid = int(row0.get("qid", 0))
        vectors = load_vectors_npy(version)
        return qid, vectors[qid].astype(np.float32)

    row0 = df.iloc[0]
    qid = int(row0.get("qid", 0))
    qvec = np.array(row0[vec_col], dtype=np.float32).reshape(-1)
    return qid, qvec

def compute_recall_at_k(pred_ids, oracle_ids, k: int) -> float:
    try:
        return float(eval_metrics.recall_at_k(oracle_ids, pred_ids, k))
    except Exception:
        # Fallback: |intersection| / k
        return float(len(set(pred_ids[:k]).intersection(oracle_ids[:k])) / float(k))

def get_backend(name: str, vectors: np.ndarray) -> SearchBackend:
    registry = {
        "random": RandomBackend,
        # Later add: "pre_filter": PreFilterBackend, "post_filter": PostFilterBackend
    }
    if name not in registry:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(registry)}")
    return registry[name](vectors)

# ----- Main -----

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="dev", choices=["dev", "full"])
    parser.add_argument("--backend", default="random")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--out", default=str(RESULTS_DIR / "results_midweek.jsonl"))
    args = parser.parse_args()

    # Load vectors for both backend & oracle
    vectors = load_vectors_npy(args.version)
    load_vectors(args.version)  # initialize oracle’s global VECTORS

    # Load one query
    qid, qvec = load_one_query(args.version)

    # Currently no filters; selectivity = 1.0
    filters: Dict[str, Any] = {}
    selectivity = 1.0

    # Run backend
    backend = get_backend(args.backend, vectors)
    result = backend.search(qvec, filters=filters, K=args.K)
    pred_ids = result["ids"]
    latency_ms = float(result["stats"].get("latency_ms", 0.0))
    scored_vectors = int(result["stats"].get("scored_vectors", 0))

    # Oracle Top-K for recall
    allowed_ids = np.arange(vectors.shape[0], dtype=np.int64)
    oracle_ids = brute_force(qvec, allowed_ids=allowed_ids, K=args.K)

    recall_at_k = compute_recall_at_k(pred_ids, oracle_ids, args.K)

    # Log one row
    row = {
        "qid": int(qid),
        "method": backend.name,
        "K": int(args.K),
        "latency_ms": latency_ms,
        "recall_at_k": float(recall_at_k),
        "filter_selectivity": float(selectivity),
        "scored_vectors": int(scored_vectors),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(row, Path(args.out))

    print("=== Run ===")
    print(f"Backend:     {backend.name}")
    print(f"Query qid:   {qid}")
    print(f"K:           {args.K}")
    print(f"Recall@{args.K}: {recall_at_k:.3f}")
    print(f"Latency:     {latency_ms:.2f} ms")
    print(f"Wrote log:   {args.out}")


if __name__ == "__main__":
    main()


# Example run: python -m src.harness.run --version dev --backend random --K 10 \
# --out results/week1_dev/results_midweek.jsonl