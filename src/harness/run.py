from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid
import json
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
from src.backends.random import RandomBackend
from src.backends.exact import ExactBackend
from src.backends.prefilter_backend import PreFilterBackend
from src.backends.post_filter_backend import PostFilterBackend
from src.backend_interface import SearchBackend
from src.logger import append_jsonl
from src.eval.oracle import brute_force, load_vectors as oracle_load_vectors
from src.eval import metrics as eval_metrics
from src.eval import oracle as ORACLE

from src.dataio.loaders import (
    load_vectors,
    load_vectors_meta,
    load_metadata,
    load_vectors_index,
)

from src.dataio.validators import (
    parse_filters,
    validate_K,
    ensure_unit_l2,
    validate_filters_schema,
    build_allowed_ids,
    ValidationError,
    FilterSpecError,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def compute_recall_at_k(pred_ids, oracle_ids, k: int) -> float:
    try:
        return float(eval_metrics.compute_recall(oracle_ids, pred_ids, k))
    except Exception:
        inter = len(set(pred_ids[:k]).intersection(oracle_ids[:k]))
        return float(inter / k)


def pick_vector_from_row(row: pd.Series) -> np.ndarray | None:
    for c in ("vector", "embedding", "qvec", "repr"):
        if c in row and row[c] is not None:
            return np.array(row[c], dtype=np.float32).reshape(-1)
    return None


# ---------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------
def get_backend(name: str, vectors: np.ndarray, metadata: pd.DataFrame, version: str) -> SearchBackend:
    registry = {
        "random": RandomBackend,
        "exact": ExactBackend,
        "pre_filter": PreFilterBackend,
        "post_filter": PostFilterBackend,
    }
    if name not in registry:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(registry)}")

    if name == "post_filter":
        artifacts_dir = ARTIFACTS / version / "v1"
        # # Example: if post-filter needs FAISS/HNSW index
        # index = load_vectors_index(artifacts_dir, prefer_ivf=True)
        return registry[name](artifacts_dir)
    else:
        return registry[name](vectors, metadata, name)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="dev", choices=["dev", "full"])
    ap.add_argument(
        "--backend",
        default="exact",
        choices=["exact", "random", "pre_filter", "post_filter"],
    )
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max_queries", type=int, default=10)
    ap.add_argument("--out", default=None, help="Optional: override output path")
    args = ap.parse_args()

    # -----------------------------------------------------------------
    # Load vectors, metadata, and meta info using loaders.py
    # -----------------------------------------------------------------
    artifact_dir = ARTIFACTS / args.version / "v1"
    vectors = load_vectors(artifact_dir)
    metadata = load_metadata(artifact_dir)
    meta_info = load_vectors_meta(artifact_dir)

    # Initialize oracle (for recall ground truth)
    oracle_load_vectors(args.version)
    # diff = np.max(np.abs(vectors - ORACLE.VECTORS))
    # print(f"Max diff between harness vectors and oracle vectors: {diff:.6f}")

    # Load queries (kept local since it depends on course harness)
    qpath = artifact_dir / "queries.parquet"
    qdf = pd.read_parquet(qpath)
    validate_K(args.K, len(vectors))
    backend = get_backend(args.backend, vectors, metadata, args.version)

    # -----------------------------------------------------------------
    # Prepare output folder per §9 Logging Contract
    # -----------------------------------------------------------------
    run_id = f"{args.version}_{args.backend}_{uuid.uuid4().hex[:8]}"
    # out_dir = RESULTS_DIR / run_id
    # out_dir.mkdir(parents=True, exist_ok=True)
    # out_path = out_dir / "results.jsonl"
    out_path = RESULTS_DIR / "results.jsonl"

    # -----------------------------------------------------------------
    # Run harness
    # -----------------------------------------------------------------
    rows = qdf.head(args.max_queries)
    for _, row in rows.iterrows():
        qid = int(row["qid"])
        qvec = pick_vector_from_row(row) or vectors[qid].astype(np.float32)
        ensure_unit_l2(qvec)

        # Strict filter parsing & schema validation
        try:
            filters = parse_filters(row.get("filters"))
            validate_filters_schema(metadata, filters)
        except (ValidationError, FilterSpecError) as e:
            raise RuntimeError(f"Filter validation failed for qid={qid}: {e}")

        allowed_ids = build_allowed_ids(metadata, filters)
        selectivity = len(allowed_ids) / len(metadata)

        # Run backend
        pred_ids, stats = backend.search(qvec, filters=filters, K=args.K)

        # Oracle recall
        oracle_ids = brute_force(qvec, allowed_ids, args.K)
        recall_at_k = compute_recall_at_k(pred_ids, oracle_ids, args.K)

        # -----------------------------------------------------------------
        # Log in required field order (§9)
        # -----------------------------------------------------------------
        row_out = {
            "qid": qid,
            "method": backend.name,
            "K": int(args.K),
            "latency_ms": float(stats.get("latency_ms", 0.0)),
            "recall_at_k": float(recall_at_k),
            "filter_selectivity": float(selectivity),
            "scored_vectors": int(stats.get("scored_vectors", 0)),
            "lists_probed": stats.get("lists_probed"),
            "nprobe": stats.get("nprobe"),
            "kth_at_stop": stats.get("kth_at_stop"),
            "bound_at_stop": stats.get("bound_at_stop"),
            "notes": stats.get("notes"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
        }

        append_jsonl(row_out, out_path)

    print(f"Wrote {len(rows)} runs to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
