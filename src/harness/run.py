# src/harness/run.py
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Collect rows for plotting
_PLOT_ROWS = []

from src.backends.exact_backend import ExactBackend
from src.backends.prefilter_backend import PreFilterBackend
from src.backends.post_filter_backend import PostFilterBackend
from src.backends.backend_interface import SearchBackend
from src.backends.hybrid_backend import HybridBackend

from src.utils.logger import append_jsonl
from src.eval import oracle as ORACLE
from src.eval import metrics as eval_metrics
from src.eval.selectivity import compute_selectivity

from src.dataio.loaders import load_vectors, load_metadata
from src.dataio.validators import (
    parse_filters,
    validate_K,
    ensure_unit_l2,
    build_allowed_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "v2"
RESULTS_ROOT = REPO_ROOT / "results"


# ----------------------------
# Backend registry (v2 only)
# ----------------------------
def get_backend(
    name: str,
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    artifacts_root: Path,
    hybrid_early_stop: str | None = None,
) -> SearchBackend:

    if name == "hybrid":
        index_path = RESULTS_ROOT / "indexes" / "faiss_ivf.index"
        return HybridBackend(
            index_path=str(index_path),
            metadata_dir=str(artifacts_root),
            nprobe_start=16,
            nprobe_step=4,
            nprobe_max=None,
            hybrid_early_stop=hybrid_early_stop,
        )

    if name == "post_filter":
        return PostFilterBackend(str(artifacts_root), k_ladder=(200, 500, 1000))

    registry = {
        "exact": ExactBackend,
        "pre_filter": PreFilterBackend,
    }
    if name not in registry:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(registry) + ['post_filter','hybrid']}")

    return registry[name](vectors, metadata, name)


# ----------------------------
# Load queries (v2 only)
# ----------------------------
def load_queries(bucket_dir: Path) -> pd.DataFrame:
    qpath = bucket_dir / "queries.parquet"
    if not qpath.exists():
        raise FileNotFoundError(f"queries.parquet not found at {qpath}")

    q = pd.read_parquet(qpath)

    # v2 stores filters as JSON strings
    if "filters_json" in q.columns:
        q["filters"] = q["filters_json"].apply(json.loads)

    return q


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--backend", default="exact",
                    choices=["exact", "pre_filter", "post_filter", "hybrid"])

    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", required=True)

    ap.add_argument(
        "--hybrid-early-stop",
        type=str,
        default=None,
        help="Early-stop policy for hybrid backend"
    )

    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # clear json output between each run
    with open(out_path, "w", encoding="utf-8") as f:
        pass

    bucket_dir = ARTIFACTS_ROOT

    # --- load core dataset artifacts
    vectors = load_vectors(bucket_dir)
    metadata = load_metadata(bucket_dir)

    # --- load precomputed query vectors (v2 only)
    qv_path = bucket_dir / "query_vectors.npy"
    if not qv_path.exists():
        raise FileNotFoundError(f"Expected {qv_path} for v2 runs.")

    query_vectors = np.load(qv_path)
    N, D = vectors.shape

    # initialize oracle
    ORACLE.VECTORS = vectors
    ORACLE.METADATA = metadata

    validate_K(args.K, N)

    backend = get_backend(
        args.backend,
        vectors,
        metadata,
        artifacts_root=bucket_dir,
        hybrid_early_stop=args.hybrid_early_stop,
    )
    run_id = uuid.uuid4().hex[:10]

    qdf = load_queries(bucket_dir)
    if args.max_queries > 0:
        qdf = qdf.head(args.max_queries)

    for _, row in qdf.iterrows():
        qid = int(row.get("qid"))
        qvec = query_vectors[qid].astype(np.float32)
        qvec = ensure_unit_l2(qvec)

        filters = parse_filters(row.get("filters") or {})
        allowed_ids = build_allowed_ids(metadata, filters)
        selectivity = compute_selectivity(filters, metadata)

        for trial in range(args.repeats):
            ids, stats = backend.search(qvec, filters=filters, K=args.K)
            oracle_ids = ORACLE.brute_force(qvec, allowed_ids, args.K)
            recall_at_k = float(eval_metrics.compute_recall(ids, oracle_ids, args.K))

            def _safe_int(x):
                return int(x) if x is not None else None

            def _safe_float(x):
                return float(x) if x is not None else None

            out_row = {
                "qid": int(qid),
                "trial": int(trial),
                "method": backend.name,
                "K": int(args.K),
                "latency_ms": float(stats.get("latency_ms", 0.0)),
                "recall_at_k": float(recall_at_k),
                "filter_selectivity": float(selectivity),
                "scored_vectors": int(stats.get("scored_vectors", 0)),

                # FIXED conversions
                "lists_probed": _safe_int(stats.get("lists_probed")),
                "nprobe": _safe_int(stats.get("nprobe")),
                "kth_at_stop": _safe_float(stats.get("kth_at_stop")),
                "bound_at_stop": _safe_float(stats.get("bound_at_stop")),
                "probes_run": _safe_int(stats.get("probes_run")),

                "early_stop_used": bool(stats.get("early_stop_used", False)),
                "early_stop_reason": stats.get("early_stop_reason"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
            }

            # hybrid-specific extras
            if backend.name == "hybrid":
                extras = stats.get("extras", {}) or {}
                out_row["hybrid_early_stop_policy"] = extras.get("early_stop_policy")
                out_row["hybrid_global_bound"] = extras.get("global_bound")
                out_row["hybrid_has_probe_order"] = bool(extras.get("has_probe_order"))
                out_row["hybrid_has_allowed_counts_per_list"] = bool(
                    extras.get("has_allowed_counts_per_list")
                )

            _PLOT_ROWS.append(out_row)
            append_jsonl(out_row, out_path)

    # plots
    if _PLOT_ROWS and False:
        df_plot = pd.DataFrame(_PLOT_ROWS)
        plt.figure(figsize=(8, 6))
        plt.scatter(df_plot["filter_selectivity"], df_plot["latency_ms"], s=10)
        plt.xlabel("Filter Selectivity")
        plt.ylabel("Latency (ms)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(out_path) + "_selectivity_vs_latency.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(df_plot["filter_selectivity"], df_plot["recall_at_k"], s=10)
        plt.xlabel("Filter Selectivity")
        plt.ylabel("Recall@K")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(out_path) + "_selectivity_vs_recall.png")
        plt.close()

        print("[PLOT] Saved plots next to results.")

    print(f"[OK] {len(qdf)} queries from v2 via '{args.backend}' → {args.out}")


if __name__ == "__main__":
    main()
