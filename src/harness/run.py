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

from src.backends.exact_backend import ExactBackend
from src.backends.prefilter_backend import PreFilterBackend
from src.backends.post_filter_backend import PostFilterBackend
from src.backends.backend_interface import SearchBackend
from src.utils.logger import append_jsonl
from src.eval import oracle as ORACLE
from src.eval import metrics as eval_metrics
from src.eval.selectivity import compute_selectivity

# central loaders & validators
from src.dataio.loaders import load_vectors, load_metadata, load_vectors_meta
from src.dataio.validators import (
    parse_filters,
    validate_K,
    ensure_unit_l2,
    build_allowed_ids,
)

# NEW: hybrid backend
from src.backends.hybrid_backend import HybridBackend  # adjust if your path differs

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
RESULTS_ROOT = REPO_ROOT / "results"


# ----------------------------
# Embedding hook (v1 only)
# ----------------------------
_EMBED_MODEL_NAME: str | None = None
_EMBED_MODEL = None


def _get_embedder(model_name: str):
    global _EMBED_MODEL, _EMBED_MODEL_NAME
    if _EMBED_MODEL is None or _EMBED_MODEL_NAME != model_name:
        from sentence_transformers import SentenceTransformer  # lazy import

        _EMBED_MODEL = SentenceTransformer(model_name)
        _EMBED_MODEL_NAME = model_name
    return _EMBED_MODEL


def embed_qtext(qtext: str, model_name: str) -> np.ndarray:
    if not isinstance(qtext, str) or not qtext.strip():
        raise ValueError("qtext must be a non-empty string")
    model = _get_embedder(model_name)
    vec = model.encode([qtext], normalize_embeddings=False)
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)  # (D,)
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        raise ValueError("Embedding norm is zero; cannot normalize")
    return (vec / n).astype(np.float32)


# ----------------------------
# Backend registry
# ----------------------------
def get_backend(
    name: str, vectors: np.ndarray, metadata: pd.DataFrame, *, artifacts_root: Path
) -> SearchBackend:
    # special-case hybrid because it uses a persisted FAISS index and reloads metadata
    if name == "hybrid":
        index_path = RESULTS_ROOT / "indexes" / "faiss_ivf.index"
        metadata_dir = artifacts_root
        return HybridBackend(
            index_path=str(index_path),
            metadata_dir=str(metadata_dir),
            nprobe_start=4,
            nprobe_step=4,
            nprobe_max=64,
        )

    if name == "post_filter":
        return PostFilterBackend(str(artifacts_root), k_ladder=(200, 500, 1000))

    registry = {
        "exact": ExactBackend,
        "pre_filter": PreFilterBackend,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(registry) + ['post_filter', 'hybrid']}"
        )
    return registry[name](vectors, metadata, name)


# ----------------------------
# Queries loader (v1 vs v2)
# ----------------------------
def load_queries(bucket_dir: Path) -> pd.DataFrame:
    qpath = bucket_dir / "queries.parquet"
    if not qpath.exists():
        raise FileNotFoundError(f"queries.parquet not found at {qpath}")
    q = pd.read_parquet(qpath)

    # v2 writes filters_json, v1 writes filters
    if "filters" not in q.columns and "filters_json" in q.columns:
        q["filters"] = q["filters_json"].apply(json.loads)

    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts",
        default="v1",
        choices=["v1", "v2"],
        help="Which artifact bucket under ./artifacts/ to use",
    )
    ap.add_argument(
        "--backend",
        default="exact",
        choices=["exact", "pre_filter", "post_filter", "hybrid"],
    )
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="If >0, only run the first N queries from the parquet",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Path to JSONL results file to write"
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to run each query for stability checks",
    )
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # pick bucket
    bucket_dir = ARTIFACTS_ROOT / args.artifacts

    # --- load core artifacts from THAT bucket ---
    vectors = load_vectors(bucket_dir)
    metadata = load_metadata(bucket_dir)
    vectors_meta = load_vectors_meta(bucket_dir)
    model_name = vectors_meta["model"]

    # initialize oracle globals for exact recall
    ORACLE.VECTORS = vectors
    ORACLE.METADATA = metadata

    N, D = vectors.shape
    validate_K(args.K, N)

    # backend (now hybrid is supported)
    backend = get_backend(
        args.backend, vectors, metadata, artifacts_root=bucket_dir
    )
    run_id = uuid.uuid4().hex[:10]

    # queries
    qdf = load_queries(bucket_dir)
    if args.max_queries > 0:
        qdf = qdf.head(args.max_queries)

    # v2 may have pre-embedded query vectors
    query_vectors = None
    if args.artifacts == "v2":
        qv_path = bucket_dir / "query_vectors.npy"
        if not qv_path.exists():
            raise FileNotFoundError(
                f"v2 run requires pre-embedded queries at {qv_path}, but file was not found."
            )
        query_vectors = np.load(qv_path)
        if query_vectors.shape[0] < len(qdf):
            raise ValueError(
                f"query_vectors.npy has only {query_vectors.shape[0]} rows but queries.parquet has {len(qdf)}"
            )
        if query_vectors.shape[1] != D:
            raise ValueError(
                f"query_vectors dim {query_vectors.shape[1]} != dataset dim {D}"
            )

    for _, row in qdf.iterrows():
        qid = int(row.get("qid", 0))
        qtext = str(row.get("qtext", ""))

        # --- choose query vector strategy based on artifact bucket ---
        if args.artifacts == "v1":
            qvec = embed_qtext(qtext, model_name) if qtext else vectors[qid]
        else:
            qvec = query_vectors[qid]

        # final safety: L2 normalize
        qvec = ensure_unit_l2(qvec)
        if qvec.shape[0] != D:
            raise ValueError(f"Query dim {qvec.shape[0]} != dataset dim {D}")

        # filters
        raw_filter: dict[str, Any] = row.get("filters", {}) or {}
        filters = parse_filters(raw_filter)
        allowed_ids = build_allowed_ids(metadata, filters)
        selectivity = compute_selectivity(filters, metadata)

        for trial in range(args.repeats):
            ids, stats = backend.search(qvec, filters=filters, K=args.K)
            oracle_ids = ORACLE.brute_force(qvec, allowed_ids, args.K)
            recall_at_k = float(eval_metrics.compute_recall(ids, oracle_ids, args.K))

            # JSON-safe normalization
            latency_ms = float(stats.get("latency_ms", 0.0))
            scored_vectors = int(stats.get("scored_vectors", 0))

            lists_probed = stats.get("lists_probed", None)
            if isinstance(lists_probed, (np.integer, np.floating)):
                lists_probed = int(lists_probed)

            nprobe = stats.get("nprobe", None)
            if isinstance(nprobe, (np.integer, np.floating)):
                nprobe = int(nprobe)

            kth_at_stop = stats.get("kth_at_stop", None)
            if isinstance(kth_at_stop, (np.integer, np.floating)):
                kth_at_stop = float(kth_at_stop)

            bound_at_stop = stats.get("bound_at_stop", None)
            if isinstance(bound_at_stop, (np.integer, np.floating)):
                bound_at_stop = float(bound_at_stop)

            out_row = {
                "qid": int(qid),
                "trial": int(trial),
                "method": backend.name,
                "K": int(args.K),
                "latency_ms": latency_ms,
                "recall_at_k": float(recall_at_k),
                "filter_selectivity": float(selectivity),
                "scored_vectors": scored_vectors,
                "lists_probed": lists_probed,
                "nprobe": nprobe,
                "kth_at_stop": kth_at_stop,
                "bound_at_stop": bound_at_stop,
                "notes": stats.get("notes", None),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
            }
            append_jsonl(out_row, out_path)

    print(
        f"[OK] {len(qdf)} queries from '{args.artifacts}' via '{args.backend}' → {args.out}"
    )


if __name__ == "__main__":
    main()