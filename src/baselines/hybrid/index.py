# src/baselines/hybrid/index.py
"""
Hybrid index utilities (HNSW version).

This module replaces the IVF index builder with a FAISS HNSW index builder.

It does the following:

1. Build a FAISS IndexHNSWFlat over canonical vectors from artifacts.
2. Load an existing FAISS HNSW index from disk.
"""

from __future__ import annotations

from typing import Optional
import os
from pathlib import Path

import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_DIR = str(PROJECT_ROOT / "results" / "indexes")
DEFAULT_INDEX_PATH = f"{DEFAULT_INDEX_DIR}/faiss_hnsw.index"

# Backward compat name
DEFAULT_FULL_INDEX_PATH = DEFAULT_INDEX_PATH


def _import_loaders():
    """
    Import canonical artifact loaders defined in src/dataio/loaders.py.
    """
    try:
        from src.dataio.loaders import load_vectors, load_vectors_meta
    except ImportError as e:
        raise ImportError(
            "Cannot import dataio.loaders. Run from project root or add 'src/' to PYTHONPATH."
        ) from e
    return load_vectors, load_vectors_meta


# ---------------------------------------------------------------------
#  HNSW Index Builder
# ---------------------------------------------------------------------

def build_hnsw_index(
    vectors: np.ndarray,
    M: int = 32,
    ef_construction: int = 200,
    metric: str = "ip",
    save_path: Optional[str] = None,
) -> faiss.IndexHNSWFlat:
    """
    Build and optionally persist a FAISS HNSW index over the given vectors.

    Args:
        vectors: (N, D) float32 canonical vectors.
        M: Number of neighbors per node (graph degree).
        ef_construction: HNSW build accuracy.
        metric: "ip" or "l2".
        save_path: where to persist the index.

    Returns:
        Trained and populated faiss.IndexHNSWFlat object.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = np.ascontiguousarray(vectors)
    n, d = vectors.shape

    if metric == "ip":
        # normalize vectors to unit length so IP = cosine similarity
        faiss.normalize_L2(vectors)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # ------------------------------------------------------------------
    # Create HNSW index
    # ------------------------------------------------------------------
    index = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index.hnsw.efConstruction = ef_construction

    # Add vectors
    index.add(vectors)

    # Recommended default search parameter
    index.hnsw.efSearch = 128

    # ------------------------------------------------------------------
    # Persist index
    # ------------------------------------------------------------------
    if save_path is None:
        save_path = DEFAULT_INDEX_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)

    return index


def build_hnsw_index_from_artifacts(
    artifacts_root: str = "/artifacts",
    bucket: str = "v2",
    M: int = 32,
    ef_construction: int = 200,
    metric: str = "ip",
    save_path: Optional[str] = None,
) -> faiss.IndexHNSWFlat:
    """
    Convenience wrapper: load canonical vectors from artifacts/<bucket>/ and
    build an HNSW index over them.

    Args:
        artifacts_root: base artifacts dir (e.g. "/artifacts").
        bucket: "v1" or "v2".
        M: graph degree.
        ef_construction: HNSW build budget.
        metric: "ip" or "l2".
        save_path: path to save the index.

    Returns:
        Populated FAISS HNSW index.
    """
    load_vectors, load_vectors_meta = _import_loaders()

    bucket_dir = os.path.join(artifacts_root, bucket)
    vectors = load_vectors(bucket_dir)
    _ = load_vectors_meta(bucket_dir)  # not strictly needed

    return build_hnsw_index(
        vectors=vectors,
        M=M,
        ef_construction=ef_construction,
        metric=metric,
        save_path=save_path,
    )


# ---------------------------------------------------------------------
#  Loader
# ---------------------------------------------------------------------

def load_hnsw_index(path: str = DEFAULT_INDEX_PATH) -> faiss.IndexHNSWFlat:
    """
    Load a previously saved HNSW index.

    Args:
        path: path to .index file.

    Returns:
        FAISS IndexHNSWFlat.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HNSW index not found at {path}. Build it via "
            "`build_hnsw_index_from_artifacts(...)` or `build_hnsw_index(...)`."
        )
    return faiss.read_index(path)


__all__ = [
    "build_hnsw_index",
    "build_hnsww_index_from_artifacts",
    "load_hnsw_index",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_FULL_INDEX_PATH",
]


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Build a FAISS HNSW index from artifacts.")
    parser.add_argument("--artifacts", type=str, default="/artifacts",
                        help="Base artifacts directory.")
    parser.add_argument("--bucket", type=str, default="v2",
                        help="Subfolder (bucket) name within artifacts.")
    parser.add_argument("--M", type=int, default=32,
                        help="HNSW graph degree (default 32).")
    parser.add_argument("--efC", type=int, default=200,
                        help="HNSW efConstruction (default 200).")
    parser.add_argument("--metric", type=str, choices=["ip", "l2"],
                        default="ip", help="Distance metric (default ip).")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save index.")

    args = parser.parse_args()

    print("Building FAISS HNSW index...")
    print(f"  Artifacts dir : {args.artifacts}")
    print(f"  Bucket        : {args.bucket}")
    print(f"  M             : {args.M}")
    print(f"  efConstruction: {args.efC}")
    print(f"  Metric        : {args.metric}")

    try:
        index = build_hnsw_index_from_artifacts(
            artifacts_root=args.artifacts,
            bucket=args.bucket,
            M=args.M,
            ef_construction=args.efC,
            metric=args.metric,
            save_path=args.save,
        )

        print(f"✅ Index built successfully!")
        print(f"   Index type  : {type(index)}")
        print(f"   ntotal      : {index.ntotal}")
        print(f"   dimension   : {index.d}")

        out_path = args.save or DEFAULT_INDEX_PATH
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"📦 Saved index → {out_path}  (built {timestamp})")

    except Exception as e:
        import traceback
        print(f"❌ Failed to build HNSW index ({type(e).__name__}): {e}")
        traceback.print_exc()
