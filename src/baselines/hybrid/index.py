# src/baselines/hybrid/index.py
"""
Hybrid index utilities.

This module does three things:

1. Build a FAISS IndexIVFFlat over the canonical vectors from artifacts
   (v1 or v2), using the official loaders from dataio.
2. Load an existing FAISS index from a path chosen by the caller.
3. Expose IVF internals (centroids, list sizes, id→list mapping) as NumPy
   arrays for downstream hybrid modules (ordering, early-stop, etc.).
"""

from __future__ import annotations

from typing import Optional, Tuple
import os

import numpy as np
import faiss
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_DIR = str(PROJECT_ROOT / "results" / "indexes")
DEFAULT_INDEX_PATH = f"{DEFAULT_INDEX_DIR}/faiss_ivf.index"

# backward-compat for older week-4 code that imported this name
DEFAULT_FULL_INDEX_PATH = DEFAULT_INDEX_PATH


def _import_loaders():
    """
    Import the canonical artifact loaders defined in src/dataio/loaders.py.
    We keep this in a helper so importing this module doesn't immediately fail
    if PYTHONPATH isn't set up yet.
    """
    try:
        from src.dataio.loaders import load_vectors, load_vectors_meta
    except ImportError as e:
        raise ImportError(
            "Cannot import dataio.loaders. Run from project root or add 'src/' to PYTHONPATH."
        ) from e
    return load_vectors, load_vectors_meta


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_ivf_index(
    vectors: np.ndarray,
    nlist: int,
    metric: str = "ip",
    save_path: Optional[str] = None,
) -> faiss.IndexIVFFlat:
    """
    Build and persist a FAISS IndexIVFFlat over the given vectors.

    Args:
        vectors: (N, D) float32, contiguous, ids implied 0..N-1.
        nlist: number of IVF lists (will be clamped to N).
        metric: "ip" or "l2".
        save_path: optional filesystem path to serialize the index. If None,
            we will write to /results/indexes/faiss_ivf.index.

    Returns:
        Trained and populated faiss.IndexIVFFlat.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = np.ascontiguousarray(vectors)
    n, d = vectors.shape

    # clamp nlist
    nlist = min(max(1, int(nlist)), n)

    # choose metric
    if metric == "ip":
        quantizer = faiss.IndexFlatIP(d)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        quantizer = faiss.IndexFlatL2(d)
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss_metric)

    # train on full canonical vectors
    index.train(vectors)
    if not index.is_trained:
        raise RuntimeError("IVF index failed to train")

    # add with IDs aligned to 0..N-1
    ids = np.arange(n, dtype=np.int64)
    index.add_with_ids(vectors, ids)

    # persist (to results/, not artifacts/) unless caller says otherwise
    if save_path is None:
        save_path = DEFAULT_INDEX_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)

    return index


def build_ivf_index_from_artifacts(
    artifacts_root: str = "/artifacts",
    bucket: str = "v2",
    nlist: int = 1024,
    metric: str = "ip",
    save_path: Optional[str] = None,
) -> faiss.IndexIVFFlat:
    """
    Convenience wrapper: load canonical vectors from artifacts/<bucket>/ and
    build an IVF index over them, using the official loaders.

    This matches your actual loaders, which take a single artifacts_root
    (e.g. '/artifacts/v2') and look for vectors.npy, vectors.meta.json, etc.

    Args:
        artifacts_root: base artifacts dir, typically "/artifacts".
        bucket: "v1" or "v2".
        nlist: IVF list count.
        metric: "ip" or "l2".
        save_path: where to write the index. If None, defaults to /results/indexes/...

    Returns:
        A trained and populated FAISS index.
    """
    load_vectors, load_vectors_meta = _import_loaders()

    bucket_dir = os.path.join(artifacts_root, bucket)

    # load canonical vectors via the real loader (one arg only)
    vectors = load_vectors(bucket_dir)

    # (optional) we could inspect meta here, but load_vectors_meta(...) is mostly
    # for consistency / future checks and compliance with the data spec.
    _ = load_vectors_meta(bucket_dir)

    return build_ivf_index(
        vectors=vectors,
        nlist=nlist,
        metric=metric,
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_ivf_index(path: str = DEFAULT_INDEX_PATH) -> faiss.IndexIVFFlat:
    """
    Load a previously persisted FAISS IVF index.

    Args:
        path: filesystem path to .index file. Defaults to /results/indexes/faiss_ivf.index
              so that harness/backends can just call this without touching artifacts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAISS index not found at {path}. Build it first using "
            "`build_ivf_index_from_artifacts(...)` or `build_ivf_index(...)`."
        )
    index = faiss.read_index(path)
    return index


# ---------------------------------------------------------------------------
# IVF internals (Person A – Task A1)
#
# These helpers expose the partition structure of the IVF index in a
# NumPy-friendly way. They are intentionally PURE (no artifacts / filters):
#
#   - get_ivf_centroids:  (L, D) coarse centroids for each list.
#   - get_ivf_list_sizes: (L,)   number of vectors in each list.
#   - get_ivf_id_to_list_map: (N,) mapping id -> list_id.
#
# Together with metadata-derived allow-lists, these are the building blocks
# for predicate-aware, list-level scheduling (à la VBASE).
# ---------------------------------------------------------------------------

def get_ivf_centroids(index: faiss.IndexIVFFlat) -> np.ndarray:
    """
    Extract IVF coarse quantizer centroids as a NumPy array.

    Returns:
        centroids: (L, D) float32 array where L = index.nlist and D = index.d.
    """
    if not isinstance(index, faiss.IndexIVFFlat):
        raise TypeError("get_ivf_centroids currently supports faiss.IndexIVFFlat only")

    nlist = int(index.nlist)
    d = int(index.d)

    centroids = np.empty((nlist, d), dtype=np.float32)
    # quantizer holds one centroid per list; reconstruct(i) fetches it
    for lid in range(nlist):
        index.quantizer.reconstruct(lid, centroids[lid])

    return centroids


def get_ivf_list_sizes(index: faiss.IndexIVFFlat) -> np.ndarray:
    """
    Extract IVF list sizes as a NumPy array.

    Returns:
        list_sizes: (L,) int64 array where L = index.nlist and
                    list_sizes[l] = number of vectors in list l.
    """
    if not isinstance(index, faiss.IndexIVFFlat):
        raise TypeError("get_ivf_list_sizes currently supports faiss.IndexIVFFlat only")

    ivf = faiss.extract_index_ivf(index)
    invlists = ivf.invlists
    if invlists is None:
        # Unpopulated index (no vectors added yet)
        return np.zeros(int(index.nlist), dtype=np.int64)

    nlist = int(index.nlist)
    list_sizes = np.empty(nlist, dtype=np.int64)
    for lid in range(nlist):
        list_sizes[lid] = invlists.list_size(lid)

    return list_sizes


__all__ = [
    "build_ivf_index",
    "build_ivf_index_from_artifacts",
    "load_ivf_index",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_FULL_INDEX_PATH",
    # IVF internals
    "get_ivf_centroids",
    "get_ivf_list_sizes",
]