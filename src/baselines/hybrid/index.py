# baselines/hybrid/index.py

from __future__ import annotations
from typing import Optional
import os
import numpy as np
import faiss

# we keep index separate from the main artifacts
DEFAULT_FULL_INDEX_DIR = "/artifacts/full/v1/hybrid"
DEFAULT_FULL_INDEX_PATH = f"{DEFAULT_FULL_INDEX_DIR}/faiss_ivf.index"


def build_ivf_index(
    vectors: np.ndarray,
    nlist: int,
    metric: str = "ip",
    save_path: Optional[str] = None,
) -> faiss.IndexIVFFlat:
    """
    Build and (optionally) persist a FAISS IndexIVFFlat over the given vectors.

    Args:
        vectors: (N, D) float32, assumed L2-normalized if using IP.
        nlist: number of IVF lists.
        metric: "ip" or "l2".
        save_path: if provided, serialize index to this path. If None, uses
            /artifacts/full/v1/hybrid/faiss_ivf.index.

    Returns:
        Trained and populated faiss.IndexIVFFlat.
    """
    # ensure correct dtype / layout
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = np.ascontiguousarray(vectors)
    n, d = vectors.shape

    # don't let nlist exceed N
    nlist = min(nlist, n)

    if metric == "ip":
        quantizer = faiss.IndexFlatIP(d)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    else:
        quantizer = faiss.IndexFlatL2(d)
        faiss_metric = faiss.METRIC_L2

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss_metric)

    # train
    index.train(vectors)
    if not index.is_trained:
        raise RuntimeError("IVF index failed to train")

    # add with IDs aligned to 0..N-1 (your pipeline guarantees this)
    ids = np.arange(n, dtype=np.int64)
    index.add_with_ids(vectors, ids)

    # pick default path under the new folder
    if save_path is None:
        save_path = DEFAULT_FULL_INDEX_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)

    return index


def load_ivf_index(path: str = DEFAULT_FULL_INDEX_PATH) -> faiss.IndexIVFFlat:
    """
    Load a previously persisted FAISS IVF index.

    Args:
        path: filesystem path to .index file.
    """
    return faiss.read_index(path)
