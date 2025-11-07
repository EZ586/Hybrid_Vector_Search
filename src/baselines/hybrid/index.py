# baselines/hybrid/index.py

from typing import Optional
import numpy as np
import faiss


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
        save_path: if provided, serialize index to this path.

    Returns:
        Trained and populated faiss.IndexIVFFlat.
    """
    # TODO: create quantizer, index, train, add, and optionally write_index
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist,faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2)
    index.train(vectors)
    ids = np.arange(vectors.shape[0])
    index.add_with_ids(vectors, ids)
    if save_path:
        faiss.write_index(index, save_path)
    return index



def load_ivf_index(path: str) -> faiss.IndexIVFFlat:
    """
    Load a previously persisted FAISS IVF index.

    Args:
        path: filesystem path to .index file.

    Returns:
        faiss.IndexIVFFlat instance.
    """
    # TODO: return faiss.read_index(path)
    return faiss.read_index(path)
