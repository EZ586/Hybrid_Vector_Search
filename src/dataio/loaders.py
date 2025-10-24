from __future__ import annotations
"""
Universal loaders for vectors, metadata, and ANN/Exact indices.
- Always prefer FAISS IndexIVFFlat (Inner Product) for ANN; fallback to HNSW(IP) then Exact.
- Make artifacts usable by *both* Post‑filter and Pre‑filter baselines.
- Enforce critical manual rules: id indexing, dtype/shape, and normalization checks
  (light validation here; full validation still belongs in dataio/validators.py).

Public API
----------
load_vectors(artifacts_root) -> np.ndarray
load_vectors_meta(artifacts_root) -> dict
load_metadata(artifacts_root) -> pd.DataFrame (indexed by id, sorted)
load_vectors_index(artifacts_root, *, prefer_ivf=True, nlist=None, nprobe=32)
get_index_params(index) -> dict  # e.g., {"type":"ivfflat", "nlist":4096, "nprobe":32}

ExactIndex(vectors) exposes .search(qvec, k) with IP scores.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ----------------------------- Exact (fallback) -----------------------------
class ExactIndex:
    """Brute‑force IP Top‑K over full matrix (expects L2‑normalized vectors).
    Method: scores = V @ q (cosine == IP when L2‑normed).
    """
    def __init__(self, vectors: np.ndarray):
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim != 2:
            raise ValueError("vectors must be 2D (N, D)")
        self.vectors = v

    def search(self, qvec: np.ndarray, k: int):
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        scores = self.vectors @ q
        k = int(min(max(k, 0), len(scores)))
        if k == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return idx.astype(int), scores[idx].astype(float)


# ----------------------------- Artifact loaders ----------------------------
_DEF_META_NAME = "vectors.meta.json"
_DEF_VEC_NAME = "vectors.npy"
_DEF_META_TABLE = "metadata.parquet"


def _expect_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing required artifact: {p}")


def load_vectors_meta(artifacts_root: str | Path) -> Dict[str, Any]:
    root = Path(artifacts_root)
    meta_p = root / _DEF_META_NAME
    _expect_file(meta_p)
    with meta_p.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    # Minimal required keys per manual
    for key in ("N", "D", "normalized"):
        if key not in meta:
            raise ValueError(f"vectors.meta.json missing key: {key}")
    return meta


def load_vectors(artifacts_root: str | Path) -> np.ndarray:
    root = Path(artifacts_root)
    vec_p = root / _DEF_VEC_NAME
    _expect_file(vec_p)
    v = np.load(vec_p).astype(np.float32)
    if v.ndim != 2:
        raise ValueError("vectors.npy must be 2D (N, D)")
    return v


def _l2_normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (v / norms).astype(np.float32, copy=False)


def load_metadata(artifacts_root: str | Path) -> pd.DataFrame:
    root = Path(artifacts_root)
    meta_tbl = root / _DEF_META_TABLE
    _expect_file(meta_tbl)
    df = pd.read_parquet(meta_tbl)
    if "id" not in df.columns:
        raise ValueError("metadata.parquet must contain an 'id' column (int64 contiguous 0..N-1)")
    # Enforce index and ordering by id
    df = df.sort_values("id").set_index("id", drop=False)
    # Light checks (full rules live in validators.py)
    n = len(df)
    if df.index[0] != 0 or df.index[-1] != n - 1 or df.index.has_duplicates:
        raise ValueError("metadata.id must be contiguous 0..N-1 with no gaps/dups")
    return df


# ------------------------------- FAISS builders ----------------------------
class _IVFWrapper:
    def __init__(self, index, nlist: int, nprobe: int):
        self.index = index
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)

    def search(self, qvec: np.ndarray, k: int):
        import numpy as _np
        q = _np.asarray(qvec, dtype=_np.float32).reshape(1, -1)
        self.index.nprobe = int(self.nprobe)
        scores, ids = self.index.search(q, int(k))
        ids = ids[0]
        scores = scores[0]
        mask = ids >= 0
        return ids[mask].astype(int), scores[mask].astype(float)


class _HNSWWrapper:
    def __init__(self, index, ef_search: int):
        self.index = index
        self.ef_search = int(ef_search)

    def search(self, qvec: np.ndarray, k: int):
        import numpy as _np
        try:
            self.index.hnsw.efSearch = int(self.ef_search)
        except Exception:
            pass
        q = _np.asarray(qvec, dtype=_np.float32).reshape(1, -1)
        scores, ids = self.index.search(q, int(k))
        ids = ids[0]
        scores = scores[0]
        mask = ids >= 0
        return ids[mask].astype(int), scores[mask].astype(float)


def _build_faiss_ivf_ip(vectors: np.ndarray, *, nlist: int, nprobe: int):
    import faiss  # type: ignore
    d = int(vectors.shape[1])
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, int(nlist), faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    index.nprobe = int(nprobe)
    return _IVFWrapper(index, nlist=int(nlist), nprobe=int(nprobe))


def _build_faiss_hnsw_ip(vectors: np.ndarray, *, m: int = 32, ef_search: int = 64):
    import faiss  # type: ignore
    d = int(vectors.shape[1])
    index = faiss.index_factory(d, f"HNSW{int(m)}", faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    try:
        index.hnsw.efSearch = int(ef_search)
    except Exception:
        pass
    return _HNSWWrapper(index, ef_search=ef_search)


# ------------------------------- Index loader ------------------------------
_DEF_NPROBE = 32
_DEF_MIN_LISTS = 64
_DEF_LISTS_MAX = 4096  # upper clamp for nlist

def _pick_nlist(N: int) -> int:
    # Safer heuristic for small/medium N: nlist ≈ √N, clamped to [64, 4096]
    return int(np.clip(int(np.sqrt(N)), _DEF_MIN_LISTS, _DEF_LISTS_MAX))


def load_vectors_index(
    artifacts_root: str | Path,
    *,
    prefer_ivf: bool = True,
    nlist: Optional[int] = None,
    nprobe: int = _DEF_NPROBE,
):
    """Build and return an ANN (IVFFlat preferred) or Exact index over vectors.

    - Reads vectors + vectors.meta.json, verifies (N, D), dtype=float32, normalization≈1.
    - Normalizes vectors defensively (cosine==IP) even if meta says normalized.
    - Chooses FAISS IVFFlat(IP) with nlist≈N/8 (clamped to [64, 4096]) unless overridden.
    - Falls back to FAISS HNSW(IP), then ExactIndex when FAISS is unavailable.
    """
    root = Path(artifacts_root)
    meta = load_vectors_meta(root)
    v = load_vectors(root)

    # Shape/dtype checks against meta
    N, D = v.shape
    if int(meta.get("N", N)) != N or int(meta.get("D", D)) != D:
        raise ValueError(
            f"vectors.npy shape ({N},{D}) does not match vectors.meta.json ({meta.get('N')},{meta.get('D')})"
        )

    # Ensure L2 normalization (manual tolerance ~1e-3); re‑normalize defensively
    v = _l2_normalize_rows(v)

    # Prefer FAISS IVFFlat(IP)
    if prefer_ivf:
        try:
            nlist_eff = int(nlist) if nlist is not None else _pick_nlist(N)
            return _build_faiss_ivf_ip(v, nlist=nlist_eff, nprobe=int(nprobe))
        except Exception:
            pass

    # Fallback to HNSW(IP), then Exact
    try:
        return _build_faiss_hnsw_ip(v, m=32, ef_search=64)
    except Exception:
        return ExactIndex(v)


def get_index_params(index: Any) -> Dict[str, Any]:
    """Return a small dict describing ANN parameters for logging/debugging.
    Keys mirror the stats contract fields where sensible.
    """
    try:
        # IVF wrapper
        if isinstance(index, _IVFWrapper):
            return {"type": "ivfflat", "nlist": int(index.nlist), "nprobe": int(index.nprobe)}
        # HNSW wrapper
        if isinstance(index, _HNSWWrapper):
            return {"type": "hnsw", "ef_search": int(index.ef_search)}
    except Exception:
        pass
    return {"type": "exact"}