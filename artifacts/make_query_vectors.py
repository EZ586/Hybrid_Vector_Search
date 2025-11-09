#!/usr/bin/env python3
"""
Materialize query vectors for a bucket (v2-style).

Usage:
    python artifacts/make_query_vectors.py --bucket_dir artifacts/v2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # lazy import
    return SentenceTransformer(model_name)


def build_text_from_filters(filters: Dict[str, Any]) -> str:
    """
    Turn a v2-style filter dict into a stable, short string that we can embed.
    Example:
        {"city": {"eq": "Clearwater"}, "stars": {"between": [3.5, 4.2]}}
    ->  "city eq Clearwater; stars between 3.5 4.2;"
    """
    parts: List[str] = []
    # sort keys for determinism
    for col in sorted(filters.keys()):
        cond = filters[col]
        if not isinstance(cond, dict):
            continue
        # there should be exactly 1 op per your generator
        for op, val in cond.items():
            if isinstance(val, list):
                val_str = " ".join(str(x) for x in val)
            else:
                val_str = str(val)
            parts.append(f"{col} {op} {val_str}".strip())
    return "; ".join(parts) + ";" if parts else "auto-filter-query"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bucket_dir",
        default="artifacts/v2",
        help="Artifact bucket containing queries.parquet and vectors.meta.json",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output path (default: <bucket_dir>/query_vectors.npy)",
    )
    args = ap.parse_args()

    bucket = Path(args.bucket_dir)
    qpath = bucket / "queries.parquet"
    meta_path = bucket / "vectors.meta.json"

    if not qpath.exists():
        raise FileNotFoundError(f"queries.parquet not found at {qpath}")
    if not meta_path.exists():
        raise FileNotFoundError(f"vectors.meta.json not found at {meta_path}")

    # 1) load queries
    qdf = pd.read_parquet(qpath)
    if "qid" not in qdf.columns:
        raise ValueError("queries.parquet must contain 'qid' column")

    # sort by qid so row index == qid
    qdf = qdf.sort_values("qid").reset_index(drop=True)

    # 2) load model name + dimension
    with open(meta_path, "r") as f:
        vmeta = json.load(f)
    model_name = vmeta["model"]

    model = load_model(model_name)

    # 3) build texts to embed
    texts: List[str] = []
    has_filters_json = "filters_json" in qdf.columns
    for _, row in qdf.iterrows():
        if has_filters_json:
            fdict = json.loads(row["filters_json"])
            text = build_text_from_filters(fdict)
        else:
            # v1-style
            text = str(row.get("qtext", "")).strip() or "empty query"
        texts.append(text)

    # 4) embed
    emb = model.encode(texts, normalize_embeddings=False)
    emb = np.asarray(emb, dtype="float32")

    # 5) L2 normalize
    emb = l2_normalize(emb)

    # 6) save
    out_path = Path(args.out) if args.out else (bucket / "query_vectors.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    print(f"[OK] wrote {emb.shape} to {out_path}")


if __name__ == "__main__":
    main()