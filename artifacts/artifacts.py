#!/usr/bin/env python3
# yelp_pipeline.py
import os, json, ast, numpy as np, pandas as pd
from datetime import datetime, timezone
from typing import Optional

# ---------- Config ----------
FILE = "data/yelp_academic_dataset_business.json"
SEED = 42
ART = "artifacts"
DEV = f"{ART}/dev/v1"
FULL = f"{ART}/full/v1"
DEV_N = 10_000
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-D

# Required / optional columns (metadata schema)
MUST = ["state","city","stars","review_count","RestaurantsPriceRange2"]
OPT  = ["categories","latitude","longitude","is_open"]

os.makedirs(DEV, exist_ok=True); os.makedirs(FULL, exist_ok=True)

# ---------- Helpers ----------
def extract_price(attr_val):
    if pd.isna(attr_val):
        return pd.NA
    try:
        if isinstance(attr_val, dict):
            d = attr_val
        elif isinstance(attr_val, str):
            d = ast.literal_eval(attr_val)
        else:
            return pd.NA
        val = d.get("RestaurantsPriceRange2", pd.NA)
        if val in (None, "", "None"):
            return pd.NA
        return int(val)
    except Exception:
        return pd.NA

def validate_schema(df: pd.DataFrame):
    for c in ["id","state","city","stars","review_count","RestaurantsPriceRange2"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    assert df["id"].dtype == "int64"
    assert df["stars"].dtype == "float32"
    assert df["review_count"].dtype == "int32"
    assert str(df["RestaurantsPriceRange2"].dtype) == "Int8"
    for col in ["state","city"]:
        assert str(df[col].dtype) == "string"
    for c in ["id","state","stars","review_count"]:
        if df[c].isna().any():
            raise ValueError(f"Column {c} has nulls; required non-null.")
    n = len(df)
    ids = df["id"].to_numpy()
    if set(ids) != set(range(n)):
        raise ValueError("id must be contiguous 0..N-1 with no gaps/dupes.")

# ---------- Build metadata artifacts ----------
def build_metadata_artifacts(src_json: str = FILE):
    df = pd.read_json(src_json, lines=True)

    df = df.copy()
    df["RestaurantsPriceRange2"] = df.get("attributes").apply(extract_price)

    keep_cols = [
        "state","city","stars","review_count","RestaurantsPriceRange2",
        "categories","latitude","longitude","is_open","name"
    ]
    meta = df.loc[:, [c for c in keep_cols if c in df.columns]].copy()

    # Dtypes
    if "state" in meta: meta["state"] = meta["state"].astype("string")
    if "city"  in meta: meta["city"]  = meta["city"].astype("string")
    meta["stars"]        = pd.to_numeric(meta["stars"], errors="coerce").astype("float32")
    meta["review_count"] = pd.to_numeric(meta["review_count"], errors="coerce").astype("Int32")
    meta["RestaurantsPriceRange2"] = pd.to_numeric(meta["RestaurantsPriceRange2"], errors="coerce").astype("Int8")
    if "is_open" in meta:
        meta["is_open"] = pd.to_numeric(meta["is_open"], errors="coerce").astype("Int8")
    if "latitude" in meta:
        meta["latitude"] = pd.to_numeric(meta["latitude"], errors="coerce").astype("float32")
    if "longitude" in meta:
        meta["longitude"] = pd.to_numeric(meta["longitude"], errors="coerce").astype("float32")
    if "categories" in meta:
        meta["categories"] = meta["categories"].astype("string")

    # Non-nulls on required before split
    meta = meta.dropna(subset=["state","stars","review_count"]).copy()
    meta["review_count"] = meta["review_count"].astype("int32")

    for c in MUST:
        if c not in meta.columns:
            raise ValueError(f"Missing required column after derivation: {c}")

    if len(meta) == 0:
        raise ValueError("No rows remain after cleaning.")

    # FULL
    meta_full = meta.reset_index(drop=True).copy()
    meta_full.insert(0, "id", np.arange(len(meta_full), dtype=np.int64))
    meta_full.drop(columns=["name"], errors="ignore", inplace=True)
    validate_schema(meta_full)

    # DEV
    if len(meta_full) < DEV_N:
        raise ValueError(f"Need at least {DEV_N} rows for dev; have {len(meta_full)}.")
    meta_dev = meta_full.sample(n=DEV_N, random_state=SEED).reset_index(drop=True).copy()
    meta_dev["id"] = np.arange(len(meta_dev), dtype=np.int64)
    validate_schema(meta_dev)

    meta_full.to_parquet(f"{FULL}/metadata.parquet", index=False, engine="pyarrow")
    meta_dev.to_parquet(f"{DEV}/metadata.parquet",  index=False, engine="pyarrow")

    schema = {
      "must_have": ["id","state","city","stars","review_count","RestaurantsPriceRange2"],
      "optional": ["categories","latitude","longitude","is_open"],
      "derived": {"RestaurantsPriceRange2": "from attributes.RestaurantsPriceRange2; nullable Int8; NA if missing"},
      "id_range": "0..N-1 contiguous (per artifact)",
      "notes": [
        "Numeric columns use exact widths: stars=float32, review_count=int32, RestaurantsPriceRange2=Int8 (nullable).",
        "state/city are pandas string dtype.",
        "Artifacts are Parquet only.",
        "FULL = entire cleaned dataset; DEV = 10,000-row sample."
      ]
    }
    with open(f"{ART}/metadata.schema.json","w") as f:
        json.dump(schema, f, indent=2)

    print(f"Done: wrote FULL (N={len(meta_full)}) and DEV (N={len(meta_dev)}) metadata + schema.")

# ---------- Helpers ----------
def read_parquet_strict(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

def write_parquet_strict(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"Wrote {path}")

def make_text(df_row):
    parts = []
    for key in ("name", "categories", "city", "state"):
        val = df_row.get(key, None)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " | ".join(parts) if parts else ""

def l2_normalize(X: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (X / n).astype("float32")

def assert_id_contiguous_zero_based(m: pd.DataFrame):
    n = len(m)
    ids = m["id"].to_numpy()
    if set(ids) != set(range(n)):
        raise AssertionError("IDs must be contiguous 0..N-1 with no gaps/dupes.")
    if not (m["id"].is_monotonic_increasing and m["id"].iloc[0] == 0):
        raise AssertionError("IDs must be sorted ascending and start at 0.")

def assert_vectors_unit_norm(vecs: np.ndarray, atol=1e-3):
    norms = np.linalg.norm(vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise AssertionError(f"All vector norms must be ≈1.0; min={norms.min():.6f}, max={norms.max():.6f}")

# ---------- Embeddings, queries, sanity ----------
def embed_and_save(meta_path: str, bucket_dir: str, seed=SEED, model_name=MODEL_NAME):
    from sentence_transformers import SentenceTransformer  # lazy import
    m = read_parquet_strict(meta_path)
    assert "id" in m.columns, "metadata must include 'id'"
    text_cols = [c for c in ("name", "categories", "city", "state") if c in m.columns]
    m = m[["id"] + text_cols].copy()
    assert_id_contiguous_zero_based(m)

    texts = m.apply(make_text, axis=1).tolist()
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, normalize_embeddings=False)
    vecs = np.asarray(vecs, dtype="float32")
    vecs = l2_normalize(vecs)

    assert_vectors_unit_norm(vecs, atol=1e-3)
    if vecs.shape[0] != len(m):
        raise AssertionError("N in vectors must equal number of metadata rows.")

    np.save(os.path.join(bucket_dir, "vectors.npy"), vecs)
    meta = {
        "N": int(vecs.shape[0]),
        "D": int(vecs.shape[1]),
        "normalized": True,
        "model": model_name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(seed),
    }
    with open(os.path.join(bucket_dir, "vectors.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{bucket_dir}: vectors.npy + vectors.meta.json (N={meta['N']}, D={meta['D']})")

def write_queries(bucket_dir: str, N_expected: Optional[int] = None):
    q = [
        {"qid":1, "qtext":"good pizza", "filters":{"state":{"eq":"FL"},"stars":{"ge":4.5},"RestaurantsPriceRange2":{"in":[2,3]}}, "K":10, "label":"strict"},
        {"qid":2, "qtext":"authentic sushi", "filters":{"state":{"eq":"TN"},"review_count":{"ge":200},"stars":{"ge":4.0}}, "K":10, "label":"strict"},
        {"qid":3, "qtext":"third wave coffee", "filters":{"city":{"eq":"Tampa"},"stars":{"ge":4.0},"categories":{"like":"Coffee"}}, "K":10, "label":"strict"},
        {"qid":4, "qtext":"breakfast", "filters":{"state":{"eq":"FL"},"review_count":{"between":[50,200]}}, "K":10, "label":"medium"},
        {"qid":5, "qtext":"family friendly pizza", "filters":{"city":{"eq":"Tampa"},"is_open":{"eq":1}}, "K":10, "label":"medium"},
        {"qid":6, "qtext":"seafood", "filters":{"state":{"eq":"PA"},"stars":{"ge":3.5}}, "K":10, "label":"medium"},
        {"qid":7, "qtext":"burgers", "filters":{"state":{"eq":"FL"}}, "K":10, "label":"broad"},
        {"qid":8, "qtext":"mexican food", "filters":{"stars":{"ge":3.0}}, "K":10, "label":"broad"},
        {"qid":9, "qtext":"steakhouse", "filters":{"review_count":{"ge":100}}, "K":10, "label":"broad"},
        {"qid":10,"qtext":"popular spots", "filters":{}, "K":10, "label":"broad"},
    ]
    dfq = pd.DataFrame(q)
    dfq["qid"] = dfq["qid"].astype("int32")
    dfq["K"] = dfq["K"].astype("int32")
    dfq["label"] = dfq["label"].astype("string")
    dfq["filters"] = dfq["filters"].apply(lambda d: json.dumps(d, separators=(",", ":")))

    if N_expected is None:
        with open(os.path.join(bucket_dir, "vectors.meta.json"), "r") as f:
            N_expected = int(json.load(f)["N"])
    if not (dfq["K"] <= N_expected).all():
        raise AssertionError("All K must be ≤ N for this bucket.")

    write_parquet_strict(dfq, os.path.join(bucket_dir, "queries.parquet"))
    print(f"{bucket_dir}/queries.parquet")

def sanity_check_bucket(bucket_dir: str, qids_to_test=(1,2,4,7,10)):
    from sentence_transformers import SentenceTransformer  # lazy import
    X = np.load(os.path.join(bucket_dir, "vectors.npy"))
    m = read_parquet_strict(os.path.join(bucket_dir, "metadata.parquet"))
    qdf = read_parquet_strict(os.path.join(bucket_dir, "queries.parquet"))
    assert_id_contiguous_zero_based(m)
    D = X.shape[1]
    model = SentenceTransformer(MODEL_NAME)

    for qid in qids_to_test:
        row = qdf.loc[qdf["qid"] == qid]
        if row.empty:
            raise AssertionError(f"Missing qid={qid} in queries.parquet")
        base = row.iloc[0]
        qtext = base["qtext"]; K = int(base["K"])
        qvec = model.encode([qtext], normalize_embeddings=False).astype("float32")
        qvec /= max(np.linalg.norm(qvec, axis=1, keepdims=True), 1e-12)
        if qvec.shape[1] != D:
            raise AssertionError(f"Query dim {qvec.shape[1]} must match dataset dim {D}.")
        scores = X @ qvec[0]
        topk = np.argsort(-scores)[:K].tolist()
        preview_cols = [c for c in ["id","city","state","stars","review_count"] if c in m.columns]
        preview = m.loc[topk, preview_cols].reset_index(drop=True)
        print(f"[{bucket_dir}] qid={qid} «{qtext}» Top-K IDs:", topk[:min(10, K)])
        print(preview.head(min(5, K)))
    print(f"Oracle sanity passed on {bucket_dir} for qids {list(qids_to_test)}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Yelp metadata → embeddings pipeline")
    p.add_argument("--stage", choices=["meta","embed","queries","sanity","all"], default="all")
    p.add_argument("--src", default=FILE, help="Path to Yelp JSON (for --stage meta)")
    args = p.parse_args()

    if args.stage in ("meta","all"):
        build_metadata_artifacts(src_json=args.src)

    if args.stage in ("embed","all"):
        embed_and_save(os.path.join(FULL, "metadata.parquet"), FULL, seed=SEED, model_name=MODEL_NAME)
        embed_and_save(os.path.join(DEV,  "metadata.parquet"), DEV,  seed=SEED, model_name=MODEL_NAME)

    if args.stage in ("queries","all"):
        write_queries(FULL)
        write_queries(DEV)

    if args.stage in ("sanity","all"):
        sanity_check_bucket(DEV, qids_to_test=(1,2,4,7,10))

    print("Pipeline complete.")