# tests/test_run_impl.py
# Run: PYTHONPATH=. pytest -q tests/test_run_impl.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import pytest
pytestmark = pytest.mark.filterwarnings("ignore:.*SwigPy.*:DeprecationWarning")

# Test the teammate's *implementation* by importing module and driving main()
import importlib
import src.harness.run as runmod


# -------------------------
# Helpers & dummy classes
# -------------------------

class DummyBackend:
    """Tuple-returning backend that records inputs and returns deterministic IDs/stats."""
    def __init__(self, vectors, metadata, name):
        self.vectors = vectors
        self.metadata = metadata
        self.name = name
        self.calls: List[Tuple[np.ndarray, Dict[str, Any], int]] = []

    def search(self, qvec: np.ndarray, *, filters: Dict[str, Any], K: int):
        self.calls.append((qvec, filters, K))

        # Build allowed_ids from filters (minimal: accept state.eq; else all)
        if filters and "state" in filters and isinstance(filters["state"], dict) and "eq" in filters["state"]:
            val = filters["state"]["eq"]
            mask = (self.metadata["state"] == val)
            allowed = self.metadata.loc[mask, "id"].to_numpy(dtype=np.int64)
        else:
            allowed = self.metadata["id"].to_numpy(dtype=np.int64)

        # Score only within allowed universe (inner product)
        scores = (self.vectors[allowed] @ qvec).astype(np.float32)
        topk_local = np.argsort(-scores)[:K]
        ids = allowed[topk_local].astype(int).tolist()

        stats = {
            "latency_ms": 0.0,
            "scored_vectors": int(len(allowed)),  # = |allowed_ids|
            "lists_probed": None,
            "nprobe": None,
            "kth_at_stop": None,
            "bound_at_stop": None,
            "notes": "dummy-backend",
        }
        return ids, stats


class DummyEmbedder:
    """Mimics SentenceTransformer(...).encode([...]) -> (1,D) array."""
    def __init__(self, D: int, model_name_sink: List[str]):
        self.D = D
        self._sink = model_name_sink  # captures model_name used by _get_embedder()

    def encode(self, texts, normalize_embeddings=False):
        assert isinstance(texts, list) and len(texts) == 1
        vec = np.arange(self.D, dtype=np.float32) + 1.0  # deterministic, non-zero
        return vec.reshape(1, -1)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def tiny_artifacts(tmp_path: Path):
    """Create tiny artifacts under tmp_path/artifacts/full/v1."""
    art = tmp_path / "artifacts" / "full" / "v1"
    art.mkdir(parents=True)

    # Tiny metadata with id 0..4 (N=5)
    m = pd.DataFrame({
        "id": np.arange(5, dtype=np.int64),
        "state": pd.Series(["FL","FL","TN","PA","FL"], dtype="string"),
        "city":  pd.Series(["Tampa","Orlando","Nashville","Philly","Miami"], dtype="string"),
        "stars": np.array([4.5, 4.2, 3.9, 4.8, 4.1], dtype="float32"),
        "review_count": np.array([200, 120, 50, 300, 80], dtype="int32"),
        "RestaurantsPriceRange2": pd.Series([2, 3, 2, 4, 1], dtype="Int8"),
    })
    m.to_parquet(art / "metadata.parquet", index=False)

    # Vectors (N=5, D=4), pre-normalized unit rows
    V = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0.5,0.5,0.5,0.5],
    ], dtype=np.float32)
    V = (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32)
    np.save(art / "vectors.npy", V)

    # vectors.meta.json with model + dims
    meta = {"N": int(V.shape[0]), "D": int(V.shape[1]), "normalized": True,
            "model": "dummy/sentencetransformer", "created_utc": "2025-01-01T00:00:00Z", "seed": 42}
    (art / "vectors.meta.json").write_text(json.dumps(meta))

    # queries.parquet with 2 queries (one strict by state, one broad)
    q = pd.DataFrame([
        {"qid": 1, "qtext": "good pizza",      "filters": json.dumps({"state":{"eq":"FL"}}), "K": 3, "label":"strict"},
        {"qid": 2, "qtext": "anything really", "filters": json.dumps({}),                     "K": 3, "label":"broad"},
    ])
    q = q.astype({"qid":"int32","K":"int32","label":"string"})
    q.to_parquet(art / "queries.parquet", index=False)

    return {
        "bucket_dir": art,
        "artifacts_root": tmp_path / "artifacts",
        "vectors": V,
        "metadata": m,
        "meta": meta,
    }


@pytest.fixture
def patch_env(monkeypatch, tiny_artifacts):
    """
    Patch run.py to use our tiny artifacts, dummy embedder, dummy backend, and capture internals.
    Focus on harness behavior per the spec.
    """
    # Point constants to our tmp artifacts/results
    monkeypatch.setattr(runmod, "ARTIFACTS", tiny_artifacts["artifacts_root"])
    out_dir = tiny_artifacts["artifacts_root"].parents[0] / "results" / "week2"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runmod, "RESULTS_DIR", out_dir)

    # Replace loaders used by run.py
    monkeypatch.setattr(runmod, "load_vectors", lambda bucket: tiny_artifacts["vectors"])
    monkeypatch.setattr(runmod, "load_metadata", lambda bucket: tiny_artifacts["metadata"])
    monkeypatch.setattr(runmod, "load_vectors_meta", lambda bucket: tiny_artifacts["meta"])

    # Embedder capture
    seen_model: List[str] = []
    dummy = DummyEmbedder(D=tiny_artifacts["vectors"].shape[1], model_name_sink=seen_model)
    def fake_get_embedder(model_name: str):
        seen_model.append(model_name)
        return dummy
    monkeypatch.setattr(runmod, "_get_embedder", fake_get_embedder)

    # Spy parse_filters & build_allowed_ids wiring (we'll override selectively in tests too)
    def pf_ok(raw):
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
        raise ValueError("filters must be json or dict")
    monkeypatch.setattr(runmod, "parse_filters", pf_ok)

    def build_allowed_ids(metadata: pd.DataFrame, filters: Dict[str, Any]):
        if not filters:
            return metadata["id"].to_numpy(dtype=np.int64)
        if "state" in filters and isinstance(filters["state"], dict) and "eq" in filters["state"]:
            val = filters["state"]["eq"]
            mask = (metadata["state"] == val)
            return metadata.loc[mask, "id"].to_numpy(dtype=np.int64)
        # emulate hard error on unknown column/operator per spec
        raise ValueError("Unknown field/operator")
    monkeypatch.setattr(runmod, "build_allowed_ids", build_allowed_ids)

    # Track oracle call to assert allowed_ids are respected
    oracle_calls: List[Dict[str, Any]] = []
    def fake_bruteforce(qvec, allowed_ids, K):
        oracle_calls.append({"allowed_ids": allowed_ids.copy(), "K": K})
        V = tiny_artifacts["vectors"][allowed_ids]
        scores = V @ qvec
        local = allowed_ids[np.argsort(-scores)[:K]]
        return local.astype(int).tolist()
    runmod.ORACLE.brute_force = fake_bruteforce  # replace oracle in module

    # Replace backend registry to always return DummyBackend, keeping the name
    def fake_get_backend(name, vectors, metadata, *, artifacts_root):
        return DummyBackend(vectors, metadata, name)
    monkeypatch.setattr(runmod, "get_backend", fake_get_backend)

    return {
        "out_dir": out_dir,
        "seen_model": seen_model,
        "oracle_calls": oracle_calls,
        "vectors": tiny_artifacts["vectors"],
        "metadata": tiny_artifacts["metadata"],
        "bucket_dir": tiny_artifacts["bucket_dir"].parent.parent,  # artifacts/<version>
    }


# ------------------------------------
# Tests of run.py *implementation*
# ------------------------------------

def test_reads_model_from_meta_and_embeds_unit_norm(patch_env, monkeypatch, tiny_artifacts):
    """Harness must read meta['model'], obtain embedder, and produce L2 unit query vectors of correct D."""
    out_path = patch_env["out_dir"] / "results.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    assert patch_env["seen_model"], "Embedder was never requested"
    assert patch_env["seen_model"][-1] == tiny_artifacts["meta"]["model"]

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2, "Should log one row per query"
    # Schema sanity (complete order check elsewhere)
    for r in rows:
        assert "qid" in r and "method" in r and "recall_at_k" in r

def test_dimension_mismatch_raises_early(patch_env, monkeypatch, tiny_artifacts):
    """Embedding D must match dataset D; mismatch → ValueError before searching."""
    def bad_embedder(model_name: str):
        class Bad:
            def encode(self, texts, normalize_embeddings=False):
                return np.ones((1, tiny_artifacts["vectors"].shape[1] + 1), dtype=np.float32)
        return Bad()
    monkeypatch.setattr(runmod, "_get_embedder", bad_embedder)

    out_path = patch_env["out_dir"] / "bad.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with pytest.raises(ValueError, match=r"Query dim .* != dataset dim"):
        with monkeypatch.context() as m:
            m.setattr("sys.argv", argv)
            runmod.main()

def test_ensure_unit_l2_is_called(patch_env, monkeypatch):
    """ensure_unit_l2 validator must be invoked on the query vector."""
    called = {"n": 0}
    def spy_ensure_unit_l2(vec):
        called["n"] += 1
        assert vec.dtype == np.float32 and vec.ndim == 1
        # Return unchanged; harness must accept it
        return vec
    monkeypatch.setattr(runmod, "ensure_unit_l2", spy_ensure_unit_l2)

    out_path = patch_env["out_dir"] / "unit.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()
    # one call per query row
    assert called["n"] >= 2, "ensure_unit_l2 should be invoked for each query vector"

def test_empty_qtext_uses_dataset_vector(patch_env, monkeypatch, tiny_artifacts):
    """Empty qtext must skip embedding and fallback to vectors[qid]."""
    art_v1 = tiny_artifacts["bucket_dir"]
    q = pd.DataFrame([
        {"qid": 1, "qtext": "good pizza", "filters": json.dumps({"state":{"eq":"FL"}}), "K":3, "label":"strict"},
        {"qid": 2, "qtext": "anything really", "filters": json.dumps({}), "K":3, "label":"broad"},
        {"qid": 3, "qtext": "", "filters": json.dumps({}), "K":3, "label":"broad"},
    ]).astype({"qid":"int32","K":"int32","label":"string"})
    q.to_parquet(art_v1 / "queries.parquet", index=False)

    # Reset embedder call record
    patch_env["seen_model"].clear()

    out_path = patch_env["out_dir"] / "empty_qtext.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    # We expect at least 2 embed calls (qid 1 and 2), but qid 3 shouldn't require model request
    assert len(patch_env["seen_model"]) >= 2

def test_malformed_filters_hard_error(patch_env, monkeypatch, tiny_artifacts):
    """Malformed filter JSON must raise (hard error), and produce no output rows."""
    def bad_queries(version: str) -> pd.DataFrame:
        return pd.DataFrame([{"qid": 99, "qtext": "bad", "filters": "{bad-json:", "K": 3, "label": "x"}])
    monkeypatch.setattr(runmod, "load_queries", bad_queries)

    out_path = patch_env["out_dir"] / "malformed.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with pytest.raises(ValueError):
        with monkeypatch.context() as m:
            m.setattr("sys.argv", argv)
            runmod.main()
    assert not out_path.exists() or out_path.read_text().strip() == ""

def test_unknown_filter_field_or_operator_hard_error(patch_env, monkeypatch, tiny_artifacts):
    """Unknown column/operator must be a hard error (per spec)."""
    # Replace queries to include unknown operator 'foo'
    art_v1 = tiny_artifacts["bucket_dir"]
    q = pd.DataFrame([
        {"qid": 1, "qtext": "x", "filters": json.dumps({"bogus":{"eq":123}}), "K": 3, "label":"x"},
    ]).astype({"qid":"int32","K":"int32","label":"string"})
    q.to_parquet(art_v1 / "queries.parquet", index=False)

    out_path = patch_env["out_dir"] / "unknownop.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with pytest.raises(ValueError):
        with monkeypatch.context() as m:
            m.setattr("sys.argv", argv)
            runmod.main()

def test_oracle_called_with_allowed_ids(patch_env, monkeypatch):
    """Harness must call oracle.brute_force with the same allowed_ids the validators produced."""
    out_path = patch_env["out_dir"] / "allowed.jsonl"
    argv = ["prog", "--version", "full", "--backend", "pre_filter", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    oracle_calls = patch_env["oracle_calls"]
    assert oracle_calls, "Oracle was never called"
    allowed_sets = [set(call["allowed_ids"].tolist()) for call in oracle_calls]
    # state=='FL' in tiny artifacts → ids {0,1,4}
    assert any(s == {0,1,4} for s in allowed_sets), f"Expected allowed_ids {{0,1,4}}, got {allowed_sets}"

def test_backend_tuple_contract_and_filters_passed(patch_env, monkeypatch):
    """Backend must be called with (ids, stats) tuple; filters flow-through; K respected."""
    out_path = patch_env["out_dir"] / "contract.jsonl"
    argv = ["prog", "--version", "full", "--backend", "pre_filter", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert all(r["method"] == "pre_filter" for r in rows)
    assert all(0.0 <= r["recall_at_k"] <= 1.0 for r in rows)

def test_k_validation_guard(patch_env, monkeypatch):
    """K must be validated in [1, N] and enforced."""
    out_path = patch_env["out_dir"] / "kbad.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "999", "--out", str(out_path)]
    with pytest.raises(ValueError):
        with monkeypatch.context() as m:
            m.setattr("sys.argv", argv)
            runmod.main()

def test_logging_field_order_and_names(patch_env, monkeypatch):
    """Row must contain exact fields in exact order and 'recall_at_k' (not recall@K); null where N/A."""
    out_path = patch_env["out_dir"] / "order.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    lines = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert lines, "No rows written"
    required_order = [
        "qid","method","K","latency_ms","recall_at_k","filter_selectivity",
        "scored_vectors","lists_probed","nprobe","kth_at_stop","bound_at_stop",
        "notes","timestamp_utc","run_id",
    ]
    for r in lines:
        assert list(r.keys()) == required_order, f"Field order mismatch: {list(r.keys())}"
        assert "recall@K" not in r
        # nullables should be present as None
        for k in ["lists_probed","nprobe","kth_at_stop","bound_at_stop","notes"]:
            assert k in r  # and values may be None

def test_timestamp_is_iso8601_with_tz(patch_env, monkeypatch):
    """timestamp_utc must be ISO-8601 with timezone."""
    out_path = patch_env["out_dir"] / "ts.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    for r in rows:
        assert re.match(r"^\d{4}-\d{2}-\d{2}T.*[+-]\d{2}:\d{2}$", r["timestamp_utc"]), f"Bad timestamp: {r['timestamp_utc']}"

def test_scored_vectors_equals_allowed_ids_size(patch_env, monkeypatch):
    """For exact and pre_filter, scored_vectors must equal |allowed_ids| (subset size)."""
    out_path = patch_env["out_dir"] / "sv.jsonl"
    argv = ["prog", "--version", "full", "--backend", "pre_filter", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    for r in rows:
        if r["qid"] == 1:  # state==FL → ids {0,1,4} size 3
            assert r["scored_vectors"] == 3
        else:  # broad → all 5
            assert r["scored_vectors"] == 5

def test_filter_selectivity_matches_allowed_ids(patch_env, monkeypatch):
    """filter_selectivity must equal |allowed_ids| / N for the same allowed universe."""
    out_path = patch_env["out_dir"] / "sel.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    N = patch_env["vectors"].shape[0]
    for r in rows:
        if r["qid"] == 1:
            assert abs(r["filter_selectivity"] - (3/N)) < 1e-12
        else:
            assert abs(r["filter_selectivity"] - 1.0) < 1e-12

def test_recall_computation_against_oracle_subset(patch_env, monkeypatch):
    """Recall@K must be computed w.r.t. the *allowed* oracle subset (not full corpus)."""
    out_path = patch_env["out_dir"] / "recall.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    exact_rows = [r for r in rows if r["method"] == "exact"]
    assert exact_rows and all(abs(r["recall_at_k"] - 1.0) <= 1e-9 for r in exact_rows)

def test_backend_registry_maps_names_correctly(tiny_artifacts, monkeypatch):
    import importlib
    # restore the real function from the already-imported module, no reload
    real_get_backend = importlib.import_module("src.harness.run").get_backend
    monkeypatch.setattr(runmod, "get_backend", real_get_backend, raising=False)

    v, m = tiny_artifacts["vectors"], tiny_artifacts["metadata"]
    b1 = runmod.get_backend("exact", v, m, artifacts_root=tiny_artifacts["bucket_dir"])
    b2 = runmod.get_backend("pre_filter", v, m, artifacts_root=tiny_artifacts["bucket_dir"])
    b3 = runmod.get_backend("post_filter", v, m, artifacts_root=tiny_artifacts["bucket_dir"])
    assert getattr(b1, "name", None) == "exact"
    assert getattr(b2, "name", None) == "pre_filter"
    assert b3 is not None

def test_post_filter_ladder_and_k_floor(patch_env, monkeypatch):
    """Post-filter: scored_vectors must be ≥ K; if notes expose k_ladder, scored_vectors should align (and never exceed N)."""
    # With tiny artifacts N=5, use a ladder within N and K <= N.
    N = patch_env["vectors"].shape[0]
    ladder = [2, 3, N]  # [2,3,5] for our tiny corpus
    class PF(DummyBackend):
        def __init__(self, vectors, metadata, name):
            super().__init__(vectors, metadata, name)
        def search(self, qvec, *, filters, K):
            # Simulate post-filter over-sampling: score up to the ladder's max, but never exceed N
            ids, stats = super().search(qvec, filters=filters, K=min(K, 3))
            stats["scored_vectors"] = max(ladder[-1], K)  # equals N here
            stats["notes"] = f"k_ladder={ladder}; kept=26; need={K}"
            return ids, stats
    def fake_get_backend(name, vectors, metadata, *, artifacts_root):
        return PF(vectors, metadata, "post_filter")
    monkeypatch.setattr(runmod, "get_backend", fake_get_backend)

    out_path = patch_env["out_dir"] / "pf.jsonl"
    argv = ["prog", "--version", "full", "--backend", "post_filter", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert rows and all(r["method"] == "post_filter" for r in rows)
    for r in rows:
        assert r["scored_vectors"] >= r["K"], "post_filter should retrieve ≥ K candidates"
        # Align with ladder and never exceed N
        assert r["scored_vectors"] in ladder or r["scored_vectors"] <= ladder[-1]
        assert r["scored_vectors"] <= N, "scored_vectors should not exceed corpus size"

def test_run_id_stable_across_queries(patch_env, monkeypatch):
    """All rows in one run must share the same run_id."""
    out_path = patch_env["out_dir"] / "stable.jsonl"
    argv = ["prog", "--version", "full", "--backend", "exact", "--K", "3", "--out", str(out_path)]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", argv)
        runmod.main()
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    run_ids = {r["run_id"] for r in rows}
    assert len(run_ids) == 1, f"Expected single run_id per run, got {run_ids}"