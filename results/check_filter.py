#!/usr/bin/env python3
"""
Check whether ground-truth top-K neighbors are excluded by the hybrid allow-list filter.
"""

import json
import numpy as np
from pathlib import Path
import faiss

# ----------- CONFIG -----------
ARTIFACTS_DIR = Path("artifacts/v1")        # or "artifacts/v2"
INDEX_PATH = Path("results/indexes/faiss_ivf.index")
HYBRID_RESULTS = Path("results/hybrid_results.jsonl")
EXACT_RESULTS = Path("results/exact_results.jsonl")

# ----------- LOAD INDEX -----------
index = faiss.read_index(str(INDEX_PATH))
ntotal = index.ntotal
print(f"[Index] nlist={index.nlist}, ntotal={ntotal}")

# ----------- LOAD RESULTS -----------
def load_jsonl(path):
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                objs.append(json.loads(line))
    return objs

hybrid = load_jsonl(HYBRID_RESULTS)
exact = load_jsonl(EXACT_RESULTS)

# ----------- ANALYSIS -----------
missing_due_to_filter = []

for h, e in zip(hybrid, exact):
    qid = h["qid"]
    hybrid_ids = set(h.get("ids", [])) if "ids" in h else set()
    exact_ids = set(e.get("ids", [])) if "ids" in e else set()

    # if hybrid recall is 0 but exact had valid IDs, check filtering
    if h["recall_at_k"] == 0.0 and exact_ids:
        # allow_ids used by hybrid were a subset of [0, ntotal)
        # we can estimate selectivity from h["filter_selectivity"]
        allowed_count = int(h["filter_selectivity"] * ntotal)
        if allowed_count < ntotal:
            missing_due_to_filter.append({
                "qid": qid,
                "filter_selectivity": h["filter_selectivity"],
                "allowed_count": allowed_count,
                "note": "likely filtered out ground-truth IDs"
            })

# ----------- REPORT -----------
print(f"\n[{len(missing_due_to_filter)} / {len(hybrid)}] queries likely zero recall due to filtering\n")

for entry in missing_due_to_filter[:10]:
    print(json.dumps(entry, indent=2))
