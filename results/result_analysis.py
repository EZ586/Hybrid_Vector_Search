#!/usr/bin/env python3
"""
filter_zero_recall.py
Reads a results JSONL file (each line = JSON object)
and prints or saves all entries with recall_at_k == 0.
"""

import json
from pathlib import Path

# ---------- Configuration ----------
INPUT_PATH = Path("results/hybrid_results.jsonl")
OUTPUT_PATH = Path("results/hybrid_less_than_1_recall.jsonl")

# ---------- Read & Filter ----------
zero_recall = []
with INPUT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if float(obj.get("recall_at_k", 0)) < 1.0:
            zero_recall.append(obj)

# ---------- Output ----------
print(f"Found {len(zero_recall)} entries with recall_at_k == 0")

# Save to file
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for obj in zero_recall:
        f.write(json.dumps(obj) + "\n")

# Optional: preview first few
for sample in zero_recall[:5]:
    print(json.dumps(sample, indent=2))
