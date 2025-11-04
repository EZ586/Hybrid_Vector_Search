# plots/hybrid_comparison.py

from typing import Sequence
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(jsonl_path: str) -> Sequence[dict]:
    """
    Load JSONL logs (Exact, Pre, Post, Hybrid) into memory.
    """
    # TODO: read line by line, json.loads
    raise NotImplementedError


def plot_latency_vs_recall(results: Sequence[dict], out_path: str) -> None:
    """
    Produce Latency vs Recall@10 plot, one curve per method.
    Expects fields: method, latency_ms, recall@10.
    """
    # TODO: group by method and scatter/line
    raise NotImplementedError


def plot_scored_vectors(results: Sequence[dict], out_path: str) -> None:
    """
    Produce bar (or grouped) chart of scored_vectors per method.
    """
    # TODO: aggregate by method and plot
    raise NotImplementedError


def main() -> None:
    """
    Entry point to regenerate:
      - /results/latency_recall_hybrid.png
      - /results/scored_vectors_hybrid.png
    """
    # TODO: wire load_results + two plot functions
    raise NotImplementedError


if __name__ == "__main__":
    main()
