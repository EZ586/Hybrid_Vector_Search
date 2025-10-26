import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"/"week2_dev"
INPUT_JSONL = RESULTS_DIR / "results_dev.jsonl"

def load_results(jsonl_path: Path) -> pd.DataFrame:
    """Load JSONL result logs into a DataFrame."""
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    return df

def plot_latency_vs_recall(df: pd.DataFrame, out_path: Path):
    """Plot Recall@10 (x-axis) vs Latency (y-axis), one curve per method."""
    plt.figure(figsize=(7, 5))
    methods = df["method"].unique()

    for method in methods:
        sub = df[df["method"] == method].sort_values("recall@K")
        plt.plot(
            sub["recall@K"],
            sub["latency_ms"],
            marker="o",
            label=method,
        )

    plt.xlabel("Recall@10")
    plt.ylabel("Latency (ms)")
    plt.title("Recall@10 vs Latency by Method")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scored_vectors_bar(df: pd.DataFrame, out_path: Path):
    """Bar chart: average scored_vectors per method."""
    plt.figure(figsize=(6, 4))
    grouped = df.groupby("method")["scored_vectors"].mean().sort_values()
    grouped.plot(kind="bar", color="skyblue", edgecolor="black")

    plt.xlabel("Method")
    plt.ylabel("Average Scored Vectors")
    plt.title("Scored Vectors by Method")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    df = load_results(INPUT_JSONL)
    if df.empty:
        print(f"No results found in {INPUT_JSONL}")
        return

    out_latency_recall = RESULTS_DIR / "latency_vs_recall.png"
    out_scored_bar = RESULTS_DIR / "scored_vectors.png"

    plot_latency_vs_recall(df, out_latency_recall)
    plot_scored_vectors_bar(df, out_scored_bar)

    print(f"Saved plots to:\n- {out_latency_recall}\n- {out_scored_bar}")

if __name__ == "__main__":
    main()
