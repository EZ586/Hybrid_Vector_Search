import argparse
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path


def load_results(path):
    """Load JSONL results and compute average recall per qid."""
    df = pd.read_json(path, lines=True)

    # average per qid
    return df.groupby("qid", as_index=False).agg({
        "filter_selectivity": "first",
        "recall_at_k": "mean"
    }).sort_values("filter_selectivity")


def loess_smooth(df, frac):
    """LOWESS smoothing."""
    return sm.nonparametric.lowess(
        df["recall_at_k"],
        df["filter_selectivity"],
        frac=frac
    )


def main():
    parser = argparse.ArgumentParser(description="Plot Recall@K vs Selectivity curves.")
    parser.add_argument("--pre", type=str, help="Path to pre-filter JSONL.")
    parser.add_argument("--post", type=str, help="Path to post-filter JSONL.")
    parser.add_argument("--hybrid", type=str, help="Path to hybrid JSONL.")
    parser.add_argument("--out", type=str, required=True, help="Output PNG file.")
    parser.add_argument("--frac", type=float, default=0.3, help="LOWESS smoothing fraction.")
    parser.add_argument("--k", type=int, default=10, help="Recall@K to label.")
    args = parser.parse_args()

    plt.figure(figsize=(8, 5))

    # === Pre-filter curve ===
    if args.pre:
        pre = load_results(args.pre)
        pre_sm = loess_smooth(pre, args.frac)
        plt.plot(
            pre_sm[:, 0], pre_sm[:, 1],
            label="Pre Filter", linewidth=2.5, color="blue"
        )

    # === Post-filter curve ===
    if args.post:
        post = load_results(args.post)
        post_sm = loess_smooth(post, args.frac)
        plt.plot(
            post_sm[:, 0], post_sm[:, 1],
            label="Post Filter", linewidth=2.5, color="red"
        )

    # === Hybrid curve ===
    if args.hybrid:
        hybrid = load_results(args.hybrid)
        hybrid_sm = loess_smooth(hybrid, args.frac)
        plt.plot(
            hybrid_sm[:, 0], hybrid_sm[:, 1],
            label="Hybrid", linewidth=2.5, color="green", linestyle="--"
        )

    # === Formatting ===
    plt.xlabel("Filter Selectivity")
    plt.ylabel(f"Recall@{args.k}")
    plt.title(f"Recall@{args.k} vs Selectivity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # === Save ===
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved recall plot to: {args.out}")


if __name__ == "__main__":
    main()
