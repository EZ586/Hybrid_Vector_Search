import argparse
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

def load_results(path):
    df = pd.read_json(path, lines=True)
    # Average latency per qid
    return df.groupby("qid", as_index=False).agg({
        "filter_selectivity": "first",
        "latency_ms": "mean"
    })

def loess_smooth(df, clip, frac):
    # Remove outliers
    df = df[df["latency_ms"] <= clip]
    return sm.nonparametric.lowess(
        df["latency_ms"],
        df["filter_selectivity"],
        frac=frac
    )

def main():
    parser = argparse.ArgumentParser(description="Plot latency vs selectivity for multiple backends.")
    parser.add_argument("--pre", type=str, default=None, help="Path to pre-filter JSONL results.")
    parser.add_argument("--post", type=str, default=None, help="Path to post-filter JSONL results.")
    parser.add_argument("--hybrid", type=str, default=None, help="Path to hybrid JSONL results.")
    parser.add_argument("--out", type=str, required=True, help="Output PNG file.")
    parser.add_argument("--clip", type=float, default=40.0, help="Latency cap for outlier removal.")
    parser.add_argument("--frac", type=float, default=0.3, help="LOESS smoothing fraction.")
    args = parser.parse_args()

    plt.figure(figsize=(8,5))

    # === Pre Filter ===
    if args.pre:
        avg_pre = load_results(args.pre)
        smooth_pre = loess_smooth(avg_pre, args.clip, args.frac)
        plt.plot(
            smooth_pre[:,0], smooth_pre[:,1],
            label="Pre Filter", linewidth=2.5, color="blue"
        )

    # === Post Filter ===
    if args.post:
        avg_post = load_results(args.post)
        smooth_post = loess_smooth(avg_post, args.clip, args.frac)
        plt.plot(
            smooth_post[:,0], smooth_post[:,1],
            label="Post Filter", linewidth=2.5, color="red"
        )

    # === Hybrid ===
    if args.hybrid:
        avg_hybrid = load_results(args.hybrid)
        smooth_hybrid = loess_smooth(avg_hybrid, args.clip, args.frac)
        plt.plot(
            smooth_hybrid[:,0], smooth_hybrid[:,1],
            label="Hybrid", linewidth=2.5, color="green", linestyle="--"
        )

    # === Labels / Formatting ===
    plt.xlabel("Filter Selectivity")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Selectivity Across Backends")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # === Save ===
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {args.out}")

if __name__ == "__main__":
    main()
