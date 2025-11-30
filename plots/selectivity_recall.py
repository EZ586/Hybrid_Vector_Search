import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# === 1. Load the jsonl files (week 7) ===
df_post = pd.read_json("results/week7/post_results.jsonl", lines=True)
df_pre = pd.read_json("results/week7/pre_results.jsonl", lines=True)
df_hybrid = pd.read_json("results/week7/hybrid_results.jsonl", lines=True)

# === 2. Average over trials for each qid ===
# post
avg_post_recall = df_post.groupby("qid", as_index=False).agg({
    "filter_selectivity": "first",   # same per qid
    "recall_at_k": "mean",           # average across trials
}).sort_values("filter_selectivity")

# pre
avg_pre_recall = df_pre.groupby("qid", as_index=False).agg({
    "filter_selectivity": "first",
    "recall_at_k": "mean",
}).sort_values("filter_selectivity")

# hybrid
avg_hybrid_recall = df_hybrid.groupby("qid", as_index=False).agg({
    "filter_selectivity": "first",
    "recall_at_k": "mean",
}).sort_values("filter_selectivity")

# === 3. LOWESS smoothing for nicer curves ===
smoothed_post_recall = sm.nonparametric.lowess(
    avg_post_recall["recall_at_k"],
    avg_post_recall["filter_selectivity"],
    frac=0.3,
)

smoothed_pre_recall = sm.nonparametric.lowess(
    avg_pre_recall["recall_at_k"],
    avg_pre_recall["filter_selectivity"],
    frac=0.3,
)

smoothed_hybrid_recall = sm.nonparametric.lowess(
    avg_hybrid_recall["recall_at_k"],
    avg_hybrid_recall["filter_selectivity"],
    frac=0.3,
)

# === 4. Plot (trend lines only) ===
plt.figure(figsize=(8, 5))

# hybrid = red
plt.plot(
    smoothed_hybrid_recall[:, 0],
    smoothed_hybrid_recall[:, 1],
    color="red",
    linewidth=2.5,
    label="Hybrid",
)

# post = blue
plt.plot(
    smoothed_post_recall[:, 0],
    smoothed_post_recall[:, 1],
    color="blue",
    linewidth=2.5,
    label="Post Filter",
)

# pre = gold
plt.plot(
    smoothed_pre_recall[:, 0],
    smoothed_pre_recall[:, 1],
    color="gold",
    linewidth=2.5,
    label="Pre Filter",
)

plt.xlabel("Filter Selectivity")
plt.ylabel("Recall@10")
plt.title("Recall@10 vs Selectivity (Pre / Post / Hybrid, LOESS)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# === 5. Export ===
out_dir = Path("results/week7")
out_dir.mkdir(parents=True, exist_ok=True)
outpath = out_dir / "recall_selectivity_loess.png"

plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to {outpath}")