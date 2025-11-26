import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# === Load JSONL files ===
df_post = pd.read_json("results/week2/post_results.jsonl", lines=True)
df_pre = pd.read_json("results/week2/pre_results.jsonl", lines=True)

# === Average latency per qid ===
avg_post = df_post.groupby("qid", as_index=False).agg({
    "filter_selectivity": "first",
    "latency_ms": "mean"
})
avg_pre = df_pre.groupby("qid", as_index=False).agg({
    "filter_selectivity": "first",
    "latency_ms": "mean"
})

# === Remove outliers above 40 ms ===
filtered_post_40 = avg_post[avg_post["latency_ms"] <= 40]
filtered_pre_40 = avg_pre[avg_pre["latency_ms"] <= 40]

# === Compute LOESS smoothing ===
smoothed_post_40 = sm.nonparametric.lowess(
    filtered_post_40["latency_ms"],
    filtered_post_40["filter_selectivity"],
    frac=0.3
)
smoothed_pre_40 = sm.nonparametric.lowess(
    filtered_pre_40["latency_ms"],
    filtered_pre_40["filter_selectivity"],
    frac=0.3
)

# === Plot only trend lines ===
plt.figure(figsize=(8,5))
plt.plot(smoothed_post_40[:, 0], smoothed_post_40[:, 1],
         color='red', linewidth=2.5, label="Post Filter")
plt.plot(smoothed_pre_40[:, 0], smoothed_pre_40[:, 1],
         color='blue', linewidth=2.5, label="Pre Filter")
plt.xlabel("Filter Selectivity")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Selectivity (Pre vs Post Filter)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# === Export plot ===
Path("results/week").mkdir(parents=True, exist_ok=True)
output_path = "results/week2/latency_selectivity.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {output_path}")