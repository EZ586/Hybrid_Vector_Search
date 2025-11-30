import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

MAX_SELECTIVITY = 0.30
# POST_PATH = "results/week2/post_results.jsonl"
# PRE_PATH = "results/week2/pre_results.jsonl"
POST_PATH = "results/week6/post_filter_shared_ivf.jsonl"
PRE_PATH = "results/week6/pre_results.jsonl"
HYBRID_PATH = "results/week7/hybrid_results_trial_8.jsonl"


def _read_and_average(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    averaged = df.groupby("qid", as_index=False).agg({
        "filter_selectivity": "first",
        value_col: "mean"
    })
    return averaged[averaged["filter_selectivity"] <= MAX_SELECTIVITY]


def _lowess(df: pd.DataFrame, y_col: str):
    if df.empty:
        return None
    filtered = df[df[y_col] <= 40]
    if filtered.empty:
        return None
    return sm.nonparametric.lowess(
        filtered[y_col],
        filtered["filter_selectivity"],
        frac=0.3
    )


avg_post = _read_and_average(POST_PATH, "latency_ms")
avg_pre = _read_and_average(PRE_PATH, "latency_ms")
avg_hybrid = _read_and_average(HYBRID_PATH, "latency_ms")

smoothed_post = _lowess(avg_post, "latency_ms")
smoothed_pre = _lowess(avg_pre, "latency_ms")
smoothed_hybrid = _lowess(avg_hybrid, "latency_ms")

plt.figure(figsize=(8, 5))
if smoothed_post is not None:
    plt.plot(smoothed_post[:, 0], smoothed_post[:, 1],
             color='red', linewidth=1.5, label="Post Filter")
if smoothed_pre is not None:
    plt.plot(smoothed_pre[:, 0], smoothed_pre[:, 1],
             color='blue', linewidth=1.5, label="Pre Filter")
if smoothed_hybrid is not None:
    plt.plot(smoothed_hybrid[:, 0], smoothed_hybrid[:, 1],
             color='green', linewidth=1.5, label="Hybrid")

plt.xlabel("Filter Selectivity")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Selectivity")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, MAX_SELECTIVITY)

Path("results/week2").mkdir(parents=True, exist_ok=True)
output_path = "results/week7/latency_selectivity_8.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {output_path}")
