"""Execute temporal analysis on updated in-domain inference outputs."""
import sys, os, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.runtime import load_config
from src.analysis.temporal import (
    merge_inference_with_processed, aggregate_monthly, aggregate_yearly,
    mann_kendall_test, chow_test, compare_events
)

cfg = load_config("configs/experiment.yaml")
PROCESSED_ROOT = Path("data/processed")
INF_ROOT = Path("data/inference")
REPORTS = Path("reports")
PLOTS = REPORTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

BEST_SENT = "roberta"
BEST_TOX = "detoxify"
SENT_LABELS = ["negative", "neutral", "positive"]
TOX_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print(f"Loading and merging {BEST_SENT}_sentiment with {BEST_TOX}_toxicity...")
df = merge_inference_with_processed(
    processed_root=PROCESSED_ROOT,
    sentiment_inference_root=INF_ROOT / f"{BEST_SENT}_sentiment",
    toxicity_inference_root=INF_ROOT / f"{BEST_TOX}_toxicity",
    sentiment_labels=SENT_LABELS,
    toxicity_labels=TOX_LABELS,
)
print(f"Merged dataset shape: {df.shape}")

# 1. Monthly aggregation (124 months)
monthly = aggregate_monthly(df)
monthly.to_csv(REPORTS / "temporal_monthly.csv", index=False)
print(f"Aggregated {len(monthly)} monthly time-series points -> reports/temporal_monthly.csv")

yearly = aggregate_yearly(df)
yearly.to_csv(REPORTS / "temporal_yearly.csv", index=False)
print(f"Aggregated {len(yearly)} yearly points -> reports/temporal_yearly.csv")

# 2. Mann-Kendall Trend Tests
mk_rows = []
for col in ["mean_sentiment_score", "pct_negative", "pct_toxic_any", "mean_toxicity_score"]:
    if col in monthly.columns:
        r = mann_kendall_test(monthly[col])
        r["series"] = col
        mk_rows.append(r)
mk_df = pd.DataFrame(mk_rows)
print("\n=== MANN-KENDALL TREND TESTS (124 months) ===")
print(mk_df.to_string(index=False))

# 3. Chow Structural Break Tests
chow_rows = []
break_points = {
    "pandemic_start": "2020-03",
    "post_dpc": "2023-01",
    "patch_7.33": "2023-04",
}
for name, bdate in break_points.items():
    for col in ["mean_sentiment_score", "pct_negative", "pct_toxic_any", "mean_toxicity_score"]:
        if col in monthly.columns:
            r = chow_test(monthly, metric=col, break_date=bdate)
            r["breakpoint_name"] = name
            chow_rows.append(r)
chow_df = pd.DataFrame(chow_rows)
print("\n=== CHOW STRUCTURAL BREAK TESTS ===")
print(chow_df.to_string(index=False))

all_tests = pd.concat([
    mk_df.assign(test_type="mann_kendall"),
    chow_df.assign(test_type="chow_test")
], ignore_index=True)
all_tests.to_csv(REPORTS / "temporal_tests.csv", index=False)

# 4. Event Comparisons
events = {
    "pre-pandemic": ("2016-01-01", "2020-02-29"),
    "pandemic-online": ("2020-03-01", "2021-12-31"),
    "post-pandemic-LAN": ("2022-01-01", "2026-04-30"),
    "dpc-era": ("2017-10-01", "2023-09-30"),
    "post-dpc": ("2023-10-01", "2026-04-30"),
}
ev_pairs = [
    ("pre-pandemic", "pandemic-online"),
    ("pandemic-online", "post-pandemic-LAN"),
    ("dpc-era", "post-dpc"),
]
ev_rows = []
for g1, g2 in ev_pairs:
    for metric in ["max_toxicity_prob", "sentiment_score"]:
        r = compare_events(df, events[g1], events[g2], metric=metric)
        r.update({"group1": g1, "group2": g2, "metric": metric})
        ev_rows.append(r)
ev_df = pd.DataFrame(ev_rows)
ev_df.to_csv(REPORTS / "event_comparison.csv", index=False)
print("\n=== EVENT COMPARISONS ===")
print(ev_df.to_string(index=False))

# 5. Generate plots
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["month_period"], monthly["mean_sentiment_score"], color="#2b5c8f", lw=2, label="Mean Sentiment Score")
ax.plot(monthly["month_period"], monthly["pct_negative"]/100, color="#d95f02", lw=1.5, ls="--", label="% Negative (scaled)")
ax.set_title("Monthly Mean Sentiment Score & % Negative (2016–2026)", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Score / Proportion")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS / "temporal_sentiment_monthly.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["month_period"], monthly["pct_toxic_any"], color="#c0392b", lw=2, label="% Toxic (any)")
ax.plot(monthly["month_period"], monthly["mean_toxicity_score"]*100, color="#8e44ad", lw=1.5, ls="--", label="Mean Toxicity Score (x100)")
ax.set_title("Monthly Toxicity Frequency and Severity (2016–2026)", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Percentage (%)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS / "temporal_toxicity_monthly.png", dpi=150)
plt.close()

print(f"\nPlots refreshed in {PLOTS}")
print("=== TEMPORAL ANALYSIS COMPLETE ===")
