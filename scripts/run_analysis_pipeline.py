"""Run complete downstream analysis pipeline (Temporal, Correlation, Dataset Characteristics)."""
import os, sys, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu

from src.runtime import load_config
from src.analysis.temporal import merge_inference_with_processed, aggregate_monthly, aggregate_yearly
from src.analysis.event_overlay import major_patches, the_internationals, pandemic_band, dpc_era
from src.analysis.trend_tests import run_trend_suite, run_chow_suite
from src.analysis.correlation import chi_square_test, mannwhitney_test, spearman_test, kruskal_test
from src.analysis.multiple_testing import apply_to_dataframe

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

print(f"--- 1. MERGING INFERENCE (In-Domain {BEST_SENT.upper()} + {BEST_TOX.upper()}) ---")
df = merge_inference_with_processed(
    processed_root=PROCESSED_ROOT,
    sentiment_inference_root=INF_ROOT / f"{BEST_SENT}_sentiment",
    toxicity_inference_root=INF_ROOT / f"{BEST_TOX}_toxicity",
    sentiment_labels=SENT_LABELS,
    toxicity_labels=TOX_LABELS,
)
print(f"Loaded merged corpus: {len(df):,} messages across {df['match_id'].nunique():,} unique matches.")

# Derive duration_bin, period, and clean outcome
df["period"] = pd.cut(
    df["year"].astype("Int64"),
    bins=[2015, 2018, 2021, 2026],
    labels=["2016-2018", "2019-2021", "2022-2026"],
    right=True,
).astype("string")

df["duration_bin"] = pd.cut(
    df["duration"].astype("Int64"),
    bins=[0, 1800, 2700, 7200],
    labels=["short", "medium", "long"],
).astype("string")

if "prob_positive" in df.columns and "prob_negative" in df.columns:
    df["sentiment_score"] = (df["prob_positive"] - df["prob_negative"]).astype(np.float32)

print("\n--- 2. TEMPORAL AGGREGATION & TESTS ---")
monthly = aggregate_monthly(df, sentiment_labels=SENT_LABELS, toxicity_labels=TOX_LABELS)
yearly = aggregate_yearly(df, sentiment_labels=SENT_LABELS, toxicity_labels=TOX_LABELS)

monthly.to_csv(REPORTS / "temporal_monthly.csv", index=False)
yearly.to_csv(REPORTS / "temporal_yearly.csv", index=False)
print(f"Wrote {len(monthly)} monthly rows and {len(yearly)} yearly rows.")

series_for_test = {
    name: monthly[name].values
    for name in ["mean_sentiment_score", "pct_negative", "pct_toxic_any", "mean_toxicity_score"]
    if name in monthly.columns
}
mk_df = run_trend_suite(series_for_test)

# Breakpoint indices
monthly_dates = pd.to_datetime(monthly["year_month"])
bp_map = {
    "Pandemic 2020-03": int(np.where(monthly_dates >= pd.Timestamp("2020-03-01"))[0][0]),
    "Post-DPC 2023-01": int(np.where(monthly_dates >= pd.Timestamp("2023-01-01"))[0][0]),
    "Patch 7.33 2023-04": int(np.where(monthly_dates >= pd.Timestamp("2023-04-01"))[0][0]),
}
chow_df = run_chow_suite(series_for_test, bp_map)

temporal_tests_df = pd.concat([mk_df, chow_df], ignore_index=True)
temporal_tests_df.to_csv(REPORTS / "temporal_tests.csv", index=False)
print("Mann-Kendall:\n", mk_df[["series", "test_statistic", "z_score", "p_value", "interpretation"]].to_string(index=False))
print("Chow:\n", chow_df[["series", "breakpoint_label", "f_statistic", "p_value", "interpretation"]].to_string(index=False))

# Event comparisons
df["_sdt"] = pd.to_datetime(df["start_date_time"], errors="coerce")
comparisons = [
    ("pre-pandemic", "2016-01-01", "2020-02-29", "pandemic-online", "2020-03-01", "2021-12-31"),
    ("pandemic-online", "2020-03-01", "2021-12-31", "post-pandemic-LAN", "2022-01-01", "2026-04-30"),
    ("dpc-era", "2017-10-01", "2023-09-30", "post-dpc", "2023-10-01", "2026-04-30"),
]
ev_rows = []
for label_a, s_a, e_a, label_b, s_b, e_b in comparisons:
    mask_a = (df["_sdt"] >= s_a) & (df["_sdt"] <= e_a)
    mask_b = (df["_sdt"] >= s_b) & (df["_sdt"] <= e_b)
    for col in ["max_toxicity_prob", "sentiment_score"]:
        if col in df.columns:
            a = df.loc[mask_a, col].dropna()
            b = df.loc[mask_b, col].dropna()
            u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
            delta = float(b.mean() - a.mean())
            ev_rows.append({
                "comparison": f"{label_a} vs {label_b}",
                "metric": col, "n_a": len(a), "n_b": len(b),
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "delta": delta, "u_statistic": float(u_stat), "p_value": float(p_val)
            })
ev_df = pd.DataFrame(ev_rows)
ev_df.to_csv(REPORTS / "event_comparison.csv", index=False)
print("Wrote reports/event_comparison.csv")

# Plots
monthly["_dt"] = pd.to_datetime(monthly["year_month"])
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["_dt"], monthly["mean_sentiment_score"], color="#2b5c8f", lw=2, label="Mean Sentiment")
ax.plot(monthly["_dt"], monthly["pct_negative"]/100, color="#d95f02", lw=1.5, ls="--", label="% Negative (scaled)")
ax.set_title("Monthly Mean Sentiment Score & % Negative (2016–2026)", fontsize=12, fontweight="bold")
ax.set_ylabel("Score / Proportion")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "temporal_sentiment_monthly.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["_dt"], monthly["pct_toxic_any"], color="#c0392b", lw=2, label="% Toxic (any)")
ax.plot(monthly["_dt"], monthly["mean_toxicity_score"]*100, color="#8e44ad", lw=1.5, ls="--", label="Mean Toxicity (x100)")
ax.set_title("Monthly Toxicity Frequency & Severity (2016–2026)", fontsize=12, fontweight="bold")
ax.set_ylabel("Percentage (%)")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS / "temporal_toxicity_monthly.png", dpi=120)
plt.close()

print("\n--- 3. CONTEXTUAL CORRELATION ANALYSIS ---")
valid = df[df["match_outcome_for_player"].isin(["win", "loss"])].copy()

corr_rows = []
if "predicted_label" in valid.columns:
    r = chi_square_test(valid, "match_outcome_for_player", "predicted_label")
    r.update({"feature": "match_outcome_for_player", "label": "sentiment"}); corr_rows.append(r)
if "max_toxicity_prob" in valid.columns:
    r = mannwhitney_test(valid, "match_outcome_for_player", "max_toxicity_prob")
    r.update({"feature": "match_outcome_for_player", "label": "max_toxicity"}); corr_rows.append(r)
if "sentiment_score" in df.columns:
    r = spearman_test(df, "duration", "sentiment_score")
    r.update({"feature": "duration", "label": "sentiment_score"}); corr_rows.append(r)
if "max_toxicity_prob" in df.columns:
    r = spearman_test(df, "duration", "max_toxicity_prob")
    r.update({"feature": "duration", "label": "max_toxicity"}); corr_rows.append(r)
if "tier" in df.columns:
    r = kruskal_test(df, "tier", "max_toxicity_prob")
    r.update({"feature": "tier", "label": "max_toxicity"}); corr_rows.append(r)
    r = chi_square_test(df, "tier", "predicted_label")
    r.update({"feature": "tier", "label": "sentiment"}); corr_rows.append(r)
if "phase" in df.columns:
    r = kruskal_test(df, "phase", "max_toxicity_prob")
    r.update({"feature": "phase", "label": "max_toxicity"}); corr_rows.append(r)
    r = chi_square_test(df, "phase", "predicted_label")
    r.update({"feature": "phase", "label": "sentiment"}); corr_rows.append(r)
if "period" in df.columns:
    r = kruskal_test(df, "period", "max_toxicity_prob")
    r.update({"feature": "period", "label": "max_toxicity"}); corr_rows.append(r)
    r = chi_square_test(df, "period", "predicted_label")
    r.update({"feature": "period", "label": "sentiment"}); corr_rows.append(r)

tests_df = pd.DataFrame(corr_rows)
tests_df = apply_to_dataframe(tests_df, p_col="p_value", q=0.05)
tests_df.to_csv(REPORTS / "correlation_tests.csv", index=False)
print("Contextual Tests after BH-FDR:\n", tests_df[["feature", "label", "test", "test_statistic", "p_adj_bh", "effect_size", "significant_after_bh"]].to_string(index=False))

# Per-match robustness
agg_per_match = df.groupby("match_id").agg(
    mean_toxicity=("max_toxicity_prob", "mean"),
    mean_sentiment=("sentiment_score", "mean"),
    duration=("duration", "first"),
    tier=("tier", "first"),
).reset_index()

robust_rows = [
    dict(spearman_test(agg_per_match, "duration", "mean_toxicity"), feature="duration", label="mean_toxicity", level="per_match"),
    dict(kruskal_test(agg_per_match, "tier", "mean_toxicity"), feature="tier", label="mean_toxicity", level="per_match"),
]
robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(REPORTS / "correlation_robustness_per_match.csv", index=False)
print("\nPer-match Robustness:\n", robust_df[["feature", "label", "test_statistic", "p_value"]].to_string(index=False))

# correlation_summary.md
lines = ["# Contextual Feature Correlation Summary (In-Domain Updated)\n", "## Test Results Table\n"]
lines.append(tests_df[["feature", "label", "test", "test_statistic", "p_value", "p_adj_bh", "effect_size", "n"]].to_markdown(index=False, floatfmt=".4f"))
lines.append("\n## Robustness at Per-Match Level (n=197,695)\n")
lines.append(robust_df.to_markdown(index=False, floatfmt=".4f"))
(REPORTS / "correlation_summary.md").write_text("\n".join(lines), encoding="utf-8")

print("\n=== PIPELINE RE-RUN COMPLETE ===")
