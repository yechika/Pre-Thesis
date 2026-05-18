"""
regenerate_plots.py
Re-runs only the plotting cells from notebooks 07 and 08
to regenerate PNG/SVG files with English titles.
Run from repo root: python scripts/regenerate_plots.py
"""
import sys
import json
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
PLOTS = REPORTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ─── shared notebook config setup ───────────────────────────────────────────
import yaml, random, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.runtime import RunLog

with open(ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEED = config.get("seed", 42)
random.seed(SEED); np.random.seed(SEED)

SENT_LABELS  = config["labels"]["sentiment_classes"]
TOX_LABELS   = config["labels"]["toxicity_labels"]

PROCESSED_ROOT = ROOT / "data" / "processed"
INF_ROOT       = ROOT / "data" / "inference"

run_log = RunLog("scripts/regenerate_plots.py", config_path="configs/experiment.yaml")

# ─── NB07: temporal plots ────────────────────────────────────────────────────
print("=== Regenerating temporal plots (NB07) ===")

from src.analysis.event_overlay import major_patches, the_internationals, pandemic_band, dpc_era

BEST_TOX_07  = "detoxify"
BEST_SENT_07 = "roberta"

monthly_path = REPORTS / "temporal_monthly.csv"
yearly_path  = REPORTS / "temporal_yearly.csv"

if not monthly_path.exists():
    print("  temporal_monthly.csv not found – skipping NB07 plots")
else:
    monthly = pd.read_csv(monthly_path)
    yearly  = pd.read_csv(yearly_path) if yearly_path.exists() else pd.DataFrame()

    monthly["_dt"] = pd.to_datetime(monthly["year_month"])
    patches  = major_patches()
    tis      = the_internationals()
    p_start, p_end = pandemic_band()
    dpc_start, dpc_end = dpc_era()

    def add_overlay(ax):
        ax.axvspan(dpc_start, dpc_end, alpha=0.06, color="blue", label="DPC era")
        ax.axvspan(p_start, p_end, alpha=0.10, color="orange", label="Pandemic (online)")
        for ev in patches:
            ax.axvline(ev.date, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.text(ev.date, ax.get_ylim()[1], ev.name, rotation=90, fontsize=7, va="top", alpha=0.6)
        for ev in tis:
            ax.axvline(ev.date, color="red", linestyle=":", alpha=0.5, linewidth=0.8)

    plot_specs = [
        ("mean_sentiment_score",  "Mean Sentiment Score (P(pos) \u2212 P(neg))", "temporal_sentiment_monthly"),
        ("pct_negative",          "% Negative Messages",                         "temporal_negative_pct_monthly"),
        ("pct_toxic_any",         "% Toxic Messages (any label)",                 "temporal_toxicity_monthly"),
        ("mean_toxicity_score",   "Mean Toxicity Score (max prob)",               "temporal_toxicity_score_monthly"),
    ]

    for col, title, fname in plot_specs:
        if col not in monthly.columns:
            print(f"  column {col!r} missing – skipping {fname}")
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monthly["_dt"], monthly[col], linewidth=1.5)
        add_overlay(ax)
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel(col)
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS / f"{fname}.png", dpi=120)
        plt.savefig(PLOTS / f"{fname}.svg")
        plt.close(fig)
        print(f"  saved {fname}.{{png,svg}}")

    # Yearly bar chart
    if not yearly.empty and "mean_sentiment_score" in yearly.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(yearly["year"], yearly["mean_sentiment_score"], alpha=0.8)
        ax.set_title("Mean Sentiment Score per Year")
        ax.set_xlabel("Year")
        plt.tight_layout()
        plt.savefig(PLOTS / "temporal_yearly_overview.png", dpi=120)
        plt.savefig(PLOTS / "temporal_yearly_overview.svg")
        plt.close(fig)
        print("  saved temporal_yearly_overview.{png,svg}")

# ─── NB08: correlation plots ─────────────────────────────────────────────────
print("\n=== Regenerating correlation plots (NB08) ===")

from src.analysis.temporal import merge_inference_with_processed

try:
    df = merge_inference_with_processed(
        processed_root=PROCESSED_ROOT,
        sentiment_inference_root=INF_ROOT / f"roberta_sentiment",
        toxicity_inference_root=INF_ROOT / f"detoxify_toxicity",
        sentiment_labels=SENT_LABELS,
        toxicity_labels=TOX_LABELS,
    )
except Exception as e:
    print(f"  merge_inference_with_processed failed: {e}")
    df = pd.DataFrame()

if df.empty:
    print("  DataFrame empty – skipping NB08 plots")
else:
    # Derived columns
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
    if "predicted_label" in df.columns:
        df["sentiment_label"] = df["predicted_label"].astype("string")

    valid = df[df["match_outcome_for_player"].isin(["win", "loss"])].copy()

    # Plot 1 – Sentiment × Match Outcome (stacked bar)
    if "sentiment_label" in df.columns and not valid.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ct = pd.crosstab(valid["match_outcome_for_player"], valid["sentiment_label"], normalize="index")
        ct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Sentiment \u00d7 Match Outcome")
        ax.set_ylabel("Proportion")
        ax.legend(title="sentiment")
        plt.tight_layout()
        plt.savefig(PLOTS / "correlation_outcome_sentiment.png", dpi=120)
        plt.savefig(PLOTS / "correlation_outcome_sentiment.svg")
        plt.close(fig)
        print("  saved correlation_outcome_sentiment.{png,svg}")

    if "max_toxicity_prob" in df.columns:
        # Plot 2 – Toxicity × Match Outcome (violin)
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.violinplot(data=valid, x="match_outcome_for_player", y="max_toxicity_prob", ax=ax)
            ax.set_title("Toxicity (max prob) \u00d7 Match Outcome")
            plt.tight_layout()
            plt.savefig(PLOTS / "correlation_outcome_toxicity.png", dpi=120)
            plt.savefig(PLOTS / "correlation_outcome_toxicity.svg")
            plt.close(fig)
            print("  saved correlation_outcome_toxicity.{png,svg}")

        # Plot 3 – Toxicity × Tournament Tier
        tier_order = ["TI", "Major", "DPC Tour", "Lainnya"]
        tier_df = df.dropna(subset=["tier"])
        if not tier_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=tier_df, x="tier", y="max_toxicity_prob",
                        order=[t for t in tier_order if t in tier_df["tier"].unique()], ax=ax)
            ax.set_title("Toxicity \u00d7 Tournament Tier")
            plt.tight_layout()
            plt.savefig(PLOTS / "correlation_tier_toxicity.png", dpi=120)
            plt.savefig(PLOTS / "correlation_tier_toxicity.svg")
            plt.close(fig)
            print("  saved correlation_tier_toxicity.{png,svg}")

        # Plot 4 – Toxicity × Tournament Phase
        phase_df = df.dropna(subset=["phase"])
        if not phase_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=phase_df, x="phase", y="max_toxicity_prob", ax=ax)
            ax.set_title("Toxicity \u00d7 Tournament Phase")
            plt.tight_layout()
            plt.savefig(PLOTS / "correlation_phase_toxicity.png", dpi=120)
            plt.savefig(PLOTS / "correlation_phase_toxicity.svg")
            plt.close(fig)
            print("  saved correlation_phase_toxicity.{png,svg}")

        # Plot 5 – Toxicity × Match Duration (LOWESS)
        sample = df.sample(min(50_000, len(df)), random_state=SEED)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(
            data=sample, x="duration", y="max_toxicity_prob",
            lowess=True, scatter_kws={"alpha": 0.05, "s": 4}, ax=ax,
        )
        ax.set_title("Toxicity \u00d7 Match Duration (LOWESS)")
        plt.tight_layout()
        plt.savefig(PLOTS / "correlation_duration_toxicity.png", dpi=120)
        plt.savefig(PLOTS / "correlation_duration_toxicity.svg")
        plt.close(fig)
        print("  saved correlation_duration_toxicity.{png,svg}")

print("\nAll plots regenerated.")
