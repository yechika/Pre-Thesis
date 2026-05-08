"""Agregat time-series sentimen + toksisitas per-bulan/per-tahun."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["match_id", "time", "player_slot"]


def merge_inference_with_processed(
    processed_root: Path,
    sentiment_inference_root: Path,
    toxicity_inference_root: Path,
    sentiment_labels: list[str],
    toxicity_labels: list[str],
) -> pd.DataFrame:
    """Gabung seluruh data/processed/*.parquet dengan inferensi sentimen + toksisitas.

    Output dataframe punya kolom kontekstual + prob/pred sentiment + prob/pred toxicity.
    """
    proc_files = sorted(
        [p for p in Path(processed_root).glob("*.parquet") if not p.stem.endswith("_non_english")]
    )
    if not proc_files:
        raise FileNotFoundError(f"Tidak ada parquet di {processed_root}")

    proc = pd.concat([pd.read_parquet(p) for p in proc_files], ignore_index=True)
    # Dedup processed: chat di detik sama oleh player sama bisa duplikat keys
    proc = proc.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)

    sent_files = sorted(Path(sentiment_inference_root).glob("*.parquet"))
    tox_files = sorted(Path(toxicity_inference_root).glob("*.parquet"))

    sent = (
        pd.concat([pd.read_parquet(p) for p in sent_files], ignore_index=True)
        if sent_files
        else None
    )
    if sent is not None:
        sent = sent.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)
    tox = (
        pd.concat([pd.read_parquet(p) for p in tox_files], ignore_index=True)
        if tox_files
        else None
    )
    if tox is not None:
        tox = tox.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)

    df = proc
    if sent is not None:
        sent_keep = KEY_COLS + [c for c in sent.columns if c.startswith(("prob_", "predicted_"))]
        df = df.merge(sent[sent_keep], on=KEY_COLS, how="left")
    if tox is not None:
        tox_keep = KEY_COLS + [
            c for c in tox.columns if c.startswith(("prob_", "pred_", "max_toxicity_", "any_"))
        ]
        df = df.merge(tox[tox_keep], on=KEY_COLS, how="left", suffixes=("", "_tox"))

    # Kolom turunan untuk agregat:
    if "predicted_label" in df.columns:
        df["sentiment_score"] = df.get("prob_positive", 0).fillna(0) - df.get("prob_negative", 0).fillna(0)
    return df


def aggregate_monthly(
    df: pd.DataFrame,
    sentiment_labels: list[str] = ["negative", "neutral", "positive"],
    toxicity_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Agregat per-bulan: pct sentimen, mean score, pct toxic, mean tox score."""
    df = df.copy()
    if "start_date_time" in df.columns:
        sdt = pd.to_datetime(df["start_date_time"], errors="coerce")
        df["year_month"] = sdt.dt.to_period("M").astype(str)
    else:
        df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    df = df[df["year_month"].notna()]
    df = df[df["year_month"] != "NaT"]

    rows = []
    for ym, sub in df.groupby("year_month", sort=True):
        n = len(sub)
        row = {"year_month": ym, "n_messages": n, "n_matches": int(sub["match_id"].nunique())}
        # Sentimen
        if "predicted_label" in sub.columns:
            for lbl in sentiment_labels:
                row[f"pct_{lbl}"] = float((sub["predicted_label"] == lbl).mean()) if n else float("nan")
            row["mean_sentiment_score"] = float(sub.get("sentiment_score", pd.Series([])).mean()) if "sentiment_score" in sub.columns else float("nan")
        # Toksisitas
        if "any_toxic" in sub.columns:
            row["pct_toxic_any"] = float(sub["any_toxic"].fillna(0).mean()) if n else float("nan")
        if "max_toxicity_prob" in sub.columns:
            row["mean_toxicity_score"] = float(sub["max_toxicity_prob"].fillna(0).mean()) if n else float("nan")
        if toxicity_labels:
            for lbl in toxicity_labels:
                col = f"pred_{lbl}"
                if col in sub.columns:
                    row[f"pct_{lbl}"] = float(sub[col].fillna(0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year_month").reset_index(drop=True)


def aggregate_yearly(
    df: pd.DataFrame,
    sentiment_labels: list[str] = ["negative", "neutral", "positive"],
    toxicity_labels: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df = df[df["year"].notna()]
    rows = []
    for year, sub in df.groupby("year", sort=True):
        n = len(sub)
        row = {"year": int(year), "n_messages": n, "n_matches": int(sub["match_id"].nunique())}
        if "predicted_label" in sub.columns:
            for lbl in sentiment_labels:
                row[f"pct_{lbl}"] = float((sub["predicted_label"] == lbl).mean()) if n else float("nan")
            row["mean_sentiment_score"] = float(sub.get("sentiment_score", pd.Series([])).mean()) if "sentiment_score" in sub.columns else float("nan")
        if "any_toxic" in sub.columns:
            row["pct_toxic_any"] = float(sub["any_toxic"].fillna(0).mean()) if n else float("nan")
        if "max_toxicity_prob" in sub.columns:
            row["mean_toxicity_score"] = float(sub["max_toxicity_prob"].fillna(0).mean()) if n else float("nan")
        if toxicity_labels:
            for lbl in toxicity_labels:
                col = f"pred_{lbl}"
                if col in sub.columns:
                    row[f"pct_{lbl}"] = float(sub[col].fillna(0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
