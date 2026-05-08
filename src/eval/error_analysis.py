"""Error analysis: sample FP/FN per-model + breakdown per-jargon."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sample_errors_sentiment(
    df: pd.DataFrame,
    label_col: str = "sentiment",
    pred_col: str = "predicted_label",
    n_per_class: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample contoh kesalahan per-class untuk inspeksi manual.

    df: gold-test merged dengan prediksi (kolom: text/key, label_col, pred_col, prob_*).
    """
    rng = np.random.default_rng(seed)
    errors = df[df[label_col] != df[pred_col]].copy()
    if len(errors) == 0:
        return errors

    samples = []
    classes = sorted(df[label_col].dropna().unique())
    for true_cls in classes:
        for pred_cls in classes:
            if true_cls == pred_cls:
                continue
            sub = errors[(errors[label_col] == true_cls) & (errors[pred_col] == pred_cls)]
            if len(sub) == 0:
                continue
            n = min(n_per_class, len(sub))
            idx = rng.choice(sub.index.values, size=n, replace=False)
            samples.append(sub.loc[idx])
    return pd.concat(samples, ignore_index=True) if samples else errors.iloc[0:0]


def sample_errors_toxicity(
    df: pd.DataFrame,
    label_cols: list[str],
    pred_cols: list[str],
    n_fp: int = 50,
    n_fn: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample FP (pred=1, true=0) dan FN (pred=0, true=1) per-label."""
    rng = np.random.default_rng(seed)
    parts = []
    for lbl, pred in zip(label_cols, pred_cols):
        if lbl not in df.columns or pred not in df.columns:
            continue
        fp = df[(df[lbl] == 0) & (df[pred] == 1)]
        fn = df[(df[lbl] == 1) & (df[pred] == 0)]
        if len(fp) > 0:
            n = min(n_fp, len(fp))
            idx = rng.choice(fp.index.values, size=n, replace=False)
            sub = fp.loc[idx].copy()
            sub["error_type"] = "FP"
            sub["error_label"] = lbl
            parts.append(sub)
        if len(fn) > 0:
            n = min(n_fn, len(fn))
            idx = rng.choice(fn.index.values, size=n, replace=False)
            sub = fn.loc[idx].copy()
            sub["error_type"] = "FN"
            sub["error_label"] = lbl
            parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def jargon_error_breakdown(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    jargon_col: str = "is_dota_jargon",
) -> pd.DataFrame:
    """Hitung error rate untuk pesan jargon vs non-jargon."""
    if jargon_col not in df.columns:
        return pd.DataFrame()

    out_rows = []
    df = df.copy()
    df["correct"] = (df[label_col] == df[pred_col]).astype(int)
    for is_jargon, sub in df.groupby(jargon_col):
        out_rows.append(
            {
                "is_jargon": int(is_jargon),
                "n": len(sub),
                "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan"),
                "error_rate": float(1 - sub["correct"].mean()) if len(sub) else float("nan"),
            }
        )
    out = pd.DataFrame(out_rows)
    if len(out) == 2:
        delta = out.loc[out["is_jargon"] == 1, "error_rate"].iloc[0] - out.loc[
            out["is_jargon"] == 0, "error_rate"
        ].iloc[0]
        out.attrs["delta_error_rate"] = float(delta)
    return out
