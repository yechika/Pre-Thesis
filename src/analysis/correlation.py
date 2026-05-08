"""Wrapper uji korelasi: chi-square, Mann-Whitney, Spearman, Kruskal-Wallis, Dunn's."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def chi_square_test(
    df: pd.DataFrame, feature: str, label: str
) -> dict:
    """Chi-square independence test untuk dua kategorikal.

    Cramér's V sebagai effect size.
    """
    sub = df[[feature, label]].dropna()
    if len(sub) == 0:
        return {"test": "chi-square", "test_statistic": np.nan, "p_value": np.nan, "df": 0,
                "effect_size": np.nan, "n": 0, "interpretation": "insufficient"}
    table = pd.crosstab(sub[feature], sub[label])
    if table.size == 0 or (table.values.sum() == 0):
        return {"test": "chi-square", "test_statistic": np.nan, "p_value": np.nan, "df": 0,
                "effect_size": np.nan, "n": 0, "interpretation": "empty_table"}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    min_dim = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
    return {
        "test": "chi-square",
        "test_statistic": float(chi2),
        "p_value": float(p),
        "df": int(dof),
        "effect_size": float(cramers_v) if not np.isnan(cramers_v) else np.nan,
        "n": int(n),
        "interpretation": "significant" if p < 0.05 else "ns",
    }


def mannwhitney_test(df: pd.DataFrame, feature: str, value_col: str) -> dict:
    """Mann-Whitney U untuk biner kategorikal × kontinu."""
    sub = df[[feature, value_col]].dropna()
    groups = sub[feature].unique()
    if len(groups) != 2:
        return {"test": "mann-whitney", "test_statistic": np.nan, "p_value": np.nan,
                "effect_size": np.nan, "n": len(sub), "interpretation": "needs_binary"}
    a = sub.loc[sub[feature] == groups[0], value_col]
    b = sub.loc[sub[feature] == groups[1], value_col]
    if len(a) < 5 or len(b) < 5:
        return {"test": "mann-whitney", "test_statistic": np.nan, "p_value": np.nan,
                "effect_size": np.nan, "n": len(sub), "interpretation": "insufficient"}
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial r sebagai effect size
    n1, n2 = len(a), len(b)
    r = 1 - (2 * stat) / (n1 * n2)
    return {
        "test": "mann-whitney",
        "test_statistic": float(stat),
        "p_value": float(p),
        "effect_size": float(r),
        "n": int(n1 + n2),
        "interpretation": "significant" if p < 0.05 else "ns",
    }


def spearman_test(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 30:
        return {"test": "spearman", "test_statistic": np.nan, "p_value": np.nan,
                "effect_size": np.nan, "n": len(sub), "interpretation": "insufficient"}
    rho, p = stats.spearmanr(sub[x_col], sub[y_col])
    return {
        "test": "spearman",
        "test_statistic": float(rho),
        "p_value": float(p),
        "effect_size": float(rho),
        "n": int(len(sub)),
        "interpretation": "significant" if p < 0.05 else "ns",
    }


def kruskal_test(df: pd.DataFrame, feature: str, value_col: str) -> dict:
    sub = df[[feature, value_col]].dropna()
    groups = [g[value_col].values for _, g in sub.groupby(feature) if len(g) >= 5]
    if len(groups) < 2:
        return {"test": "kruskal-wallis", "test_statistic": np.nan, "p_value": np.nan,
                "effect_size": np.nan, "n": len(sub), "interpretation": "insufficient"}
    h, p = stats.kruskal(*groups)
    # Eta squared epsilon as effect size
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps_sq = (h - k + 1) / (n - k) if n > k else np.nan
    return {
        "test": "kruskal-wallis",
        "test_statistic": float(h),
        "p_value": float(p),
        "effect_size": float(eps_sq) if not np.isnan(eps_sq) else np.nan,
        "n": int(n),
        "df": int(k - 1),
        "interpretation": "significant" if p < 0.05 else "ns",
    }


def dunn_posthoc(df: pd.DataFrame, feature: str, value_col: str) -> pd.DataFrame:
    """Dunn's post-hoc untuk Kruskal-Wallis.

    Menggunakan implementasi sederhana berbasis rank-based pairwise Mann-Whitney
    (alternatif: pakai scikit-posthocs jika di-install).
    """
    try:
        import scikit_posthocs as sp  # type: ignore
        return sp.posthoc_dunn(df[[feature, value_col]].dropna(), val_col=value_col, group_col=feature, p_adjust='holm')
    except ImportError:
        # Fallback: pairwise Mann-Whitney
        sub = df[[feature, value_col]].dropna()
        groups = sorted(sub[feature].unique())
        rows = []
        for a, b in combinations(groups, 2):
            ya = sub.loc[sub[feature] == a, value_col]
            yb = sub.loc[sub[feature] == b, value_col]
            if len(ya) < 5 or len(yb) < 5:
                continue
            stat, p = stats.mannwhitneyu(ya, yb, alternative="two-sided")
            rows.append({"group_a": a, "group_b": b, "u": float(stat), "p_value": float(p)})
        return pd.DataFrame(rows)
