"""Benjamini-Hochberg FDR correction (q = 0.05 default)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """BH-FDR step-up.

    Return: (p_adjusted, significant_after_bh) — keduanya array sepanjang p_values.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    valid = ~np.isnan(p)
    p_valid = p[valid]
    m = len(p_valid)

    if m == 0:
        return np.full_like(p, np.nan), np.zeros_like(p, dtype=bool)

    order = np.argsort(p_valid)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)

    # BH-adjusted p: p_adj_i = min over k>=i of (m * p_(k) / k)
    sorted_p = p_valid[order]
    bh = sorted_p * m / np.arange(1, m + 1)
    # Enforce monotonicity from the top: cumulative min from right
    bh_monotone = np.minimum.accumulate(bh[::-1])[::-1]
    bh_monotone = np.clip(bh_monotone, 0, 1)

    p_adj_valid = np.empty(m)
    p_adj_valid[order] = bh_monotone

    p_adj = np.full_like(p, np.nan)
    p_adj[valid] = p_adj_valid

    sig = np.zeros_like(p, dtype=bool)
    sig[valid] = p_adj_valid <= q
    return p_adj, sig


def apply_to_dataframe(
    df: pd.DataFrame,
    p_col: str = "p_value",
    q: float = 0.05,
    out_col_prefix: str = "p_adj_bh",
) -> pd.DataFrame:
    """Tambahkan kolom `p_adj_bh` dan `significant_after_bh` ke df."""
    out = df.copy()
    p_adj, sig = benjamini_hochberg(out[p_col].values, q=q)
    out[out_col_prefix] = p_adj
    out["significant_after_bh"] = sig
    return out
