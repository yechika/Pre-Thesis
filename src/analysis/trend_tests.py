"""Mann-Kendall trend test + Chow structural break test untuk series bulanan."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mann_kendall(values: np.ndarray) -> dict:
    """Mann-Kendall trend test (nonparametric).

    Return dict: {tau, p_value, trend ∈ {increasing, decreasing, no-trend}}.
    """
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = len(values)
    if n < 4:
        return {"tau": float("nan"), "p_value": float("nan"), "trend": "insufficient", "n": n}

    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(values[i + 1 :] - values[i]))

    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    if p < 0.05 and z > 0:
        trend = "increasing"
    elif p < 0.05 and z < 0:
        trend = "decreasing"
    else:
        trend = "no-trend"
    return {"tau": float(tau), "p_value": float(p), "trend": trend, "n": n, "z": float(z)}


def chow_test(values: np.ndarray, breakpoint: int) -> dict:
    """Chow test untuk perubahan struktur sebelum vs sesudah breakpoint.

    Sederhanakan: regresi linear y = a + b*t pada series penuh vs dua segmen.
    Return F-statistic + p-value.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if breakpoint < 3 or breakpoint > n - 3:
        return {"f_statistic": float("nan"), "p_value": float("nan"), "n": n, "breakpoint": breakpoint}

    valid = ~np.isnan(values)
    t = np.arange(n)

    def _ssr(y: np.ndarray, x: np.ndarray) -> float:
        if len(y) < 3:
            return float("nan")
        slope, intercept, *_ = stats.linregress(x, y)
        pred = slope * x + intercept
        return float(np.sum((y - pred) ** 2))

    full_y = values[valid]
    full_x = t[valid]
    ssr_full = _ssr(full_y, full_x)

    seg1_mask = valid & (t < breakpoint)
    seg2_mask = valid & (t >= breakpoint)
    ssr1 = _ssr(values[seg1_mask], t[seg1_mask])
    ssr2 = _ssr(values[seg2_mask], t[seg2_mask])

    if any(np.isnan([ssr_full, ssr1, ssr2])):
        return {"f_statistic": float("nan"), "p_value": float("nan"), "n": n, "breakpoint": breakpoint}

    k = 2  # parameter per segmen (slope + intercept)
    n_total = int(valid.sum())
    numerator = (ssr_full - (ssr1 + ssr2)) / k
    denominator = (ssr1 + ssr2) / (n_total - 2 * k)
    if denominator <= 0:
        return {"f_statistic": float("nan"), "p_value": float("nan"), "n": n_total, "breakpoint": breakpoint}
    f = numerator / denominator
    p = 1 - stats.f.cdf(f, k, n_total - 2 * k)
    return {"f_statistic": float(f), "p_value": float(p), "n": n_total, "breakpoint": breakpoint}


def run_trend_suite(series_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    """Jalankan Mann-Kendall untuk setiap series, kembali sebagai DataFrame siap CSV."""
    rows = []
    for name, vals in series_dict.items():
        mk = mann_kendall(vals)
        rows.append(
            {
                "series": name,
                "test_name": "Mann-Kendall",
                "test_statistic": mk.get("tau", float("nan")),
                "z_score": mk.get("z", float("nan")),
                "p_value": mk["p_value"],
                "interpretation": mk["trend"],
                "n": mk["n"],
            }
        )
    return pd.DataFrame(rows)


def run_chow_suite(
    series_dict: dict[str, np.ndarray],
    breakpoints: dict[str, int],
) -> pd.DataFrame:
    """breakpoints: dict[label → index posisi breakpoint di series]."""
    rows = []
    for series_name, vals in series_dict.items():
        for bp_label, bp_idx in breakpoints.items():
            r = chow_test(vals, bp_idx)
            rows.append(
                {
                    "series": series_name,
                    "test_name": "Chow",
                    "breakpoint_label": bp_label,
                    "breakpoint_index": bp_idx,
                    "f_statistic": r["f_statistic"],
                    "p_value": r["p_value"],
                    "interpretation": "structural_break" if r["p_value"] < 0.05 else "no_break",
                    "n": r["n"],
                }
            )
    return pd.DataFrame(rows)
