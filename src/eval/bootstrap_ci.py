"""Bootstrap CI 95% untuk metrik klasifikasi."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI.

    metric_fn: fungsi yang menerima resampled arrays dan kembali float.
    *arrays: arrays panjangnya sama; di-resample bersama-sama (paired).

    Return: (mean, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("Semua array harus punya panjang sama.")

    results = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        try:
            v = metric_fn(*[a[idx] for a in arrays])
            results.append(float(v))
        except (ValueError, ZeroDivisionError):
            continue

    if not results:
        return float("nan"), float("nan"), float("nan")

    arr = np.array(results)
    alpha = (1 - ci_level) / 2
    lower = float(np.percentile(arr, alpha * 100))
    upper = float(np.percentile(arr, (1 - alpha) * 100))
    mean = float(arr.mean())
    return mean, lower, upper
