"""Stratified sampling deterministik untuk gold-standard set."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STRATA = ["year", "match_outcome_for_player", "tier"]


@dataclass
class SamplingStats:
    target_size: int
    n_strata: int
    n_strata_below_min: int
    actual_size: int
    small_strata: list[tuple[str, int]] = field(default_factory=list)


def _stratum_key(row: pd.Series, strata_cols: list[str]) -> str:
    return "|".join(str(row[c]) for c in strata_cols)


def stratified_sample(
    df: pd.DataFrame,
    target_size: int = 8000,
    min_per_stratum: int = 50,
    strata_cols: list[str] | None = None,
    seed: int = 42,
    exclude_unknown_outcome: bool = True,
) -> tuple[pd.DataFrame, SamplingStats]:
    """Stratified sampling.

    Aturan:
    - Setiap stratum (kombinasi `strata_cols`) mendapat minimal `min_per_stratum` baris.
      Jika populasi stratum < min, ambil seluruhnya dan catat di stats.small_strata.
    - Sisa kuota dialokasikan proporsional ke ukuran populasi stratum yang masih tersedia.
    - Sampling deterministik dengan seed.
    """
    if strata_cols is None:
        strata_cols = DEFAULT_STRATA

    df_in = df.copy()
    if exclude_unknown_outcome and "match_outcome_for_player" in df_in.columns:
        df_in = df_in[df_in["match_outcome_for_player"].isin(["win", "loss"])]

    df_in = df_in.dropna(subset=strata_cols)

    rng = np.random.default_rng(seed)

    # Hitung ukuran tiap stratum.
    grouped = df_in.groupby(strata_cols, sort=True, observed=True)
    sizes = grouped.size().to_dict()
    n_strata = len(sizes)

    if n_strata == 0:
        return df_in.iloc[0:0].copy(), SamplingStats(
            target_size=target_size,
            n_strata=0,
            n_strata_below_min=0,
            actual_size=0,
        )

    small_strata: list[tuple[str, int]] = []

    # Tahap 1: setiap stratum dapat min(min_per_stratum, populasi).
    base_alloc: dict[tuple, int] = {}
    for key, size in sizes.items():
        alloc = min(min_per_stratum, size)
        base_alloc[key] = alloc
        if size < min_per_stratum:
            small_strata.append(("|".join(map(str, key)), size))

    base_total = sum(base_alloc.values())
    remaining_quota = max(0, target_size - base_total)

    # Tahap 2: alokasi proporsional dari sisa kuota ke stratum yang masih punya
    # kapasitas (populasi - base_alloc > 0).
    extra_alloc: dict[tuple, int] = {key: 0 for key in sizes}
    if remaining_quota > 0:
        capacity = {k: sizes[k] - base_alloc[k] for k in sizes}
        cap_total = sum(max(0, v) for v in capacity.values())
        if cap_total > 0:
            for key, cap in capacity.items():
                if cap <= 0:
                    continue
                extra = round(remaining_quota * (cap / cap_total))
                extra = min(extra, cap)
                extra_alloc[key] = int(extra)

    # Sample per stratum.
    parts: list[pd.DataFrame] = []
    for key, sub in grouped:
        n = base_alloc[key] + extra_alloc.get(key, 0)
        n = min(n, len(sub))
        if n <= 0:
            continue
        idx = rng.choice(sub.index.values, size=n, replace=False)
        parts.append(df_in.loc[idx])

    sampled = pd.concat(parts, ignore_index=True) if parts else df_in.iloc[0:0].copy()

    # Reproducible ordering: shuffle dengan seed yang sama supaya output deterministik.
    if len(sampled) > 0:
        order = rng.permutation(len(sampled))
        sampled = sampled.iloc[order].reset_index(drop=True)

    stats = SamplingStats(
        target_size=target_size,
        n_strata=n_strata,
        n_strata_below_min=len(small_strata),
        actual_size=len(sampled),
        small_strata=sorted(small_strata, key=lambda x: x[1]),
    )
    return sampled, stats


def load_processed_concat(processed_root: Path, exclude_non_english: bool = True) -> pd.DataFrame:
    """Load + concat seluruh data/processed/*.parquet (kecuali _non_english)."""
    files = sorted(Path(processed_root).glob("*.parquet"))
    if exclude_non_english:
        files = [f for f in files if not f.stem.endswith("_non_english")]
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file *.parquet di {processed_root} — jalankan notebook 01 dulu."
        )
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)
