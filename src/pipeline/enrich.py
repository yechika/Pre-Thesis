"""Bangun mapping leagueid→tier dan patch_name dari Constants/, lalu enrich processed."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Pattern untuk derivasi tier dari leaguename (case-insensitive).
# Urutan penting: TI dicek dulu, lalu Major, lalu DPC Tour, lalu Lainnya.
_TI_RE = re.compile(r"\bThe International\b", re.IGNORECASE)
_TI_NUMBER_RE = re.compile(r"\bThe International\s+(\d{2,4}|[1-9])\b", re.IGNORECASE)
_MAJOR_RE = re.compile(r"\bmajor\b", re.IGNORECASE)
_DPC_RE = re.compile(r"\bDPC\b|Dota Pro Circuit|\bTour [123]\b", re.IGNORECASE)
_QUALIFIER_RE = re.compile(r"qualif|qualifier", re.IGNORECASE)


def _derive_dota_tier(leaguename: str | None, opendota_tier: str | None) -> str:
    """Map (leaguename, opendota_tier) → {TI, Major, DPC Tour, Lainnya}.

    "excluded" dan "amateur" tier dari OpenDota difilter ke "Lainnya". Open
    Qualifiers dianggap "Lainnya" walau namanya mengandung "International"/"Major".
    """
    name = (leaguename or "").strip()
    tier = (opendota_tier or "").strip().lower()

    if tier in {"excluded", "amateur"} or tier == "":
        return "Lainnya"
    if _QUALIFIER_RE.search(name):
        return "Lainnya"
    if _TI_RE.search(name):
        # Pastikan ini TI utama (mengandung tahun/nomor), bukan "International Dota 2 League" dll.
        if _TI_NUMBER_RE.search(name) or re.search(r"International \d", name):
            return "TI"
        return "Lainnya"
    if _MAJOR_RE.search(name):
        return "Major"
    if _DPC_RE.search(name):
        return "DPC Tour"
    return "Lainnya"


def build_league_tier_map(constants_root: Path, out_path: Path) -> pd.DataFrame:
    """Baca Constants.Leagues.csv, derive dota_tier, simpan ke league_tier_map.csv."""
    src = Path(constants_root) / "Constants.Leagues.csv"
    df = pd.read_csv(src)
    df["leaguename"] = df["leaguename"].fillna("").astype(str)
    df["opendota_tier"] = df["tier"].fillna("").astype(str)
    df["dota_tier"] = df.apply(
        lambda r: _derive_dota_tier(r["leaguename"], r["opendota_tier"]), axis=1
    )
    out = df[["leagueid", "leaguename", "opendota_tier", "dota_tier"]].copy()
    out.to_csv(out_path, index=False)
    return out


def build_patch_map(constants_root: Path, out_path: Path) -> pd.DataFrame:
    """Baca Constants.Patch.csv → patch_map.csv dengan kolom patch, date_start, date_end."""
    src = Path(constants_root) / "Constants.Patch.csv"
    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce", utc=True, format="ISO8601"
    ).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    df["date_start"] = df["date"]
    df["date_end"] = df["date_start"].shift(-1) - pd.Timedelta(seconds=1)
    df.loc[df.index[-1], "date_end"] = pd.Timestamp("2099-12-31")
    out = df[["patch", "date_start", "date_end"]].copy()
    out.to_csv(out_path, index=False)
    return out


def attach_tier(df: pd.DataFrame, league_map: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom `tier` (= dota_tier) dan `leaguename` ke df berdasarkan leagueid."""
    lm = league_map.set_index("leagueid")[["leaguename", "dota_tier"]]
    out = df.copy()
    if "leagueid" not in out.columns:
        out["tier"] = "Lainnya"
        out["leaguename"] = ""
        return out
    joined = out["leagueid"].map(lm["dota_tier"]).fillna("Lainnya")
    out["tier"] = joined.astype("string")
    out["leaguename"] = out["leagueid"].map(lm["leaguename"]).fillna("").astype("string")
    return out


def attach_patch_name(df: pd.DataFrame, patch_map: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom `patch_name` (e.g. '7.34') berdasarkan start_date_time × patch_map.

    Pakai date-range lookup karena kolom `patch` di metadata berupa internal int
    (mis. 54) yang tidak langsung dapat dipetakan ke versi tanpa metadata Valve.
    """
    out = df.copy()
    if "start_date_time" not in out.columns:
        out["patch_name"] = pd.Series([pd.NA] * len(out), dtype="string")
        return out
    pm = patch_map.copy()
    pm["date_start"] = pd.to_datetime(pm["date_start"])
    pm["date_end"] = pd.to_datetime(pm["date_end"])
    pm = pm.sort_values("date_start").reset_index(drop=True)

    sdt = pd.to_datetime(out["start_date_time"], errors="coerce")
    # Searchsorted: untuk tiap sdt, cari index patch yang date_start ≤ sdt.
    # `searchsorted` kembalikan numpy array — pakai np.clip (bukan pandas .clip).
    import numpy as np
    idx = pm["date_start"].searchsorted(sdt, side="right") - 1
    idx = np.clip(np.asarray(idx), 0, len(pm) - 1)
    patch_names = pm["patch"].iloc[idx].values
    # Set NaT-derived rows to NA.
    patch_names = pd.Series(patch_names, index=out.index, dtype="string")
    patch_names[sdt.isna()] = pd.NA
    out["patch_name"] = patch_names
    return out


# ---- Phase derivation ---------------------------------------------------------
# Skema: (leaguename, series_type) → phase ∈ {group_stage, playoffs, grand_final, lainnya}.
# Heuristik: dataset Dota 2 publik tidak menyertakan phase eksplisit per match. Kita
# menetapkan default "lainnya" dan memberi marker untuk leaguename yang mengandung
# kata kunci "Grand Final" / "Final" / "Playoffs" / "Group Stage" — JIKA leaguename
# tersedia (kebanyakan turnamen profesional tidak memecah leagueid per fase, jadi ini
# fallback minimal saja). Lihat reports/contextual_features.md untuk catatan
# limitasi.

_PHASE_PATTERNS = [
    (re.compile(r"grand final", re.IGNORECASE), "grand_final"),
    (re.compile(r"\bfinals?\b", re.IGNORECASE), "playoffs"),
    (re.compile(r"playoff", re.IGNORECASE), "playoffs"),
    (re.compile(r"group stage", re.IGNORECASE), "group_stage"),
]


def derive_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom `phase` dari leaguename (default 'lainnya')."""
    out = df.copy()
    if "leaguename" not in out.columns:
        out["phase"] = "lainnya"
        return out
    names = out["leaguename"].fillna("").astype(str)
    phase = pd.Series("lainnya", index=out.index, dtype="string")
    for pat, label in _PHASE_PATTERNS:
        mask = names.str.contains(pat, regex=True, na=False)
        phase = phase.mask(mask & (phase == "lainnya"), label)
    out["phase"] = phase
    return out


@dataclass
class EnrichStats:
    n_total: int
    n_known_league: int
    n_ti: int
    n_major: int
    n_dpc_tour: int
    n_lainnya: int
    n_patch_resolved: int


def enrich_dataframe(
    df: pd.DataFrame,
    league_map: pd.DataFrame,
    patch_map: pd.DataFrame,
) -> tuple[pd.DataFrame, EnrichStats]:
    out = attach_tier(df, league_map)
    out = attach_patch_name(out, patch_map)
    out = derive_phase(out)
    stats = EnrichStats(
        n_total=len(out),
        n_known_league=int((out["leaguename"].fillna("").astype(str).str.len() > 0).sum()),
        n_ti=int((out["tier"] == "TI").sum()),
        n_major=int((out["tier"] == "Major").sum()),
        n_dpc_tour=int((out["tier"] == "DPC Tour").sum()),
        n_lainnya=int((out["tier"] == "Lainnya").sum()),
        n_patch_resolved=int(out["patch_name"].notna().sum()),
    )
    return out, stats
