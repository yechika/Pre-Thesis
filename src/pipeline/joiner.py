"""Join chat × main_metadata + kolom turunan kontekstual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Konvensi Dota 2: player_slot < 128 → Radiant; >= 128 → Dire.
RADIANT_SLOT_BOUNDARY = 128


@dataclass
class JoinStats:
    n_in: int
    n_with_metadata: int
    n_missing_metadata: int


def _outcome_for_player(player_slot: float, radiant_win: object) -> str:
    if pd.isna(player_slot) or pd.isna(radiant_win):
        return "unknown"
    is_radiant = player_slot < RADIANT_SLOT_BOUNDARY
    won = bool(radiant_win) if is_radiant else not bool(radiant_win)
    return "win" if won else "loss"


def _quarter_from_month(month: int) -> int:
    return ((month - 1) // 3) + 1


def join_with_metadata(chat: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, JoinStats]:
    n_in = len(chat)
    # Metadata ber-leagueid sudah dipakai untuk drop duplikasi sebelum join — di-Dota,
    # match_id unik global, tapi leagueid bisa muncul di beberapa baris metadata yang
    # sama; pakai unique by match_id.
    md = metadata.drop_duplicates(subset=["match_id"], keep="first")

    # Hindari kolom leagueid duplikat di hasil join — pakai metadata sebagai sumber.
    chat_to_join = chat.drop(columns=[c for c in ["leagueid"] if c in chat.columns], errors="ignore")
    merged = chat_to_join.merge(md, on="match_id", how="left", validate="many_to_one")

    # Kolom turunan
    merged["match_outcome_for_player"] = merged.apply(
        lambda r: _outcome_for_player(r.get("player_slot"), r.get("radiant_win")), axis=1
    )
    # match_minute = waktu chat dari awal pertandingan (kolom `time` dalam detik)
    merged["match_minute"] = (merged["time"].astype("float64") / 60.0).round(2)

    # Year/month/quarter dari start_date_time
    sdt = pd.to_datetime(merged["start_date_time"], errors="coerce")
    merged["year"] = sdt.dt.year.astype("Int16")
    merged["month"] = sdt.dt.month.astype("Int8")
    merged["quarter"] = sdt.dt.month.map(
        lambda m: _quarter_from_month(int(m)) if pd.notna(m) else np.nan
    ).astype("Int8")

    n_missing = int(merged["start_date_time"].isna().sum())
    stats = JoinStats(
        n_in=n_in,
        n_with_metadata=n_in - n_missing,
        n_missing_metadata=n_missing,
    )
    return merged, stats
