"""Load chat.csv dan main_metadata.csv per-folder dari dota2_dataset_bersih/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CHAT_DTYPES = {
    "time": "float64",
    "type": "string",
    "key": "string",
    "slot": "float64",
    "player_slot": "float64",
    "match_id": "int64",
    "leagueid": "Int64",
    "unit": "string",
}

METADATA_KEEP = [
    "match_id",
    "leagueid",
    "start_date_time",
    "duration",
    "radiant_win",
    "lobby_type",
    "game_mode",
    "radiant_team_id",
    "dire_team_id",
    "patch",
    "region",
    "first_blood_time",
]


def folder_path(raw_root: Path, folder: str) -> Path:
    return Path(raw_root) / folder


def load_chat(raw_root: Path, folder: str) -> pd.DataFrame:
    path = folder_path(raw_root, folder) / "chat.csv"
    if not path.exists():
        raise FileNotFoundError(f"chat.csv tidak ditemukan untuk folder={folder} ({path})")
    df = pd.read_csv(path, dtype=CHAT_DTYPES, low_memory=False)
    df["source_folder"] = folder
    return df


def load_metadata(raw_root: Path, folder: str) -> pd.DataFrame:
    path = folder_path(raw_root, folder) / "main_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"main_metadata.csv tidak ditemukan untuk folder={folder} ({path})")
    df = pd.read_csv(path, low_memory=False)
    keep = [c for c in METADATA_KEEP if c in df.columns]
    df = df[keep].copy()
    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").astype("Int64")
    if "radiant_win" in df.columns:
        df["radiant_win"] = df["radiant_win"].astype("boolean")
    if "patch" in df.columns:
        df["patch"] = pd.to_numeric(df["patch"], errors="coerce").astype("Int64")
    return df


def folder_exists(raw_root: Path, folder: str) -> bool:
    return (folder_path(raw_root, folder) / "chat.csv").exists() and (
        folder_path(raw_root, folder) / "main_metadata.csv"
    ).exists()
