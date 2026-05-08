"""Clean chat dataframe: drop chatwheel, normalisasi unicode, lowercase."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import pandas as pd

CHATWHEEL_TYPE = "chatwheel"


@dataclass
class CleanStats:
    n_in: int
    n_chatwheel_dropped: int
    n_empty_dropped: int
    n_out: int


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def clean_chat(df: pd.DataFrame, lowercase: bool = True) -> tuple[pd.DataFrame, CleanStats]:
    n_in = len(df)
    chatwheel_mask = df["type"].fillna("").str.lower() == CHATWHEEL_TYPE
    n_chatwheel = int(chatwheel_mask.sum())
    df = df.loc[~chatwheel_mask].copy()

    df["key"] = df["key"].fillna("").astype(str).map(normalize_unicode).str.strip()
    if lowercase:
        df["key_lower"] = df["key"].str.lower()
    else:
        df["key_lower"] = df["key"]

    empty_mask = df["key"].str.len() == 0
    n_empty = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()

    return df, CleanStats(
        n_in=n_in,
        n_chatwheel_dropped=n_chatwheel,
        n_empty_dropped=n_empty,
        n_out=len(df),
    )
