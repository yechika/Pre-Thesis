"""Tulis output processed parquet + idempotent hash check + schema stable."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Schema kolom yang DIWAJIBKAN ada di output processed parquet supaya seluruh folder
# dapat di-concat tanpa friction (lihat spec chat-data-pipeline).
PROCESSED_SCHEMA = {
    "match_id": "int64",
    "time": "float64",
    "type": "string",
    "key": "string",
    "key_lower": "string",
    "slot": "float64",
    "player_slot": "float64",
    "lang": "string",
    "lang_source": "string",
    "match_outcome_for_player": "string",
    "match_minute": "float64",
    "tier": "string",
    "leaguename": "string",
    "patch_name": "string",
    "phase": "string",
    "start_date_time": "datetime64[ns]",
    "duration": "Int64",
    "radiant_win": "boolean",
    "leagueid": "Int64",
    "patch": "Int64",
    "region": "float64",
    "lobby_type": "Int64",
    "game_mode": "Int64",
    "radiant_team_id": "float64",
    "dire_team_id": "float64",
    "year": "Int16",
    "month": "Int8",
    "quarter": "Int8",
    "source_folder": "string",
}


@dataclass
class WriteStats:
    n_rows: int
    out_path: Path
    output_sha256: str
    skipped: bool


def _coerce_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, dtype in PROCESSED_SCHEMA.items():
        if col not in df.columns:
            df[col] = pd.NA
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # Fallback: coerce numerik via pd.to_numeric, datetime via pd.to_datetime
            if dtype.startswith("Int") or dtype in {"int64", "float64"}:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype, errors="ignore")
            elif dtype.startswith("datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype, errors="ignore")
    # Reorder kolom sesuai schema; kolom ekstra ditolak supaya schema stable.
    return df[list(PROCESSED_SCHEMA.keys())]


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_sources(*paths: Path) -> str:
    """Hash gabungan SHA256 dari beberapa file sumber, untuk idempotency check."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
        h.update(sha256_of_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def write_processed(
    df: pd.DataFrame,
    out_path: Path,
    sources_hash: str,
    *,
    overwrite: bool = False,
) -> WriteStats:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")

    # Idempoten: skip jika output ada dan sources_hash cocok dengan sidecar.
    if out_path.exists() and sidecar.exists() and not overwrite:
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("sources_hash") == sources_hash:
                return WriteStats(
                    n_rows=int(meta.get("n_rows", 0)),
                    out_path=out_path,
                    output_sha256=meta.get("output_sha256", ""),
                    skipped=True,
                )
        except Exception:
            pass  # sidecar rusak → tulis ulang.

    df = _coerce_to_schema(df)
    df.to_parquet(out_path, compression="snappy", index=False)
    output_sha = sha256_of_file(out_path)
    sidecar.write_text(
        json.dumps(
            {
                "sources_hash": sources_hash,
                "n_rows": int(len(df)),
                "output_sha256": output_sha,
                "schema": {k: v for k, v in PROCESSED_SCHEMA.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return WriteStats(
        n_rows=len(df),
        out_path=out_path,
        output_sha256=output_sha,
        skipped=False,
    )


def write_non_english(df: pd.DataFrame, out_path: Path) -> WriteStats:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        # Tetap tulis file kosong supaya diketahui telah diproses.
        empty = _coerce_to_schema(df)
        empty.to_parquet(out_path, compression="snappy", index=False)
    else:
        df_out = _coerce_to_schema(df)
        df_out.to_parquet(out_path, compression="snappy", index=False)
    return WriteStats(
        n_rows=len(df),
        out_path=out_path,
        output_sha256=sha256_of_file(out_path),
        skipped=False,
    )
