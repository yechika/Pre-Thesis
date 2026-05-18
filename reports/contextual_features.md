# Contextual Feature Definitions

This document provides operational definitions for the five contextual features used in the correlation analysis (notebook 08) and time-series analysis (notebook 07). Data source: `data/processed/<folder>.parquet`.

## 1. `match_outcome_for_player` — Match outcome from the perspective of the chat sender

- **Type**: binary categorical.
- **Values**: `win` / `loss` / `unknown`.
- **Raw column source**: `radiant_win` (bool, dari `main_metadata.csv`) × `player_slot` (float, dari `chat.csv`).
- **Derivation rules**:
  - Radiant players have `player_slot < 128`; Dire players have `player_slot >= 128`.
  - If Radiant player AND `radiant_win=True` → `win`.
  - If Dire player AND `radiant_win=False` → `win`.
  - Otherwise → `loss`.
  - `radiant_win` or `player_slot` NaN → `unknown` (excluded from statistical tests).
- **Implementation**: `src/pipeline/joiner.py::_outcome_for_player`.

## 2. `duration` — Match duration

- **Type**: continuous (seconds).
- **Source**: `duration` column in `main_metadata.csv`.
- **Categorical bins** (for visualization):
  - `short`: `< 1800` (< 30 minutes)
  - `medium`: `1800 ≤ duration ≤ 2700` (30–45 minutes)
  - `long`: `> 2700` (> 45 minutes)
- **Note**: professional matches typically last 25–60 minutes; matches < 15 minutes likely indicate early-GG or forfeit.

## 3. `tier` — Tournament tier (DERIVED)

- **Type**: ordinal categorical.
- **Values**: `TI` > `Major` > `DPC Tour` > `Lainnya`.
- **Raw column source**: `leagueid` (`main_metadata.csv`) × `Constants.Leagues.csv` (`leaguename`, OpenDota `tier`).
- **Derivation rules** (`src/pipeline/enrich.py::_derive_dota_tier`):
  - OpenDota `tier ∈ {excluded, amateur, ""}` → `Lainnya` (direct).
  - `leaguename` contains "Qualifier" → `Lainnya` (regardless of tier).
  - `leaguename` contains "The International" + year/number → `TI`.
  - `leaguename` contains "Major" → `Major`.
  - `leaguename` contains "DPC", "Dota Pro Circuit", or "Tour [123]" → `DPC Tour`.
  - Otherwise → `Lainnya`.
- **Mapping distribution** (as of April 2026, from Constants.Leagues.csv, 8,812 entries): TI=14, Major=24, DPC Tour=97, Other=8,677.
- **Limitation**: leagueid mapped to `Lainnya` includes non-DPC tournaments (e.g., Esports World Cup, Riyadh Masters, Snow Sweet Snow). For finer granularity, extend the regex in `_derive_dota_tier`.

## 4. `phase` — Tournament phase (DERIVED, SEVERE LIMITATION)

- **Type**: categorical.
- **Values**: `group_stage` / `playoffs` / `grand_final` / `lainnya`.
- **Source**: regex heuristic on `leaguename` (`src/pipeline/enrich.py::derive_phase`).
  - `leaguename` contains "Grand Final" → `grand_final`.
  - contains "Final" or "Playoff" → `playoffs`.
  - contains "Group Stage" → `group_stage`.
  - default → `lainnya`.
- **LIMITATION**: The public Dota 2 dataset (Kaggle `bwandowando/dota-2-pro-league-matches-2023`) does not include an explicit per-match `phase` column. Many tournaments use a single `leagueid` for all phases (group + playoffs bundled together), so the majority of rows fall to `lainnya`. Consequences:
  - The `phase × toxicity_score` correlation test may be under-powered.
  - Interpretation of `phase` findings MUST be cautious — confounders likely exist.
- **Mitigations considered and rejected**:
  - **Match-level positional inference** (match order within leagueid → group=early, playoffs=late): rejected because match order does not reliably reflect phase and requires external bracket data.
  - **Manual labeling per leagueid**: rejected because 8,812 leagueid entries are not feasible for a 3-student thesis.
  - **Cross-reference Liquipedia per match_id**: rejected because it requires scraping/API access outside the project scope.
- **Report recommendation**: report `phase` results as exploratory analysis, not a primary claim. Main claims should focus on `tier`, `match_outcome_for_player`, `duration`, and `period`.

## 5. `period` — Time period

- **Type**: ordinal categorical.
- **Primary granularity**: three bins — `2016-2018`, `2019-2021`, `2022-2026`.
- **Alternative granularity** (optional): per-year (2016 to 2026).
- **Raw column source**: `start_date_time` (timestamp) → `year` (Int16, dari `joiner.py`).
- **Derivation rules**: `period` di-compute on-the-fly di notebook 08 dari kolom `year`:
  - `year ∈ {2016, 2017, 2018}` → `2016-2018`.
  - `year ∈ {2019, 2020, 2021}` → `2019-2021`.
  - `year ∈ {2022, 2023, 2024, 2025, 2026}` → `2022-2026`.
- **Bin rationale**: three eras each encompassing significant events (early TI post-DPC formation, pandemic era, post-DPC restructuring era).

## 6. `patch_name` — Patch version at match time

- **Type**: categorical.
- **Source**: `start_date_time` × `data/processed/patch_map.csv` (date-range lookup).
- **Rule**: for each match, select the patch where `date_start ≤ start_date_time ≤ date_end`.
- **Note**: the `patch` column in `main_metadata.csv` is a Valve internal integer not directly mapped to human-readable versions (e.g., `7.34`); we use date-range lookup from `Constants.Patch.csv` as the authoritative source.

---

## Processed Output Column Summary

| Column | Source | Type |
|--------|--------|------|
| `match_outcome_for_player` | derived | string `{win, loss, unknown}` |
| `duration` | metadata | Int64 (seconds) |
| `tier` | derived | string `{TI, Major, DPC Tour, Lainnya}` |
| `leaguename` | Constants.Leagues | string |
| `phase` | derived (limited) | string `{group_stage, playoffs, grand_final, lainnya}` |
| `patch_name` | Constants.Patch + start_date_time | string |
| `year`, `month`, `quarter` | derived | Int16/Int8/Int8 |

`period` is computed on-the-fly during analysis (not stored in parquet so bins can be changed without re-running the pipeline).
