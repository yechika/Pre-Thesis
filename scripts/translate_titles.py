"""
translate_titles.py
Translate Indonesian titles/labels in notebooks and report markdown files to English.
Run from repo root: python scripts/translate_titles.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def replace_in_cell_sources(cells: list, pairs: list[tuple[str, str]]) -> int:
    """Apply string replacements to all source lines of all cells. Returns change count."""
    count = 0
    for cell in cells:
        lines = cell.get("source", [])
        for j, line in enumerate(lines):
            orig = line
            for old, new in pairs:
                line = line.replace(old, new)
            if line != orig:
                count += 1
            lines[j] = line
        cell["source"] = lines
    return count


def patch_notebook(path: Path, pairs: list[tuple[str, str]]) -> None:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    n = replace_in_cell_sources(nb["cells"], pairs)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  [{path.name}] {n} line(s) changed")


def patch_text(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    orig = text
    for old, new in pairs:
        text = text.replace(old, new)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"  [{path.name}] updated")
    else:
        print(f"  [{path.name}] no changes needed")


# ---------------------------------------------------------------------------
# Notebook 07 — temporal analysis plots
# ---------------------------------------------------------------------------
NB07_PAIRS = [
    # Plot titles
    ("'% Pesan Negatif'", "'% Negative Messages'"),
    ('"% Pesan Negatif"', '"% Negative Messages"'),
    ("'% Pesan Toksik (any-label)'", "'% Toxic Messages (any label)'"),
    ('"% Pesan Toksik (any-label)"', '"% Toxic Messages (any label)"'),
    ("'Mean Sentiment Score per Tahun'", "'Mean Sentiment Score per Year'"),
    ('"Mean Sentiment Score per Tahun"', '"Mean Sentiment Score per Year"'),
    # Axis labels
    ("ax.set_xlabel('Bulan')", "ax.set_xlabel('Month')"),
    ('ax.set_xlabel("Bulan")', 'ax.set_xlabel("Month")'),
    ("ax.set_xlabel('Tahun')", "ax.set_xlabel('Year')"),
    ('ax.set_xlabel("Tahun")', 'ax.set_xlabel("Year")'),
]

# ---------------------------------------------------------------------------
# Notebook 08 — correlation analysis plots + summary file content
# ---------------------------------------------------------------------------
NB08_PAIRS = [
    # Plot titles (× is the actual multiplication sign U+00D7)
    ("'Sentimen \u00d7 Hasil Pertandingan'", "'Sentiment \u00d7 Match Outcome'"),
    ('"Sentimen \u00d7 Hasil Pertandingan"', '"Sentiment \u00d7 Match Outcome"'),
    ("'Toksisitas (max prob) \u00d7 Hasil'", "'Toxicity (max prob) \u00d7 Match Outcome'"),
    ('"Toksisitas (max prob) \u00d7 Hasil"', '"Toxicity (max prob) \u00d7 Match Outcome"'),
    ("'Toksisitas \u00d7 Tier Turnamen'", "'Toxicity \u00d7 Tournament Tier'"),
    ('"Toksisitas \u00d7 Tier Turnamen"', '"Toxicity \u00d7 Tournament Tier"'),
    ("'Toksisitas \u00d7 Fase Turnamen'", "'Toxicity \u00d7 Tournament Phase'"),
    ('"Toksisitas \u00d7 Fase Turnamen"', '"Toxicity \u00d7 Tournament Phase"'),
    ("'Toksisitas \u00d7 Durasi Pertandingan (LOWESS)'", "'Toxicity \u00d7 Match Duration (LOWESS)'"),
    ('"Toksisitas \u00d7 Durasi Pertandingan (LOWESS)"', '"Toxicity \u00d7 Match Duration (LOWESS)"'),
    # ylabel
    ("ax.set_ylabel('Proporsi')", "ax.set_ylabel('Proportion')"),
    ('ax.set_ylabel("Proporsi")', 'ax.set_ylabel("Proportion")'),
    # correlation_summary.md text generated inside the notebook
    ("'# Ringkasan Korelasi Fitur Kontekstual\\n'", "'# Contextual Feature Correlation Summary\\n'"),
    ('"# Ringkasan Korelasi Fitur Kontekstual\\n"', '"# Contextual Feature Correlation Summary\\n"'),
    ("'## Tabel Hasil Uji\\n'", "'## Test Results Table\\n'"),
    ('"## Tabel Hasil Uji\\n"', '"## Test Results Table\\n"'),
    (
        "'## Finding Signifikan setelah BH-FDR (urut effect size desc)\\n'",
        "'## Significant Findings after BH-FDR (sorted by effect size, desc)\\n'",
    ),
    (
        '"## Finding Signifikan setelah BH-FDR (urut effect size desc)\\n"',
        '"## Significant Findings after BH-FDR (sorted by effect size, desc)\\n"',
    ),
    (
        "'_Tidak ada finding signifikan setelah koreksi BH-FDR._'",
        "'_No significant findings after BH-FDR correction._'",
    ),
    (
        '"_Tidak ada finding signifikan setelah koreksi BH-FDR._"',
        '"_No significant findings after BH-FDR correction._"',
    ),
    ("'\\n## Catatan Limitasi\\n'", "'\\n## Limitations\\n'"),
    ('"\\n## Catatan Limitasi\\n"', '"\\n## Limitations\\n"'),
    (
        "'- Pesan dalam satu pertandingan TIDAK independen (clustering effect). Analisis robustness pada level per-pertandingan dilaporkan di `correlation_robustness_per_match.csv`.'",
        "'- Messages within the same match are NOT independent (clustering effect). Robustness analysis at per-match level is reported in `correlation_robustness_per_match.csv`.'",
    ),
    (
        "'- Kolom `phase` mayoritas bernilai `lainnya` karena dataset Dota 2 publik tidak menyertakan fase per-match secara eksplisit (lihat `reports/contextual_features.md` \\u00a74).'",
        "'- The `phase` column is predominantly `lainnya` because the public Dota 2 dataset does not include explicit per-match phase labels (see `reports/contextual_features.md` \u00a74).'",
    ),
    (
        "'- Effect size yang sangat kecil (mis. Cram\\u00e9r\\'s V < 0.1, |rho| < 0.1) menunjukkan signifikansi statistik tanpa relevansi praktis pada n besar.'",
        "'- Very small effect sizes (e.g., Cram\u00e9r\\'s V < 0.1, |rho| < 0.1) indicate statistical significance without practical relevance at large n.'",
    ),
]

# ---------------------------------------------------------------------------
# Notebook 06 — comparison_summary.md generated content
# ---------------------------------------------------------------------------
NB06_PAIRS = [
    ("'# Ringkasan Perbandingan 4 Model\\n'", "'# Comparison Summary: 4 Models\\n'"),
    ('"# Ringkasan Perbandingan 4 Model\\n"', '"# Comparison Summary: 4 Models\\n"'),
    ("'## Sentimen 3-class\\n'", "'## 3-Class Sentiment\\n'"),
    ('"## Sentimen 3-class\\n"', '"## 3-Class Sentiment\\n"'),
    ("'\\n## Toksisitas Multi-Label\\n'", "'\\n## Multi-Label Toxicity\\n'"),
    ('"\\n## Toksisitas Multi-Label\\n"', '"\\n## Multi-Label Toxicity\\n"'),
]

# ---------------------------------------------------------------------------
# Markdown files — already-generated reports
# ---------------------------------------------------------------------------
COMPARISON_SUMMARY_PAIRS = [
    ("# Ringkasan Perbandingan 4 Model", "# Comparison Summary: 4 Models"),
    ("## Sentimen 3-class", "## 3-Class Sentiment"),
    ("## Toksisitas Multi-Label", "## Multi-Label Toxicity"),
]

CORRELATION_SUMMARY_PAIRS = [
    ("# Ringkasan Korelasi Fitur Kontekstual", "# Contextual Feature Correlation Summary"),
    ("## Tabel Hasil Uji", "## Test Results Table"),
    ("## Finding Signifikan setelah BH-FDR (urut effect size desc)", "## Significant Findings after BH-FDR (sorted by effect size, desc)"),
    ("_Tidak ada finding signifikan setelah koreksi BH-FDR._", "_No significant findings after BH-FDR correction._"),
    ("## Catatan Limitasi", "## Limitations"),
    (
        "- Pesan dalam satu pertandingan TIDAK independen (clustering effect). Analisis robustness pada level per-pertandingan dilaporkan di `correlation_robustness_per_match.csv`.",
        "- Messages within the same match are NOT independent (clustering effect). Robustness analysis at per-match level is reported in `correlation_robustness_per_match.csv`.",
    ),
    (
        "- Kolom `phase` mayoritas bernilai `lainnya` karena dataset Dota 2 publik tidak menyertakan fase per-match secara eksplisit (lihat `reports/contextual_features.md` \u00a74).",
        "- The `phase` column is predominantly `lainnya` because the public Dota 2 dataset does not include explicit per-match phase labels (see `reports/contextual_features.md` \u00a74).",
    ),
    (
        "- Effect size yang sangat kecil (mis. Cram\u00e9r's V < 0.1, |rho| < 0.1) menunjukkan signifikansi statistik tanpa relevansi praktis pada n besar.",
        "- Very small effect sizes (e.g., Cram\u00e9r's V < 0.1, |rho| < 0.1) indicate statistical significance without practical relevance at large n.",
    ),
]

CONTEXTUAL_FEATURES_PAIRS = [
    ("# Definisi Fitur Kontekstual", "# Contextual Feature Definitions"),
    (
        "Dokumen ini mendefinisikan secara operasional kelima fitur kontekstual yang dipakai di analisis korelasi (notebook 08) dan time-series (notebook 07). Sumber data: `data/processed/<folder>.parquet`.",
        "This document provides operational definitions for the five contextual features used in the correlation analysis (notebook 08) and time-series analysis (notebook 07). Data source: `data/processed/<folder>.parquet`.",
    ),
    ("## 1. `match_outcome_for_player` \u2014 Hasil pertandingan dari sudut pandang pengirim chat", "## 1. `match_outcome_for_player` \u2014 Match outcome from the perspective of the chat sender"),
    ("- **Tipe**: kategorikal biner.", "- **Type**: binary categorical."),
    ("- **Nilai**: `win` / `loss` / `unknown`.", "- **Values**: `win` / `loss` / `unknown`."),
    ("- **Sumber kolom mentah**:", "- **Raw column source**:"),
    ("- **Aturan turunan**:", "- **Derivation rules**:"),
    ("  - Pemain Radiant memiliki `player_slot < 128`; Dire `player_slot >= 128`.", "  - Radiant players have `player_slot < 128`; Dire players have `player_slot >= 128`."),
    ("  - Jika pemain Radiant DAN `radiant_win=True` \u2192 `win`.", "  - If Radiant player AND `radiant_win=True` \u2192 `win`."),
    ("  - Jika pemain Dire DAN `radiant_win=False` \u2192 `win`.", "  - If Dire player AND `radiant_win=False` \u2192 `win`."),
    ("  - Lainnya \u2192 `loss`.", "  - Otherwise \u2192 `loss`."),
    ("  - `radiant_win` atau `player_slot` NaN \u2192 `unknown` (dieksklusi dari uji statistik).", "  - `radiant_win` or `player_slot` NaN \u2192 `unknown` (excluded from statistical tests)."),
    ("- **Implementasi**:", "- **Implementation**:"),
    ("## 2. `duration` \u2014 Durasi pertandingan", "## 2. `duration` \u2014 Match duration"),
    ("- **Tipe**: kontinu (detik).", "- **Type**: continuous (seconds)."),
    ("- **Sumber**: kolom `duration` di `main_metadata.csv`.", "- **Source**: `duration` column in `main_metadata.csv`."),
    ("- **Bin kategorikal** (untuk visualisasi):", "- **Categorical bins** (for visualization):"),
    ("  - `short`: `< 1800` (< 30 menit)", "  - `short`: `< 1800` (< 30 minutes)"),
    ("  - `medium`: `1800 \u2264 duration \u2264 2700` (30-45 menit)", "  - `medium`: `1800 \u2264 duration \u2264 2700` (30\u201345 minutes)"),
    ("  - `long`: `> 2700` (> 45 menit)", "  - `long`: `> 2700` (> 45 minutes)"),
    ("- **Catatan**: pertandingan profesional umumnya 25-60 menit; pertandingan < 15 menit kemungkinan early-GG atau forfeit.", "- **Note**: professional matches typically last 25\u201360 minutes; matches < 15 minutes likely indicate early-GG or forfeit."),
    ("## 3. `tier` \u2014 Tier turnamen (DERIVED)", "## 3. `tier` \u2014 Tournament tier (DERIVED)"),
    ("- **Tipe**: kategorikal ordinal.", "- **Type**: ordinal categorical."),
    ("- **Nilai**: `TI` > `Major` > `DPC Tour` > `Lainnya`.", "- **Values**: `TI` > `Major` > `DPC Tour` > `Lainnya`."),
    ("- **Aturan turunan** (`src/pipeline/enrich.py::_derive_dota_tier`):", "- **Derivation rules** (`src/pipeline/enrich.py::_derive_dota_tier`):"),
    ("  - OpenDota `tier \u2208 {excluded, amateur, \"\"}` \u2192 `Lainnya` (langsung).", "  - OpenDota `tier \u2208 {excluded, amateur, \"\"}` \u2192 `Lainnya` (direct)."),
    ('  - `leaguename` mengandung "Qualifier" \u2192 `Lainnya` (terlepas dari tier).', '  - `leaguename` contains "Qualifier" \u2192 `Lainnya` (regardless of tier).'),
    ('  - `leaguename` mengandung "The International" + tahun/nomor \u2192 `TI`.', '  - `leaguename` contains "The International" + year/number \u2192 `TI`.'),
    ('  - `leaguename` mengandung "Major" \u2192 `Major`.', '  - `leaguename` contains "Major" \u2192 `Major`.'),
    ('  - `leaguename` mengandung "DPC", "Dota Pro Circuit", atau "Tour [123]" \u2192 `DPC Tour`.', '  - `leaguename` contains "DPC", "Dota Pro Circuit", or "Tour [123]" \u2192 `DPC Tour`.'),
    ("  - Sisanya \u2192 `Lainnya`.", "  - Otherwise \u2192 `Lainnya`."),
    ("- **Distribusi mapping** (per April 2026, dari Constants.Leagues.csv 8.812 entries): TI=14, Major=24, DPC Tour=97, Lainnya=8.677.", "- **Mapping distribution** (as of April 2026, from Constants.Leagues.csv, 8,812 entries): TI=14, Major=24, DPC Tour=97, Other=8,677."),
    ("- **Limitasi**: leagueid di `Lainnya` mencakup turnamen non-DPC (mis. Esports World Cup, Riyadh Masters, Snow Sweet Snow). Jika diperlukan finer-grained, perluas regex di `_derive_dota_tier`.", "- **Limitation**: leagueid mapped to `Lainnya` includes non-DPC tournaments (e.g., Esports World Cup, Riyadh Masters, Snow Sweet Snow). For finer granularity, extend the regex in `_derive_dota_tier`."),
    ("## 4. `phase` \u2014 Fase turnamen (DERIVED, LIMITASI BERAT)", "## 4. `phase` \u2014 Tournament phase (DERIVED, SEVERE LIMITATION)"),
    ("- **Tipe**: kategorikal.", "- **Type**: categorical."),
    ("- **Nilai**: `group_stage` / `playoffs` / `grand_final` / `lainnya`.", "- **Values**: `group_stage` / `playoffs` / `grand_final` / `lainnya`."),
    ("- **Sumber**: heuristik regex pada `leaguename` (`src/pipeline/enrich.py::derive_phase`).", "- **Source**: regex heuristic on `leaguename` (`src/pipeline/enrich.py::derive_phase`)."),
    ("  - `leaguename` mengandung \"Grand Final\" \u2192 `grand_final`.", "  - `leaguename` contains \"Grand Final\" \u2192 `grand_final`."),
    ("  - mengandung \"Final\" atau \"Playoff\" \u2192 `playoffs`.", "  - contains \"Final\" or \"Playoff\" \u2192 `playoffs`."),
    ("  - mengandung \"Group Stage\" \u2192 `group_stage`.", "  - contains \"Group Stage\" \u2192 `group_stage`."),
    ("  - default \u2192 `lainnya`.", "  - default \u2192 `lainnya`."),
    ("- **LIMITASI**: Dataset Dota 2 publik", "- **LIMITATION**: The public Dota 2 dataset"),
    ("tidak menyertakan kolom `phase` per-match secara eksplisit. Banyak turnamen memakai satu `leagueid` untuk seluruh fase (group + playoffs di-bundle), sehingga sebagian besar baris jatuh ke `lainnya`. Konsekuensi:", "does not include an explicit per-match `phase` column. Many tournaments use a single `leagueid` for all phases (group + playoffs bundled together), so the majority of rows fall to `lainnya`. Consequences:"),
    ("  - Uji korelasi `phase \u00d7 toxicity_score` MUNGKIN under-powered.", "  - The `phase \u00d7 toxicity_score` correlation test may be under-powered."),
    ("  - Interpretasi finding `phase` HARUS hati-hati \u2014 kemungkinan ada konfounder.", "  - Interpretation of `phase` findings MUST be cautious \u2014 confounders likely exist."),
    ("- **Mitigasi yang dipertimbangkan dan ditolak**:", "- **Mitigations considered and rejected**:"),
    ("  - **Match-level positional inference** (urutan match dalam leagueid \u2192 group=awal, playoffs=akhir): ditolak karena urutan match tidak selalu mencerminkan fase, dan butuh data bracket eksternal.", "  - **Match-level positional inference** (match order within leagueid \u2192 group=early, playoffs=late): rejected because match order does not reliably reflect phase and requires external bracket data."),
    ("  - **Manual labeling per leagueid**: ditolak karena 8.812 leagueid tidak feasible untuk skripsi 3 mahasiswa.", "  - **Manual labeling per leagueid**: rejected because 8,812 leagueid entries are not feasible for a 3-student thesis."),
    ("  - **Cross-reference Liquipedia per match_id**: ditolak karena membutuhkan scraping/API yang di luar ruang lingkup.", "  - **Cross-reference Liquipedia per match_id**: rejected because it requires scraping/API access outside the project scope."),
    ("- **Rekomendasi laporan**: melaporkan hasil `phase` sebagai analisis eksploratori, bukan klaim utama. Fokus klaim utama pada `tier`, `match_outcome_for_player`, `duration`, dan `period`.", "- **Report recommendation**: report `phase` results as exploratory analysis, not a primary claim. Main claims should focus on `tier`, `match_outcome_for_player`, `duration`, and `period`."),
    ("## 5. `period` \u2014 Periode waktu", "## 5. `period` \u2014 Time period"),
    ("- **Tipe**: kategorikal ordinal.", "- **Type**: ordinal categorical."),
    ("- **Granularitas utama**: tiga bin \u2014 `2016-2018`, `2019-2021`, `2022-2026`.", "- **Primary granularity**: three bins \u2014 `2016-2018`, `2019-2021`, `2022-2026`."),
    ("- **Granularitas alternatif** (opsional): per-tahun (2016 sd 2026).", "- **Alternative granularity** (optional): per-year (2016 to 2026)."),
    ("- **Sumber kolom mentah**: `start_date_time` (timestamp) \u2192 `year` (Int16, dari `joiner.py`).", "- **Raw column source**: `start_date_time` (timestamp) \u2192 `year` (Int16, from `joiner.py`)."),
    ("- **Aturan turunan**: `period` di-compute on-the-fly di notebook 08 dari kolom `year`:", "- **Derivation rules**: `period` is computed on-the-fly in notebook 08 from the `year` column:"),
    ("  - `year \u2208 {2016, 2017, 2018}` \u2192 `2016-2018`.", "  - `year \u2208 {2016, 2017, 2018}` \u2192 `2016-2018`."),
    ("  - `year \u2208 {2019, 2020, 2021}` \u2192 `2019-2021`.", "  - `year \u2208 {2019, 2020, 2021}` \u2192 `2019-2021`."),
    ("  - `year \u2208 {2022, 2023, 2024, 2025, 2026}` \u2192 `2022-2026`.", "  - `year \u2208 {2022, 2023, 2024, 2025, 2026}` \u2192 `2022-2026`."),
    ("- **Rasional bin**: tiga era yang masing-masing menampung peristiwa signifikan (TI awal pasca-DPC formation, era pandemi, era post-DPC restructuring).", "- **Bin rationale**: three eras each encompassing significant events (early TI post-DPC formation, pandemic era, post-DPC restructuring era)."),
    ("## 6. `patch_name` \u2014 Versi patch saat pertandingan", "## 6. `patch_name` \u2014 Patch version at match time"),
    ("- **Tipe**: kategorikal.", "- **Type**: categorical."),
    ("- **Sumber**: `start_date_time` \u00d7 `data/processed/patch_map.csv` (date-range lookup).", "- **Source**: `start_date_time` \u00d7 `data/processed/patch_map.csv` (date-range lookup)."),
    ("- **Aturan**: untuk setiap pertandingan, ambil patch yang `date_start \u2264 start_date_time \u2264 date_end`.", "- **Rule**: for each match, select the patch where `date_start \u2264 start_date_time \u2264 date_end`."),
    ("- **Catatan**: kolom `patch` di `main_metadata.csv` adalah integer internal Valve yang tidak langsung dipetakan ke versi (mis. `7.34`); kami pakai date-range lookup dari `Constants.Patch.csv` sebagai sumber otoritatif.", "- **Note**: the `patch` column in `main_metadata.csv` is a Valve internal integer not directly mapped to human-readable versions (e.g., `7.34`); we use date-range lookup from `Constants.Patch.csv` as the authoritative source."),
    ("---", "---"),
    ("## Ringkasan kolom keluaran processed", "## Processed Output Column Summary"),
    ("| Kolom | Sumber | Tipe |", "| Column | Source | Type |"),
    ("|-------|--------|------|", "|--------|--------|------|"),
    ("| `match_outcome_for_player` | derived | string `{win, loss, unknown}` |", "| `match_outcome_for_player` | derived | string `{win, loss, unknown}` |"),
    ("| `duration` | metadata | Int64 (detik) |", "| `duration` | metadata | Int64 (seconds) |"),
    ("| `tier` | derived | string `{TI, Major, DPC Tour, Lainnya}` |", "| `tier` | derived | string `{TI, Major, DPC Tour, Lainnya}` |"),
    ("| `leaguename` | Constants.Leagues | string |", "| `leaguename` | Constants.Leagues | string |"),
    ("| `phase` | derived (terbatas) | string `{group_stage, playoffs, grand_final, lainnya}` |", "| `phase` | derived (limited) | string `{group_stage, playoffs, grand_final, lainnya}` |"),
    ("| `patch_name` | Constants.Patch + start_date_time | string |", "| `patch_name` | Constants.Patch + start_date_time | string |"),
    ("| `year`, `month`, `quarter` | derived | Int16/Int8/Int8 |", "| `year`, `month`, `quarter` | derived | Int16/Int8/Int8 |"),
    ("`period` di-compute on-the-fly di analisis (tidak disimpan di parquet supaya bin bisa diubah tanpa re-run pipeline).", "`period` is computed on-the-fly during analysis (not stored in parquet so bins can be changed without re-running the pipeline)."),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Patching notebooks ===")
    patch_notebook(ROOT / "notebooks" / "07_temporal_analysis.ipynb", NB07_PAIRS)
    patch_notebook(ROOT / "notebooks" / "08_correlation_analysis.ipynb", NB08_PAIRS)
    patch_notebook(ROOT / "notebooks" / "06_comparative_evaluation.ipynb", NB06_PAIRS)

    print("\n=== Patching existing markdown reports ===")
    patch_text(ROOT / "reports" / "comparison_summary.md", COMPARISON_SUMMARY_PAIRS)
    patch_text(ROOT / "reports" / "correlation_summary.md", CORRELATION_SUMMARY_PAIRS)
    patch_text(ROOT / "reports" / "contextual_features.md", CONTEXTUAL_FEATURES_PAIRS)

    print("\nDone. Re-run notebooks 06-08 to regenerate plots with English titles.")


if __name__ == "__main__":
    main()
