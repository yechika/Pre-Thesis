# Definisi Fitur Kontekstual

Dokumen ini mendefinisikan secara operasional kelima fitur kontekstual yang dipakai di analisis korelasi (notebook 08) dan time-series (notebook 07). Sumber data: `data/processed/<folder>.parquet`.

## 1. `match_outcome_for_player` — Hasil pertandingan dari sudut pandang pengirim chat

- **Tipe**: kategorikal biner.
- **Nilai**: `win` / `loss` / `unknown`.
- **Sumber kolom mentah**: `radiant_win` (bool, dari `main_metadata.csv`) × `player_slot` (float, dari `chat.csv`).
- **Aturan turunan**:
  - Pemain Radiant memiliki `player_slot < 128`; Dire `player_slot >= 128`.
  - Jika pemain Radiant DAN `radiant_win=True` → `win`.
  - Jika pemain Dire DAN `radiant_win=False` → `win`.
  - Lainnya → `loss`.
  - `radiant_win` atau `player_slot` NaN → `unknown` (dieksklusi dari uji statistik).
- **Implementasi**: `src/pipeline/joiner.py::_outcome_for_player`.

## 2. `duration` — Durasi pertandingan

- **Tipe**: kontinu (detik).
- **Sumber**: kolom `duration` di `main_metadata.csv`.
- **Bin kategorikal** (untuk visualisasi):
  - `short`: `< 1800` (< 30 menit)
  - `medium`: `1800 ≤ duration ≤ 2700` (30-45 menit)
  - `long`: `> 2700` (> 45 menit)
- **Catatan**: pertandingan profesional umumnya 25-60 menit; pertandingan < 15 menit kemungkinan early-GG atau forfeit.

## 3. `tier` — Tier turnamen (DERIVED)

- **Tipe**: kategorikal ordinal.
- **Nilai**: `TI` > `Major` > `DPC Tour` > `Lainnya`.
- **Sumber kolom mentah**: `leagueid` (`main_metadata.csv`) × `Constants.Leagues.csv` (`leaguename`, OpenDota `tier`).
- **Aturan turunan** (`src/pipeline/enrich.py::_derive_dota_tier`):
  - OpenDota `tier ∈ {excluded, amateur, ""}` → `Lainnya` (langsung).
  - `leaguename` mengandung "Qualifier" → `Lainnya` (terlepas dari tier).
  - `leaguename` mengandung "The International" + tahun/nomor → `TI`.
  - `leaguename` mengandung "Major" → `Major`.
  - `leaguename` mengandung "DPC", "Dota Pro Circuit", atau "Tour [123]" → `DPC Tour`.
  - Sisanya → `Lainnya`.
- **Distribusi mapping** (per April 2026, dari Constants.Leagues.csv 8.812 entries): TI=14, Major=24, DPC Tour=97, Lainnya=8.677.
- **Limitasi**: leagueid di `Lainnya` mencakup turnamen non-DPC (mis. Esports World Cup, Riyadh Masters, Snow Sweet Snow). Jika diperlukan finer-grained, perluas regex di `_derive_dota_tier`.

## 4. `phase` — Fase turnamen (DERIVED, LIMITASI BERAT)

- **Tipe**: kategorikal.
- **Nilai**: `group_stage` / `playoffs` / `grand_final` / `lainnya`.
- **Sumber**: heuristik regex pada `leaguename` (`src/pipeline/enrich.py::derive_phase`).
  - `leaguename` mengandung "Grand Final" → `grand_final`.
  - mengandung "Final" atau "Playoff" → `playoffs`.
  - mengandung "Group Stage" → `group_stage`.
  - default → `lainnya`.
- **LIMITASI**: Dataset Dota 2 publik (Kaggle `bwandowando/dota-2-pro-league-matches-2023`) tidak menyertakan kolom `phase` per-match secara eksplisit. Banyak turnamen memakai satu `leagueid` untuk seluruh fase (group + playoffs di-bundle), sehingga sebagian besar baris jatuh ke `lainnya`. Konsekuensi:
  - Uji korelasi `phase × toxicity_score` MUNGKIN under-powered.
  - Interpretasi finding `phase` HARUS hati-hati — kemungkinan ada konfounder.
- **Mitigasi yang dipertimbangkan dan ditolak**:
  - **Match-level positional inference** (urutan match dalam leagueid → group=awal, playoffs=akhir): ditolak karena urutan match tidak selalu mencerminkan fase, dan butuh data bracket eksternal.
  - **Manual labeling per leagueid**: ditolak karena 8.812 leagueid tidak feasible untuk skripsi 3 mahasiswa.
  - **Cross-reference Liquipedia per match_id**: ditolak karena membutuhkan scraping/API yang di luar ruang lingkup.
- **Rekomendasi laporan**: melaporkan hasil `phase` sebagai analisis eksploratori, bukan klaim utama. Fokus klaim utama pada `tier`, `match_outcome_for_player`, `duration`, dan `period`.

## 5. `period` — Periode waktu

- **Tipe**: kategorikal ordinal.
- **Granularitas utama**: tiga bin — `2016-2018`, `2019-2021`, `2022-2026`.
- **Granularitas alternatif** (opsional): per-tahun (2016 sd 2026).
- **Sumber kolom mentah**: `start_date_time` (timestamp) → `year` (Int16, dari `joiner.py`).
- **Aturan turunan**: `period` di-compute on-the-fly di notebook 08 dari kolom `year`:
  - `year ∈ {2016, 2017, 2018}` → `2016-2018`.
  - `year ∈ {2019, 2020, 2021}` → `2019-2021`.
  - `year ∈ {2022, 2023, 2024, 2025, 2026}` → `2022-2026`.
- **Rasional bin**: tiga era yang masing-masing menampung peristiwa signifikan (TI awal pasca-DPC formation, era pandemi, era post-DPC restructuring).

## 6. `patch_name` — Versi patch saat pertandingan

- **Tipe**: kategorikal.
- **Sumber**: `start_date_time` × `data/processed/patch_map.csv` (date-range lookup).
- **Aturan**: untuk setiap pertandingan, ambil patch yang `date_start ≤ start_date_time ≤ date_end`.
- **Catatan**: kolom `patch` di `main_metadata.csv` adalah integer internal Valve yang tidak langsung dipetakan ke versi (mis. `7.34`); kami pakai date-range lookup dari `Constants.Patch.csv` sebagai sumber otoritatif.

---

## Ringkasan kolom keluaran processed

| Kolom | Sumber | Tipe |
|-------|--------|------|
| `match_outcome_for_player` | derived | string `{win, loss, unknown}` |
| `duration` | metadata | Int64 (detik) |
| `tier` | derived | string `{TI, Major, DPC Tour, Lainnya}` |
| `leaguename` | Constants.Leagues | string |
| `phase` | derived (terbatas) | string `{group_stage, playoffs, grand_final, lainnya}` |
| `patch_name` | Constants.Patch + start_date_time | string |
| `year`, `month`, `quarter` | derived | Int16/Int8/Int8 |

`period` di-compute on-the-fly di analisis (tidak disimpan di parquet supaya bin bisa diubah tanpa re-run pipeline).
