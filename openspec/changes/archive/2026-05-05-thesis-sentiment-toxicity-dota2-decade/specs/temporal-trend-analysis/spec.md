## ADDED Requirements

### Requirement: Time-series sentimen dan toksisitas per-bulan dan per-tahun

Sistem SHALL menghasilkan time-series agregat dari hasil inferensi best model (atau ensemble — keputusan didokumentasikan) di seluruh `data/inference/` lintas dekade 2016-2026.

Granularitas yang DIWAJIBKAN: per-bulan (utama) dan per-tahun (ringkasan).

Metrik per-bin yang DIWAJIBKAN:
- `pct_positive`, `pct_neutral`, `pct_negative` (persentase pesan).
- `mean_sentiment_score` (skor sentimen kontinu, mis. P(positive) - P(negative)).
- `pct_toxic_any` (persentase pesan dengan minimal satu label toksisitas = 1).
- `mean_toxicity_score` (rata-rata `max(toxic_label_probs)` per pesan).
- `n_messages` (jumlah pesan di-bin) dan `n_matches` (jumlah pertandingan unik).

#### Scenario: Time-series bulanan dihasilkan
- **WHEN** analisis temporal selesai
- **THEN** `reports/temporal_monthly.csv` berisi baris per-bulan dari 2016-01 sd 2026-MM dengan seluruh metrik di atas

#### Scenario: Time-series tahunan dihasilkan
- **WHEN** analisis temporal selesai
- **THEN** `reports/temporal_yearly.csv` berisi 11 baris (2016 sd 2026) dengan metrik agregat

#### Scenario: Bin kosong ditangani
- **WHEN** suatu bulan tidak memiliki pesan (mis. off-season)
- **THEN** baris bulan tersebut tetap ada dengan metrik `NaN` dan `n_messages=0`, plot menampilkan gap tanpa interpolasi palsu

### Requirement: Plot time-series dengan event overlay

Sistem SHALL menghasilkan plot time-series berikut di `reports/plots/`:
- `temporal_sentiment_monthly.png/svg`: line chart `mean_sentiment_score` dan `pct_negative` per-bulan.
- `temporal_toxicity_monthly.png/svg`: line chart `pct_toxic_any` dan `mean_toxicity_score` per-bulan.
- `temporal_yearly_overview.png/svg`: bar chart agregat tahunan.

Setiap plot bulanan MUST menyertakan event overlay sebagai vertical lines atau horizontal bands, minimal:
- Pergantian patch major (dari kolom `patch` di metadata): patch 7.00 (2016-12), 7.20 (2018-11), 7.30 (2021-04), 7.33 (2023-04), 7.34+ (2023-08+).
- Era pandemi COVID-19 daring (band 2020-03 sd 2021-12).
- The International tahunan (vertical line tipis).
- Era format DPC (2017-2022) vs post-DPC (2023-2026) sebagai band warna latar.

#### Scenario: Plot dihasilkan dengan event overlay
- **WHEN** notebook analisis temporal dijalankan
- **THEN** seluruh plot di atas ada di `reports/plots/` dengan event overlay yang terlihat dan legenda yang jelas

#### Scenario: Plot dapat dipakai di laporan
- **WHEN** plot SVG dibuka
- **THEN** plot scalable tanpa pixelation untuk dimasukkan ke laporan skripsi

### Requirement: Uji tren statistik

Sistem SHALL menjalankan **Mann-Kendall trend test** pada series bulanan untuk:
- `mean_sentiment_score`
- `pct_negative`
- `pct_toxic_any`
- `mean_toxicity_score`

Sistem SHALL menjalankan **Chow test** (atau structural break test alternatif) untuk mendeteksi titik break struktural di series, dengan kandidat breakpoint: 2020-03 (mulai pandemi), 2023-01 (transisi post-DPC), 2023-04 (patch 7.33).

Hasil uji MUST dilaporkan di `reports/temporal_tests.csv` dengan kolom: `series`, `test_name`, `test_statistic`, `p_value`, `interpretation`.

#### Scenario: Mann-Kendall dilaporkan
- **WHEN** uji tren selesai
- **THEN** `reports/temporal_tests.csv` berisi baris untuk setiap kombinasi (series × Mann-Kendall) dengan p-value dan arah tren (increasing/decreasing/no-trend)

#### Scenario: Structural break dilaporkan
- **WHEN** Chow test pada series `pct_toxic_any` di breakpoint 2020-03 menghasilkan p < 0.05
- **THEN** laporan mencatat secara eksplisit "structural break terdeteksi di awal pandemi COVID-19 (p=...)"

### Requirement: Breakdown per-event ringkasan

Sistem SHALL menghasilkan tabel ringkasan komparatif `reports/event_comparison.csv` yang membandingkan agregat metrik antar era/event:
- Pre-pandemic (2016-2019) vs Pandemic-era online (2020-03 sd 2021-12) vs Post-pandemic LAN (2022-01+).
- DPC era (2017-2022) vs Post-DPC era (2023+).
- Per-patch era major (7.00 era / 7.20 era / 7.30 era / 7.33+ era).

Setiap perbandingan MUST menyertakan: rata-rata metrik, jumlah pesan, jumlah pertandingan, dan uji Mann-Whitney U dengan p-value (untuk kontinu) atau chi-square (untuk proporsi).

#### Scenario: Ringkasan event dihasilkan
- **WHEN** analisis event-comparison selesai
- **THEN** `reports/event_comparison.csv` berisi baris untuk setiap pasangan era dengan delta metrik dan p-value uji
