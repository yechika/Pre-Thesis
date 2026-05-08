## ADDED Requirements

### Requirement: Definisi fitur kontekstual yang dianalisis

Sistem SHALL menganalisis korelasi label sentimen dan toksisitas terhadap fitur kontekstual berikut:

1. **Hasil pertandingan (`match_outcome_for_player`)**: kategorikal biner — `win` / `loss`.
2. **Durasi pertandingan (`duration`)**: kontinu — detik. Juga dianalisis dalam bin kategorikal: short (<30 menit), medium (30-45 menit), long (>45 menit).
3. **Tier turnamen (`tier`)**: ordinal — `TI` > `Major` > `DPC Tour` > `Lainnya`.
4. **Fase turnamen (`phase`)**: kategorikal — `group_stage`, `playoffs`, `grand_final`, `lainnya`. Dapat diturunkan dari nama liga atau metadata jika tersedia.
5. **Periode waktu (`period`)**: kategorikal — `2016-2018`, `2019-2021`, `2022-2026`. Granularitas alternatif (per-tahun) opsional.

Setiap fitur MUST didokumentasikan di `reports/contextual_features.md` dengan definisi operasional, sumber kolom mentah, dan langkah turunan.

#### Scenario: Fitur tersedia di dataset
- **WHEN** join data inferensi + processed selesai
- **THEN** kolom `match_outcome_for_player`, `duration`, `tier`, `phase`, `period` tersedia di unit analisis (per-pesan atau per-pertandingan agregat)

#### Scenario: Tier diturunkan dari leagueid
- **WHEN** `leagueid` cocok dengan ID turnamen TI tahunan
- **THEN** `tier == "TI"`; mapping `leagueid → tier` didokumentasikan di `data/processed/league_tier_map.csv`

### Requirement: Uji statistik per-tipe variabel

Sistem SHALL menjalankan uji statistik yang sesuai untuk setiap fitur kontekstual × label:

- **Hasil (biner) × sentimen (kategorikal 3-class)** → chi-square test of independence.
- **Hasil (biner) × toxicity_score (kontinu)** → Mann-Whitney U test.
- **Durasi (kontinu) × sentiment_score (kontinu)** → Spearman rank correlation.
- **Durasi (kontinu) × toxicity_score (kontinu)** → Spearman rank correlation.
- **Tier (ordinal) × toxicity_score** → Kruskal-Wallis dengan post-hoc Dunn's test.
- **Tier (ordinal) × sentimen** → chi-square (jika n besar) atau Cramér's V.
- **Fase × toxicity_score** → Kruskal-Wallis + Dunn's.
- **Fase × sentimen** → chi-square + Cramér's V.
- **Periode × toxicity_score** → Kruskal-Wallis + Dunn's.

Hasil setiap uji MUST dilaporkan di `reports/correlation_tests.csv` dengan kolom: `feature`, `label`, `test_name`, `test_statistic`, `p_value`, `effect_size`, `df`, `n`, `interpretation`.

#### Scenario: Chi-square hasil × sentimen
- **WHEN** uji chi-square independensi `match_outcome_for_player × sentiment` dijalankan
- **THEN** `reports/correlation_tests.csv` berisi baris dengan test_statistic, df, p_value, dan effect_size (Cramér's V)

#### Scenario: Spearman durasi × toxicity
- **WHEN** Spearman correlation `duration × toxicity_score` dijalankan di seluruh pertandingan agregat
- **THEN** koefisien rho, p_value, dan jumlah n dilaporkan; nilai n MUST cocok dengan jumlah unique match_id

### Requirement: Koreksi multiple-testing dengan Benjamini-Hochberg FDR

Sistem SHALL menerapkan koreksi Benjamini-Hochberg FDR dengan q = 0.05 di seluruh suite uji korelasi (≈20-30 uji).

Tabel `reports/correlation_tests.csv` MUST berisi kolom `p_adj_bh` (p-value setelah koreksi) dan `significant_after_bh` (boolean).

Interpretasi di laporan akhir HANYA mengklaim signifikansi untuk hasil dengan `significant_after_bh == True`.

#### Scenario: BH koreksi diterapkan
- **WHEN** seluruh uji selesai dan tabel ditulis
- **THEN** kolom `p_adj_bh` ada untuk setiap baris dan urutan p_adj monoton naik sesuai p_value

#### Scenario: Klaim signifikan hanya pada p_adj < 0.05
- **WHEN** uji `tier × toxicity_score` memiliki p = 0.03 tapi p_adj_bh = 0.07
- **THEN** laporan TIDAK mengklaim hasil tersebut signifikan setelah koreksi

### Requirement: Visualisasi korelasi

Sistem SHALL menghasilkan plot pendukung di `reports/plots/`:
- `correlation_outcome_sentiment.png/svg`: bar chart proporsi sentimen per outcome (win/loss).
- `correlation_outcome_toxicity.png/svg`: violin/box plot toxicity_score per outcome.
- `correlation_duration_toxicity.png/svg`: scatter dengan trend line LOWESS dari duration vs toxicity.
- `correlation_tier_toxicity.png/svg`: box plot toxicity_score per tier dengan post-hoc significance bracket.
- `correlation_phase_toxicity.png/svg`: box plot toxicity_score per fase.

Setiap plot MUST menyertakan label sumbu, judul, jumlah n, dan p-value uji yang relevan.

#### Scenario: Plot violin outcome × toxicity
- **WHEN** notebook korelasi dijalankan
- **THEN** `reports/plots/correlation_outcome_toxicity.svg` ada dan menampilkan distribusi toxicity per outcome dengan p-value Mann-Whitney di subtitle

### Requirement: Ringkasan korelasi siap-kutip

Sistem SHALL menghasilkan `reports/correlation_summary.md` berisi:
- Tabel ringkasan: untuk setiap fitur, hasil uji utama, p_adj, effect size, interpretasi singkat (1 kalimat).
- Daftar finding signifikan setelah BH dengan urutan dari effect size terbesar.
- Catatan kelemahan (mis. konfounder, sampel tidak independen antar pesan dalam satu pertandingan).

#### Scenario: Ringkasan ditulis
- **WHEN** seluruh uji + plot selesai
- **THEN** `reports/correlation_summary.md` ada dengan tabel + daftar finding + catatan kelemahan

#### Scenario: Catatan independensi
- **WHEN** uji per-pesan dijalankan
- **THEN** ringkasan menyebutkan secara eksplisit bahwa pesan dalam satu pertandingan tidak independen (clustering effect) dan analisis robustness pada level per-pertandingan dilaporkan sebagai kontrol
