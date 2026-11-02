## ADDED Requirements

### Requirement: Full-Corpus Re-Inference dan Analisis Tren Temporal 124 Bulan
Sistem SHALL menjalankan inferensi ulang pada seluruh dataset chat 2016–2026 (1.38M+ pesan) menggunakan checkpoint in-domain terpilih, lalu mengagregasi time-series bulanan (124 bulan) untuk menghitung mean sentiment score, persentase pesan negatif, persentase pesan toksik, dan mean toxicity score.

Sistem MUST menjalankan uji tren non-parametrik Mann-Kendall dan uji structural break Chow test pada titik break pre-specified (Pandemic Maret 2020, Post-DPC Januari 2023, Patch 7.33 April 2023).

#### Scenario: Analisis Temporal Diperbarui dari Inferensi In-Domain
- **WHEN** `notebooks/05_inference.ipynb` dan `notebooks/07_temporal_analysis.ipynb` dijalankan
- **THEN** output `reports/temporal_monthly.csv`, `reports/temporal_tests.csv`, dan seluruh grafik tren `reports/plots/temporal_*.png` diperbarui secara otomatis
