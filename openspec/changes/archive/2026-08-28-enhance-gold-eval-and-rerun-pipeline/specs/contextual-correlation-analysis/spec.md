## ADDED Requirements

### Requirement: Analisis Korelasi Kontekstual dengan Koreksi BH-FDR dan Match-Level Robustness
Sistem SHALL menguji korelasi antara lima faktor kontekstual (`match_outcome`, `duration`, `tier`, `phase`, `period`) terhadap sentimen dan toksisitas dengan menerapkan koreksi multiple testing Benjamini-Hochberg (FDR q=0.05).

Sistem MUST menjalankan analisis robustness pada level per-match untuk mengontrol efek clustering pesan.

#### Scenario: Analisis Korelasi Kontekstual Diperbarui
- **WHEN** `notebooks/08_correlation_analysis.ipynb` dijalankan
- **THEN** output `reports/correlation_tests.csv`, `reports/correlation_robustness_per_match.csv`, `reports/correlation_summary.md`, dan plot `reports/plots/correlation_*.png` diperbarui
