## ADDED Requirements

### Requirement: Evaluasi Ganda pada Hold-out Population dan Manual Enriched Probe
Sistem SHALL mengevaluasi performa model komparatif pada dua set uji terpisah:
1. Hold-out population gold-test set (n≈2,392) untuk mengukur estimasi performa pada prevalensi natural.
2. Manual labeled enriched probe (n=150) untuk mengukur daya diskriminasi murni pada data seimbang bebas bias kelangkaan kelas.

Metrik evaluasi MUST mencakup Balanced Accuracy, Matthews Correlation Coefficient (MCC), Macro F1, Precision-Recall AUC (PR-AUC), dan per-label F1.

#### Scenario: Evaluasi Komparatif Lengkap Dihasilkan
- **WHEN** `notebooks/06_comparative_evaluation.ipynb` dan `notebooks/09_imbalance_metrics.ipynb` dijalankan
- **THEN** sistem menuliskan tabel perbandingan komprehensif ke `reports/comparison_summary.md`, `reports/eval_sentiment.csv`, dan `reports/eval_toxicity.csv`
