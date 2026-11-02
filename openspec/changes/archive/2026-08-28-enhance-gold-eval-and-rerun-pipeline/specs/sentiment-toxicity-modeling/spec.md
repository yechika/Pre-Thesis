## ADDED Requirements

### Requirement: In-Domain Sentiment Fine-Tuning dengan Bilingual Slang Normalization
Sistem SHALL melakukan fine-tuning arsitektur transformer (BERT, RoBERTa, DistilBERT) untuk tugas sentimen menggunakan data in-domain Dota 2 chat yang telah dinormalisasi dengan kamus slang dwibahasa (Inggris + Indonesia) dan diberi penyeimbang kelas (class-weighted loss / oversampling minoritas).

#### Scenario: In-Domain Sentiment Fine-Tuning Berhasil
- **WHEN** training sentimen in-domain dijalankan pada `notebooks/04_training.ipynb`
- **THEN** model menghasilkan Balanced Accuracy ≥ 0.70 dan MCC ≥ 0.80 pada hold-out test set dan checkpoint tersimpan di `models/<model_name>-sentiment`

### Requirement: Optimasi Decision Threshold untuk Multi-Label Toksisitas
Sistem SHALL mengoptimasi ambang batas probabilitas keputusan (decision threshold) per-label secara empiris pada data validasi untuk memaksimalkan skor F1 pada masing-masing label toksisitas yang mengalami imbalance.

#### Scenario: Threshold Tuning per Label
- **WHEN** proses tuning threshold dijalankan
- **THEN** sistem menghasilkan ambang batas spesifik per label (toxic, obscene, insult, severe_toxic, threat, identity_hate) yang mengungguli threshold default 0.5
