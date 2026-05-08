# Agreement Report — Gold-Standard Anotasi (Mode Single-Annotator)

Seed: 42  |  Eksekusi: 2026-05-04T08:50:43+00:00
Total sample: 7,971

## Mode anotasi: AI-only (Single-Annotator, Pilihan B)

Sample diberi label oleh **1 anotator AI** (Claude, model claude-opus-4-7) menggunakan
kombinasi rule-based matching + reasoning LLM. **Cohen's kappa dan Krippendorff's
alpha tidak dapat dihitung** karena membutuhkan minimal 2 rater independen.

### Implikasi metodologi (untuk skripsi BAB III):

1. **Inter-annotator agreement** TIDAK dilaporkan untuk gold-test set.
2. **Reliability** gold-test set bergantung sepenuhnya pada konsistensi 1 anotator AI.
3. **Bias risk**: jika model evaluasi (BERT/RoBERTa/DistilBERT) punya bias representasi
   yang mirip dengan anotator AI, hasil F1 mungkin over-optimistic. Gunakan
   `tox_identity_hate` dan kolom `is_ambiguous` sebagai proxy untuk audit.
4. **Mitigasi**: 38% sample ditandai `is_ambiguous=1` — ini bisa di-eksklusi atau
   di-evaluasi terpisah di notebook 06 untuk subgroup analysis.

### Untuk validitas skripsi yang lebih kuat:

Direkomendasikan migrasi ke **Pilihan A (multi-annotator)** sebelum publikasi —
tim peneliti (Daniel/Dhitan/Aldiaz) label minimal 200 sample bersama, hitung kappa
antara AI dan masing-masing anotator manusia. Threshold publikasi: kappa pair-wise
AI-vs-human >= 0.6 untuk sentimen.

## Distribusi label gold_final

### Sentiment

| Label | n | % |
|-------|---|---|
| positive | 4,512 | 56.6% |
| neutral | 3,357 | 42.1% |
| negative | 102 | 1.3% |

### Toxicity (per-label, jumlah baris dengan label = 1)

| Label | n | % |
|-------|---|---|
| toxic | 68 | 0.85% |
| severe_toxic | 0 | 0.00% |
| obscene | 53 | 0.66% |
| threat | 0 | 0.00% |
| insult | 66 | 0.83% |
| identity_hate | 2 | 0.03% |

### Markers

| Marker | n | % |
|--------|---|---|
| is_dota_jargon=1 | 4,257 | 53.4% |
| is_ambiguous=1 | 3,037 | 38.1% |

## Catatan untuk reviewer

Sample dengan `is_ambiguous=1` adalah baris yang anotator AI tidak yakin (mis. sarcasm
tidak terdeteksi, frasa multi-bahasa, atau token tidak dikenal seperti "gwr"). Saran:
eksklusi dari uji metrik utama atau lakukan analisis terpisah.