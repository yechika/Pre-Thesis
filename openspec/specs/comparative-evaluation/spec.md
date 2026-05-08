# comparative-evaluation

## Purpose

Evaluasi komparatif empat model di gold-test held-out dengan bootstrap CI, confusion matrix, error analysis, dan ringkasan siap-kutip.

## Requirements

### Requirement: Evaluasi keempat model di gold-standard test set

Sistem SHALL mengevaluasi setiap model (BERT, RoBERTa, DistilBERT, Detoxify) pada test split gold-standard (30% dari `data/gold/gold_final.csv`) untuk kedua tugas (sentimen 3-class dan toksisitas multi-label).

Test split MUST tetap held-out — tidak boleh dipakai untuk fine-tuning, hyperparameter selection, atau early stopping.

#### Scenario: Test split held-out
- **WHEN** training/fine-tuning dijalankan
- **THEN** kode training MUST tidak membaca `data/gold/test/*` (validasi kode + log mencatat hash file gold yang dibaca)

#### Scenario: Evaluasi seluruh model di test sama
- **WHEN** seluruh evaluasi selesai
- **THEN** setiap model dievaluasi pada test split yang IDENTIK (cek hash baris)

### Requirement: Metrik standar untuk perbandingan

Sistem SHALL melaporkan metrik berikut untuk **sentimen 3-class** per-model: akurasi, F1-macro, F1 per-class (positive/neutral/negative), presisi-macro, recall-macro, confusion matrix, dan ROC-AUC One-vs-Rest.

Sistem SHALL melaporkan metrik berikut untuk **toksisitas multi-label** per-model: F1-micro, F1-macro, F1 per-label (toxic/severe_toxic/obscene/threat/insult/identity_hate), Hamming loss, Subset accuracy (exact match), Average Precision per-label.

Setiap metrik MUST disertai **bootstrap 95% confidence interval** dengan minimal 1000 resample.

#### Scenario: Metrik sentimen lengkap
- **WHEN** evaluasi sentimen BERT selesai
- **THEN** `reports/eval_sentiment.csv` berisi baris untuk BERT dengan akurasi, F1-macro, F1 per-class, dan kolom CI lower/upper

#### Scenario: Metrik toksisitas lengkap
- **WHEN** evaluasi toksisitas RoBERTa selesai
- **THEN** `reports/eval_toxicity.csv` berisi baris untuk RoBERTa dengan F1-micro, F1-macro, F1 per-label (6 label), Hamming loss, Subset accuracy, dengan CI lower/upper

#### Scenario: Bootstrap CI dilaporkan
- **WHEN** metrik akurasi BERT dilaporkan sebagai 0.84
- **THEN** kolom `ci_lower` dan `ci_upper` (mis. 0.82, 0.86) ada di tabel hasil dan dicatat dengan jumlah resample = 1000

### Requirement: Confusion matrix dan analisis error

Sistem SHALL menghasilkan confusion matrix per-model untuk sentimen (3×3) dan per-label untuk toksisitas (2×2 per label).

Sistem SHALL menyediakan analisis error: minimal 50 contoh false-positive dan 50 false-negative per-model per-tugas, disampling acak dari test set, dengan kolom: pesan, label gold, prediksi, probabilitas. Output di `reports/error_samples_<model>_<task>.csv`.

Sistem MUST melakukan breakdown error per-jargon: hitung error rate untuk pesan dengan `is_dota_jargon == 1` vs `is_dota_jargon == 0` dan laporkan beda.

#### Scenario: Confusion matrix dihasilkan
- **WHEN** evaluasi sentimen DistilBERT selesai
- **THEN** `reports/confusion_distilbert_sentiment.png` dan `.csv` ada

#### Scenario: Error sample ditulis
- **WHEN** evaluasi BERT-toxicity selesai
- **THEN** `reports/error_samples_bert_toxicity.csv` berisi 50+ FP dan 50+ FN dengan pesan asli dan probabilitas

#### Scenario: Breakdown jargon
- **WHEN** analisis error selesai
- **THEN** `reports/jargon_error_breakdown.csv` berisi error rate per-model per-tugas dengan dan tanpa jargon, plus delta error rate

### Requirement: Tabel perbandingan akhir untuk laporan

Sistem SHALL menghasilkan tabel ringkasan akhir di `reports/comparison_summary.md` berisi:
- Tabel sentimen: baris = model, kolom = akurasi (CI), F1-macro (CI), F1 per-class (CI).
- Tabel toksisitas: baris = model, kolom = F1-micro (CI), F1-macro (CI), Hamming loss, Subset accuracy.
- Highlight (bold) untuk skor tertinggi tiap kolom.
- Catatan signifikansi: pasangan model dengan beda F1-macro signifikan (CI tidak tumpang tindih) ditandai.

Format MUST siap-kutip ke laporan skripsi BAB IV — tabel markdown yang dapat dikonversi ke LaTeX/Word tanpa edit besar.

#### Scenario: Ringkasan komparatif lengkap
- **WHEN** seluruh evaluasi selesai
- **THEN** `reports/comparison_summary.md` ada dengan tabel sentimen + tabel toksisitas + highlight skor terbaik

#### Scenario: Pasangan signifikan ditandai
- **WHEN** F1-macro BERT dan F1-macro RoBERTa memiliki CI yang tidak tumpang tindih
- **THEN** ringkasan menyebutkan secara eksplisit: "RoBERTa > BERT pada F1-macro sentimen (CI tidak tumpang tindih)"
