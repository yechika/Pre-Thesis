# HANDOFF — Re-run Paper 1 di mesin RTX 3060

Dokumen ini untuk sesi Claude Code (atau orang) di mesin RTX 3060. Baca semua sebelum mulai.

## 1. Konteks singkat

- **Paper**: "Comparative Sentiment Analysis and Toxicity Detection on Professional Dota 2 In-Game Chat: A Decade Review (2016–2026)" — ICoDiT 2026, **Paper ID 64**, sudah *accepted* (Springer LNCS). Penulis: Daniel Dirgantara (corresponding), Muhammad Dhitan Imam Sakti, Aldiaz Kusuma Ramadhan, Eko Setyo Purwanto (BINUS).
- **Deadline revisi di CMT: 9 November 2026** (rebuttal letter + artikel track changes + artikel clean). Hasil run harus kembali ke laptop utama paling lambat **7 November** supaya sempat mengisi naskah.
- Naskah revisi + rebuttal sudah disiapkan di laptop utama, dengan placeholder kuning untuk semua angka yang bergantung pada run ulang. Tugas mesin ini **hanya menghasilkan angka/plot**, bukan menulis paper.

## 2. Kenapa harus run ulang

1. **Label gold baru.** Ketiga penulis (Daniel, Dhitan, Aldiaz) melabel 8.121 pesan gold **secara manual dan independen**; semua angka model harus dihitung ulang dengan label ini. File versi pilot (lokal saja, tidak di-push) tidak boleh dipakai.
   - Pesan identik dalam konteks identik (teks dinormalisasi × 20% terakhir match × hasil tim pengirim) dilabel sekali sebagai satu unit, lalu labelnya disalin ke semua kemunculan: 8.121 pesan = 2.441 unit.
   - 300 unit dilabel ketiga annotator. Fleiss' κ sentimen **0,773**, toxic **0,792** (lihat `data/gold/manual_relabel/agreement.json`). Konsensus pakai majority vote; 2 unit seri (tiga label berbeda) sementara diisi `neutral`.
   - Split train/test **sama dengan sebelumnya** (5.684 / 2.437 baris yang sama), hanya labelnya yang baru.
2. **Bug merge diperbaiki** di `src/analysis/temporal.py`. Dulu inferensi digabung dengan metadata pakai `(match_id, time, player_slot)` + `drop_duplicates`, sehingga 222.700 pesan berbeda (pemain sama, detik sama) terbuang. Sekarang dipakai nomor urut per key (`_seq`), jadi **semua 1.603.567 pesan** ikut. Untuk tes `match_outcome` n = 1.568.863 (34.704 pesan tanpa `player_slot` = observer/caster, outcome `unknown`).
3. **Model out-of-domain (TweetEval) hilang.** Folder `models/{bert,roberta,distilbert}-sentiment` sudah tertimpa model in-domain, jadi model TweetEval harus dilatih ulang.

## 3. Aturan (wajib)

- **Jangan pernah mengubah label** di `data/gold/*.csv` atau `manual_relabel/returned/*.xlsx`. Jangan melabel ulang secara otomatis.
- Hyperparameter sama seperti di paper: 3 epoch, lr 2e-5, batch 32, AdamW (weight decay 0.01), warmup 10%, max_len 128 (sentimen) / 192 (toksisitas), fp16. Seed 42 kecuali run multi-seed.
- Kalau ada hasil yang aneh (mis. MCC in-domain jauh turun), **laporkan apa adanya**, jangan di-tune supaya mirip angka lama.
- Catat versi `torch`, `transformers`, `CUDA`, nama GPU, dan wall-clock tiap tahap.

## 4. File yang perlu disalin ke mesin 3060

Salin folder `pre-thesis/` **kecuali**:
- `dota2_dataset_bersih/` (901 MB, tidak diperlukan; korpus bersih ada di `data/processed/`)
- `models/*-sentiment`, `models/_rebuilt/`, `models/indomain-sentiment-gpu/` (semuanya akan dilatih ulang)

Yang **wajib** ada: `src/`, `scripts/`, `configs/`, `data/processed/`, `data/gold/`, `data/external/` (Jigsaw + cache TweetEval), `data/inference/detoxify_toxicity/`, `data/inference/{bert,roberta,distilbert}_toxicity/`, `models/{bert,roberta,distilbert}-toxicity/`, `models/detoxify/`.

Environment: Python 3.10–3.12, `torch` CUDA build untuk 3060, `transformers`, `datasets`, `detoxify`, `scikit-learn`, `scipy`, `statsmodels`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, `pyyaml`, `pymannkendall` jika dipakai (cek import di `src/analysis/trend_tests.py`). `.venv` lama di repo rusak, jadi buat environment baru.

## 5. Daftar pekerjaan (urut)

Semua dijalankan dari root `pre-thesis/`.

### T1. Sentimen in-domain, 3 model, seed 42
`python scripts/train_indomain.py`
- Melatih BERT/RoBERTa/DistilBERT ke `models/{key}-sentiment` (slang normalization + oversampling + class weights), lalu evaluasi di gold-test.
- Output: `reports/eval_sentiment.csv` / `eval_rebuilt_sentiment.csv` (accuracy, F1-macro, F1 per kelas, balanced accuracy + CI, MCC + CI, `train_sec`).
- Bagian threshold Detoxify di script yang sama: biarkan jalan, outputnya `reports/eval_toxicity_threshold_tuned.csv`.

### T2. Multi-seed DistilBERT in-domain (reviewer R2b)
Script baru (misalnya `scripts/train_indomain_seeds.py`), meniru T1 untuk **DistilBERT saja**, seed **42, 43, 44, 45, 46**.
- Output model ke `models/seeds/distilbert/seed_{s}` (**jangan** menimpa `models/distilbert-sentiment`).
- Split train/validation juga pakai seed yang sama (`load_gold_sentiment(..., seed=s)`, `oversample_minority(..., seed=s)`), test tetap `data/gold/test.csv`.
- Output: `reports/eval_indomain_seeds.csv` (satu baris per seed: accuracy, f1_macro, balanced_accuracy, mcc, train_sec) plus baris ringkasan mean ± SD.
- Opsional, jika waktu cukup: ulangi untuk BERT dan RoBERTa.

### T3. Sentimen out-of-domain (TweetEval), 3 model, seed 42
Script baru (misalnya `scripts/train_ood_tweeteval.py`):
- Data: TweetEval `sentiment` dari cache `data/external/_hf_cache/tweet_eval/sentiment` (`datasets.load_dataset("tweet_eval", "sentiment", cache_dir="data/external/_hf_cache")`). Label TweetEval 0=negative, 1=neutral, 2=positive (sama urutannya dengan `SENTIMENT_LABELS`; tetap verifikasi).
- Train = TweetEval train, validation = TweetEval validation, **tanpa** slang normalization, **tanpa** oversampling/class weights (paper: "standard cross-entropy").
- Pakai `src.training.sentiment_trainer.fine_tune_sentiment`, output `models/ood/{key}-sentiment`.
- Evaluasi di `data/gold/test.csv` dengan teks mentah (tanpa slang). Output: `reports/eval_ood_sentiment.csv` (balanced accuracy + CI, MCC + CI, accuracy, F1-macro, F1 per kelas, recall per kelas), confusion matrix `reports/confusion_ood_{key}_sentiment.csv`, dan validation F1-macro TweetEval per model.
- **Jargon breakdown**: error rate model OOD pada pesan `is_dota_jargon==1` vs `0` di gold-test → `reports/jargon_error_breakdown.csv`. Catatan: label jargon manual hanya ±2% pesan, jadi subset jargon kecil; tetap laporkan n-nya.

### T4. Inferensi full korpus (1.603.567 pesan)
- In-domain: `python scripts/run_inference.py` → `data/inference/{roberta,distilbert,bert}_sentiment/` (pakai slang normalization). Minimal **RoBERTa** wajib, karena analisis temporal memakai `BEST_SENT = "roberta"`.
- Out-of-domain: inferensi **RoBERTa OOD** (`models/ood/roberta-sentiment`, teks mentah) → `data/inference/ood_roberta_sentiment/`, format kolom sama (`match_id, time, player_slot, prob_negative, prob_neutral, prob_positive, predicted_label`), **satu baris per pesan dengan urutan sama** seperti `data/processed/*.parquet` (merge mengandalkan urutan baris).
- Toksisitas **tidak** perlu inferensi ulang (model Jigsaw dan Detoxify tidak bergantung label gold).

### T5. Evaluasi komparatif toksisitas + probe
`python scripts/eval_comparative.py`, dengan label gold baru:
- `reports/eval_toxicity.csv`: F1-micro + CI, F1-macro, MCC micro, PR-AUC macro, F1 per label, hamming loss, subset accuracy untuk BERT/RoBERTa/DistilBERT/Detoxify.
- `reports/eval_enriched_toxicity.csv`: probe 150 pesan (`data/gold/manual_labeled_probe.csv`). Probe ini adalah 150 pesan tambahan di dalam gold set; label di file probe sudah disamakan dengan `gold_final.csv` yang baru. Laporkan juga recall Detoxify pada profanity Indonesia jika script mendukung.
- Detoxify zero-shot pada tugas sentimen 3 kelas (accuracy, F1-macro, balanced accuracy).
- Statistik test set: jumlah positif per label toksisitas di gold-test, dan jumlah `negative` di gold-test.

### T6. Analisis temporal + korelasi
`python scripts/run_analysis_pipeline.py` (pakai `merge_inference_with_processed` yang sudah diperbaiki; cek bahwa `len(df) == 1603567`):
- `reports/temporal_monthly.csv`, `temporal_yearly.csv`, `temporal_tests.csv` (Mann–Kendall τ, z, p untuk 4 seri), Chow test di 2020-03, 2023-01, 2023-04 (F, p), `event_comparison.csv`, `correlation_tests.csv` (10 tes + BH-adjusted p + effect size + **n per tes**), `correlation_robustness_per_match.csv`, plus plot (Fig. 1 dan Fig. 2) di `reports/plots/`.
- Ulangi Mann–Kendall `mean_sentiment_score` dengan inferensi **OOD RoBERTa** (paper membandingkan τ in-domain vs τ out-of-domain) → `reports/temporal_tests_ood.csv`.
- Era means untuk caption Fig. 2: rata-rata sentiment score 2016–2017, 2018–2019, pandemic-online (2020-03 s.d. 2021-12), ≥2023, serta bulan minimum.
- Sensitivity: re-cek statistik utama pada subset `is_ambiguous==0` (cukup ringkasan perubahan).

## 6. Yang harus dikirim balik ke laptop utama

Zip lalu kirim:
- seluruh `reports/` (CSV, log, `plots/`)
- `data/inference/ood_roberta_sentiment/` dan `data/inference/roberta_sentiment/` (opsional, untuk verifikasi)
- script baru (`scripts/train_ood_tweeteval.py`, `scripts/train_indomain_seeds.py`, dan perubahan script lain)
- `RUN_NOTES.md`: versi software, GPU, wall-clock per tahap (training in-domain per model, OOD per model, inferensi), warning/error, dan setiap penyimpangan dari dokumen ini

## 7. Angka paper yang akan diisi dari hasil ini

| Bagian paper | Sumber |
|---|---|
| Abstrak, kontribusi, kesimpulan (semua angka model & tren) | T1, T3, T5, T6 |
| Table 2 (sentimen external vs in-domain + CI + per-class F1 + train time) | T1, T3 |
| Catatan Table 2: mean ± SD 5 seed DistilBERT | T2 |
| Table 3 (toksisitas 4 model) | T5 |
| Table 4 (probe 150) | T5 |
| Table 5 (Mann–Kendall), Table 6 (Chow), Table 7 (korelasi) | T6 |
| τ out-of-domain pembanding, Fig. 2 + caption era | T6 |
| Jargon: persentase & error-rate gap OOD | T3 + `agreement.json` |
| Hardware/runtime di Sect. 4 | `RUN_NOTES.md` |
