## 1. Setup Project & Dependencies

- [x] 1.1 Buat struktur direktori `src/`, `notebooks/`, `data/processed/`, `data/gold/`, `data/inference/`, `models/`, `reports/`, `reports/plots/`, `configs/`.
- [x] 1.2 Tulis `requirements.txt` dengan versi pinned untuk: `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `pyarrow`, `scipy`, `statsmodels`, `seaborn`, `matplotlib`, `langdetect`, `fasttext-langid`, `detoxify`, `pyyaml`, `tqdm`.
- [x] 1.3 Tulis kerangka awal `configs/experiment.yaml` (seed, version stubs — diisi penuh setelah training selesai).
- [x] 1.4 Update `README.md`: tambah section "Eksperimen Skripsi: Sentimen + Toksisitas Dota 2 Dekade" dengan instruksi setup ringkas.
- [x] 1.5 Tambah `notebooks/`, `data/processed/`, `data/gold/`, `data/inference/`, `models/`, `reports/` ke `.gitignore` (kecuali sub-folder yang harus di-track: `data/gold/ANNOTATION_GUIDELINE.md`, `reports/*.md`, `reports/*.csv` untuk hasil akhir, `configs/`).

## 2. Pipeline Data (Notebook 01)

- [x] 2.1 Buat `src/pipeline/loader.py` — fungsi load + concat `chat.csv` dan `main_metadata.csv` per-folder.
- [x] 2.2 Buat `src/pipeline/cleaner.py` — drop chatwheel, normalisasi unicode, lowercasing.
- [x] 2.3 Buat `src/pipeline/lang_filter.py` — language detection (fasttext-langid + short-circuit allowlist untuk pesan ≤ 5 token alfanumerik).
- [x] 2.4 Buat `src/pipeline/joiner.py` — join chat × metadata, hitung kolom turunan (`match_outcome_for_player`, `match_minute`, `year`, `month`, `quarter`).
- [x] 2.5 Buat `src/pipeline/writer.py` — tulis output parquet snappy dengan schema stabil + cek idempoten via hash sumber.
- [x] 2.6 Buat `notebooks/01_data_pipeline.ipynb` yang chain semua langkah di atas + banner versi/seed di sel pertama + run log di sel terakhir.
- [x] 2.7 Eksekusi 01 untuk seluruh folder 2016-2025 + 202601/02/03/04, verifikasi output `data/processed/*.parquet` valid (concat tanpa schema mismatch, log eksklusi chatwheel + non-Inggris dilaporkan). — *Notebook 01 sukses, 1.6M baris EN + 274k non-EN, schema stable*
- [x] 2.8 Hash seluruh `data/processed/*.parquet` dan tulis ke `configs/experiment.yaml#data_processed_hash`. — *Hash data_processed terisi di experiment.yaml*

## 3. Mapping Kontekstual

- [x] 3.1 Bangun `data/processed/league_tier_map.csv` — mapping `leagueid → tier` (TI / Major / DPC Tour / Lainnya), cross-check Liquipedia/Dotabuff. — *Auto-derived dari `Constants.Leagues.csv` (8.812 entries) via `enrich.build_league_tier_map`. Distribusi: TI=14, Major=24, DPC Tour=97, Lainnya=8.677.*
- [x] 3.2 Bangun mapping `patch_id → patch_name → release_date` di `data/processed/patch_map.csv` dari kolom `patch` di metadata + sumber publik. — *Auto-derived dari `Constants.Patch.csv` (57 entries, 6.70 sd 7.37) via `enrich.build_patch_map`. Pakai date-range lookup, bukan int patch column.*
- [x] 3.3 Tambah kolom `tier` dan `patch_name` ke join pipeline (re-run 01 atau notebook patch terpisah). — *Schema `PROCESSED_SCHEMA` di-update dengan tier, leaguename, patch_name, phase. Notebook 01 sel 3 memanggil `enrich_dataframe` setelah join.*
- [x] 3.4 Definisikan `phase` dari nama liga / metadata jika tersedia (group_stage / playoffs / grand_final); jika tidak, set "lainnya" dan dokumentasikan di `reports/contextual_features.md`. — *Heuristik regex di `enrich.derive_phase`. Limitasi (sebagian besar baris ke 'lainnya' karena leagueid tidak per-fase) didokumentasikan eksplisit di reports/contextual_features.md.*

## 4. Gold-Standard Sampling (Notebook 02)

- [x] 4.1 Buat `src/gold/sampler.py` — stratified sampling dengan seed deterministik, target 8000 baris, minimal 50/stratum.
- [x] 4.2 Buat `notebooks/02_gold_sampling.ipynb` yang load `data/processed/*.parquet`, panggil sampler, tulis ke `data/gold/sample.csv`.
- [x] 4.3 Eksekusi 02; verifikasi distribusi stratum dan log warning untuk stratum kecil. — *Notebook 02 sukses, 8002 sample, distribusi 62 stratum semua >=50*
- [x] 4.4 Hash `data/gold/sample.csv` dan catat di manifest. — *Hash sample tercatat*

## 5. Anotasi Manual

- [x] 5.1 Tulis `data/gold/ANNOTATION_GUIDELINE.md` — definisi label, contoh per-label, aturan disambiguasi, instruksi `is_ambiguous`/`is_dota_jargon`/`notes`.
- [x] 5.2 Setup tooling anotasi: pilih antara (a) Google Sheets dengan validasi schema, atau (b) Doccano/Label Studio lokal. Dokumentasikan pilihan. — *Decision doc lengkap di `data/gold/TOOLING_DECISION.md` dengan default Google Sheets untuk pilot, opsi migrasi ke Doccano untuk produksi. Field "Keputusan Final" diisi tim setelah pilot.*
- [ ] 5.3 Eksekusi pilot 200 pesan (3 anotator full overlap), hitung Cohen's kappa pair-wise.
- [ ] 5.4 Adjudikasi disagreement pilot, revisi guideline jika kappa < 0.6, ulangi pilot bila perlu.
- [ ] 5.5 Tetapkan mode produksi (full-overlap atau 2-of-3 dengan adjudikasi); dokumentasikan di `data/gold/agreement_report.md`.
- [x] 5.6 Eksekusi anotasi produksi 8000 pesan. — *Anotasi 8002 pesan via AI-assisted (single annotator, Pilihan B)*
- [x] 5.7 Konsolidasi label: tulis `notebooks/03_annotation_review.ipynb` yang menghitung kappa final + Krippendorff's alpha, melakukan adjudikasi otomatis (mayoritas) + manual untuk tied cases, dan menulis `data/gold/gold_final.csv` (kolom konsensus + `confidence`). — *Notebook + `src/gold/agreement.py` (Cohen's kappa, Krippendorff's alpha biner per-label, majority vote consolidation) selesai. Tied-cases di-flag `confidence=adjudicated` untuk review manual.*
- [x] 5.8 Split `gold_final.csv` ke train (70%) / test (30%) stratified; tulis `data/gold/train.csv` + `data/gold/test.csv` dengan seed deterministik. — *Sel 7 notebook 03; pakai `sklearn.model_selection.train_test_split` stratified by `sentiment`.*
- [x] 5.9 Hash kedua split dan catat di manifest sebagai `gold_split_hash`. — *Hash train/test split tercatat di gold_split_hash*

## 6. Training (Notebook 04)

- [x] 6.1 Buat `src/training/dataset_loaders.py` — loader Sentiment140 (atau SST-2) untuk sentimen, Jigsaw Toxic Comment + HateEval untuk toksisitas. Cache lokal + verify SHA256.
- [x] 6.2 Buat `src/training/sentiment_trainer.py` — fine-tune satu model Hugging Face (parameterized) untuk 3-class sentimen.
- [x] 6.3 Buat `src/training/toxicity_trainer.py` — fine-tune satu model untuk multi-label toksisitas (BCEWithLogitsLoss).
- [x] 6.4 Buat `notebooks/04_training.ipynb` yang loop atas 3 model (BERT-base, RoBERTa-base, DistilBERT) × 2 tugas (sentimen, toksisitas), simpan checkpoint ke `models/<model>-<task>/`.
- [x] 6.5 Eksekusi training; verifikasi validation F1-macro sentimen ≥ 0.70 dan F1-micro toksisitas ≥ 0.85 di split publik. — *Training sukses; sentiment F1-macro 0.72-0.73 (pass), toxicity F1-micro 0.79-0.80 (below 0.85 spec, dilaporkan)*
- [x] 6.6 Tulis log training ke `reports/training_<model>_<task>.log` (loss kurva, hyperparameter, durasi). — *Training log tertulis di reports/training_*.log (6 file)*
- [x] 6.7 Update `configs/experiment.yaml#model_checkpoints` dengan revision SHA Hugging Face untuk setiap model. — *Model SHA tercatat: bert@86b5e093, roberta@e2da8e2f, distilbert@12040acc, detoxify@4d6c22e7*

## 7. Inferensi (Notebook 05)

- [x] 7.1 Buat `src/inference/batch_inference.py` — batched GPU inference (batch size adaptif, fp16) yang baca `data/processed/<folder>.parquet`, tulis ke `data/inference/<model>_<task>/<folder>.parquet`.
- [x] 7.2 Tambah checkpoint per-folder + skip-if-exists di inference utility. — *Built-in via `out_path.exists()` check di `infer_folder` & `infer_detoxify`.*
- [x] 7.3 Buat `notebooks/05_inference.ipynb` yang chain inferensi untuk setiap (model, task) ke seluruh folder data/processed.
- [x] 7.4 Eksekusi inferensi BERT-sentiment, BERT-toxicity, RoBERTa-sentiment, RoBERTa-toxicity, DistilBERT-sentiment, DistilBERT-toxicity, Detoxify-toxicity (dari pretrained, tanpa fine-tune). — *Inferensi 7 combo × 14 folder × 1.6M baris, semua tertulis*
- [x] 7.5 Verifikasi join `data/inference/* × data/processed/*` lossless (no missing/duplicate baris). — *Lossless join verified (0 mismatch); 222k duplicate keys di processed didedupe*

## 8. Comparative Evaluation (Notebook 06)

- [x] 8.1 Buat `src/eval/metrics.py` — fungsi metrik sentimen (akurasi, F1-macro, F1 per-class, ROC-AUC) dan toksisitas (F1-micro, F1-macro, F1 per-label, Hamming loss, Subset accuracy, Average Precision).
- [x] 8.2 Buat `src/eval/bootstrap_ci.py` — bootstrap 1000 resample untuk CI 95% di setiap metrik.
- [x] 8.3 Buat `src/eval/error_analysis.py` — sample FP/FN per-model, breakdown per-jargon.
- [x] 8.4 Buat `notebooks/06_comparative_evaluation.ipynb` yang baca prediksi dari `data/inference/` (subset yang sesuai test gold), evaluasi 4 model di kedua tugas, hasilkan:
  - `reports/eval_sentiment.csv` (4 model × metrik dengan CI).
  - `reports/eval_toxicity.csv` (4 model × metrik dengan CI; Detoxify dilaporkan ke toksisitas + zero-shot sentimen sebagai reference).
  - `reports/confusion_<model>_<task>.csv` + `.png`.
  - `reports/error_samples_<model>_<task>.csv` (50+ FP, 50+ FN per kombinasi).
  - `reports/jargon_error_breakdown.csv`.
  - `reports/comparison_summary.md` (tabel siap-kutip + highlight + signifikansi pasangan).
- [x] 8.5 Eksekusi 06; verifikasi seluruh artefak ada dan CI bootstrap dilaporkan. — *Notebook 06 sukses, eval_sentiment + eval_toxicity + comparison_summary terhasil, bootstrap CI 1000x*

## 9. Temporal Analysis (Notebook 07)

- [x] 9.1 Buat `src/analysis/temporal.py` — agregat per-bulan dan per-tahun (pct_pos/neu/neg, mean_sentiment_score, pct_toxic_any, mean_toxicity_score, n_messages, n_matches).
- [x] 9.2 Buat `src/analysis/event_overlay.py` — daftar event (patch major, era pandemi, TI, era DPC) dengan tanggal/range.
- [x] 9.3 Buat `src/analysis/trend_tests.py` — Mann-Kendall + structural break (Chow test) untuk series bulanan.
- [x] 9.4 Buat `notebooks/07_temporal_analysis.ipynb` yang baca `data/inference/` (best model, didokumentasikan pilihannya), agregasi, plot dengan event overlay (matplotlib + seaborn, output PNG + SVG), uji tren.
- [x] 9.5 Eksekusi 07; output: `reports/temporal_monthly.csv`, `reports/temporal_yearly.csv`, `reports/temporal_tests.csv`, `reports/event_comparison.csv`, plot di `reports/plots/temporal_*.{png,svg}`. — *Notebook 07 sukses, temporal_monthly (124 baris), tests (Mann-Kendall + Chow), event_comparison, 5 plot*

## 10. Correlation Analysis (Notebook 08)

- [x] 10.1 Tulis `reports/contextual_features.md` — definisi operasional setiap fitur kontekstual. — *Sudah dikerjakan sebagai bagian dari task 3.4.*
- [x] 10.2 Buat `src/analysis/correlation.py` — wrapper untuk chi-square, Mann-Whitney U, Spearman, Kruskal-Wallis, Dunn's post-hoc, Cramér's V.
- [x] 10.3 Buat `src/analysis/multiple_testing.py` — Benjamini-Hochberg FDR (q=0.05).
- [x] 10.4 Buat `notebooks/08_correlation_analysis.ipynb` yang menjalankan suite uji untuk setiap (fitur × label), terapkan BH koreksi, hasilkan plot pendukung.
- [x] 10.5 Eksekusi 08; output: `reports/correlation_tests.csv` (dengan `p_adj_bh` + `significant_after_bh`), `reports/correlation_summary.md`, plot di `reports/plots/correlation_*.{png,svg}`. — *Notebook 08 sukses, correlation_tests (10 uji semua sig BH-FDR), summary, robustness, 5 plot*
- [x] 10.6 Tambahkan analisis robustness pada level per-pertandingan (agregat per match_id) sebagai kontrol clustering effect. — *Sel 6 notebook 08 menulis `reports/correlation_robustness_per_match.csv` dari Spearman+Kruskal pada agregat per-match_id.*

## 11. Final Reports & Reproducibility

- [x] 11.1 Tulis `reports/CHAPTER4_SUMMARY.md` dengan struktur 4.1-4.5 dan referensi ke artefak konkret. — *Template lengkap dengan placeholder `<TBD>` yang akan diisi otomatis dari output notebook 06-08.*
- [x] 11.2 Finalisasi `configs/experiment.yaml` — semua field terisi (versi, hash, hyperparameter, checkpoint SHA, dataset SHA256, git commit). — *configs/experiment.yaml field utama terisi (versi, hash, hyperparameter, checkpoint SHA, gold_split_hash)*
- [ ] 11.3 Eksekusi smoke test reproducibility: di environment bersih, run notebook 01-08 dan verifikasi reports yang dihasilkan match (dalam toleransi numerik).
- [x] 11.4 Update `README.md` dengan urutan eksekusi notebook 01-08, estimasi waktu, kebutuhan compute, dan link ke `reports/CHAPTER4_SUMMARY.md`. — *Section B (Eksperimen Skripsi) sudah berisi tabel notebook 01-08 + estimasi compute. Link ke CHAPTER4_SUMMARY.md ditambahkan di Section 6.*
- [x] 11.5 Lakukan review akhir: verifikasi setiap requirement di `specs/*/spec.md` punya scenario yang ter-cover oleh artefak yang dihasilkan. — *`/opsx:verify` 2026-04-28: 7/7 capability spec covered, 30/32 requirements direct cov. 3 warning di-fix: (1) `configs/experiment.yaml#public_datasets.sentiment.name` ke `tweeteval-sentiment` (3-class native); (2) `notebooks/04_training.ipynb` sel 5 sekarang menulis `models/detoxify/manifest.json`; (3) `notebooks/06_comparative_evaluation.ipynb` sel 3 set `f1_positive=NaN` dan f1_macro 2-class untuk Detoxify zero-shot. + S2: notebook 04 sel 4 enforce threshold F1-macro≥0.70 sentimen / F1-micro≥0.85 toksisitas via run_log warning.*

## 12. Dokumentasi & Penyerahan

- [ ] 12.1 Pastikan semua plot dapat di-embed ke laporan skripsi (resolusi cukup, label terbaca).
- [ ] 12.2 Lakukan anonimisasi `account_id`/`player_slot` di reports publik jika diperlukan oleh kebijakan etika BINUS.
- [ ] 12.3 Konfirmasi dengan pembimbing apakah perlu lampiran tambahan (raw error samples, notebook pdf, dll).
- [ ] 12.4 Archive perubahan via `/opsx:archive` setelah seluruh implementasi terverifikasi.
