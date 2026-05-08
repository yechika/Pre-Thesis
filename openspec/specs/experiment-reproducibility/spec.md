# experiment-reproducibility

## Purpose

Manifest versi terpusat, notebook bernomor deterministik, environment setup pinned, dan output siap-kutip untuk reproducibility.

## Requirements

### Requirement: Manifest versi terpusat

Sistem SHALL menyediakan `configs/experiment.yaml` sebagai single source of truth untuk seluruh konfigurasi eksperimen, minimal berisi:

- `seed`: integer untuk seluruh komponen stochastic.
- `python_version`, `transformers_version`, `torch_version`, `datasets_version`, `scikit_learn_version`.
- `model_checkpoints`: untuk setiap model {`name`, `hf_repo`, `revision_sha`}.
- `public_datasets`: untuk setiap dataset {`name`, `source_url`, `sha256`, `local_path`}.
- `data_processed_hash`: SHA256 dari concat seluruh `data/processed/*.parquet`.
- `gold_split_hash`: SHA256 dari `data/gold/gold_final.csv` (atau hash per-split train/val/test).
- `hyperparameters`: hyperparameter training per (model × tugas).
- `created_at` dan `git_commit`.

#### Scenario: Manifest lengkap
- **WHEN** seluruh artefak eksperimen selesai dihasilkan
- **THEN** `configs/experiment.yaml` lolos validasi schema dan tidak ada field yang null/missing

#### Scenario: Hash mendeteksi perubahan data
- **WHEN** file `data/processed/2024.parquet` di-modify dan manifest tidak diupdate
- **THEN** notebook eksperimen MUST memberi peringatan `[WARN] data_processed_hash mismatch — kemungkinan data berubah`

### Requirement: Notebook eksperimen ber-nomor dan deterministik

Sistem SHALL menyediakan notebook eksperimen di `notebooks/` dengan nomor urut yang menentukan urutan eksekusi:

- `01_data_pipeline.ipynb`: ingest + clean + filter bahasa + merge metadata → `data/processed/`.
- `02_gold_sampling.ipynb`: stratified sampling → `data/gold/sample.csv`.
- `03_annotation_review.ipynb`: konsolidasi anotasi → `data/gold/gold_final.csv` + agreement metrics.
- `04_training.ipynb`: fine-tune BERT/RoBERTa/DistilBERT untuk sentimen + toksisitas → `models/`.
- `05_inference.ipynb`: inferensi seluruh model ke `data/processed/*.parquet` → `data/inference/`.
- `06_comparative_evaluation.ipynb`: evaluasi 4 model di gold-test → `reports/comparison_*.csv`.
- `07_temporal_analysis.ipynb`: time-series + event overlay + uji tren → `reports/temporal_*.csv` + plots.
- `08_correlation_analysis.ipynb`: uji korelasi fitur kontekstual + BH koreksi → `reports/correlation_*.csv` + plots.

Setiap notebook MUST di sel pertama:
1. Load `configs/experiment.yaml`.
2. Set seed Python/NumPy/PyTorch/CUDA.
3. Cetak banner: versi paket utama + git commit hash + timestamp.

Setiap notebook MUST di sel terakhir mencatat ringkasan run (durasi, file output yang dihasilkan, peringatan) ke `reports/run_log.csv`.

#### Scenario: Notebook urut deterministik
- **WHEN** seluruh notebook 01-08 dieksekusi berurutan dengan konfig yang sama
- **THEN** seluruh artefak (data/processed, data/gold, models, data/inference, reports) terhasilkan tanpa error dependency

#### Scenario: Banner versi tercetak
- **WHEN** sel pertama notebook 04_training dijalankan
- **THEN** output sel mencetak versi `transformers`, `torch`, `python`, `git_commit`, dan `seed` aktif

#### Scenario: Run log tertulis
- **WHEN** notebook 06_comparative_evaluation selesai
- **THEN** baris baru ditambahkan ke `reports/run_log.csv` dengan kolom `notebook`, `start_at`, `end_at`, `duration_sec`, `outputs`, `warnings`, `git_commit`

### Requirement: Setup environment dapat di-reproduce

Sistem SHALL menyediakan `requirements.txt` (atau `environment.yml`) yang men-pin versi setiap dependency utama (transformers, torch, datasets, scikit-learn, pandas, pyarrow, scipy, statsmodels, seaborn, matplotlib, langdetect/fasttext-langid, detoxify).

Sistem MAY menyediakan `Dockerfile` opsional untuk eksperimen yang sepenuhnya berjalan di container.

`README.md` MUST diupdate dengan instruksi setup yang menyertakan:
- Install dependency.
- Konfigurasi `.env` (kredensial Kaggle untuk re-download data).
- Urutan eksekusi notebook 01-08.
- Estimasi waktu dan kebutuhan compute (CPU/GPU/RAM).

#### Scenario: Reproducer eksternal dapat menjalankan
- **WHEN** seseorang dengan environment bersih clone repo dan ikuti README
- **THEN** mereka dapat menjalankan seluruh notebook 01-08 dan menghasilkan reports yang konsisten dengan hasil tim peneliti (dalam toleransi numerik fp16)

#### Scenario: requirements.txt pinned
- **WHEN** `pip install -r requirements.txt` dijalankan di environment baru
- **THEN** versi paket yang ter-install identik dengan versi yang tercatat di `configs/experiment.yaml`

### Requirement: Output siap-kutip ke laporan skripsi

Sistem SHALL menghasilkan `reports/CHAPTER4_SUMMARY.md` yang merangkum hasil seluruh eksperimen dalam struktur yang dapat langsung dipakai sebagai kerangka BAB IV skripsi:

- 4.1 Karakteristik Dataset (volume, distribusi tahunan, distribusi tier, agreement metrics)
- 4.2 Hasil Perbandingan Model (tabel + interpretasi pemenang per-tugas)
- 4.3 Tren Temporal Sentimen dan Toksisitas (plot + uji tren + interpretasi event)
- 4.4 Korelasi Faktor Kontekstual (tabel + interpretasi finding signifikan)
- 4.5 Pembahasan dan Limitasi

Setiap section MUST merujuk ke file artefak konkret (CSV/PNG/SVG di `reports/` dan `reports/plots/`) yang dapat di-cite.

#### Scenario: Ringkasan BAB IV lengkap
- **WHEN** seluruh notebook 01-08 selesai dijalankan
- **THEN** `reports/CHAPTER4_SUMMARY.md` ada dengan kelima section terisi dan referensi ke artefak konkret

#### Scenario: Tabel siap copy ke Word
- **WHEN** mahasiswa menyalin tabel komparatif dari `reports/comparison_summary.md`
- **THEN** tabel markdown dapat di-render ke Word/LaTeX tanpa edit struktur (header tabel + alignment terjaga)
