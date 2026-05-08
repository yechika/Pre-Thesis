## Skripsi: Sentimen + Toksisitas Chat In-Game Dota 2 (2016-2026)

Repository skripsi BINUS Teknik Informatika oleh Daniel Dirgantara, Muhammad Dhitan Imam Sakti, dan Aldiaz Kusuma Ramadhan.

Topik: analisis sentimen komparatif dan deteksi toksisitas pada chat in-game pertandingan profesional Dota 2 selama satu dekade (2016-2026), membandingkan **4 model deep learning**: BERT-base, RoBERTa-base, DistilBERT, dan Detoxify (`unitary/toxic-bert`).

## A. Setup Awal — Unduh Dataset

### 1. Install Dependencies
```bash
pip install kagglehub python-dotenv pandas kagglesdk
```

### 2. Config Kaggle API
1. API Key dari [Kaggle Settings](https://www.kaggle.com/settings) -> Create New Token.
2. Salin file `.env.example` menjadi `.env`.
3. Isi `KAGGLE_USERNAME` dan `KAGGLE_KEY` di file `.env`.

### 3. Run
```bash
python import.py
python import_constants.py
```

Atau eksekusi notebook:
- `import.ipynb` — folder tahunan + kuartal 2026.
- `import_constants.ipynb` — folder Constants.

## B. Eksperimen Skripsi: Sentimen + Toksisitas Dota 2 Dekade

Lihat openspec change `thesis-sentiment-toxicity-dota2-decade` (di `openspec/changes/`) untuk proposal lengkap, design, specs, dan tasks.

### 1. Install dependency tambahan eksperimen
```bash
pip install -r requirements.txt
```

Membutuhkan Python 3.10+ dan, untuk training, GPU CUDA (Colab/Kaggle/lokal).

### 2. Konfigurasi
Edit `configs/experiment.yaml` jika perlu mengubah seed, hyperparameter, atau path. File ini adalah single source of truth untuk eksperimen.

### 3. Urutan eksekusi notebook (di `notebooks/`)

| #  | Notebook | Tujuan | Estimasi waktu | Compute |
|----|----------|--------|----------------|---------|
| 01 | `01_data_pipeline.ipynb` | Ingest, clean, filter EN, join metadata → `data/processed/` | 30-90 menit | CPU |
| 02 | `02_gold_sampling.ipynb` | Stratified sampling 8000 pesan → `data/gold/sample.csv` | < 5 menit | CPU |
| 03 | `03_annotation_review.ipynb` | Konsolidasi anotasi manual + IAA → `data/gold/gold_final.csv` | 5-10 menit | CPU |
| 04 | `04_training.ipynb` | Fine-tune BERT/RoBERTa/DistilBERT × {sentimen, toksisitas} | 4-12 jam | **GPU** |
| 05 | `05_inference.ipynb` | Inferensi 4 model ke seluruh `data/processed/` | 2-8 jam | **GPU** |
| 06 | `06_comparative_evaluation.ipynb` | Evaluasi 4 model di gold-test → `reports/comparison_*` | 10-30 menit | CPU |
| 07 | `07_temporal_analysis.ipynb` | Time-series + event overlay + Mann-Kendall/Chow → `reports/temporal_*` | 15-30 menit | CPU |
| 08 | `08_correlation_analysis.ipynb` | Uji korelasi fitur kontekstual + BH-FDR → `reports/correlation_*` | 15-30 menit | CPU |

> Notebook 03 dan 05.6 (anotasi manual) adalah aktivitas tim peneliti bertingkat — bukan otomasi; lihat `data/gold/ANNOTATION_GUIDELINE.md`.

### 4. Output utama
- `data/processed/<folder>.parquet` — chat bersih per-folder.
- `data/gold/gold_final.csv` — gold-standard label konsensus.
- `models/<model>-<task>/` — checkpoint hasil fine-tune.
- `data/inference/<model>_<task>/<folder>.parquet` — prediksi per-pesan.
- `reports/comparison_summary.md` — tabel komparatif siap-kutip.
- `reports/temporal_*.csv` + plots — analisis tren dekade.
- `reports/correlation_*.csv` + plots — uji korelasi faktor kontekstual.
- `reports/CHAPTER4_SUMMARY.md` — kerangka BAB IV skripsi.

### 5. Reproducibility
- Seluruh konfigurasi terpusat di `configs/experiment.yaml`.
- Seed di-set di setiap notebook.
- Hash SHA256 dataset publik dan output processed dicatat di manifest.
- Setiap run notebook menulis ringkasan ke `reports/run_log.csv`.

### 6. Laporan
- **Kerangka BAB IV skripsi**: [reports/CHAPTER4_SUMMARY.md](reports/CHAPTER4_SUMMARY.md) — tabel placeholder yang akan diisi dari output notebook 06-08.
- Hasil komparatif 4 model: `reports/comparison_summary.md`.
- Definisi fitur kontekstual: `reports/contextual_features.md`.
