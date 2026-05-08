## Why

Penelitian skripsi ini perlu kerangka kerja eksperimen yang reproducible untuk membandingkan empat model deep learning pada chat in-game pertandingan profesional Dota 2 (2016-2026), karena (a) belum ada studi yang membandingkan model-model transformer modern secara head-to-head pada teks chat MOBA yang pendek, informal, dan penuh jargon; (b) tren temporal sentimen dan toksisitas selama satu dekade belum pernah dianalisis di domain ini; dan (c) korelasi faktor kontekstual pertandingan (hasil, durasi, tier, fase turnamen) terhadap dinamika komunikasi belum terkuantifikasi. Tanpa kerangka spec yang eksplisit, eksperimen rentan terhadap inkonsistensi tokenization, evaluasi yang tidak comparable, dan hasil yang sulit diaudit untuk publikasi ilmiah.

## What Changes

- Tambahkan pipeline ingest & cleaning untuk data chat dari `dota2_dataset_bersih/` (2016-2026) yang menggabungkan `chat.csv` dengan `main_metadata.csv` per-folder dan menormalisasi kolom temporal/kontekstual.
- Tambahkan filter bahasa Inggris dan eksklusi `chatwheel`/emote, sehingga hanya teks chat alami yang masuk ke pipeline analisis.
- Tambahkan modul anotasi gold-standard manual: stratified sampling 5-10k pesan per dimensi (tahun × hasil pertandingan × tier) dengan inter-annotator agreement (Cohen's kappa) untuk dual-axis label (sentiment polarity 3-class + toxicity multi-label).
- Tambahkan eksperimen perbandingan **empat model deep learning**: BERT-base-uncased, RoBERTa-base, DistilBERT, dan Detoxify (unitary/toxic-bert) — dengan strategi **transfer learning** (fine-tune di Sentiment140/SST-2 untuk sentimen dan Jigsaw Toxic Comment + HateEval untuk toksisitas) lalu inferensi ke chat Dota 2.
- Tambahkan evaluasi komparatif berbasis akurasi, presisi, recall, dan F1-score (macro & per-class) di gold-standard set, plus matriks konfusi dan analisis error per-model.
- Tambahkan analisis temporal sentimen & toksisitas dekade 2016-2026 dengan time-series resampling per-bulan/per-tahun, breakdown per-event (pergantian patch/meta major, era pandemi COVID-19 daring 2020-2021).
- Tambahkan analisis korelasi faktor kontekstual (hasil menang/kalah, durasi pertandingan, tier turnamen, fase turnamen, periode waktu) terhadap label sentimen & toksisitas memakai uji statistik (chi-square, Spearman, Mann-Whitney) dengan koreksi multiple-testing.
- Tambahkan artefak reproducibility: notebook eksperimen end-to-end, file konfigurasi seed, manifest versi model & dataset, plus laporan hasil siap-kutip untuk skripsi BAB IV.

## Capabilities

### New Capabilities
- `chat-data-pipeline`: Ingest, clean, filter (bahasa Inggris, non-chatwheel), dan menggabungkan chat.csv dengan metadata pertandingan untuk seluruh rentang 2016-2026.
- `gold-standard-annotation`: Protokol stratified sampling, dual-axis labeling (sentiment 3-class + toxicity multi-label), dan pengukuran inter-annotator agreement.
- `sentiment-toxicity-modeling`: Fine-tuning empat model deep learning (BERT, RoBERTa, DistilBERT, Detoxify/toxic-bert) di dataset publik dan inferensi ke chat Dota 2 dengan output probabilitas per-class.
- `comparative-evaluation`: Evaluasi keempat model di gold-standard set memakai akurasi, presisi, recall, F1 (macro + per-class), matriks konfusi, dan analisis error.
- `temporal-trend-analysis`: Time-series sentimen & toksisitas 2016-2026, breakdown per-tahun/per-bulan/per-patch, dan studi event-impact (pandemi COVID-19, pergantian meta).
- `contextual-correlation-analysis`: Uji korelasi statistik antara fitur pertandingan (menang/kalah, durasi, tier, fase, periode) dan label sentimen/toksisitas dengan koreksi multiple-testing.
- `experiment-reproducibility`: Manifest versi (model checkpoint, dataset hash, seed), notebook end-to-end, dan laporan hasil siap-kutip untuk BAB IV skripsi.

### Modified Capabilities
<!-- Tidak ada — capability `dataset-import-notebooks` yang sudah ada hanya menyediakan data mentah dan tidak diubah perilakunya. -->

## Impact

- **File baru**:
  - `notebooks/` — notebook tahap pipeline, anotasi, training, evaluasi, analisis temporal, analisis korelasi (urut bernomor 01-07).
  - `src/` — modul Python untuk preprocessing, language detection, label loader, training loop, dan utilitas evaluasi.
  - `data/gold/` — gold-standard CSV hasil anotasi manual + guideline anotasi.
  - `data/processed/` — chat gabungan terclean (parquet) per-tahun.
  - `models/` — checkpoint hasil fine-tune untuk tiap model × tugas.
  - `reports/` — tabel evaluasi, plot temporal, hasil uji korelasi, dan ringkasan untuk BAB IV.
  - `configs/experiment.yaml` — konfigurasi seed, hyperparameter, manifest versi.
- **File tidak disentuh**: `import.py`, `import_constants.py`, `import.ipynb`, `import_constants.ipynb`, dan struktur `dota2_dataset_bersih/` (read-only sebagai sumber).
- **Dependencies baru**: `transformers`, `torch` (atau `tensorflow`), `datasets`, `scikit-learn`, `langdetect` atau `fasttext-langid`, `detoxify`, `seaborn`/`matplotlib`, `scipy`, `statsmodels`, `pyarrow` (parquet).
- **Compute**: Fine-tuning 4 model di dataset publik membutuhkan GPU (Colab/Kaggle/lokal CUDA); inferensi ke seluruh chat Dota 2 (estimasi ~5-50 juta baris) di-batch dan di-cache.
- **Dataset eksternal**: Sentiment140 / SST-2, Jigsaw Toxic Comment Classification, HateEval — diunduh sekali untuk fine-tuning.
- **Etika & privasi**: Chat profesional umumnya publik, namun anotasi manual harus mengikuti pedoman penelitian Bina Nusantara (anonimisasi player_slot di laporan publik).
