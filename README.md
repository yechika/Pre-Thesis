## Thesis: Sentiment + Toxicity of Dota 2 In-Game Chat (2016-2026)

Undergraduate thesis repository, BINUS Computer Science, by Daniel Dirgantara, Muhammad Dhitan Imam Sakti, and Aldiaz Kusuma Ramadhan.

Topic: comparative sentiment analysis and toxicity detection on in-game chat from professional Dota 2 matches across one decade (2016-2026), comparing **4 deep learning models**: BERT-base, RoBERTa-base, DistilBERT, and Detoxify (`unitary/toxic-bert`).

## A. Initial Setup — Download Dataset

### 1. Install Dependencies
```bash
pip install kagglehub python-dotenv pandas kagglesdk
```

### 2. Configure Kaggle API
1. Get an API key from [Kaggle Settings](https://www.kaggle.com/settings) -> Create New Token.
2. Copy `.env.example` to `.env`.
3. Fill in `KAGGLE_USERNAME` and `KAGGLE_KEY` in the `.env` file.

### 3. Run
```bash
python import.py
python import_constants.py
```

Or execute the notebooks:
- `import.ipynb` — yearly folders + Q1 2026.
- `import_constants.ipynb` — Constants folder.

## B. Thesis Experiment: Dota 2 Decade Sentiment + Toxicity

See the archived openspec change `thesis-sentiment-toxicity-dota2-decade` (under `openspec/changes/archive/`) for the full proposal, design, specs, and tasks.

### 1. Install additional experiment dependencies
```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and, for training, a CUDA GPU (Colab/Kaggle/local).

### 2. Configuration
Edit `configs/experiment.yaml` to change the seed, hyperparameters, or paths. This file is the single source of truth for the experiment.

### 3. Notebook execution order (in `notebooks/`)

| #  | Notebook | Purpose | Est. time | Compute |
|----|----------|---------|-----------|---------|
| 01 | `01_data_pipeline.ipynb` | Ingest, clean, filter EN, join metadata → `data/processed/` | 30-90 min | CPU |
| 02 | `02_gold_sampling.ipynb` | Stratified sampling of 8000 messages → `data/gold/sample.csv` | < 5 min | CPU |
| 03 | `03_annotation_review.ipynb` | Consolidate manual annotations + IAA → `data/gold/gold_final.csv` | 5-10 min | CPU |
| 04 | `04_training.ipynb` | Fine-tune BERT/RoBERTa/DistilBERT × {sentiment, toxicity} | 4-12 hr | **GPU** |
| 05 | `05_inference.ipynb` | Inference of 4 models over all of `data/processed/` | 2-8 hr | **GPU** |
| 06 | `06_comparative_evaluation.ipynb` | Evaluate 4 models on gold-test → `reports/comparison_*` | 10-30 min | CPU |
| 07 | `07_temporal_analysis.ipynb` | Time-series + event overlay + Mann-Kendall/Chow → `reports/temporal_*` | 15-30 min | CPU |
| 08 | `08_correlation_analysis.ipynb` | Contextual-feature correlation tests + BH-FDR → `reports/correlation_*` | 15-30 min | CPU |

> Notebook 03 and step 05.6 (manual annotation) are tiered research-team activities — not automated; see `data/gold/ANNOTATION_GUIDELINE.md`.

### 4. Main outputs
- `data/processed/<folder>.parquet` — cleaned chat per folder.
- `data/gold/gold_final.csv` — consensus gold-standard labels (split into `train.csv` / `test.csv`).
- `models/<model>-<task>/` — fine-tuned checkpoints.
- `data/inference/<model>_<task>/<folder>.parquet` — per-message predictions.
- `reports/comparison_summary.md` — citation-ready comparison table.
- `reports/temporal_*.csv` + plots — decade trend analysis.
- `reports/correlation_*.csv` + plots — contextual-factor correlation tests.
- `reports/CHAPTER4_SUMMARY.md` — thesis Chapter IV outline.

### 5. Reproducibility
- All configuration is centralized in `configs/experiment.yaml`.
- The seed is set in every notebook.
- SHA256 hashes of the public dataset and processed outputs are recorded in the manifest.
- Each notebook run writes a summary to `reports/run_log.csv`.

### 6. Reports
The experiment has been executed end-to-end (NB01-NB08); the report files below are populated from the notebook outputs.
- **Thesis Chapter IV**: [reports/CHAPTER4_SUMMARY.md](reports/CHAPTER4_SUMMARY.md) — filled from notebook 06-08 outputs.
- 4-model comparison results: `reports/comparison_summary.md`.
- Contextual correlation results: `reports/correlation_summary.md`.
- Contextual feature definitions: `reports/contextual_features.md`.
