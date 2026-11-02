## 1. Data Consolidation & Gold Review

- [x] 1.1 Update `notebooks/03_annotation_review.ipynb` to support user-annotated data without strict single-AI filter.
- [x] 1.2 Merge `data/gold/sample_for_labeling.csv` into gold dataset splits (`train.csv`, `test.csv`) and save as dedicated benchmark probe `data/gold/manual_labeled_probe.csv`.
- [x] 1.3 Verify zero leakage/overlap between train and test splits and update SHA256 hashes in `configs/experiment.yaml`.

## 2. In-Domain Training & Slang Augmentation

- [x] 2.1 Enhance `src/preprocessing/slang.py` and slang translation dictionary with Indonesian gaming profanity and toxic slang terms.
- [x] 2.2 Configure and execute in-domain sentiment fine-tuning for BERT, RoBERTa, and DistilBERT with class-weighted oversampling in `notebooks/04_training.ipynb`.
- [x] 2.3 Tune per-label decision thresholds for multi-label toxicity models to maximize F1 across rare labels.
- [x] 2.4 Save all in-domain model checkpoints to `models/<model_key>-sentiment/` and `models/<model_key>-toxicity/`.

## 3. Comparative Evaluation & Imbalance Metrics

- [x] 3.1 Execute `notebooks/06_comparative_evaluation.ipynb` on the held-out gold test set (n=2,392).
- [x] 3.2 Execute `notebooks/09_imbalance_metrics.ipynb` to compute Balanced Accuracy, MCC, PR-AUC, and per-label F1 scores on both population test set and manual probe.
- [x] 3.3 Update `reports/comparison_summary.md`, `reports/eval_sentiment.csv`, and `reports/eval_toxicity.csv`.

## 4. Full-Corpus GPU Inference

- [x] 4.1 Execute batch GPU inference in `notebooks/05_inference.ipynb` over 1.38M messages using in-domain sentiment model and Detoxify toxicity.
- [x] 4.2 Validate inference parquet outputs in `data/inference/` across all year partitions (2016–2026).

## 5. Temporal Trend Analysis Refresh

- [x] 5.1 Execute `notebooks/07_temporal_analysis.ipynb` to aggregate 124 monthly time series using in-domain predictions.
- [x] 5.2 Recompute Mann-Kendall monotonic trend tests and Chow structural break tests at key events (Pandemic 2020-03, Post-DPC 2023-01, Patch 7.33 2023-04).
- [x] 5.3 Refresh temporal trend plots in `reports/plots/temporal_*.png`.

## 6. Contextual Correlation Analysis Refresh

- [x] 6.1 Execute `notebooks/08_correlation_analysis.ipynb` across five contextual features (`match_outcome`, `duration`, `tier`, `phase`, `period`).
- [x] 6.2 Apply Benjamini-Hochberg FDR correction (q=0.05) and execute per-match aggregation robustness tests.
- [x] 6.3 Refresh correlation summary tables and violin/box plots in `reports/plots/correlation_*.png`.

## 7. Report & Chapter 4 Synthesis

- [x] 7.1 Execute `notebooks/10_dataset_characteristics.ipynb` to update dataset distribution figures for Chapter 4.1.
- [x] 7.2 Update `reports/CHAPTER4_SUMMARY.md` with final empirical metrics, tables, and discussions.
