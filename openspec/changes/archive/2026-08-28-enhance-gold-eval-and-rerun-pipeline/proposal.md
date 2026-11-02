## Why

The original thesis experiment exhibited critical model evaluation bottlenecks in Chapter 4:
1. Out-of-domain Twitter-trained sentiment models collapsed on Dota chat (Balanced Accuracy ~0.33, MCC negative).
2. Toxicity evaluation suffered from extreme population sparsity (0.85% prevalence) and language blindspots (>76% missed Indonesian slang in zero-shot Detoxify).
3. Temporal and contextual analyses relied on legacy out-of-domain inferences rather than calibrated in-domain predictions.

Now that the author has provided a high-signal, balanced manual annotation sample (150 rows), we need to integrate this sample into the gold/benchmark dataset, re-train in-domain models with Indonesian slang awareness, tune toxicity thresholds, re-run full-corpus inference, and refresh the entire analysis suite across Chapters 4.1–4.4.

## What Changes

- **Gold Dataset & Annotation Review**: Update `notebooks/03_annotation_review.ipynb` to support user-annotated data alongside baseline annotations, consolidating `data/gold/sample_for_labeling.csv` into gold splits and secondary evaluation probes.
- **In-Domain Training & Slang Expansion**: Update `notebooks/04_training.ipynb` with enhanced bilingual slang dictionary (`translate_slang` with Indonesian toxic terms) and class-weighted in-domain sentiment training.
- **Toxicity Threshold Optimization**: Fit per-label decision thresholds on training data to maximize F1 across multi-label toxicity classes.
- **Full-Corpus Re-Inference**: Re-run batch inference in `notebooks/05_inference.ipynb` over the 1.38M message dataset using calibrated in-domain models.
- **Comparative Evaluation Refresh**: Update `notebooks/06_comparative_evaluation.ipynb` and `notebooks/09_imbalance_metrics.ipynb` to report both population test metrics and enriched manual benchmark metrics (Balanced Accuracy, MCC, PR-AUC).
- **Temporal & Contextual Analysis Refresh**: Re-execute `notebooks/07_temporal_analysis.ipynb`, `notebooks/08_correlation_analysis.ipynb`, and `notebooks/10_dataset_characteristics.ipynb` to update all CSV reports and visualization plots.

## Capabilities

### New Capabilities
<!-- No brand new capabilities; modifying existing pipeline capabilities -->

### Modified Capabilities
- `gold-standard-annotation`: Integrate user manual sample into gold dataset and secondary benchmark probe, update filtering in `03_annotation_review.ipynb`.
- `sentiment-toxicity-modeling`: Fine-tune in-domain sentiment models with slang translation and class weights; tune per-label toxicity decision thresholds; augment toxicity training with in-domain/Indonesian slang.
- `comparative-evaluation`: Evaluate models on both standard test set and enriched manual probe; generate updated metrics (Balanced Accuracy, MCC, PR-AUC, F1-macro).
- `temporal-trend-analysis`: Re-infer 1.38M full corpus with in-domain models, recompute monthly metrics, Mann-Kendall and Chow tests.
- `contextual-correlation-analysis`: Recompute contextual correlations (duration, outcome, tier, phase, period) with in-domain inference outputs.

## Impact

- **Affected Notebooks**: `03_annotation_review.ipynb`, `04_training.ipynb`, `05_inference.ipynb`, `06_comparative_evaluation.ipynb`, `07_temporal_analysis.ipynb`, `08_correlation_analysis.ipynb`, `09_imbalance_metrics.ipynb`, `10_dataset_characteristics.ipynb`.
- **Affected Data/Reports**: `data/gold/`, `models/`, `data/inference/`, `reports/`, `CHAPTER4_SUMMARY.md`.
- **Dependencies**: Hugging Face Transformers, PyTorch, Scikit-learn, Detoxify, Pandas, NumPy.
