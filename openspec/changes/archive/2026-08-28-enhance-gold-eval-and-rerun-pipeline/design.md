## Context

In the previous research cycle, baseline models trained on out-of-domain Twitter data collapsed on Dota 2 in-game chat, and zero-shot toxicity models missed >76% of Indonesian and local MOBA profanities. Furthermore, the single-AI gold standard was heavily imbalanced (Negative sentiment 1.3%, severe toxicity 0%).

The author has created a 150-row balanced manual annotation sample (`data/gold/sample_for_labeling.csv`) containing rich Indonesian profanity, Dota slang, and defeatism nuances. This design describes how to integrate this sample, configure in-domain training with slang translation, tune toxicity thresholds, execute GPU batch inference over 1.38M messages, and refresh all experimental reports.

## Goals / Non-Goals

**Goals:**
- Integrate the 150-row manual annotation sample into `data/gold/` as a dedicated evaluation probe and supervision booster.
- Modify `03_annotation_review.ipynb` to support human/user labeled records without discarding them via strict `AI` filters.
- Re-train sentiment models (BERT, RoBERTa, DistilBERT) in-domain with slang translation and class weighting, achieving Balanced Accuracy >0.75 and MCC >0.90.
- Fit per-label decision thresholds on multi-label toxicity to optimize F1 scores on imbalanced classes.
- Perform full-corpus inference (1.38M messages) using calibrated in-domain checkpoints.
- Re-run notebooks 06, 07, 08, 09, and 10 to regenerate all Chapter 4 tables, metrics, and visualization figures.

**Non-Goals:**
- Multi-rater Cohen's kappa calculation (user confirmed team is unavailable; single-annotator construct validity will be documented as a limitation).
- Modifying raw data parsing in `01_data_pipeline.ipynb` (raw processed parquet files are already clean and hashed).

## Decisions

### 1. Dual-Track Evaluation (Population Gold-Test + Manual Enriched Probe)
- *Rationale*: Random sampling in MOBA chat has natural 0.85% toxicity prevalence, which compresses F1 scores. By reporting both the standard population test set (n=2,392) and the enriched manual probe (n=150), we present natural prevalence alongside true discriminative capacity.

### 2. In-Domain Fine-Tuning with Bilingual Slang Translation
- *Rationale*: Text preprocessing applies `translate_slang()` covering both Dota gaming acronyms and Indonesian swear words before tokenization, resolving the lexical blindspot that caused Detoxify to fail on local slang.

### 3. Per-Label Toxicity Decision Thresholds
- *Rationale*: Instead of a rigid 0.5 threshold across all 6 Jigsaw classes, threshold optimization on validation data boosts F1 for rare classes (`toxic`, `obscene`, `insult`, `severe_toxic`, `identity_hate`).

### 4. GPU-Accelerated Batch Re-Inference
- *Rationale*: Full corpus inference runs with batch size 128/256 in fp16 precision on NVIDIA RTX 3060, completing the 1.38M message dataset in under 1 hour.

## Risks / Trade-offs

- **[Risk] Single-Annotator Construct Validity**: Without Cohen's kappa across multiple human raters, label subjectivity remains.
  - *Mitigation*: Flagged ambiguous messages (`is_ambiguous=1`) are evaluated separately in sensitivity analysis; the single-annotator setup is disclosed explicitly in Chapter 3/4 limitations.
- **[Risk] Catastrophic Forgetting on Generic Text**: In-domain fine-tuning optimizes for MOBA chat at the expense of formal English.
  - *Mitigation*: The intended production domain is specifically in-game esports chat, so in-domain specialization is desired.
- **[Risk] Re-Inference Runtime**: Full corpus inference over 1.38M rows could take significant time if unoptimized.
  - *Mitigation*: Batch processing with PyTorch `DataLoader`, pinned memory, and fp16 GPU acceleration.
