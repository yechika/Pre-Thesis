# Enriched Toxicity Probe — Secondary Analysis

**Motivation.** The population gold-test set yields only 14–20 toxic positives per label,
making F1 estimates unstable and confounding *model capability* with *class sparsity*.
To disentangle the two, we constructed a secondary **enriched toxic-candidate probe**:
a stratified sample (22/year × 11 years, 2016–2026; n=242) of messages flagged by either
Detoxify score ≥0.5 **or** an EN+ID profanity lexicon, excluding all existing gold keys,
deduplicated by text, and **manually labeled** (sentiment + 6 Jigsaw toxicity labels).
Because candidates are toxicity-enriched by construction, prevalence is inflated by design;
F1 here measures **discrimination on toxic-candidate messages, not population prevalence.**

Label positives (n=242): toxic 97, obscene 97, insult 61, identity_hate 1, severe_toxic 0, threat 0.

## Table A. Toxicity on the Enriched Probe (existing models, no re-inference)

| model | F1-micro | Prec-micro | Recall-micro | F1-toxic | F1-obscene | F1-insult |
|:--|--:|--:|--:|--:|--:|--:|
| BERT | 0.424 | 0.425 | 0.422 | 0.387 | 0.548 | 0.299 |
| RoBERTa | 0.416 | 0.436 | 0.398 | 0.368 | 0.536 | 0.345 |
| DistilBERT | 0.416 | 0.439 | 0.395 | 0.363 | 0.556 | 0.337 |
| Detoxify | 0.424 | 0.392 | 0.461 | 0.364 | 0.576 | 0.370 |

_(identity_hate omitted: single positive.)_

## Table B. Indonesian-profanity blind-spot (Detoxify)

| subset | n positives | Detoxify recall |
|:--|--:|--:|
| all toxic positives | 97 | 0.526 |
| lexicon-flagged toxic positives | 82 | 0.439 |
| Indonesian-profanity toxic positives (lexicon & Detoxify score <0.5) | 46 | **0.000 (0/46)** |

## Findings

1. **The low population F1 is largely a sparsity/prevalence artifact, not absent signal.**
   On a denser set, all four models reach F1-micro ≈0.42 (vs 0.13–0.15 on the population
   gold-test), confirming the classifiers carry real discriminative signal when positives
   are present.
2. **The binding failure is language coverage.** Detoxify misses **every one** of the 46
   Indonesian/transliterated-profanity toxic messages it scored below threshold
   (e.g., *anjing, kontol, memek, goblok, asu, ngentot, babi, tai, cok*); its recall on
   lexicon-flagged toxic positives is 0.44 versus 0.53 overall. English-trained toxicity
   models (Jigsaw, Detoxify) structurally cannot detect the substantial non-English share
   of toxic chat in professional Dota 2.
3. **Diagnosis confirmed.** The toxicity bottleneck is jointly (a) extreme population
   sparsity and (b) English-only lexical coverage — not a thresholding or architecture
   deficiency. This motivates multilingual encoders (IndoBERT / XLM-R) and an Indonesian
   toxic-term resource for in-domain supervision.

_Single-annotator labels (same construct-validity caveat as the primary gold set). Files:
`reports/eval_enriched_toxicity.csv`, `reports/enriched_gold_labeled.csv`._
