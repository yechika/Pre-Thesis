# Rebuilt vs Original — Imbalance-Robust Metrics (gold-test)

CPU re-run: sentiment **fine-tuned in-domain on gold + slang translation + class-weighted oversampling** (max_seq_length=64 proxy). Toxicity = Detoxify slang ablation (no retrain).

## Sentiment — Balanced Accuracy + MCC (OLD external-trained → NEW in-domain)

| model | bal_acc OLD | bal_acc NEW | Δ | MCC OLD | MCC NEW | Δ |
|:--|--:|--:|--:|--:|--:|--:|
| bert | 0.315 | **0.784** | +0.469 | -0.127 | **0.952** | +1.079 |
| roberta | 0.343 | **0.813** | +0.471 | -0.088 | **0.945** | +1.033 |
| distilbert | 0.331 | **0.802** | +0.471 | -0.098 | **0.910** | +1.008 |

_Chance bal_acc (3-class) ≈ 0.33. MCC: 0=random, <0=worse-than-chance, 1=perfect._

## Toxicity — Detoxify slang ablation (gold-test, supported labels)

| variant | mcc_macro | pr_auc_macro | pr_auc_micro |
|:--|--:|--:|--:|
| detoxify_no_slang | 0.227 | 0.080 | 0.057 |
| detoxify_slang | 0.220 | 0.113 | 0.075 |