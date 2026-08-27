# Extended Comparative Metrics: Imbalance-Robust (4 Models)

From cached gold-test labels + stored inference scores (no re-training / re-inference).

## 3-Class Sentiment — Balanced Accuracy + MCC

| model              |   balanced_accuracy |   balanced_acc_ci_lo |   balanced_acc_ci_hi |    mcc |   mcc_ci_lo |   mcc_ci_hi |
|:-------------------|--------------------:|---------------------:|---------------------:|-------:|------------:|------------:|
| bert               |               0.315 |                0.299 |                0.340 | -0.127 |      -0.161 |      -0.093 |
| roberta            |               0.343 |                0.309 |                0.383 | -0.088 |      -0.121 |      -0.054 |
| distilbert         |               0.331 |                0.306 |                0.363 | -0.098 |      -0.131 |      -0.062 |
| detoxify_zero_shot |               0.373 |              nan     |              nan     |  0.008 |     nan     |     nan     |

_MCC<0 (CI excluding 0) = worse-than-chance agreement. Balanced acc ~0.33 = random for 3 classes._


## Multi-Label Toxicity — MCC + PR-AUC

| model      |   mcc_macro |   mcc_macro_ci_lo |   mcc_macro_ci_hi |   mcc_micro |   pr_auc_macro |   pr_auc_micro |
|:-----------|------------:|------------------:|------------------:|------------:|---------------:|---------------:|
| bert       |       0.090 |             0.014 |             0.156 |       0.123 |          0.061 |          0.035 |
| roberta    |       0.079 |             0.015 |             0.143 |       0.139 |          0.101 |          0.059 |
| distilbert |       0.082 |             0.006 |             0.155 |       0.137 |          0.067 |          0.043 |
| detoxify   |       0.096 |             0.015 |             0.166 |       0.161 |          0.058 |          0.034 |

_PR-AUC macro over labels with positive support only (severe_toxic & threat excluded). Threshold-free; complements F1 / Hamming._
