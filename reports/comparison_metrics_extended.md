# Extended Comparative Metrics: Imbalance-Robust (4 Models)

From cached gold-test labels + stored inference scores (no re-training / re-inference).

## 3-Class Sentiment — Balanced Accuracy + MCC

| model              |   balanced_accuracy |   balanced_acc_ci_lo |   balanced_acc_ci_hi |   mcc |   mcc_ci_lo |   mcc_ci_hi |
|:-------------------|--------------------:|---------------------:|---------------------:|------:|------------:|------------:|
| bert               |               0.836 |                0.791 |                0.882 | 0.891 |       0.874 |       0.909 |
| roberta            |               0.837 |                0.792 |                0.883 | 0.891 |       0.874 |       0.909 |
| distilbert         |               0.842 |                0.798 |                0.888 | 0.891 |       0.874 |       0.909 |
| detoxify_zero_shot |               0.376 |              nan     |              nan     | 0.026 |     nan     |     nan     |

_MCC<0 (CI excluding 0) = worse-than-chance agreement. Balanced acc ~0.33 = random for 3 classes._


## Multi-Label Toxicity — MCC + PR-AUC

| model      |   mcc_macro |   mcc_macro_ci_lo |   mcc_macro_ci_hi |   mcc_micro |   pr_auc_macro |   pr_auc_micro |
|:-----------|------------:|------------------:|------------------:|------------:|---------------:|---------------:|
| bert       |       0.096 |             0.040 |             0.148 |       0.154 |          0.089 |          0.064 |
| roberta    |       0.085 |             0.032 |             0.133 |       0.165 |          0.115 |          0.092 |
| distilbert |       0.093 |             0.031 |             0.151 |       0.163 |          0.134 |          0.075 |
| detoxify   |       0.098 |             0.039 |             0.151 |       0.176 |          0.079 |          0.050 |

_PR-AUC macro over labels with positive support only (severe_toxic & threat excluded). Threshold-free; complements F1 / Hamming._
