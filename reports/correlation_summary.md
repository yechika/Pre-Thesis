# Contextual Feature Correlation Summary

## Test Results Table

| feature                  | label             | test           |    test_statistic |   p_value |   p_adj_bh | significant_after_bh   |   effect_size |       n |
|:-------------------------|:------------------|:---------------|------------------:|----------:|-----------:|:-----------------------|--------------:|--------:|
| match_outcome_for_player | sentiment_label   | chi-square     |         5674.7952 |    0.0000 |     0.0000 | True                   |        0.0647 | 1355212 |
| match_outcome_for_player | max_toxicity_prob | mann-whitney   | 215116358257.0000 |    0.0000 |     0.0000 | True                   |        0.0597 | 1355212 |
| duration                 | sentiment_score   | spearman       |           -0.0784 |    0.0000 |     0.0000 | True                   |       -0.0784 | 1380867 |
| duration                 | max_toxicity_prob | spearman       |           -0.0805 |    0.0000 |     0.0000 | True                   |       -0.0805 | 1380867 |
| tier                     | max_toxicity_prob | kruskal-wallis |          887.4146 |    0.0000 |     0.0000 | True                   |        0.0006 | 1380867 |
| tier                     | sentiment_label   | chi-square     |         6855.5770 |    0.0000 |     0.0000 | True                   |        0.0498 | 1380867 |
| phase                    | max_toxicity_prob | kruskal-wallis |           40.4378 |    0.0000 |     0.0000 | True                   |        0.0000 | 1380867 |
| phase                    | sentiment_label   | chi-square     |           88.2961 |    0.0000 |     0.0000 | True                   |        0.0057 | 1380867 |
| period                   | max_toxicity_prob | kruskal-wallis |        14672.6320 |    0.0000 |     0.0000 | True                   |        0.0106 | 1380867 |
| period                   | sentiment_label   | chi-square     |        24780.8746 |    0.0000 |     0.0000 | True                   |        0.0947 | 1380867 |

## Finding Signifikan setelah BH-FDR (urut effect size desc)

- `period` × `sentiment_label` (chi-square): stat=24780.875, p_adj=0.0000, effect=0.095, n=1380867
- `duration` × `max_toxicity_prob` (spearman): stat=-0.080, p_adj=0.0000, effect=-0.080, n=1380867
- `duration` × `sentiment_score` (spearman): stat=-0.078, p_adj=0.0000, effect=-0.078, n=1380867
- `match_outcome_for_player` × `sentiment_label` (chi-square): stat=5674.795, p_adj=0.0000, effect=0.065, n=1355212
- `match_outcome_for_player` × `max_toxicity_prob` (mann-whitney): stat=215116358257.000, p_adj=0.0000, effect=0.060, n=1355212
- `tier` × `sentiment_label` (chi-square): stat=6855.577, p_adj=0.0000, effect=0.050, n=1380867
- `period` × `max_toxicity_prob` (kruskal-wallis): stat=14672.632, p_adj=0.0000, effect=0.011, n=1380867
- `phase` × `sentiment_label` (chi-square): stat=88.296, p_adj=0.0000, effect=0.006, n=1380867
- `tier` × `max_toxicity_prob` (kruskal-wallis): stat=887.415, p_adj=0.0000, effect=0.001, n=1380867
- `phase` × `max_toxicity_prob` (kruskal-wallis): stat=40.438, p_adj=0.0000, effect=0.000, n=1380867

## Limitations

- Messages within the same match are NOT independent (clustering effect). Robustness analysis at per-match level is reported in `correlation_robustness_per_match.csv`.
- Kolom `phase` mayoritas bernilai `lainnya` karena dataset Dota 2 publik tidak menyertakan fase per-match secara eksplisit (lihat `reports/contextual_features.md` §4).
- Effect size yang sangat kecil (mis. Cramér's V < 0.1, |rho| < 0.1) menunjukkan signifikansi statistik tanpa relevansi praktis pada n besar.