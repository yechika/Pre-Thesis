# Toxicity Threshold Tuning (gold-test)

Per-label decision thresholds fit on the **gold-train** split (max F1) and applied to the held-out **gold-test** split (no test leakage). Default = 0.5 for all labels. Labels with <5 train positives (severe_toxic, threat) are untuned. F1-macro is over labels with test support (toxic, obscene, insult).

## F1-micro & F1-macro: default 0.5 vs tuned

| model | F1-micro @0.5 [95% CI] | F1-micro tuned [95% CI] | Δ | F1-macro @0.5 | F1-macro tuned | Δ |
|:--|--:|--:|--:|--:|--:|--:|
| bert | 0.125 [0.020, 0.248] | **0.139 [0.021, 0.276]** | +0.014 | 0.150 | **0.141** | -0.009 |
| roberta | 0.141 [0.032, 0.259] | **0.165 [0.020, 0.329]** | +0.023 | 0.154 | **0.168** | +0.015 |
| distilbert | 0.128 [0.021, 0.256] | **0.135 [0.022, 0.272]** | +0.007 | 0.137 | **0.142** | +0.006 |
| detoxify | 0.152 [0.024, 0.294] | **0.136 [0.021, 0.272]** | -0.016 | 0.162 | **0.142** | -0.020 |

## Per-label tuned thresholds & F1 (gold-test)

| model | label | threshold | F1 @0.5 | F1 tuned | Δ |
|:--|:--|--:|--:|--:|--:|
| bert | toxic | 0.93 | 0.075 | 0.080 | +0.005 |
| bert | obscene | 0.36 | 0.200 | 0.182 | -0.018 |
| bert | insult | 0.26 | 0.174 | 0.160 | -0.014 |
| roberta | toxic | 0.94 | 0.120 | 0.129 | +0.009 |
| roberta | obscene | 0.27 | 0.261 | 0.222 | -0.039 |
| roberta | insult | 0.31 | 0.080 | 0.154 | +0.074 |
| distilbert | toxic | 0.94 | 0.108 | 0.074 | -0.034 |
| distilbert | obscene | 0.44 | 0.211 | 0.211 | +0.000 |
| distilbert | insult | 0.04 | 0.091 | 0.143 | +0.052 |
| detoxify | toxic | 0.83 | 0.111 | 0.143 | +0.032 |
| detoxify | obscene | 0.25 | 0.200 | 0.174 | -0.026 |
| detoxify | insult | 0.04 | 0.174 | 0.108 | -0.066 |