"""Stratified sample from the enriched candidate pool for inline labeling.
~22 per year, mixing lexicon hits and detox-only candidates. Deterministic (seed 42)."""
import numpy as np
import pandas as pd

SEED = 42
PER_YEAR = 22
pool = pd.read_parquet("reports/_enriched_pool.parquet")
rng = np.random.RandomState(SEED)

picks = []
for yr, grp in pool.groupby("year"):
    lex = grp[grp.lex]
    det = grp[~grp.lex]
    n_lex = min(len(lex), PER_YEAR // 2)
    n_det = min(len(det), PER_YEAR - n_lex)
    parts = []
    if n_lex:
        parts.append(lex.sample(n=n_lex, random_state=rng))
    if n_det:
        parts.append(det.sample(n=n_det, random_state=rng))
    picks.append(pd.concat(parts))
samp = pd.concat(picks, ignore_index=True)
samp = samp.sample(frac=1.0, random_state=rng).reset_index(drop=True)
samp.insert(0, "idx", range(len(samp)))

out = samp[["idx", "year", "key", "max_toxicity_prob", "lex"]].copy()
out["key"] = out["key"].astype(str).str.replace("\n", " ", regex=False).str.slice(0, 200)
out.to_csv("reports/_enriched_sample.csv", index=False)
# keep the full keyed sample (with match keys) for later join to model predictions
samp.to_parquet("reports/_enriched_sample_keyed.parquet")
print("sampled:", len(samp), "| lex:", int(samp.lex.sum()), "| detox-only:", int((~samp.lex).sum()))
print(out.groupby("year").size().to_string())
