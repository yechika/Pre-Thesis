"""Build an enriched toxic-candidate pool for a secondary toxicity probe set.

Candidates = messages with Detoxify max_toxicity_prob >= 0.5 OR a hit on an
EN+ID profanity lexicon. Excludes existing gold keys, dedups by lowercased text,
spreads across years. Output: reports/_enriched_pool.parquet (+ a CSV preview).
"""
import glob, os, re
from pathlib import Path
import pandas as pd

PROF = r"\b(fuck\w*|shit\w*|bitch\w*|asshole\w*|cunt\w*|dick\w*|retard\w*|idiot\w*|stupid|trash|noob\w*|garbage|moron|nigger\w*|faggot\w*|gay|whore|slut|asu|anjing|anjg|anj|kontol|kntl|memek|mmk|bangsat|ngentot|jancok|jancuk|cok|cuk|babi|goblok|tolol|tll|kampret|tai|taik|bgst|bego|kntol|pepek|pler|monyet|ngewe|pukimak|puki)\b"
prog = re.compile(PROF, re.I)

g = pd.concat([pd.read_csv("data/gold/train.csv"), pd.read_csv("data/gold/test.csv")],
              ignore_index=True)
gold_keys = set(zip(g.match_id, g.time, g.player_slot))

rows = []
for pp in sorted(glob.glob("data/processed/[0-9]*.parquet")):
    folder = os.path.splitext(os.path.basename(pp))[0]
    d = pd.read_parquet(pp, columns=["match_id", "time", "player_slot", "key", "year",
                                     "month", "tier", "match_outcome_for_player", "duration"])
    dp = f"data/inference/detoxify_toxicity/{folder}.parquet"
    if not os.path.exists(dp):
        continue
    det = pd.read_parquet(dp)
    d["k"] = list(zip(d.match_id, d.time, d.player_slot))
    det["k"] = list(zip(det.match_id, det.time, det.player_slot))
    det = det.drop_duplicates("k")
    m = d.merge(det[["k", "max_toxicity_prob"]], on="k", how="left")
    kl = m.key.astype(str).str.lower()
    is_lex = kl.str.contains(prog)
    cand = (m.max_toxicity_prob.fillna(0) >= 0.5) | is_lex
    keep = cand & ~m.k.isin(gold_keys)
    sub = m[keep].copy()
    sub["lex"] = is_lex[keep].values
    rows.append(sub)

allc = pd.concat(rows, ignore_index=True)
allc["kl"] = allc.key.astype(str).str.lower().str.strip()
allc = allc.drop_duplicates("kl")
allc = allc[allc.kl.str.len() > 0]

print("total dedup candidates:", len(allc))
print("  lexicon hits:", int(allc.lex.sum()),
      " detox>=0.5 only:", int((~allc.lex & (allc.max_toxicity_prob >= 0.5)).sum()))
print("by year:\n", allc.groupby("year").size().to_string())
Path("reports").mkdir(exist_ok=True)
allc.to_parquet("reports/_enriched_pool.parquet")
print("saved reports/_enriched_pool.parquet")
