"""Per-label threshold tuning for multi-label toxicity (no retrain / no re-inference).

Thresholds are fit on the gold TRAIN split (maximize per-label F1) and applied to
the held-out gold TEST split, so there is no test leakage. Compares F1@0.5 (default)
against F1@tuned and writes a thesis-ready table.

Reads stored inference probabilities from data/inference/<model>_toxicity/*.parquet.
Run: python _threshold_tune.py
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.runtime import load_config
from src.eval.bootstrap_ci import bootstrap_ci

cfg = load_config("configs/experiment.yaml")
SEED = int(cfg["seed"])
BOOT = int(cfg["evaluation"]["bootstrap_resamples"])
TOX = cfg["labels"]["toxicity_labels"]               # 6 labels, fixed order
GOLD = Path(cfg["data"]["gold_root"])
INF = Path(cfg["data"]["inference_root"])
MODELS = ["bert", "roberta", "distilbert", "detoxify"]
MIN_TRAIN_POS = 5                                    # below this -> cannot tune, keep 0.5

LOG = open("reports/_threshold_tune.log", "w", encoding="utf-8")
def log(*a):
    m = " ".join(str(x) for x in a)
    print(m); LOG.write(m + "\n"); LOG.flush()

def keyed(df):
    df = df.copy()
    df["k"] = list(zip(df["match_id"], df["time"], df["player_slot"]))
    return df

train = keyed(pd.read_csv(GOLD / "train.csv"))
test = keyed(pd.read_csv(GOLD / "test.csv"))
tcols = [f"tox_{l}" for l in TOX]
y_tr = train[tcols].astype(int).values
y_te = test[tcols].astype(int).values

log("train support:", dict(zip(TOX, y_tr.sum(0).tolist())))
log("test  support:", dict(zip(TOX, y_te.sum(0).tolist())))
supported = [i for i in range(len(TOX)) if y_te[:, i].sum() > 0]   # labels scored in macro
log("macro over supported labels:", [TOX[i] for i in supported])

def load_probs(folder, base):
    root = INF / folder
    if not root.exists():
        return None
    inf = keyed(pd.concat([pd.read_parquet(p) for p in sorted(root.glob("*.parquet"))],
                          ignore_index=True)).drop_duplicates(subset="k", keep="first")
    m = base.merge(inf, on="k", how="left", suffixes=("", "_inf"))
    pcols = [f"prob_{l}" for l in TOX]
    return m[pcols].fillna(0.0).values

GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

def f1_micro(yt, yp):
    return f1_score(yt, yp, average="micro", zero_division=0)
def f1_macro_sup(yt, yp):
    return f1_score(yt[:, supported], yp[:, supported], average="macro", zero_division=0)

rows, per_label_rows = [], []
for mk in MODELS:
    p_tr = load_probs(f"{mk}_toxicity", train)
    p_te = load_probs(f"{mk}_toxicity", test)
    if p_tr is None or p_te is None:
        log(f"[SKIP] {mk}: no inference"); continue

    thr = np.full(len(TOX), 0.5)
    for i in range(len(TOX)):
        if y_tr[:, i].sum() < MIN_TRAIN_POS:
            continue
        best_t, best_f = 0.5, -1.0
        for t in GRID:
            f = f1_score(y_tr[:, i], (p_tr[:, i] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        thr[i] = best_t

    pred_def = (p_te >= 0.5).astype(int)
    pred_tun = (p_te >= thr).astype(int)

    fmi_def, fmi_tun = f1_micro(y_te, pred_def), f1_micro(y_te, pred_tun)
    fma_def, fma_tun = f1_macro_sup(y_te, pred_def), f1_macro_sup(y_te, pred_tun)
    _, dlo, dhi = bootstrap_ci(f1_micro, y_te, pred_def, n_resamples=BOOT, seed=SEED)
    _, tlo, thi = bootstrap_ci(f1_micro, y_te, pred_tun, n_resamples=BOOT, seed=SEED)

    rows.append({"model": mk,
                 "f1_micro_default": fmi_def, "f1_micro_def_lo": dlo, "f1_micro_def_hi": dhi,
                 "f1_micro_tuned": fmi_tun, "f1_micro_tun_lo": tlo, "f1_micro_tun_hi": thi,
                 "delta_micro": fmi_tun - fmi_def,
                 "f1_macro_default": fma_def, "f1_macro_tuned": fma_tun,
                 "delta_macro": fma_tun - fma_def,
                 **{f"thr_{TOX[i]}": thr[i] for i in supported}})
    log(f"\n{mk}: F1-micro {fmi_def:.3f} -> {fmi_tun:.3f} ({fmi_tun-fmi_def:+.3f}) | "
        f"F1-macro(sup) {fma_def:.3f} -> {fma_tun:.3f} ({fma_tun-fma_def:+.3f})")
    for i in supported:
        f_d = f1_score(y_te[:, i], pred_def[:, i], zero_division=0)
        f_t = f1_score(y_te[:, i], pred_tun[:, i], zero_division=0)
        per_label_rows.append({"model": mk, "label": TOX[i], "threshold": thr[i],
                               "f1_default": f_d, "f1_tuned": f_t, "delta": f_t - f_d})
        log(f"    {TOX[i]:14s} thr={thr[i]:.2f}  F1 {f_d:.3f} -> {f_t:.3f} ({f_t-f_d:+.3f})")

df = pd.DataFrame(rows)
df.to_csv("reports/eval_toxicity_threshold_tuned.csv", index=False)
pd.DataFrame(per_label_rows).to_csv("reports/eval_toxicity_threshold_perlabel.csv", index=False)

# ---- thesis-ready markdown ----
md = ["# Toxicity Threshold Tuning (gold-test)\n",
      "Per-label decision thresholds fit on the **gold-train** split (max F1) and applied "
      "to the held-out **gold-test** split (no test leakage). Default = 0.5 for all labels. "
      f"Labels with <{MIN_TRAIN_POS} train positives (severe_toxic, threat) are untuned. "
      "F1-macro is over labels with test support "
      f"({', '.join(TOX[i] for i in supported)}).\n",
      "## F1-micro & F1-macro: default 0.5 vs tuned\n",
      "| model | F1-micro @0.5 [95% CI] | F1-micro tuned [95% CI] | Δ | F1-macro @0.5 | F1-macro tuned | Δ |",
      "|:--|--:|--:|--:|--:|--:|--:|"]
for r in rows:
    md.append("| %s | %.3f [%.3f, %.3f] | **%.3f [%.3f, %.3f]** | %+.3f | %.3f | **%.3f** | %+.3f |" % (
        r["model"], r["f1_micro_default"], r["f1_micro_def_lo"], r["f1_micro_def_hi"],
        r["f1_micro_tuned"], r["f1_micro_tun_lo"], r["f1_micro_tun_hi"], r["delta_micro"],
        r["f1_macro_default"], r["f1_macro_tuned"], r["delta_macro"]))
md.append("\n## Per-label tuned thresholds & F1 (gold-test)\n")
md.append("| model | label | threshold | F1 @0.5 | F1 tuned | Δ |")
md.append("|:--|:--|--:|--:|--:|--:|")
for r in per_label_rows:
    md.append("| %s | %s | %.2f | %.3f | %.3f | %+.3f |" %
              (r["model"], r["label"], r["threshold"], r["f1_default"], r["f1_tuned"], r["delta"]))
open("reports/toxicity_threshold_tuning.md", "w", encoding="utf-8").write("\n".join(md))
log("\nwrote reports/eval_toxicity_threshold_tuned.csv + _perlabel.csv + toxicity_threshold_tuning.md")
LOG.close()
