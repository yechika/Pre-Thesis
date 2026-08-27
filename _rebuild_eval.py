"""Retrain sentiment in-domain (+slang, class-weighted) and evaluate on the gold
TEST set; ablate slang for Detoxify toxicity. Writes a new-vs-old comparison.

CPU proxy: max_seq_length=64 (messages stay short even after slang expansion),
otherwise the same config hyperparameters. Run: python _rebuild_eval.py
"""
import os, sys, json, time, warnings
sys.path.insert(0, ".")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, average_precision_score

from src.runtime import load_config
from src.training.gold_loaders import (
    load_gold_sentiment, load_gold_toxicity, oversample_minority,
    compute_class_weights, SENTIMENT_LABELS, TOXICITY_LABELS,
)
from src.training.sentiment_trainer import fine_tune_sentiment
from src.eval.bootstrap_ci import bootstrap_ci

LOG = open("reports/_rebuild_eval.log", "w", encoding="utf-8")
def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); LOG.write(msg + "\n"); LOG.flush()

cfg = load_config("configs/experiment.yaml")
SEED = int(cfg["seed"])
prep = cfg.get("preprocessing", {})
APPLY_SLANG = bool(prep.get("slang_translation", True))
OVERSAMPLE = float(prep.get("oversample_minority_frac", 0.30))
CW_SCHEME = str(prep.get("class_weight_scheme", "balanced"))
MODELS = {
    "bert": cfg["model_checkpoints"]["bert"]["hf_repo"],
    "roberta": cfg["model_checkpoints"]["roberta"]["hf_repo"],
    "distilbert": cfg["model_checkpoints"]["distilbert"]["hf_repo"],
}
HP = dict(cfg["hyperparameters"]["sentiment"])
HP["max_seq_length"] = 64          # CPU proxy
HP["precision"] = "fp32"           # no cuda here

log("=== REBUILD EVAL ===  slang=%s oversample=%s cw=%s" % (APPLY_SLANG, OVERSAMPLE, CW_SCHEME))

# ---------------- SENTIMENT (in-domain + slang) ----------------
sp = load_gold_sentiment("data/gold", apply_slang=APPLY_SLANG, val_frac=0.1, seed=SEED)
train_bal = oversample_minority(sp.train, "label", target_frac=OVERSAMPLE, seed=SEED)
cw = compute_class_weights(train_bal["label"], 3, CW_SCHEME)
log("train(after oversample)=%d val=%d test=%d cw=%s"
    % (len(train_bal), len(sp.validation), len(sp.test), np.round(cw, 3).tolist()))

y_true = sp.test["label"].to_numpy()
test_texts = sp.test["text"].tolist()

def predict_argmax(model_dir, texts, max_len=64, batch=64):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)

rows = []
for key, repo in MODELS.items():
    out_dir = "models/_rebuilt/%s-sentiment" % key
    log("\n--- TRAIN %s sentiment (in-domain+slang) ---" % key)
    t0 = time.time()
    try:
        res = fine_tune_sentiment(repo, train_bal, sp.validation, SENTIMENT_LABELS,
                                  out_dir, HP, seed=SEED, class_weights=cw.tolist())
        dt = time.time() - t0
        y_pred = predict_argmax(out_dir, test_texts, max_len=HP["max_seq_length"])
        ba = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        _, ba_lo, ba_hi = bootstrap_ci(lambda a, b: balanced_accuracy_score(a, b),
                                       y_true, y_pred, n_resamples=1000, seed=SEED)
        _, mcc_lo, mcc_hi = bootstrap_ci(lambda a, b: matthews_corrcoef(a, b),
                                         y_true, y_pred, n_resamples=1000, seed=SEED)
        rows.append({"model": key, "n_test": len(y_true),
                     "balanced_accuracy": ba, "balanced_acc_ci_lo": ba_lo, "balanced_acc_ci_hi": ba_hi,
                     "mcc": mcc, "mcc_ci_lo": mcc_lo, "mcc_ci_hi": mcc_hi,
                     "train_sec": round(dt, 1)})
        log("  %s: bal_acc=%.3f [%.3f,%.3f]  mcc=%.3f [%.3f,%.3f]  (%.0fs)"
            % (key, ba, ba_lo, ba_hi, mcc, mcc_lo, mcc_hi, dt))
    except Exception as e:
        log("  [FAIL] %s: %r" % (key, e))

sent_df = pd.DataFrame(rows)
sent_df.to_csv("reports/eval_rebuilt_sentiment.csv", index=False)
log("\nwrote reports/eval_rebuilt_sentiment.csv")

# ---------------- TOXICITY (Detoxify slang ablation, no training) ----------------
log("\n=== TOXICITY: Detoxify slang ablation on gold test ===")
tox = load_gold_toxicity("data/gold", apply_slang=False, label_cols=TOXICITY_LABELS)
y_tox = tox.test[TOXICITY_LABELS].to_numpy().astype(int)
support = y_tox.sum(axis=0)
sup_idx = [i for i in range(len(TOXICITY_LABELS)) if support[i] > 0]
log("gold-test label support: %s" % dict(zip(TOXICITY_LABELS, support.tolist())))

def detox_scores(texts):
    from detoxify import Detoxify
    m = Detoxify("original", device="cpu")
    ren = {"toxicity": "toxic", "severe_toxicity": "severe_toxic", "identity_attack": "identity_hate"}
    rows = []
    B = 128
    for i in range(0, len(texts), B):
        r = m.predict(texts[i:i+B])
        labs = list(r.keys())
        for j in range(len(texts[i:i+B])):
            rows.append({ren.get(l, l): float(r[l][j]) for l in labs})
    df = pd.DataFrame(rows)
    return df[TOXICITY_LABELS].to_numpy()

from src.preprocessing import translate_slang
raw_texts = tox.test["text"].tolist()                       # already raw (apply_slang=False)
slang_texts = [translate_slang(t) for t in raw_texts]

tox_rows = []
for tag, txts in [("detoxify_no_slang", raw_texts), ("detoxify_slang", slang_texts)]:
    sc = detox_scores(txts)
    pred = (sc >= 0.5).astype(int)
    mccs = [matthews_corrcoef(y_tox[:, i], pred[:, i]) for i in sup_idx]
    mcc_macro = float(np.mean(mccs)) if mccs else 0.0
    pr_macro = float(average_precision_score(y_tox[:, sup_idx], sc[:, sup_idx], average="macro"))
    pr_micro = float(average_precision_score(y_tox[:, sup_idx], sc[:, sup_idx], average="micro"))
    tox_rows.append({"variant": tag, "mcc_macro": mcc_macro,
                     "pr_auc_macro": pr_macro, "pr_auc_micro": pr_micro})
    log("  %-20s mcc_macro=%.3f pr_auc_macro=%.3f pr_auc_micro=%.3f"
        % (tag, mcc_macro, pr_macro, pr_micro))
pd.DataFrame(tox_rows).to_csv("reports/eval_rebuilt_toxicity_detox.csv", index=False)

# ---------------- COMPARISON MARKDOWN ----------------
old = pd.read_csv("reports/eval_imbalance_sentiment.csv").set_index("model")
md = ["# Rebuilt vs Original — Imbalance-Robust Metrics (gold-test)\n",
      "CPU re-run: sentiment **fine-tuned in-domain on gold + slang translation + "
      "class-weighted oversampling** (max_seq_length=64 proxy). Toxicity = Detoxify "
      "slang ablation (no retrain).\n",
      "## Sentiment — Balanced Accuracy + MCC (OLD external-trained → NEW in-domain)\n",
      "| model | bal_acc OLD | bal_acc NEW | Δ | MCC OLD | MCC NEW | Δ |",
      "|:--|--:|--:|--:|--:|--:|--:|"]
for _, r in sent_df.iterrows():
    k = r["model"]
    ob = old.loc[k, "balanced_accuracy"] if k in old.index else float("nan")
    om = old.loc[k, "mcc"] if k in old.index else float("nan")
    md.append("| %s | %.3f | **%.3f** | %+.3f | %.3f | **%.3f** | %+.3f |"
              % (k, ob, r["balanced_accuracy"], r["balanced_accuracy"] - ob,
                 om, r["mcc"], r["mcc"] - om))
md.append("\n_Chance bal_acc (3-class) ≈ 0.33. MCC: 0=random, <0=worse-than-chance, 1=perfect._\n")
md.append("## Toxicity — Detoxify slang ablation (gold-test, supported labels)\n")
md.append("| variant | mcc_macro | pr_auc_macro | pr_auc_micro |")
md.append("|:--|--:|--:|--:|")
for r in tox_rows:
    md.append("| %s | %.3f | %.3f | %.3f |"
              % (r["variant"], r["mcc_macro"], r["pr_auc_macro"], r["pr_auc_micro"]))
open("reports/comparison_metrics_rebuilt.md", "w", encoding="utf-8").write("\n".join(md))
log("\nwrote reports/comparison_metrics_rebuilt.md")
log("\n=== DONE ===")
LOG.close()
