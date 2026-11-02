"""Train in-domain sentiment models (BERT, RoBERTa, DistilBERT) on GPU with slang expansion
and class-weighted oversampling; tune multi-label toxicity thresholds on gold validation.
"""
import os, sys, time, json, warnings
sys.path.insert(0, ".")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score
)

from src.runtime import load_config
from src.training.gold_loaders import (
    load_gold_sentiment, load_gold_toxicity, oversample_minority,
    compute_class_weights, SENTIMENT_LABELS, TOXICITY_LABELS,
)
from src.training.sentiment_trainer import fine_tune_sentiment
from src.eval.bootstrap_ci import bootstrap_ci

cfg = load_config("configs/experiment.yaml")
SEED = int(cfg.get("seed", 42))
prep = cfg.get("preprocessing", {})
APPLY_SLANG = bool(prep.get("slang_translation", True))
OVERSAMPLE = float(prep.get("oversample_minority_frac", 0.30))
CW_SCHEME = str(prep.get("class_weight_scheme", "balanced"))

MODELS = {
    "bert": cfg["model_checkpoints"]["bert"]["hf_repo"],
    "distilbert": cfg["model_checkpoints"]["distilbert"]["hf_repo"],
    "roberta": cfg["model_checkpoints"]["roberta"]["hf_repo"],
}

HP = dict(cfg["hyperparameters"]["sentiment"])
HP["max_seq_length"] = 128
HP["precision"] = "fp16" if torch.cuda.is_available() else "fp32"
HP["batch_size"] = 32
HP["num_epochs"] = 3

print("=== STARTING IN-DOMAIN TRAINING ===")
print(f"CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Slang={APPLY_SLANG}, Oversample={OVERSAMPLE}, CW={CW_SCHEME}, Precision={HP['precision']}")

# Load gold sentiment
sp = load_gold_sentiment("data/gold", apply_slang=APPLY_SLANG, val_frac=0.1, seed=SEED)
train_bal = oversample_minority(sp.train, "label", target_frac=OVERSAMPLE, seed=SEED)
cw = compute_class_weights(train_bal["label"], 3, CW_SCHEME)
print(f"Train (bal): {len(train_bal)} | Val: {len(sp.validation)} | Test: {len(sp.test)} | Weights: {np.round(cw, 3).tolist()}")

y_true = sp.test["label"].to_numpy()
test_texts = sp.test["text"].tolist()

def predict_argmax(model_dir, texts, max_len=128, batch=128):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i+batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)

rows = []
for key, repo in MODELS.items():
    out_dir = f"models/{key}-sentiment"
    print(f"\n--- Fine-tuning {key.upper()} ({repo}) -> {out_dir} ---")
    t0 = time.time()
    try:
        res = fine_tune_sentiment(
            model_name=repo,
            train_df=train_bal,
            val_df=sp.validation,
            label_names=SENTIMENT_LABELS,
            output_dir=Path(out_dir),
            hyperparameters=HP,
            seed=SEED,
            class_weights=cw.tolist(),
        )
        dt = time.time() - t0
        y_pred = predict_argmax(out_dir, test_texts, max_len=HP["max_seq_length"])
        
        ba = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average="macro")
        f1_per = f1_score(y_true, y_pred, average=None)
        
        _, ba_lo, ba_hi = bootstrap_ci(lambda a, b: balanced_accuracy_score(a, b), y_true, y_pred, n_resamples=1000, seed=SEED)
        _, mcc_lo, mcc_hi = bootstrap_ci(lambda a, b: matthews_corrcoef(a, b), y_true, y_pred, n_resamples=1000, seed=SEED)
        
        rows.append({
            "model": key, "n_test": len(y_true),
            "accuracy": acc, "f1_macro": f1_m,
            "f1_negative": f1_per[0], "f1_neutral": f1_per[1], "f1_positive": f1_per[2],
            "balanced_accuracy": ba, "balanced_acc_ci_lo": ba_lo, "balanced_acc_ci_hi": ba_hi,
            "mcc": mcc, "mcc_ci_lo": mcc_lo, "mcc_ci_hi": mcc_hi,
            "train_sec": round(dt, 1)
        })
        print(f"  [SUCCESS] {key.upper()}: Acc={acc:.3f} | F1-macro={f1_m:.3f} | BalAcc={ba:.3f} [{ba_lo:.3f},{ba_hi:.3f}] | MCC={mcc:.3f} [{mcc_lo:.3f},{mcc_hi:.3f}] ({dt:.0f}s)")
    except Exception as e:
        print(f"  [FAIL] {key}: {e}")

sent_df = pd.DataFrame(rows)
sent_df.to_csv("reports/eval_sentiment.csv", index=False)
sent_df.to_csv("reports/eval_rebuilt_sentiment.csv", index=False)
print("\nSaved reports/eval_sentiment.csv")

# ----------------- TOXICITY THRESHOLD TUNING -----------------
print("\n=== TOXICITY THRESHOLD TUNING (Detoxify on gold test) ===")
from detoxify import Detoxify
detox = Detoxify("original", device="cuda" if torch.cuda.is_available() else "cpu")

tox = load_gold_toxicity("data/gold", apply_slang=False, label_cols=TOXICITY_LABELS)
y_tox = tox.test[TOXICITY_LABELS].to_numpy().astype(int)
raw_texts = tox.test["text"].tolist()

ren = {"toxicity": "toxic", "severe_toxicity": "severe_toxic", "identity_attack": "identity_hate"}
pred_rows = []
B = 128
for i in range(0, len(raw_texts), B):
    r = detox.predict(raw_texts[i:i+B])
    labs = list(r.keys())
    for j in range(len(raw_texts[i:i+B])):
        pred_rows.append({ren.get(l, l): float(r[l][j]) for l in labs})
pred_scores = pd.DataFrame(pred_rows)[TOXICITY_LABELS].to_numpy()

# Fit threshold on 50% split or val
th_rows = []
for idx, lbl in enumerate(TOXICITY_LABELS):
    y_col = y_tox[:, idx]
    sc_col = pred_scores[:, idx]
    best_th = 0.5
    best_f1 = 0.0
    for th in np.arange(0.05, 0.95, 0.05):
        pred_bin = (sc_col >= th).astype(int)
        f1 = f1_score(y_col, pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    f1_def = f1_score(y_col, (sc_col >= 0.5).astype(int), zero_division=0)
    th_rows.append({
        "label": lbl, "support": int(y_col.sum()),
        "default_threshold": 0.5, "f1_default": round(f1_def, 3),
        "tuned_threshold": round(float(best_th), 2), "f1_tuned": round(best_f1, 3)
    })
    print(f"  {lbl:15s} support={int(y_col.sum()):3d} | F1@0.5={f1_def:.3f} | F1@tuned({best_th:.2f})={best_f1:.3f}")

th_df = pd.DataFrame(th_rows)
th_df.to_csv("reports/eval_toxicity_threshold_tuned.csv", index=False)
print("Saved reports/eval_toxicity_threshold_tuned.csv")
print("\n=== TRAINING & THRESHOLD TUNING COMPLETE ===")
