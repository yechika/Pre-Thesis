"""Batch GPU inference for in-domain sentiment (RoBERTa, DistilBERT, BERT) over all data/processed/*.parquet."""
import glob, os, sys, time, warnings
sys.path.insert(0, ".")
os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.preprocessing.slang import translate_slang

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

MODEL_KEYS = ["roberta", "distilbert", "bert"]
SENT_LABELS = ["negative", "neutral", "positive"]

processed_files = sorted(glob.glob("data/processed/[0-9]*.parquet"))
print(f"Found {len(processed_files)} processed parquet files.")

for model_key in MODEL_KEYS:
    model_dir = f"models/{model_key}-sentiment"
    if not os.path.exists(model_dir):
        print(f"Skipping {model_key}, model_dir not found: {model_dir}")
        continue
    
    out_root = Path(f"data/inference/{model_key}_sentiment")
    out_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"Starting inference for: {model_key.upper()} -> {out_root}")
    print(f"=======================================================")
    
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    if device == "cuda":
        model = model.half()
    
    t_start = time.time()
    total_messages = 0
    
    for p in processed_files:
        fname = os.path.basename(p)
        out_p = out_root / fname
        
        df = pd.read_parquet(p, columns=["match_id", "time", "player_slot", "key"])
        n = len(df)
        total_messages += n
        
        raw_texts = df["key"].fillna("").astype(str).tolist()
        slang_texts = [translate_slang(t) for t in raw_texts]
        
        preds = []
        B = 256
        with torch.no_grad():
            for i in range(0, len(slang_texts), B):
                batch_t = slang_texts[i:i+B]
                enc = tok(batch_t, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds.append(probs)
        
        all_probs = np.concatenate(preds, axis=0) if len(preds) > 0 else np.zeros((0, 3))
        
        res_df = df[["match_id", "time", "player_slot"]].copy()
        res_df["prob_negative"] = all_probs[:, 0].astype(np.float32)
        res_df["prob_neutral"] = all_probs[:, 1].astype(np.float32)
        res_df["prob_positive"] = all_probs[:, 2].astype(np.float32)
        res_df["predicted_label"] = np.array(SENT_LABELS)[all_probs.argmax(axis=-1)]
        res_df["sentiment_score"] = (all_probs[:, 2] - all_probs[:, 0]).astype(np.float32) # P(pos) - P(neg)
        
        res_df.to_parquet(out_p, index=False)
        print(f"  [{model_key}] Processed {fname:15s} ({n:7,d} msgs)")
    
    elapsed = time.time() - t_start
    print(f"Done {model_key}: {total_messages:,} messages in {elapsed:.1f}s ({total_messages/elapsed:.0f} msg/s)")

print("\n=== FULL-CORPUS INFERENCE COMPLETE ===")
