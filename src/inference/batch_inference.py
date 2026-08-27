"""Batched GPU inference: model fine-tuned → probabilitas per-pesan ke parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import translate_slang

KEY_COLS = ["match_id", "time", "player_slot"]


def _prepare_texts(df: pd.DataFrame, text_col: str, apply_slang: bool) -> list[str]:
    """Raw chat -> model input. Applies slang translation for train/infer parity.

    Models are fine-tuned on slang-translated text (see gold_loaders /
    notebook 04), so inference MUST translate too or the input distribution
    shifts and predictions degrade.
    """
    raw = df[text_col].fillna("").astype(str)
    if apply_slang:
        return raw.map(translate_slang).tolist()
    return raw.tolist()


@dataclass
class InferenceStats:
    n_input: int
    n_output: int
    skipped: bool
    out_path: Path


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _make_pipeline(
    model_dir: Path,
    task: str,
    batch_size: int,
    fp16: bool = True,
):
    """Build a transformers pipeline (text-classification or multi-label).

    task ∈ {'sentiment', 'toxicity'}.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    device_idx = 0 if _device() == "cuda" else -1
    if fp16 and device_idx == 0:
        model = model.half()

    if task == "sentiment":
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_idx,
            top_k=None,  # return all class scores
            batch_size=batch_size,
        )
    elif task == "toxicity":
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_idx,
            top_k=None,  # multi-label: kembalikan semua skor
            function_to_apply="sigmoid",
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def infer_folder(
    processed_path: Path,
    model_dir: Path,
    task: str,
    out_path: Path,
    batch_size: int = 128,
    fp16: bool = True,
    text_col: str = "key",
    apply_slang: bool = True,
) -> InferenceStats:
    """Inferensi satu file `data/processed/<folder>.parquet`.

    Skip jika `out_path` sudah ada (idempotent).
    apply_slang: translate Dota slang sebelum inference (paritas dengan training).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        return InferenceStats(
            n_input=0, n_output=len(existing), skipped=True, out_path=out_path
        )

    df = pd.read_parquet(processed_path)
    texts = _prepare_texts(df, text_col, apply_slang)

    pipe = _make_pipeline(model_dir, task=task, batch_size=batch_size, fp16=fp16)
    raw_results = pipe(texts, truncation=True)

    # Setiap raw result adalah list of dict[label, score].
    # Bangun DataFrame: kolom prob_<label> untuk setiap label, plus predicted_label.
    if task == "sentiment":
        prob_df = pd.DataFrame([
            {f"prob_{r['label'].lower()}": float(r["score"]) for r in row}
            for row in raw_results
        ])
        # predicted_label = argmax label (lowercase)
        prob_cols = [c for c in prob_df.columns if c.startswith("prob_")]
        if prob_cols:
            argmax_idx = prob_df[prob_cols].values.argmax(axis=1)
            label_names = [c.replace("prob_", "") for c in prob_cols]
            prob_df["predicted_label"] = [label_names[i] for i in argmax_idx]
        else:
            prob_df["predicted_label"] = ""
    else:  # toxicity multi-label
        prob_df = pd.DataFrame([
            {f"prob_{r['label'].lower()}": float(r["score"]) for r in row}
            for row in raw_results
        ])
        prob_cols = [c for c in prob_df.columns if c.startswith("prob_")]
        # threshold 0.5 untuk binary prediksi per-label
        for c in prob_cols:
            label_name = c.replace("prob_", "")
            prob_df[f"pred_{label_name}"] = (prob_df[c] >= 0.5).astype(int)
        # max toxicity prob
        if prob_cols:
            prob_df["max_toxicity_prob"] = prob_df[prob_cols].max(axis=1)
            prob_df["any_toxic"] = (prob_df[prob_cols] >= 0.5).any(axis=1).astype(int)

    # Output: KEY_COLS + kolom prob/pred. Tidak duplikasi text supaya kecil.
    keys = df[KEY_COLS].reset_index(drop=True)
    prob_df = prob_df.reset_index(drop=True)
    out = pd.concat([keys, prob_df], axis=1)
    out.to_parquet(out_path, compression="snappy", index=False)

    return InferenceStats(
        n_input=len(df), n_output=len(out), skipped=False, out_path=out_path
    )


def infer_detoxify(
    processed_path: Path,
    out_path: Path,
    batch_size: int = 128,
    text_col: str = "key",
    apply_slang: bool = True,
) -> InferenceStats:
    """Inferensi via `unitary/toxic-bert` pretrained (tanpa fine-tune tambahan).

    Output 6 label Detoxify default: toxic, severe_toxic, obscene, threat,
    insult, identity_hate.
    apply_slang: translate Dota slang dulu supaya input seragam dengan model
    fine-tuned (fair comparison) + Detoxify mengenali jargon toksik Dota.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        return InferenceStats(
            n_input=0, n_output=len(existing), skipped=True, out_path=out_path
        )

    from detoxify import Detoxify

    df = pd.read_parquet(processed_path)
    texts = _prepare_texts(df, text_col, apply_slang)

    device = _device()
    model = Detoxify("original", device=device)

    # Detoxify pakai nama label berbeda dari Jigsaw config — remap supaya
    # schema seragam dengan fine-tuned models di evaluator NB06.
    DETOX_RENAME = {
        "toxicity": "toxic",
        "severe_toxicity": "severe_toxic",
        "identity_attack": "identity_hate",
    }

    # Batch inference manual (Detoxify accepts list).
    rows = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = model.predict(batch)
        # result: dict[label → list[score]]
        labels = list(result.keys())
        for j in range(len(batch)):
            rows.append({
                f"prob_{DETOX_RENAME.get(lbl, lbl)}": float(result[lbl][j])
                for lbl in labels
            })

    prob_df = pd.DataFrame(rows)
    prob_cols = [c for c in prob_df.columns if c.startswith("prob_")]
    for c in prob_cols:
        label_name = c.replace("prob_", "")
        prob_df[f"pred_{label_name}"] = (prob_df[c] >= 0.5).astype(int)
    if prob_cols:
        prob_df["max_toxicity_prob"] = prob_df[prob_cols].max(axis=1)
        prob_df["any_toxic"] = (prob_df[prob_cols] >= 0.5).any(axis=1).astype(int)

    keys = df[KEY_COLS].reset_index(drop=True)
    out = pd.concat([keys, prob_df.reset_index(drop=True)], axis=1)
    out.to_parquet(out_path, compression="snappy", index=False)

    return InferenceStats(
        n_input=len(df), n_output=len(out), skipped=False, out_path=out_path
    )
