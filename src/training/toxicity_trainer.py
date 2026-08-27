"""Fine-tune transformer untuk toksisitas multi-label (BCEWithLogitsLoss)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainingResult:
    model_dir: Path
    eval_metrics: dict[str, float]
    history: list[dict[str, Any]]


def fine_tune_toxicity(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_names: list[str],
    output_dir: Path,
    hyperparameters: dict[str, Any],
    seed: int = 42,
    pos_weights: list[float] | None = None,
) -> TrainingResult:
    """Fine-tune satu transformer untuk multi-label toxicity classification.

    train_df, val_df: kolom `text` + satu kolom per label di `label_names` (0/1).
    Text diasumsikan SUDAH melewati ``translate_slang`` di layer notebook/loader
    supaya kosakata toksik jargon Dota (noob/trash/ez/kys) ter-ekspansi dan
    paritas dengan inference.
    pos_weights: bobot positif per-label untuk BCEWithLogitsLoss (tangani
        imbalance label — mayoritas pesan non-toksik). None = unweighted.

    Metrik kini termasuk PR-AUC (macro/micro) + MCC macro — threshold-free /
    imbalance-robust, selaras reports/comparison_metrics_extended.md.
    """
    import torch
    from datasets import Dataset
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        hamming_loss,
        matthews_corrcoef,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    class WeightedBCETrainer(Trainer):
        """Trainer dengan BCEWithLogitsLoss + pos_weight per-label."""

        def __init__(self, *args, pos_weight_tensor=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._pw = pos_weight_tensor

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            pw = self._pw.to(logits.device) if self._pw is not None else None
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            loss = loss_fn(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
    )

    max_len = int(hyperparameters.get("max_seq_length", 192))

    def to_dataset(df: pd.DataFrame) -> Dataset:
        labels = df[label_names].astype(float).values  # float untuk BCE
        ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": labels.tolist()})
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_len, padding=False),
            batched=True,
        )

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)

    def compute_metrics(eval_pred: tuple) -> dict[str, float]:
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        labels_i = labels.astype(int)
        # MCC macro: per-label MCC averaged over labels with >=1 positive.
        mccs = []
        for j in range(labels_i.shape[1]):
            if labels_i[:, j].sum() > 0:
                mccs.append(matthews_corrcoef(labels_i[:, j], preds[:, j]))
        mcc_macro = float(np.mean(mccs)) if mccs else 0.0
        # PR-AUC only over labels with positive support (else undefined).
        support = labels_i.sum(axis=0) > 0
        pr_auc_macro = (
            float(average_precision_score(labels_i[:, support], probs[:, support], average="macro"))
            if support.any()
            else 0.0
        )
        return {
            "f1_micro": f1_score(labels_i, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels_i, preds, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(labels_i, preds),
            "mcc_macro": mcc_macro,
            "pr_auc_macro": pr_auc_macro,
            "pr_auc_micro": float(
                average_precision_score(labels_i, probs, average="micro")
            ),
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(hyperparameters.get("num_epochs", 3)),
        per_device_train_batch_size=int(hyperparameters.get("batch_size", 32)),
        per_device_eval_batch_size=int(hyperparameters.get("batch_size", 32)),
        learning_rate=float(hyperparameters.get("learning_rate", 2e-5)),
        weight_decay=float(hyperparameters.get("weight_decay", 0.01)),
        warmup_ratio=float(hyperparameters.get("warmup_ratio", 0.1)),
        gradient_accumulation_steps=int(hyperparameters.get("gradient_accumulation_steps", 1)),
        fp16=hyperparameters.get("precision", "fp16") == "fp16" and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=seed,
        save_total_limit=2,
    )

    pw_tensor = (
        torch.tensor(list(pos_weights), dtype=torch.float)
        if pos_weights is not None
        else None
    )
    trainer = WeightedBCETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # transformers v5+ (was 'tokenizer' di v4)
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        pos_weight_tensor=pw_tensor,
    )
    trainer.train()
    # Skip trainer.evaluate() — bug v5 + Jupyter (lihat sentiment_trainer.py).
    history = list(trainer.state.log_history)
    eval_metrics = {}
    for log in reversed(history):
        if any(k.startswith("eval_") for k in log):
            eval_metrics = {k: v for k, v in log.items() if k.startswith("eval_")}
            break
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return TrainingResult(model_dir=output_dir, eval_metrics=eval_metrics, history=history)
