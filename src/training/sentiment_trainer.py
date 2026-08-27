"""Fine-tune transformer untuk sentimen 3-class (atau 2-class fallback)."""

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


def fine_tune_sentiment(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_names: list[str],
    output_dir: Path,
    hyperparameters: dict[str, Any],
    seed: int = 42,
    class_weights: list[float] | None = None,
) -> TrainingResult:
    """Fine-tune satu transformer untuk sentimen sequence classification.

    train_df, val_df: kolom `text`, `label` (int). Text diasumsikan SUDAH
    melewati ``translate_slang`` (lihat src/training/gold_loaders.py) supaya
    train/inference paritas.
    label_names: contoh ['negative', 'neutral', 'positive'] atau ['negative', 'positive'].
    class_weights: bobot per-kelas untuk weighted cross-entropy (tangani
        imbalance, mis. kelas 'negative' yang langka). None = unweighted.

    Metrik yang dilaporkan kini termasuk balanced_accuracy + MCC — keduanya
    imbalance-robust dan dipakai di reports/comparison_metrics_extended.md.
    """
    import torch
    from datasets import Dataset
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    class WeightedTrainer(Trainer):
        """Trainer dengan weighted cross-entropy untuk imbalance kelas."""

        def __init__(self, *args, class_weight_tensor=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._cw = class_weight_tensor

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = (
                self._cw.to(logits.device) if self._cw is not None else None
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
    )

    max_len = int(hyperparameters.get("max_seq_length", 128))

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)

    def compute_metrics(eval_pred: tuple) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "mcc": matthews_corrcoef(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=seed,
        save_total_limit=2,
    )

    cw_tensor = (
        torch.tensor(list(class_weights), dtype=torch.float)
        if class_weights is not None
        else None
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # transformers v5+ (was 'tokenizer' di v4)
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weight_tensor=cw_tensor,
    )
    trainer.train()
    # Skip trainer.evaluate() — bug di transformers v5 + Jupyter
    # (NotebookProgressCallback raise "on_train_begin must be called before on_evaluate").
    # Eval metrics sudah ada di state.log_history (eval_strategy="epoch") — ambil dari sana.
    history = list(trainer.state.log_history)
    eval_metrics = {}
    for log in reversed(history):
        if any(k.startswith("eval_") for k in log):
            eval_metrics = {k: v for k, v in log.items() if k.startswith("eval_")}
            break
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return TrainingResult(model_dir=output_dir, eval_metrics=eval_metrics, history=history)
