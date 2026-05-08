"""Metrik sentimen 3-class + toksisitas multi-label."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class SentimentMetrics:
    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]
    precision_macro: float
    recall_macro: float
    roc_auc_ovr: float | None
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToxicityMetrics:
    f1_micro: float
    f1_macro: float
    f1_per_label: dict[str, float]
    hamming_loss: float
    subset_accuracy: float
    avg_precision_per_label: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_sentiment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    y_score: np.ndarray | None = None,
) -> SentimentMetrics:
    """y_true, y_pred: array of int label indices. y_score: prob (n × n_classes) opsional."""
    f1_per_class_arr = f1_score(y_true, y_pred, labels=range(len(label_names)), average=None, zero_division=0)
    f1_per_class = {label_names[i]: float(f1_per_class_arr[i]) for i in range(len(label_names))}

    roc_ovr: float | None = None
    if y_score is not None and len(set(y_true)) >= 2:
        try:
            roc_ovr = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except ValueError:
            roc_ovr = None

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names))).tolist()
    return SentimentMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_per_class=f1_per_class,
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        roc_auc_ovr=roc_ovr,
        confusion=cm,
    )


def compute_toxicity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    y_score: np.ndarray | None = None,
) -> ToxicityMetrics:
    """y_true, y_pred, y_score: shape (n_samples, n_labels), values 0/1 atau prob."""
    f1_per_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = {label_names[i]: float(f1_per_arr[i]) for i in range(len(label_names))}

    if y_score is not None:
        ap_arr = []
        for i in range(len(label_names)):
            try:
                ap_arr.append(float(average_precision_score(y_true[:, i], y_score[:, i])))
            except ValueError:
                ap_arr.append(float("nan"))
    else:
        ap_arr = [float("nan")] * len(label_names)
    ap_per = {label_names[i]: ap_arr[i] for i in range(len(label_names))}

    return ToxicityMetrics(
        f1_micro=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_per_label=f1_per,
        hamming_loss=float(hamming_loss(y_true, y_pred)),
        subset_accuracy=float((np.all(y_true == y_pred, axis=1)).mean()),
        avg_precision_per_label=ap_per,
    )
