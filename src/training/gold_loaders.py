"""In-domain loaders for the Dota 2 gold-standard set + imbalance helpers.

Why in-domain
-------------
The original pipeline fine-tuned sentiment on *external* corpora (TweetEval /
SST-2) and only evaluated on the Dota gold set. Tweet/movie "positive vs
negative" polarity is anti-correlated with Dota chat labelling (``gg`` =
positive sportsmanship, ``g`` = neutral call-out), which is exactly why the
external-trained sentiment model scored a *negative* MCC (worse than chance).

Fixing it means training **on the gold set itself** (`data/gold/train.csv`),
with the same ``translate_slang`` preprocessing used at inference time.

Toxicity is handled differently — see ``load_gold_toxicity``: the gold set has
0 positive examples for ``severe_toxic`` and ``threat`` (and only 2 for
``identity_hate``), so it cannot supervise a 6-label classifier. Toxicity stays
trained on Jigsaw (external, plenty of positives) + slang bridging; the gold
toxicity loader here is for *evaluation* and optional positive-example
augmentation only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import translate_slang
from src.training.dataset_loaders import DatasetSplit

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
TOXICITY_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
TEXT_SOURCE_COL = "key"  # the raw chat message lives in `key`


# ---------------------------------------------------------------------------
# Sentiment — in-domain, 3-class
# ---------------------------------------------------------------------------
def _prep_text(series: pd.Series, apply_slang: bool) -> pd.Series:
    text = series.fillna("").astype(str)
    if apply_slang:
        text = text.map(translate_slang)
    return text.str.strip()


def load_gold_sentiment(
    gold_root: str | Path = "data/gold",
    apply_slang: bool = True,
    val_frac: float = 0.1,
    seed: int = 42,
    label_names: list[str] | None = None,
) -> DatasetSplit:
    """Load the gold set as a 3-class sentiment ``DatasetSplit``.

    Returns frames with columns ``text`` (slang-translated) and ``label`` (int).
    ``train.csv`` is split into train/validation (stratified on label);
    ``test.csv`` is the held-out gold-test used by notebook 06.
    """
    label_names = label_names or SENTIMENT_LABELS
    gold_root = Path(gold_root)
    label2id = {n: i for i, n in enumerate(label_names)}

    def prep(csv_name: str) -> pd.DataFrame:
        df = pd.read_csv(gold_root / csv_name)
        out = pd.DataFrame(
            {
                "text": _prep_text(df[TEXT_SOURCE_COL], apply_slang),
                "label": df["sentiment"].map(label2id),
            }
        )
        out = out.dropna(subset=["label"])
        out = out[out["text"] != ""]
        out["label"] = out["label"].astype(int)
        return out.reset_index(drop=True)

    full_train = prep("train.csv")
    test = prep("test.csv")

    # Stratified validation split from the train portion.
    val = full_train.groupby("label", group_keys=False).sample(
        frac=val_frac, random_state=seed
    )
    train = full_train.drop(val.index).reset_index(drop=True)
    val = val.reset_index(drop=True)

    return DatasetSplit(
        train=train, validation=val, test=test, label_names=label_names
    )


# ---------------------------------------------------------------------------
# Toxicity — in-domain (eval / augmentation only)
# ---------------------------------------------------------------------------
def load_gold_toxicity(
    gold_root: str | Path = "data/gold",
    apply_slang: bool = True,
    label_cols: list[str] | None = None,
) -> DatasetSplit:
    """Load the gold set as a multi-label toxicity ``DatasetSplit``.

    NOTE: intended for evaluation and as a *source of positive examples* to mix
    into Jigsaw training — NOT as a standalone training set (severe_toxic and
    threat have zero positives in gold).
    """
    label_cols = label_cols or TOXICITY_LABELS
    gold_root = Path(gold_root)
    src_to_label = {f"tox_{c}": c for c in label_cols}

    def prep(csv_name: str) -> pd.DataFrame:
        df = pd.read_csv(gold_root / csv_name)
        out = pd.DataFrame({"text": _prep_text(df[TEXT_SOURCE_COL], apply_slang)})
        for src, lbl in src_to_label.items():
            out[lbl] = df[src].fillna(0).astype(float).clip(0, 1)
        out = out[out["text"] != ""].reset_index(drop=True)
        return out

    full_train = prep("train.csv")
    test = prep("test.csv")
    # No internal val split needed here (toxicity trains on Jigsaw); expose train
    # as both train+validation for API symmetry.
    return DatasetSplit(
        train=full_train, validation=full_train, test=test, label_names=label_cols
    )


def gold_toxicity_positives(
    gold_root: str | Path = "data/gold",
    apply_slang: bool = True,
    label_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return only gold rows with >=1 positive toxicity label.

    Use to augment Jigsaw training with in-domain toxic vocabulary
    (e.g. ``noob``, ``trash``, ``ez``, ``kys`` after slang translation).
    """
    label_cols = label_cols or TOXICITY_LABELS
    split = load_gold_toxicity(gold_root, apply_slang=apply_slang, label_cols=label_cols)
    df = split.train
    mask = df[label_cols].sum(axis=1) > 0
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Imbalance helpers
# ---------------------------------------------------------------------------
def oversample_minority(
    df: pd.DataFrame,
    label_col: str = "label",
    target_frac: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Random oversample minority classes up to ``target_frac`` x majority size.

    Default 0.30 keeps duplication moderate (avoids over-fitting on the 71
    `negative` examples) while giving the optimizer enough minority signal.
    Pair with class weights for residual imbalance. Returns a shuffled frame.
    """
    counts = df[label_col].value_counts()
    n_max = int(counts.max())
    target = int(round(target_frac * n_max))
    parts = [df]
    for lbl, c in counts.items():
        if c < target:
            extra = df[df[label_col] == lbl].sample(
                target - int(c), replace=True, random_state=seed
            )
            parts.append(extra)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def compute_class_weights(
    labels, n_classes: int, scheme: str = "balanced"
) -> np.ndarray:
    """Class weights for a weighted cross-entropy.

    scheme:
      * ``balanced`` — sklearn style: n / (k * count_c).
      * ``sqrt``     — softer: sqrt of balanced (less aggressive on rare class).
    Zero-count classes are floored to weight 1.0.
    """
    labels = np.asarray(list(labels))
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    balanced = counts.sum() / (n_classes * counts)
    if scheme == "sqrt":
        return np.sqrt(balanced)
    return balanced


def compute_pos_weights(
    df: pd.DataFrame, label_cols: list[str], cap: float = 50.0
) -> np.ndarray:
    """Per-label ``pos_weight`` for BCEWithLogitsLoss = n_neg / n_pos, capped.

    Cap prevents explosion for ultra-rare labels. Labels with 0 positives get
    the cap (they still cannot be learned — reported as no-support downstream).
    """
    pw = []
    n = len(df)
    for c in label_cols:
        pos = float(df[c].sum())
        neg = n - pos
        pw.append(min(cap, neg / pos) if pos > 0 else cap)
    return np.asarray(pw, dtype=np.float32)
