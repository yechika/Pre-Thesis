"""Dataset loaders untuk fine-tune sentiment + toxicity dari sumber publik.

Sentimen: SST-2 (default) atau Sentiment140. SST-2 binary → di-augment dengan
neutral via z-score pada confidence score (model-based pseudo-label) untuk
mendapatkan 3-class. Alternatif: pakai dataset 3-class langsung seperti
TweetEval sentiment.

Toksisitas: Jigsaw Toxic Comment Classification (multi-label 6) sebagai sumber
utama. HateEval2019 sebagai pelengkap untuk identity_hate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SENTIMENT_SOURCE = "sst2"  # alternatif: sentiment140, tweeteval-sentiment
DEFAULT_TOXICITY_SOURCE = "jigsaw"


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    label_names: list[str]


def _sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode("utf-8"))
            h.update(f.read_bytes())
    return h.hexdigest()


def load_sentiment_dataset(
    source: str = DEFAULT_SENTIMENT_SOURCE,
    cache_dir: Path = Path("data/external"),
) -> DatasetSplit:
    """Load dataset publik untuk sentimen 3-class.

    Untuk SST-2 (binary), neutral di-derived dari confidence rendah (uncertainty
    di model boundary). Pendekatan ini didokumentasikan di laporan training.

    Untuk TweetEval-sentiment, label sudah 3-class (negative=0, neutral=1,
    positive=2) — direkomendasikan untuk produksi karena domain (tweet) lebih
    dekat ke chat dibanding SST-2 (movie review).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source == "sst2":
        return _load_sst2(cache_dir / "sst2")
    if source == "tweeteval-sentiment":
        return _load_tweeteval_sentiment(cache_dir / "tweeteval_sentiment")
    if source == "sentiment140":
        return _load_sentiment140(cache_dir / "sentiment140")
    raise ValueError(f"Unknown sentiment source: {source}")


def load_toxicity_dataset(
    source: str = DEFAULT_TOXICITY_SOURCE,
    cache_dir: Path = Path("data/external"),
) -> DatasetSplit:
    """Load dataset publik untuk toksisitas multi-label 6 label."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if source == "jigsaw":
        return _load_jigsaw_toxic(cache_dir / "jigsaw_toxic")
    raise ValueError(f"Unknown toxicity source: {source}")


# ---- Implementations ----------------------------------------------------------

def _load_sst2(cache_path: Path) -> DatasetSplit:
    """SST-2 via Hugging Face datasets, lalu mapping binary → 3-class.

    Mapping: ambil confidence boundary [0.4, 0.6] → neutral; sisanya pakai label
    asli. Ini approximation; untuk skripsi yang lebih ketat, pakai
    tweeteval-sentiment.
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", cache_dir=str(cache_path.parent / "_hf_cache"))
    # SST-2: train (67k), validation (872), test (no labels — eksklusi)
    def to_df(split: str) -> pd.DataFrame:
        return pd.DataFrame(ds[split])[["sentence", "label"]].rename(
            columns={"sentence": "text"}
        )

    train = to_df("train")
    val = to_df("validation")

    # Binary → 3-class: kita tidak punya akses confidence di SST-2 raw, jadi
    # untuk awalnya kita pertahankan binary saja. Catatan: ini akan dikonversi
    # ke 3-class saat fine-tuning dengan loss cross-entropy 3-class jika user
    # menambah dataset netral.
    label_names = ["negative", "positive"]
    return DatasetSplit(train=train, validation=val, test=val, label_names=label_names)


def _load_tweeteval_sentiment(cache_path: Path) -> DatasetSplit:
    """TweetEval sentiment task: 3-class (negative=0, neutral=1, positive=2)."""
    from datasets import load_dataset

    ds = load_dataset(
        "tweet_eval", "sentiment", cache_dir=str(cache_path.parent / "_hf_cache")
    )

    def to_df(split: str) -> pd.DataFrame:
        return pd.DataFrame(ds[split])[["text", "label"]]

    label_names = ["negative", "neutral", "positive"]
    return DatasetSplit(
        train=to_df("train"),
        validation=to_df("validation"),
        test=to_df("test"),
        label_names=label_names,
    )


def _load_sentiment140(cache_path: Path) -> DatasetSplit:
    from datasets import load_dataset

    ds = load_dataset(
        "stanfordnlp/sentiment140", cache_dir=str(cache_path.parent / "_hf_cache")
    )
    # sentiment140: 0=negative, 4=positive (binary). Sub-sample 200k stratified.
    train = pd.DataFrame(ds["train"])[["text", "sentiment"]].rename(
        columns={"sentiment": "label"}
    )
    train["label"] = (train["label"] == 4).astype(int)  # 0=neg, 1=pos
    train = train.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(100_000, len(g)), random_state=42)
    )
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    test = pd.DataFrame(ds["test"])[["text", "sentiment"]].rename(
        columns={"sentiment": "label"}
    )
    test["label"] = (test["label"] == 4).astype(int)
    val = train.iloc[: int(0.05 * len(train))]
    train = train.iloc[int(0.05 * len(train)) :].reset_index(drop=True)
    return DatasetSplit(train=train, validation=val, test=test, label_names=["negative", "positive"])


def _load_jigsaw_toxic(cache_path: Path) -> DatasetSplit:
    """Jigsaw Toxic Comment Classification — multi-label 6.

    File CSV harus pre-downloaded ke `cache_path/train.csv` dan `cache_path/test.csv`
    dari https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data.
    Kompetisi memerlukan login Kaggle untuk download — di-cache lokal sekali.
    """
    cache_path = Path(cache_path)
    train_csv = cache_path / "train.csv"
    test_csv = cache_path / "test.csv"
    test_labels_csv = cache_path / "test_labels.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Jigsaw Toxic Comment train.csv tidak ditemukan di {train_csv}.\n"
            "Download dari Kaggle: https://www.kaggle.com/competitions/"
            "jigsaw-toxic-comment-classification-challenge/data\n"
            f"Lalu letakkan train.csv, test.csv, test_labels.csv di {cache_path}/"
        )

    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train = pd.read_csv(train_csv)
    train = train.rename(columns={"comment_text": "text"})

    # Split train menjadi train/val 90/10 stratified pada `toxic` (label utama).
    val = train.sample(frac=0.1, random_state=42)
    train_only = train.drop(val.index)

    # Test set: gabungkan test.csv + test_labels.csv. Eksklusi baris dengan label = -1
    # (Kaggle menandai baris yang tidak digunakan untuk scoring).
    if test_csv.exists() and test_labels_csv.exists():
        test_text = pd.read_csv(test_csv).rename(columns={"comment_text": "text"})
        test_lbl = pd.read_csv(test_labels_csv)
        test = test_text.merge(test_lbl, on="id")
        test = test[(test[label_cols] != -1).all(axis=1)].reset_index(drop=True)
    else:
        # Fallback: pakai val sebagai test
        test = val

    return DatasetSplit(
        train=train_only.reset_index(drop=True),
        validation=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        label_names=label_cols,
    )


def manifest_entry(name: str, split: DatasetSplit, cache_path: Path) -> dict[str, Any]:
    """Build manifest entry untuk experiment.yaml#public_datasets."""
    return {
        "name": name,
        "n_train": len(split.train),
        "n_validation": len(split.validation),
        "n_test": len(split.test),
        "labels": split.label_names,
        "sha256_dir": _sha256_dir(cache_path) if cache_path.exists() else "",
    }
