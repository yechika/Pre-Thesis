"""Cohen's kappa pair-wise + Krippendorff's alpha multi-label + train/test split."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

KEY_COLS = ["match_id", "time", "player_slot"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def _normalize_annotations(df: pd.DataFrame, toxicity_labels: list[str]) -> pd.DataFrame:
    """Normalize tipe + sanitize value: sentiment lowercase string, tox_* sebagai int 0/1."""
    out = df.copy()
    out["sentiment"] = out["sentiment"].fillna("").astype(str).str.lower().str.strip()
    for lbl in toxicity_labels:
        col = f"tox_{lbl}"
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    for col in ("is_dota_jargon", "is_ambiguous"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return out


def load_annotations(
    raw_root: Path,
    toxicity_labels: list[str],
) -> dict[str, pd.DataFrame]:
    """Baca raw_annotations/sample_*.csv → dict[annotator_id → df]."""
    raw_root = Path(raw_root)
    files = sorted(raw_root.glob("sample_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada sample_*.csv di {raw_root}. Anotator export ke sini setelah selesai."
        )
    out: dict[str, pd.DataFrame] = {}
    for f in files:
        ann_id = f.stem.replace("sample_", "")
        df = pd.read_csv(f)
        df = _normalize_annotations(df, toxicity_labels)
        out[ann_id] = df
    return out


def _align_on_keys(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ambil baris yang KEY_COLS-nya cocok di kedua df, urut sama."""
    keys_a = a[KEY_COLS].apply(tuple, axis=1)
    keys_b = b[KEY_COLS].apply(tuple, axis=1)
    common = sorted(set(keys_a) & set(keys_b))
    a2 = a.set_index(KEY_COLS).loc[common].reset_index()
    b2 = b.set_index(KEY_COLS).loc[common].reset_index()
    return a2, b2


@dataclass
class AgreementReport:
    pairwise_kappa_sentiment: dict[tuple[str, str], float]
    overall_alpha_toxicity: float
    n_pairs: int
    n_overlap: int


def cohen_kappa_pairwise(annotations: dict[str, pd.DataFrame]) -> dict[tuple[str, str], float]:
    """Hitung Cohen's kappa pair-wise untuk kolom `sentiment`."""
    out: dict[tuple[str, str], float] = {}
    for a, b in combinations(sorted(annotations.keys()), 2):
        df_a, df_b = _align_on_keys(annotations[a], annotations[b])
        if len(df_a) == 0:
            out[(a, b)] = float("nan")
            continue
        k = cohen_kappa_score(df_a["sentiment"], df_b["sentiment"], labels=SENTIMENT_LABELS)
        out[(a, b)] = float(k)
    return out


def krippendorff_alpha_binary(values: np.ndarray) -> float:
    """Krippendorff's alpha untuk data biner (rater × item) dengan nilai 0/1.

    Implementasi sederhana berbasis distance metric nominal. NaN diabaikan.

    values: array shape (n_raters, n_items), nilai 0/1 atau np.nan.
    """
    values = np.asarray(values, dtype=float)
    n_raters, n_items = values.shape

    # Coincidence count (skema Krippendorff): untuk tiap pasang rater × item dengan
    # kedua nilai tidak NaN, hitung kontribusi ke matriks coincidence.
    coincidence = np.zeros((2, 2), dtype=float)
    for j in range(n_items):
        col = values[:, j]
        valid = col[~np.isnan(col)]
        m = len(valid)
        if m < 2:
            continue
        # Tiap pasang ordered (i, k) dengan i != k berkontribusi 1/(m-1).
        for v1 in valid:
            for v2 in valid:
                if v1 == v2:
                    coincidence[int(v1), int(v2)] += 1.0 / (m - 1)
                else:
                    coincidence[int(v1), int(v2)] += 1.0 / (m - 1)

    n_total = coincidence.sum()
    if n_total == 0:
        return float("nan")
    # Marginal proportions
    marg = coincidence.sum(axis=1) / n_total

    # Observed disagreement
    do = (coincidence[0, 1] + coincidence[1, 0]) / n_total
    # Expected disagreement (nominal distance: 0 if same, 1 if diff).
    de = 2 * marg[0] * marg[1] * (n_total / max(1, n_total - 1)) if n_total > 1 else 0.0
    if de == 0:
        return float("nan")
    return float(1 - do / de)


def krippendorff_alpha_multilabel(
    annotations: dict[str, pd.DataFrame], toxicity_labels: list[str]
) -> tuple[float, dict[str, float]]:
    """Hitung alpha untuk setiap label toxicity + alpha agregat (rata-rata).

    Asumsi: annotations sudah dialign (semua anotator anotasi item yang sama).
    Untuk item dengan partial-overlap, kita pakai NaN untuk anotator yang tidak ada.
    """
    # Build union of keys across all annotators.
    all_keys: set[tuple] = set()
    for df in annotations.values():
        for k in df[KEY_COLS].apply(tuple, axis=1):
            all_keys.add(k)
    keys = sorted(all_keys)
    annotators = sorted(annotations.keys())

    per_label: dict[str, float] = {}
    for lbl in toxicity_labels:
        col = f"tox_{lbl}"
        # values shape: (n_annotators, n_items)
        mat = np.full((len(annotators), len(keys)), np.nan, dtype=float)
        for i, ann in enumerate(annotators):
            df = annotations[ann]
            if col not in df.columns:
                continue
            df_idx = df.set_index(KEY_COLS)
            for j, key in enumerate(keys):
                if key in df_idx.index:
                    val = df_idx.loc[key, col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    if pd.notna(val):
                        mat[i, j] = float(val)
        per_label[lbl] = krippendorff_alpha_binary(mat)
    overall = float(np.nanmean(list(per_label.values()))) if per_label else float("nan")
    return overall, per_label


# ---- Konsensus ---------------------------------------------------------------

def majority_vote_sentiment(values: list[str]) -> tuple[str, str]:
    """Return (label_konsensus, confidence). values: list label dari anotator."""
    cnt = pd.Series(values).value_counts()
    if len(cnt) == 0:
        return "", "missing"
    top = cnt.iloc[0]
    if top == len(values):
        return cnt.index[0], "unanimous"
    if top > len(values) / 2:
        return cnt.index[0], "majority"
    # Tied
    return cnt.index[0], "adjudicated"


def majority_vote_binary(values: list[int]) -> tuple[int, str]:
    cnt = pd.Series(values).value_counts()
    if len(cnt) == 0:
        return 0, "missing"
    top = cnt.iloc[0]
    if top == len(values):
        return int(cnt.index[0]), "unanimous"
    if top > len(values) / 2:
        return int(cnt.index[0]), "majority"
    return int(cnt.index[0]), "adjudicated"


def consolidate(
    annotations: dict[str, pd.DataFrame],
    toxicity_labels: list[str],
) -> pd.DataFrame:
    """Gabung anotasi dari beberapa anotator → 1 baris per pesan, label konsensus."""
    all_keys: set[tuple] = set()
    for df in annotations.values():
        for k in df[KEY_COLS].apply(tuple, axis=1):
            all_keys.add(k)
    keys = sorted(all_keys)
    annotators = sorted(annotations.keys())

    # Reference baris untuk metadata pesan (ambil dari anotator pertama yang punya).
    rows = []
    for key in keys:
        # Cari ref df
        ref_row = None
        for ann in annotators:
            df_idx = annotations[ann].set_index(KEY_COLS)
            if key in df_idx.index:
                row = df_idx.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                ref_row = row
                break
        if ref_row is None:
            continue

        # Sentiment vote
        sent_values = []
        for ann in annotators:
            df_idx = annotations[ann].set_index(KEY_COLS)
            if key in df_idx.index:
                v = df_idx.loc[key, "sentiment"]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                if isinstance(v, str) and v in SENTIMENT_LABELS:
                    sent_values.append(v)
        sent_label, sent_conf = majority_vote_sentiment(sent_values)

        # Tox votes per-label
        tox_labels: dict[str, int] = {}
        tox_confs: dict[str, str] = {}
        for lbl in toxicity_labels:
            col = f"tox_{lbl}"
            vals = []
            for ann in annotators:
                df_idx = annotations[ann].set_index(KEY_COLS)
                if key in df_idx.index and col in annotations[ann].columns:
                    v = df_idx.loc[key, col]
                    if isinstance(v, pd.Series):
                        v = v.iloc[0]
                    if pd.notna(v):
                        vals.append(int(v))
            label, conf = majority_vote_binary(vals)
            tox_labels[col] = label
            tox_confs[f"conf_{col}"] = conf

        out_row = {
            "match_id": key[0],
            "time": key[1],
            "player_slot": key[2],
            "key": ref_row.get("key", ""),
            "sentiment": sent_label,
            "confidence_sentiment": sent_conf,
            **tox_labels,
            **tox_confs,
            "n_annotators": len(sent_values),
        }
        rows.append(out_row)
    return pd.DataFrame(rows)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split berdasarkan kolom `sentiment` (≥ 2 anggota tiap kelas)."""
    if "sentiment" not in df.columns:
        raise KeyError("Kolom 'sentiment' tidak ada di df.")
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["sentiment"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
