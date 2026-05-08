"""Language detection + short-circuit allowlist untuk pesan pendek alfanumerik."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

ASCII_ALNUM_RE = re.compile(r"^[A-Za-z0-9 \t._\-!?']+$")


@dataclass
class LangFilterStats:
    n_in: int
    n_short_circuit_en: int
    n_detected_en: int
    n_non_en: int


def _short_circuit_is_english(text: str, max_tokens: int) -> bool:
    if not text:
        return False
    tokens = text.split()
    if len(tokens) > max_tokens:
        return False
    return bool(ASCII_ALNUM_RE.match(text))


def _detect_lang_fasttext(texts: Iterable[str]) -> list[str]:
    """Lazy import — fasttext-langdetect (ftlangdetect).

    Jika `ftlangdetect` tidak terinstal (mis. fasttext gagal compile di Windows),
    fallback ke `langdetect` (pure Python) dengan log peringatan satu kali.
    """
    try:
        from ftlangdetect import detect  # type: ignore
    except ImportError:
        import warnings
        warnings.warn(
            "ftlangdetect tidak terinstal — fallback ke langdetect (pure Python). "
            "Untuk akurasi lebih baik di teks pendek, install: "
            "`pip install fasttext-langdetect` (butuh C++ compiler di Windows).",
            RuntimeWarning,
            stacklevel=2,
        )
        return _detect_lang_langdetect(texts)

    out: list[str] = []
    for t in texts:
        if not t:
            out.append("und")
            continue
        try:
            # ftlangdetect tidak suka newline — strip dulu untuk hindari ValueError
            t_clean = t.replace("\n", " ").replace("\r", " ").strip()
            if not t_clean:
                out.append("und")
                continue
            res = detect(text=t_clean, low_memory=True)
            out.append(res.get("lang", "und"))
        except Exception:
            out.append("und")
    return out


def _detect_lang_langdetect(texts: Iterable[str]) -> list[str]:
    try:
        from langdetect import DetectorFactory, detect  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Tidak ada language detector terinstal. Install salah satu:\n"
            "  1) langdetect (pure Python, recommended): `pip install langdetect`\n"
            "  2) ftlangdetect (akurat tapi butuh C++ compiler): `pip install fasttext-langdetect`\n"
            "Lalu re-run sel ini."
        ) from exc

    DetectorFactory.seed = 0
    out: list[str] = []
    for t in texts:
        if not t:
            out.append("und")
            continue
        try:
            out.append(detect(t))
        except Exception:
            out.append("und")
    return out


def detect_language(
    texts: pd.Series,
    detector: str = "fasttext",
) -> pd.Series:
    if detector == "fasttext":
        langs = _detect_lang_fasttext(texts.fillna("").tolist())
    elif detector == "langdetect":
        langs = _detect_lang_langdetect(texts.fillna("").tolist())
    else:
        raise ValueError(f"Unknown detector: {detector}")
    return pd.Series(langs, index=texts.index, dtype="string")


def filter_english(
    df: pd.DataFrame,
    text_col: str = "key",
    detector: str = "fasttext",
    short_circuit_max_tokens: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, LangFilterStats]:
    """Pisahkan dataframe menjadi (df_en, df_non_en, stats).

    Pesan ≤ short_circuit_max_tokens token alfanumerik di-anggap EN tanpa
    memanggil detector. Sisanya dilewatkan ke detector.
    """
    n_in = len(df)
    texts = df[text_col].fillna("").astype(str)

    short_mask = texts.map(lambda t: _short_circuit_is_english(t, short_circuit_max_tokens))
    n_short = int(short_mask.sum())

    long_idx = df.index[~short_mask]
    detected = pd.Series("en", index=df.index, dtype="string")
    if len(long_idx) > 0:
        detected.loc[long_idx] = detect_language(texts.loc[long_idx], detector=detector)

    df = df.copy()
    df["lang"] = detected
    df["lang_source"] = pd.Series("detector", index=df.index, dtype="string")
    df.loc[short_mask, "lang_source"] = "short_circuit"

    en_mask = df["lang"] == "en"
    df_en = df.loc[en_mask].copy()
    df_non_en = df.loc[~en_mask].copy()

    stats = LangFilterStats(
        n_in=n_in,
        n_short_circuit_en=n_short,
        n_detected_en=int(en_mask.sum()) - n_short,
        n_non_en=int((~en_mask).sum()),
    )
    return df_en, df_non_en, stats
