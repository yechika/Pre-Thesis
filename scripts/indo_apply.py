"""Apply Indonesian labels: borderline (7) + high (5) + medium (58)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---- Borderline confirmed ----
BORDERLINE_LABELS = {
    "mati aja": dict(
        sentiment="negative", tox_toxic=0, tox_obscene=0, tox_insult=0,
        is_jargon=0, is_ambig=1,
        notes="(Indo) defeatism rage 'mati aja', tone self-vent, tidak diarahkan eksplisit ke target",
    ),
    "gue muted lo pada": dict(
        sentiment="negative", tox_toxic=1, tox_obscene=0, tox_insult=0,
        is_jargon=0, is_ambig=1,
        notes="(Indo) konfrontasional ke teammate ('saya mute kalian semua'), mild toxic",
    ),
    "kami jahat": dict(
        sentiment="positive", tox_toxic=0, tox_obscene=0, tox_insult=0,
        is_jargon=0, is_ambig=1,
        notes="(Indo) joking self-description dari pemenang, banter ringan",
    ),
    "sampe tantgan lu bisulan": dict(
        sentiment="negative", tox_toxic=0, tox_obscene=0, tox_insult=0,
        is_jargon=0, is_ambig=1,
        notes="(Indo) playful curse 'sampai tanganmu bisulan', tone banter (tidak murni toxic)",
    ),
    "fun sma bodoh beda bro": dict(
        sentiment="negative", tox_toxic=1, tox_obscene=0, tox_insult=1,
        is_jargon=0, is_ambig=1,
        notes="(Indo) menggunakan 'bodoh' (idiot) untuk kontras dengan 'fun', mild insult",
    ),
    "udah muncrat": dict(
        sentiment="negative", tox_toxic=0, tox_obscene=1, tox_insult=0,
        is_jargon=0, is_ambig=1,
        notes="(Indo) vulgar slang 'muncrat', kemungkinan obscene tapi bisa juga descriptive",
    ),
    "kamu ngmng doang": dict(
        sentiment="negative", tox_toxic=1, tox_obscene=0, tox_insult=1,
        is_jargon=0, is_ambig=1,
        notes="(Indo) dismissive 'kamu ngomong doang', mild insult ke seseorang",
    ),
}

# ---- HIGH confidence labels ----
HIGH_LABELS = {
    "marahan ya bang wkwk": dict(
        sentiment="positive", is_ambig=1,
        notes="(Indo) banter dengan tawa wkwk, playful",
    ),
    "ketemu lagi wkwk": dict(
        sentiment="positive", is_ambig=1,
        notes="(Indo) reuni player + tawa wkwk, positif/banter",
    ),
    "ORACLE GW CUPU": dict(
        sentiment="negative", is_ambig=1,
        notes="(Indo) self-deprecating: 'oracle saya cupu (lemah)'",
    ),
    "parah tdi maen ama gua": dict(
        sentiment="negative", is_ambig=1,
        notes="(Indo) 'parah tadi main sama saya', mengakui buruk-nya permainan",
    ),
    "okee gue maghrib dl": dict(
        sentiment="neutral", is_ambig=0,
        notes="(Indo) AFK announcement untuk sholat maghrib",
    ),
}


def make_full_label(sentiment, tox_toxic=0, tox_severe=0, tox_obscene=0,
                    tox_threat=0, tox_insult=0, tox_identity=0,
                    is_jargon=0, is_ambig=0, notes=""):
    return {
        "annotator_id": "AI",
        "sentiment": sentiment,
        "tox_toxic": tox_toxic, "tox_severe_toxic": tox_severe,
        "tox_obscene": tox_obscene, "tox_threat": tox_threat,
        "tox_insult": tox_insult, "tox_identity_hate": tox_identity,
        "is_dota_jargon": is_jargon, "is_ambiguous": is_ambig, "notes": notes,
    }


def main():
    proposal_path = Path("data/gold/_indo_proposal.csv")
    sample_path = Path("data/gold/sample.csv")
    pdf = pd.read_csv(proposal_path)
    df = pd.read_csv(sample_path)

    # Pastikan dtype object supaya tidak warning
    for col in ["annotator_id", "sentiment", "notes"]:
        df[col] = df[col].astype("object")
    for col in ["tox_toxic", "tox_severe_toxic", "tox_obscene", "tox_threat",
                "tox_insult", "tox_identity_hate", "is_dota_jargon", "is_ambiguous"]:
        df[col] = df[col].astype("object")

    n_applied = {"borderline": 0, "high": 0, "medium": 0}

    for _, p in pdf.iterrows():
        idx = int(p["index"])
        key = p["key"]
        # Skip kalau sudah dilabel
        if df.at[idx, "annotator_id"] == "AI":
            continue

        if p["confidence"] == "ASK":
            # Borderline
            if key in BORDERLINE_LABELS:
                lbl = make_full_label(**BORDERLINE_LABELS[key])
                for k, v in lbl.items():
                    df.at[idx, k] = v
                n_applied["borderline"] += 1
            else:
                # Fallback: neutral ambig
                lbl = make_full_label(
                    "neutral", is_ambig=1,
                    notes=f"(Indo) borderline tidak terdefinisi default: {key}",
                )
                for k, v in lbl.items():
                    df.at[idx, k] = v
                n_applied["borderline"] += 1

        elif p["confidence"] == "high":
            if key in HIGH_LABELS:
                lbl = make_full_label(**HIGH_LABELS[key])
                for k, v in lbl.items():
                    df.at[idx, k] = v
                n_applied["high"] += 1

        elif p["confidence"] == "medium":
            # Default: neutral ambig=1 banter
            sent = p["sentiment"] if p["sentiment"] in ("positive", "negative", "neutral") else "neutral"
            lbl = make_full_label(
                sent, is_ambig=1,
                notes=f"(Indo) {p['reason']}: \"{key}\"",
            )
            for k, v in lbl.items():
                df.at[idx, k] = v
            n_applied["medium"] += 1

    df.to_csv(sample_path, index=False, encoding="utf-8")

    print(f"Applied:")
    for k, n in n_applied.items():
        print(f"  {k}: {n}")
    total = sum(n_applied.values())
    print(f"  TOTAL: {total}")

    cumul = (df["annotator_id"] == "AI").sum()
    print(f"\nCumulative dilabel: {cumul:,}/{len(df):,} ({100*cumul/len(df):.1f}%)")
    print(f"Sisa unlabeled: {len(df) - cumul:,}")

    print("\n=== Distribusi sentiment terbaru ===")
    print(df.loc[df["annotator_id"] == "AI", "sentiment"].value_counts())

    print("\n=== Tox flags ===")
    for tcol in ["tox_toxic", "tox_obscene", "tox_insult"]:
        n = (df[tcol] == 1).sum()
        print(f"  {tcol}: {n}")


if __name__ == "__main__":
    main()
