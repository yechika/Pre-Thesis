"""Propose labels untuk Indonesian unlabeled rows + flag borderline."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INDO_FUNCTION_WORDS = {
    "yang", "sama", "sm", "dgn", "utk", "untuk", "dari", "dr", "ke", "di",
    "saya", "aku", "kamu", "anda", "kita", "kami", "mereka", "dia",
    "lu", "lo", "elu", "gw", "gue", "gua", "gwa",
    "tidak", "ngga", "nggak", "gak", "ga", "kagak", "engga", "enggak", "gk", "tak",
    "sih", "dong", "deh", "kok", "kan", "aja", "doang", "lagi", "lg",
    "masih", "msh", "terus", "trs", "udah", "udh", "dah", "belum", "blm", "blom",
    "bgt", "banget", "bisa", "bs", "mau", "pengen", "pgn", "kalo", "kalau", "klo",
    "ada", "gada", "punya", "jadi", "jd", "biar",
    "kenapa", "knp", "gimana", "gmn", "apa", "apaan", "siapa", "mana",
    "main", "maen", "menang", "mng", "kalah", "klh", "lawan", "lwn",
    "parah", "mantap", "mantul", "mantab", "cupu", "gabut",
    "anjr", "njir", "lah", "kek",
    "gas", "gass", "mabar",
    "bro", "brok", "bg", "gan", "om", "mas", "mbak", "kak", "cuy", "beng", "bang",
    "sabar", "tenang", "santai",
    "maaf", "mf", "punten", "makasih", "mksh", "mksih",
    "salah", "gausah", "percuma",
}
INDO_STRONG = {
    "wkwk", "wkwkwk", "mantap", "cupu", "gabut", "mabar", "gass",
    "lu", "gw", "gue", "gua", "lo", "sih", "dong", "deh", "kok",
    "banget", "bgt", "aja", "doang", "kalo", "klo", "nggak", "ngga", "gak",
    "udah", "udh", "dah", "belum", "blm", "jangan", "jgn",
    "kenapa", "knp", "gimana", "gmn", "gausah", "percuma",
    "bro", "gan", "om", "mas", "kak", "cuy", "beng",
    "mksh", "makasih", "maaf", "punten", "parah",
}


def detect_indo(text):
    if not isinstance(text, str):
        return False
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return False
    if any(t in INDO_STRONG for t in tokens):
        return True
    indo_count = sum(1 for t in tokens if t in INDO_FUNCTION_WORDS)
    return indo_count / len(tokens) >= 0.30 and len(tokens) >= 2


def propose_label(s):
    if not isinstance(s, str):
        return ("neutral", "low", "tidak ada teks")
    n = s.lower()

    if re.search(r"\bwkwk", n):
        return ("positive", "high", "mengandung tawa wkwk")

    NEG_KEYS = ["cupu", "parah", "gabut", "percuma", "gausah"]
    if any(k in n for k in NEG_KEYS):
        return ("negative", "high", "kata negatif Indonesia (cupu/parah/dst)")

    BORDER_TOXIC = [
        "mati aja", "mati lu", "gue muted", "gua muted",
        "bisulan", "ngmng doang", "ngomong doang", "ngomong aja",
        "kamu ngmng", "kamu doang", "muncrat", "jahat", "goblog", "bodoh",
        "jelek", "ngebacot", "ngebacooot",
    ]
    if any(k in n for k in BORDER_TOXIC):
        return ("borderline", "ASK", "borderline negative/toxic - perlu konfirmasi")

    AFK_KEYS = [
        "maghrib", "sholat", "mandi dulu", "mandi dl", "mandi sek",
        "makan dl", "makan dulu", "makan sek", "pulang dulu",
        "pergi dl", "pergi dulu", "wc dl", "wc dulu", "toilet",
    ]
    if any(k in n for k in AFK_KEYS):
        return ("neutral", "high", "AFK/break announcement")

    POS_KEYS = [
        "mantap", "mantul", "mantab", "sabar bang", "sabar bro",
        "tenang", "sippp", "sip ", "sip!", "oke deh", "okee deh",
        "takpe", "mksh", "makasih", "maaf", "punten",
    ]
    if any(k in n for k in POS_KEYS):
        return ("positive", "high", "kata positif Indonesia")

    if re.search(r"\b(?:kenapa|knp|gimana|gmn|siapa|mana)\b", n) or "?" in s:
        return ("neutral", "medium", "pertanyaan/info-seeking")

    return ("neutral", "medium", "banter/percakapan kasual")


def main():
    df = pd.read_csv("data/gold/sample.csv")
    unlabeled = df[df["annotator_id"].fillna("") == ""].copy()
    indo = unlabeled[unlabeled["key"].apply(detect_indo)].copy()
    print(f"Total Indonesian unlabeled: {len(indo)}")

    rows = []
    for _, r in indo.iterrows():
        sent, conf, reason = propose_label(r["key"])
        rows.append({
            "index": r.name, "key": r["key"],
            "outcome": r["match_outcome_for_player"],
            "sentiment": sent, "confidence": conf, "reason": reason,
        })

    pdf = pd.DataFrame(rows)
    print("\n=== Distribusi proposal ===")
    print(pdf.groupby(["sentiment", "confidence"]).size())

    print("\n\n========================================")
    print("=== HIGH-CONFIDENCE (saya label langsung) ===")
    print("========================================")
    for sent in ["positive", "negative", "neutral"]:
        sub = pdf[(pdf["sentiment"] == sent) & (pdf["confidence"] == "high")]
        if len(sub) == 0:
            continue
        print(f"\n--- {sent.upper()} (n={len(sub)}) ---")
        for _, r in sub.iterrows():
            print(f'  [{r["outcome"][:4]:4s}] "{r["key"]}"  ({r["reason"]})')

    print("\n\n=== MEDIUM-CONFIDENCE (label tapi ambiguous=1) ===")
    sub = pdf[pdf["confidence"] == "medium"]
    print(f"Total: {len(sub)}")
    for _, r in sub.iterrows():
        print(f'  [{r["sentiment"]:8s}] [{r["outcome"][:4]:4s}] "{r["key"]}"')

    print("\n\n========================================")
    print("=== ASK USER (borderline, butuh konfirmasi) ===")
    print("========================================")
    sub = pdf[pdf["confidence"] == "ASK"]
    print(f"Total: {len(sub)}")
    for _, r in sub.iterrows():
        print(f'  [{r["outcome"][:4]:4s}] "{r["key"]}"')

    pdf.to_csv("data/gold/_indo_proposal.csv", index=False)
    print(f'\n\nProposal disimpan ke data/gold/_indo_proposal.csv')


if __name__ == "__main__":
    main()
