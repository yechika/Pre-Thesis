"""Final pass: label sisa unlabeled dengan rule longgar + flag uncertain."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Re-use konstanta dari label_batch.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_batch import (  # type: ignore
    norm_full, norm_strict, make_label,
    TOXIC_FOREIGN, TOXIC_IDENTITY_SLURS, TOXIC_BLOCKER_BASE,
    INSULT_WORDS, OBSCENE_STANDALONE,
    SPORT_EXACT, SPORT_PHRASES_REGEX,
    APOLOGY_EXACT, APOLOGY_TOKENS,
    TECH_EXACT, TECH_TOKENS_BROAD,
    EZ_VARIANTS, FF_VARIANTS,
    has_toxic_blocker,
    _indo_toxic_regex_match,
    TOXIC_MULTIWORD_PHRASES,
)


# Load Dota tokens
def load_dota_tokens(constants_root: Path):
    heroes = pd.read_csv(constants_root / "Constants.Heroes.csv")
    items = pd.read_csv(constants_root / "Constants.Items.csv")
    hero_names = set()
    for n in heroes["localized_name"].dropna():
        parts = re.findall(r"[a-z]+", n.lower())
        hero_names.add(n.lower())
        if len(parts) > 1:
            hero_names.add("".join(parts))
        for p in parts:
            if len(p) >= 4:
                hero_names.add(p)
    hero_names |= {
        "am", "aa", "wd", "cm", "od", "es", "np", "qop", "sk", "ta", "sf",
        "sd", "dk", "ns", "vs", "kotl", "ds", "wr", "wk", "pa", "pl", "pp",
        "ck", "lc", "ls", "spec", "mk", "tb", "ww", "fv", "necro", "centaur",
        "io", "tiny", "lina", "lion", "zeus", "tinker", "rubick", "puck",
        "riki", "ogre", "bane", "earth", "sven", "kunkka", "skywrath", "sm",
        "broodmother", "bm", "muerta", "primal", "marci", "void",
    }
    item_names = set()
    for n in items["dname"].dropna():
        parts = re.findall(r"[a-z]+", n.lower())
        item_names.add(n.lower())
        if len(parts) > 1:
            item_names.add("".join(parts))
        for p in parts:
            if len(p) >= 4:
                item_names.add(p)
    item_names |= {
        "bkb", "manta", "bf", "battlefury", "midas", "vlad", "vlads", "pipe",
        "meka", "mek", "eul", "euls", "force", "glimmer", "shard", "aghs",
        "agh", "aghanim", "octarine", "hex", "blink", "drum", "wand", "stick",
        "ward", "wards", "obs", "sentry", "tp", "tps", "salve", "tango",
        "mango", "clarity", "courier", "stout", "armlet", "abyssal",
        "rapier", "diffusal", "diff", "sange", "yasha", "treads", "phase",
        "veil", "lotus", "lens", "satanic", "ac", "mom", "deso", "desolator",
        "halberd", "boots", "tranquils", "smoke",
    }
    return hero_names, item_names


HERO_SET, ITEM_SET = load_dota_tokens(Path("dota2_dataset_bersih/Constants"))


# ---- Helper checks ----------
def is_only_dota_tokens(s):
    """True kalau ALL alpha tokens adalah Dota hero/item."""
    tokens = re.findall(r"[a-z]+", s.lower()) if isinstance(s, str) else []
    if not tokens or len(tokens) > 4:
        return False
    return all(t in HERO_SET or t in ITEM_SET for t in tokens)


def has_dota_token(s):
    tokens = re.findall(r"[a-z]+", s.lower()) if isinstance(s, str) else []
    return any(t in HERO_SET or t in ITEM_SET for t in tokens)


def has_sport_token_loose(s):
    parts = (norm_full(s).split() if isinstance(s, str) else [])
    sport_tokens = {"gl", "hf", "wp", "gj", "np", "ty", "thx", "thanks", "tysm",
                    "tyvm", "glhf", "gz", "gratz", "congrats", "grats",
                    "respect", "respects", "nice", "good", "well"}
    return any(p.strip(",.!?\"'") in sport_tokens for p in parts)


def has_apology_token_loose(s):
    parts = (norm_full(s).split() if isinstance(s, str) else [])
    return any(p.strip(",.!?\"'") in APOLOGY_TOKENS for p in parts)


def has_tech_token_loose(s):
    parts = (norm_full(s).split() if isinstance(s, str) else [])
    return any(p.strip(",.!?\"'") in TECH_TOKENS_BROAD for p in parts)


# ---- Extra patterns (Tier 1 — very confident) ----
EMOTICON_POS = {":)", ":-)", ":d", ":-d", "=)", ":p", ":-p", "<3", ":3", "^^",
                "^_^", ":>", "=d", "(:"}
EMOTICON_NEG = {":(", ":-(", ":/", "=(", ":<", "):"}
GREETINGS_LOOSE = {"hi", "hi all", "hi guys", "hi all good luck",
                   "hello", "hello all", "hello guys", "hey", "hey guys",
                   "hey all", "halo", "haii", "haiiii", "salam",
                   "bye", "byee", "byeee", "byeeee", "cya", "see you",
                   "see ya", "see u", "morning", "good morning", "morn",
                   "evening", "good evening", "afternoon"}
FOREIGN_ACK = {"da", "нет", "ni hao", "ni hao ma", "hola", "si", "oui",
               "non", "ja", "nein", "wala", "meron", "opo", "oo", "hindi",
               "yes sir", "no sir"}


def label_remaining(s, time, duration, outcome):
    """Return (label_dict, confidence: high/medium/low/ASK)."""
    if not isinstance(s, str):
        return None, None
    s_strip = s.strip()
    if not s_strip:
        return None, None
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    parts = n_full.split()

    # === TIER 1: Very confident ===

    # Pure punctuation/symbols
    if re.fullmatch(r"[\.\,\:\;\!\?\-\_\=\+\*\&\^\%\#\@\$~`\\\(\)\[\]\{\}\<\>\/]+",
                    s_strip):
        return make_label("neutral", is_ambig=1,
                          notes=f'filler punctuation/symbols ("{s_strip}")'), "high"

    # Emoticons
    if n_full in EMOTICON_POS or n_strict in {x.replace(" ", "") for x in EMOTICON_POS}:
        return make_label("positive",
                          notes=f'emoticon positive ("{n_full}")'), "high"
    if n_full in EMOTICON_NEG or n_strict in {x.replace(" ", "") for x in EMOTICON_NEG}:
        return make_label("negative", is_ambig=1,
                          notes=f'emoticon negative ("{n_full}")'), "high"

    # Single alpha character
    if re.fullmatch(r"[a-zA-Z]", s_strip):
        return make_label("neutral", is_ambig=1,
                          notes=f'single character filler ("{s_strip}")'), "high"

    # Greetings
    if n_full in GREETINGS_LOOSE or n_strict in {x.replace(" ", "") for x in GREETINGS_LOOSE}:
        return make_label("positive",
                          notes=f'greeting/farewell ("{n_full}")'), "high"

    # Foreign acknowledgment
    if n_full in FOREIGN_ACK or n_strict in {x.replace(" ", "") for x in FOREIGN_ACK}:
        return make_label("neutral", is_ambig=1,
                          notes=f'foreign acknowledgment ("{n_full}")'), "high"

    # Pure numbers / timer
    if re.fullmatch(r"[\d\s\.,]+", s_strip):
        return make_label("neutral", is_ambig=1,
                          notes=f'numbers/timer ("{n_full}")'), "high"

    # Pure Dota token only
    if is_only_dota_tokens(s):
        return make_label("neutral", is_jargon=1,
                          notes=f'Dota reference only ("{n_full}")'), "high"

    # === TIER ASK: GWR family — UNKNOWN ===
    if "gwr" in n_strict and len(n_strict) <= 6:
        return make_label("neutral", is_ambig=1,
                          notes=f'"gwr" arti tidak jelas, default neutral'), "ASK"

    # Other unknown short keys
    UNKNOWN_AMBIG = {"kale", "kalee", "ka le", "dx", "ts", "wsp", "ne", "nya",
                     "ya", "y2", "tu", "ge", "gege"}
    if n_strict in UNKNOWN_AMBIG:
        return make_label("neutral", is_ambig=1,
                          notes=f'token ambiguous ("{n_full}"), default neutral'), "ASK"

    # === TIER 2: Toxic detection (longer messages with toxic words) ===
    if has_toxic_blocker(s):
        # Cek apakah ada juga slur identitas
        tokens = re.findall(r"[a-z]+", s.lower())
        has_identity = any(t in TOXIC_IDENTITY_SLURS for t in tokens)
        return make_label(
            "negative", tox_toxic=1, tox_obscene=1, tox_insult=1,
            tox_identity=1 if has_identity else 0, is_ambig=1,
            notes=f'mengandung kata toxic dalam pesan panjang',
        ), "medium"

    # === TIER 2: Sportsmanship/apology/tech long form ===
    if 1 <= len(parts) <= 10:
        if has_apology_token_loose(s):
            return make_label("positive", is_ambig=1,
                              notes=f'apology dalam pesan panjang ("{n_full}")'), "medium"
        if has_sport_token_loose(s):
            return make_label("positive", is_ambig=1,
                              notes=f'sportsmanship dalam pesan panjang ("{n_full}")'), "medium"
        if has_tech_token_loose(s):
            return make_label("neutral",
                              notes=f'tech issue dalam pesan panjang ("{n_full}")'), "medium"

    # === TIER 2: Dota coord with hero/item ===
    if has_dota_token(s) and len(parts) <= 8 and not has_toxic_blocker(s):
        return make_label("neutral", is_jargon=1, is_ambig=1,
                          notes=f'Dota coordination/observation ("{n_full[:50]}")'), "medium"

    # === TIER 3: Default fallback ===
    if len(parts) >= 6:
        return make_label("neutral", is_ambig=1,
                          notes=f'banter/percakapan kasual panjang'), "low"

    if len(parts) <= 5:
        return make_label("neutral", is_ambig=1,
                          notes=f'banter/percakapan kasual pendek'), "low"

    return make_label("neutral", is_ambig=1,
                      notes=f'fallback ambiguous'), "low"


def main():
    sample_path = Path("data/gold/sample.csv")
    df = pd.read_csv(sample_path)
    print(f"Total: {len(df):,}  |  sudah dilabel: {(df['annotator_id']=='AI').sum():,}")

    for col in ["annotator_id", "sentiment", "notes"]:
        df[col] = df[col].astype("object")
    for col in ["tox_toxic", "tox_severe_toxic", "tox_obscene", "tox_threat",
                "tox_insult", "tox_identity_hate", "is_dota_jargon", "is_ambiguous"]:
        df[col] = df[col].astype("object")

    unlabeled_mask = df["annotator_id"].fillna("") == ""
    print(f"Akan diproses: {unlabeled_mask.sum():,}")

    n_per_conf = {"high": 0, "medium": 0, "low": 0, "ASK": 0}
    uncertain_rows = []

    for i in df[unlabeled_mask].index:
        key = df.at[i, "key"]
        time = df.at[i, "time"]
        duration = df.at[i, "duration"]
        outcome = df.at[i, "match_outcome_for_player"]
        result = label_remaining(key, time, duration, outcome)
        if result is None or result[0] is None:
            continue
        lbl, conf = result
        df.at[i, "annotator_id"] = "AI"
        for k, v in lbl.items():
            df.at[i, k] = v
        n_per_conf[conf] += 1
        if conf == "ASK":
            uncertain_rows.append({
                "index": i, "key": key, "outcome": outcome,
                "sentiment": lbl["sentiment"],
                "notes": lbl["notes"],
            })

    df.to_csv(sample_path, index=False, encoding="utf-8")

    print("\n=== Hasil per confidence ===")
    for c, n in n_per_conf.items():
        print(f"  {c:8s}: {n:5,}")
    total = sum(n_per_conf.values())
    print(f"  {'TOTAL':8s}: {total:5,}")

    cumul = (df["annotator_id"] == "AI").sum()
    print(f"\nCumulative dilabel: {cumul:,}/{len(df):,} ({100*cumul/len(df):.1f}%)")
    print(f"Sisa unlabeled    : {len(df) - cumul:,}")

    print("\n=== Distribusi sentiment final ===")
    print(df.loc[df["annotator_id"] == "AI", "sentiment"].value_counts())

    print("\n=== Tox flags ===")
    for tcol in ["tox_toxic", "tox_severe_toxic", "tox_obscene", "tox_threat",
                 "tox_insult", "tox_identity_hate"]:
        n = (df[tcol] == 1).sum()
        print(f"  {tcol:20s}: {n}")

    # Save uncertain cases
    if uncertain_rows:
        unc = pd.DataFrame(uncertain_rows)
        unc.to_csv("data/gold/_uncertain_cases.csv", index=False)
        print(f"\n=== UNCERTAIN ({len(unc)} cases) — disimpan ke data/gold/_uncertain_cases.csv ===")
        # Summary by category
        print("\nBreakdown:")
        for k_pat, label in [
            (r"\bgwr", "GWR family"),
            (r"kale|ka le", "kale/ka le"),
            (r"^dx$|^Dx$|^DX$", "dx"),
            (r"^ts$", "ts"),
            (r"^wsp$", "wsp"),
            (r"^nya$", "nya"),
            (r"^ne$", "ne"),
            (r"^tu$", "tu"),
        ]:
            subset = unc[unc["key"].astype(str).str.contains(k_pat, regex=True, na=False)]
            if len(subset) > 0:
                print(f"  {label:20s}: {len(subset):3} baris")


if __name__ == "__main__":
    main()
