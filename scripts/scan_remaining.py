"""Scan sisa unlabeled untuk: bahasa lain + Dota hero/item/ability tokens."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---- Bahasa lain (markers yang khas) ----
# Russian (transliterated common ones in latin script)
RUSSIAN_MARKERS = {
    "blyat", "blya", "blyad", "suka", "pidor", "pizdec", "huy", "yebat",
    "nahuy", "ebal", "kurwa", "ny", "nu", "da", "kak", "shto",
    "rakom", "loh", "lokh", "uebok", "huesos", "huevos",
    "rus", "russian", "русский", "хуй", "блять", "сука", "пизда",
}
# Filipino (Tagalog markers + already-toxic)
FILIPINO_MARKERS = {
    "ang", "ng", "ako", "ikaw", "tayo", "kayo", "sila", "ito", "iyan",
    "puta", "tanga", "bobo", "ulol", "gago", "tarantado", "pi",
    "lintik", "putik", "salbahe", "gago", "putangina", "putang ina",
    "kupal", "siraulo", "hayop", "leche", "kamukha",
    "talaga", "naman", "lang", "pala", "kasi", "pero", "tapos",
    "magaling", "mabuti", "salamat",
}
# Spanish (common)
SPANISH_MARKERS = {
    "que", "como", "donde", "porque", "para", "esto", "eso", "esta",
    "este", "tienes", "tiene", "tengo", "tenemos", "estoy", "estas",
    "no", "si", "muy", "mas", "menos", "todo", "nada", "alguien",
    "puta", "mierda", "carajo", "joder", "cabron", "pendejo",
    "amigo", "hermano", "hijo", "perro", "gato",
    "vamos", "vamonos", "ahora", "luego", "antes", "despues",
    "hora", "tiempo", "rato", "minuto",
    "pide", "puede", "quiero", "tengo", "es", "son",
}
# Vietnamese / Thai / Chinese transliterated (pinyin) - less common
VIETNAMESE_MARKERS = {
    "khong", "co", "khong co", "duoc", "khong duoc", "thua", "ngu",
    "ban", "toi", "anh", "em",
}
CHINESE_PINYIN_MARKERS = {
    "shen", "hao", "gei", "shi", "wo", "ni", "ta", "buhao", "mei", "you",
    "diao", "haide", "haode", "buyao", "ai",
}


def detect_lang_other(text):
    """Return list of detected non-EN/non-ID languages."""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return []
    detected = []
    if any(t in RUSSIAN_MARKERS for t in tokens) or re.search(r"[а-яА-Я]", text):
        detected.append("russian")
    if sum(1 for t in tokens if t in FILIPINO_MARKERS) >= 1:
        detected.append("filipino")
    if sum(1 for t in tokens if t in SPANISH_MARKERS) >= 2:
        detected.append("spanish")
    if any(t in VIETNAMESE_MARKERS for t in tokens):
        detected.append("vietnamese")
    if sum(1 for t in tokens if t in CHINESE_PINYIN_MARKERS) >= 2:
        detected.append("chinese_pinyin")
    return detected


# ---- Dota tokens ----
def load_dota_tokens(constants_root: Path):
    heroes = pd.read_csv(constants_root / "Constants.Heroes.csv")
    items = pd.read_csv(constants_root / "Constants.Items.csv")

    hero_names = []
    for n in heroes["localized_name"].dropna().tolist():
        # Tokenize hero name: "Anti-Mage" → ["anti", "mage", "antimage"]
        parts = re.findall(r"[a-z]+", n.lower())
        if not parts:
            continue
        hero_names.append(n.lower())  # full lowercase
        if len(parts) > 1:
            hero_names.append("".join(parts))  # joined
        for p in parts:
            if len(p) >= 4:  # individual token kalau panjang ≥4
                hero_names.append(p)
    # Tambah abbreviations populer manual
    hero_abbrev = {
        "am", "anti-mage", "antimage",
        "aa", "ancient apparition",
        "wd", "witch doctor",
        "cm", "crystal maiden",
        "od", "outworld",
        "es", "earthshaker", "earth spirit",
        "np", "natures prophet", "nature's prophet",
        "qop", "queen of pain",
        "sk", "sand king",
        "ta", "templar assassin",
        "sf", "shadow fiend",
        "sd", "shadow demon", "slardar",
        "dk", "dragon knight",
        "ns", "night stalker",
        "vs", "vengeful spirit", "vengeful",
        "kotl", "keeper of the light",
        "ds", "dark seer",
        "wr", "windranger",
        "wk", "wraith king",
        "pa", "phantom assassin",
        "pl", "phantom lancer",
        "pp", "puck",
        "ck", "chaos knight",
        "lc", "legion commander",
        "lifestealer", "ls",
        "spec", "spectre",
        "mk", "monkey king",
        "tb", "terrorblade",
        "ti", "tinker",  # juga bisa "TI" turnamen, hati2
        "ww", "winter wyvern",
        "voids", "void", "faceless void", "fv",
        "necro", "necrophos",
        "centaur", "io", "tiny", "lina", "lion", "zeus", "tinker",
        "rubick", "puck", "riki", "ogre", "bane", "earth", "sven", "kunkka",
        "skywrath", "sm",
        "broodmother", "bm",
    }
    hero_set = set(hero_names) | hero_abbrev

    item_names = []
    for n in items["dname"].dropna().tolist():
        parts = re.findall(r"[a-z]+", n.lower())
        if not parts:
            continue
        item_names.append(n.lower())
        if len(parts) > 1:
            item_names.append("".join(parts))
        for p in parts:
            if len(p) >= 4:
                item_names.append(p)
    item_set = set(item_names)
    item_abbrev = {
        "bkb", "manta", "bf", "battlefury", "midas", "vlad", "vlads",
        "pipe", "meka", "mek", "eul", "euls", "force", "glimmer",
        "shard", "aghs", "agh", "aghanim", "aghs scepter", "aghs blessing",
        "octarine", "hex", "blink", "drum", "wand", "stick",
        "mjollnir", "skadi", "rapier", "diffusal", "diff", "sange", "yasha",
        "pms", "tranquils", "treads", "phase", "veil", "lotus", "lens",
        "satanic", "ac", "mom", "deso", "desolator", "halberd",
        "boots", "wards", "ward", "obs", "sentry", "smoke", "tp", "tps",
        "tpscroll", "salve", "tango", "mango", "clarity", "courier",
        "stout", "vlads", "armlet", "abyss", "abyssal",
    }
    return hero_set, item_set | item_abbrev


def has_dota_tokens(s, hero_set, item_set):
    """Return (has_hero, has_item, hits)."""
    if not isinstance(s, str):
        return False, False, []
    tokens = re.findall(r"[a-z]+", s.lower())
    norm = re.sub(r"\s+", " ", s.lower().strip())
    hero_hits = [t for t in tokens if t in hero_set]
    item_hits = [t for t in tokens if t in item_set]
    # Cek phrase 2-token "anti mage" dst
    parts = norm.split()
    for i in range(len(parts) - 1):
        bi = " ".join(parts[i:i + 2])
        if bi in hero_set:
            hero_hits.append(bi)
    return bool(hero_hits), bool(item_hits), hero_hits + item_hits


def main():
    df = pd.read_csv("data/gold/sample.csv")
    unlabeled = df[df["annotator_id"].fillna("") == ""].copy()
    print(f"Total unlabeled: {len(unlabeled):,}")

    # Bahasa lain
    unlabeled["_lang_other"] = unlabeled["key"].apply(detect_lang_other)

    print("\n=== Distribusi bahasa terdeteksi ===")
    lang_counts = {}
    for langs in unlabeled["_lang_other"]:
        for l in langs:
            lang_counts[l] = lang_counts.get(l, 0) + 1
    for l, c in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {l}: {c}")
    none_lang = (unlabeled["_lang_other"].apply(len) == 0).sum()
    print(f"  (none/EN/Dota): {none_lang}")

    # Sample per language
    for lang in ["russian", "filipino", "spanish", "vietnamese", "chinese_pinyin"]:
        sub = unlabeled[unlabeled["_lang_other"].apply(lambda x: lang in x)]
        if len(sub) == 0:
            continue
        print(f"\n--- Sample {lang.upper()} (n={len(sub)}) ---")
        for _, r in sub.head(15).iterrows():
            print(f'  "{r["key"]}"  outcome={r["match_outcome_for_player"]}')

    # Dota tokens
    hero_set, item_set = load_dota_tokens(Path("dota2_dataset_bersih/Constants"))
    print(f"\nHero set: {len(hero_set)} entries")
    print(f"Item set: {len(item_set)} entries")

    unlabeled["_dota"] = unlabeled["key"].apply(
        lambda s: has_dota_tokens(s, hero_set, item_set)
    )
    unlabeled["_has_hero"] = unlabeled["_dota"].apply(lambda x: x[0])
    unlabeled["_has_item"] = unlabeled["_dota"].apply(lambda x: x[1])
    unlabeled["_dota_hits"] = unlabeled["_dota"].apply(lambda x: x[2])

    has_dota = unlabeled[unlabeled["_has_hero"] | unlabeled["_has_item"]]
    print(f"\n=== Sisa mengandung Dota token (hero/item): {len(has_dota)} ===")

    # Top patterns
    print("\nSample hero+item hits (top 30):")
    for _, r in has_dota.head(30).iterrows():
        print(f'  "{r["key"]}"  | hits: {r["_dota_hits"][:5]}')


if __name__ == "__main__":
    main()
