"""Dota 2 in-game chat slang / jargon translator.

Why this exists
---------------
Pro Dota 2 chat messages are extremely short (median ~2 chars, 1 token) and
dominated by jargon: ``gg``, ``g``, ``ez``, ``ss``, ``rax``, ``noob``. A
pretrained transformer sees these as near-empty sub-word tokens and cannot
recover any sentiment / toxicity signal — which is the main reason the original
external-trained pipeline scored *worse than chance* on the Dota gold set.

This module rewrites slang into plain emotional / descriptive English **before**
tokenization, so the model's pretrained embeddings actually fire. The polarity
itself is still learned from the gold labels — the dictionary only *expands*
terse jargon into words a sentiment/toxicity model already understands.

The same ``translate_slang`` MUST be applied at both training and inference time
(train/infer parity), otherwise the model sees a different input distribution at
serving time.

Sources for the vocabulary:
    * Dota 2 Wiki — Glossary (gg, ggwp, wp, gj, ez, ss, mia, rax, rosh, gank,
      throw, feed, bkb, smoke, ...).
    * Red Bull "An urban dictionary for Dota 2".
    * Common Twitch / gaming chat slang (pog, kekw, sadge, goat, clutch, kys).

Categories are kept only for documentation / analysis (``SLANG_BY_CATEGORY``);
at runtime everything is flattened into a single lookup (``DOTA_SLANG``).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Multi-word phrases — matched FIRST (before single-token lookup) because the
# whole phrase carries meaning that the individual tokens lose.
# Keys must be lowercase; matching is whitespace-insensitive on word boundaries.
# ---------------------------------------------------------------------------
PHRASE_SLANG: dict[str, str] = {
    "gg ez": "good game but easy mocking the enemy",
    "ez gg": "easy game mocking the enemy",
    "ez game": "easy game mocking the enemy",
    "ez mid": "easy lane mocking the enemy",
    "gg wp": "good game well played respect",
    "gg well played": "good game well played respect",
    "good game": "good game well played",
    "well played": "well played respect",
    "good luck have fun": "good luck have fun friendly",
    "good job": "good job nice well done",
    "nice job": "nice job well done",
    "nice try": "nice try encouraging",
    "well done": "well done great job",
    "shut up": "shut up hostile aggressive",
    "shut the fuck up": "shut the fuck up hostile aggressive toxic",
    "kill yourself": "kill yourself extreme toxic threat self harm",
    "kill your self": "kill yourself extreme toxic threat self harm",
    "end yourself": "end yourself extreme toxic threat",
    "report mid": "report the mid player toxic blaming",
    "report him": "report him toxic blaming",
    "report this": "report this player toxic blaming",
    "report pls": "report please toxic blaming",
    "report please": "report please toxic blaming",
    "uninstall the game": "uninstall the game you are terrible toxic insult",
    "bad game": "bad game disappointed",
    "easy game": "easy game mocking the enemy",
    "good game everyone": "good game everyone sportsmanship",
    "what a throw": "what a throw threw the game frustrated",
}

# ---------------------------------------------------------------------------
# Single-token slang, grouped by intent for readability.
# ---------------------------------------------------------------------------
SLANG_BY_CATEGORY: dict[str, dict[str, str]] = {
    # --- Sportsmanship / positive closing ---------------------------------
    "sportsmanship": {
        "gg": "good game well played",
        "ggwp": "good game well played respect",
        "wp": "well played respect",
        "gj": "good job nice",
        "nj": "nice job",
        "gl": "good luck",
        "hf": "have fun",
        "glhf": "good luck have fun friendly",
        "ty": "thank you",
        "tyvm": "thank you very much",
        "tysm": "thank you so much",
        "thx": "thanks",
        "thanks": "thanks grateful",
        "respect": "respect admiration",
        "commend": "commend praise good behavior",
        "commended": "commended praised",
    },
    # --- Hype / excitement / praise ---------------------------------------
    "hype": {
        "pog": "incredible amazing hype",
        "poggers": "amazing exciting hype",
        "pogchamp": "epic amazing hype",
        "pogu": "amazing surprise hype",
        "pepega": "silly funny",
        "kekw": "hilarious laughing",
        "lul": "funny laughing",
        "lulw": "laughing very hard",
        "omegalul": "extremely funny",
        "lol": "laughing out loud funny",
        "lmao": "hilarious laughing",
        "lmfao": "extremely hilarious laughing",
        "rofl": "rolling laughing hilarious",
        "haha": "laughing funny",
        "hahaha": "laughing hard funny",
        "sheesh": "impressive amazing excited",
        "goat": "greatest of all time legendary best",
        "clutch": "clutch incredible high pressure play",
        "cracked": "insanely skilled brilliant",
        "insane": "insane amazing impressive",
        "nuts": "incredible crazy good",
        "godlike": "godlike dominant unstoppable",
        "ggez": "good game easy mocking",
        "diff": "skill gap superior performance",
        "gigachad": "legendary peak performance hero",
        "chad": "confident strong admirable",
        "sheeesh": "very impressive excited",
        "letsgo": "lets go excited hype",
        "lesgo": "lets go excited hype",
        "hype": "hype excitement",
        "wagh": "excited cheering",
    },
    # --- Sadness / disappointment (mild negative) -------------------------
    "sad": {
        "sadge": "sad disappointed",
        "pepehands": "crying sad",
        "monkas": "nervous anxious",
        "monkaw": "scared worried",
        "copium": "denial coping disappointed",
        "hopium": "desperate hope",
        "f": "respect paying respects sad",
        "rip": "dead unfortunate sad",
        "sadKEK": "bitterly funny sad",
        "unlucky": "unlucky unfortunate",
        "unluko": "unlucky unfortunate",
        "feelsbad": "feeling bad disappointed",
        "feelsbadman": "feeling bad disappointed",
        "oof": "ouch unfortunate",
        "damn": "damn frustrated",
    },
    # --- Neutral callouts / coordination (game-speak, no polarity) --------
    "callout": {
        "g": "go engage attack now",
        "b": "back retreat fall back",
        "care": "be careful warning",
        "ss": "enemy missing from lane warning",
        "mia": "enemy missing in action warning",
        "miss": "enemy missing warning",
        "inc": "incoming enemy warning",
        "omw": "on my way coming",
        "reuse": "use the courier again",
        "recall": "teleport back to base",
        "tp": "teleport",
        "def": "defend the base",
        "push": "push the lane",
        "gank": "ambush enemy hero",
        "smoke": "smoke gank surprise attack",
        "rosh": "roshan objective",
        "roshan": "roshan objective",
        "rax": "barracks objective",
        "racks": "barracks objective",
        "hg": "high ground push",
        "mid": "middle lane",
        "top": "top lane",
        "bot": "bottom lane",
        "lane": "lane",
        "jungle": "jungle farm",
        "farm": "farm gold and experience",
        "stack": "stack the neutral camp",
        "pull": "pull the creeps",
        "deny": "deny the creep",
        "ward": "place a ward vision",
        "obs": "observer ward vision",
        "sentry": "sentry ward detection",
        "dust": "dust of appearance detection",
        "bkb": "black king bar item",
        "rune": "power rune",
        "bounty": "bounty rune gold",
        "creep": "lane creep",
        "cs": "creep score last hits",
        "oom": "out of mana",
        "mana": "mana",
        "ulti": "ultimate ability",
        "ult": "ultimate ability",
        "cd": "ability on cooldown",
        "wait": "wait hold position",
        "go": "go engage attack",
        "back": "back retreat",
        "regroup": "regroup together",
        "split": "split push",
        "rat": "split push strategy",
    },
    # --- Chat shorthand (no game polarity, but very high frequency) -------
    "shorthand": {
        "sry": "sorry apologetic",
        "soz": "sorry apologetic",
        "pls": "please",
        "pls": "please",
        "plz": "please",
        "pliss": "please",
        "np": "no problem",
        "yw": "you are welcome",
        "u": "you",
        "ur": "your",
        "r": "are",
        "y": "why",
        "k": "okay",
        "ok": "okay",
        "okay": "okay",
        "kk": "okay",
        "yy": "yes",
        "ya": "yes",
        "yep": "yes",
        "nah": "no",
        "brb": "be right back",
        "omg": "oh my god surprised",
        "omfg": "oh my god shocked",
        "xd": "laughing funny",
        "xdd": "laughing hard funny",
        "rdy": "ready",
        "ready": "ready",
        "sec": "wait a second",
        "min": "wait a minute",
        "lag": "lag frustrated",
        "laggy": "laggy frustrated",
        "glgl": "good luck good luck",
        "gege": "good game good game",
        "hfhf": "have fun have fun",
        "gngn": "good night good game",
        "gn": "good night",
        "afaik": "as far as i know",
        "imo": "in my opinion",
        "fyi": "for your information",
    },
    # --- Generic positive words sometimes typed standalone ----------------
    "positive_word": {
        "w": "big win success",
        "dub": "win success",
        "win": "win success",
        "nice": "nice good",
        "great": "great good",
        "good": "good positive",
        "ggs": "good games well played",
        "ez": "easy win mocking",
        "easy": "easy win mocking",
        "clean": "clean nicely done",
        "smooth": "smooth nicely done",
    },
    # --- Toxic: insults ----------------------------------------------------
    "toxic_insult": {
        "noob": "noob incompetent bad terrible player insult",
        "nub": "noob bad terrible player insult",
        "newb": "noob bad player insult",
        "trash": "trash terrible awful insult",
        "garbage": "garbage worthless terrible insult",
        "dogwater": "awful terrible insult",
        "dog": "trash terrible insult",
        "bot": "brainless terrible player insult",
        "bots": "brainless terrible players insult",
        "boosted": "fake skill carried terrible insult",
        "shitter": "terrible useless player insult",
        "bum": "useless terrible player insult",
        "clown": "ridiculous foolish insult",
        "clown0": "ridiculous foolish insult",
        "idiot": "idiot stupid insult",
        "idiots": "idiots stupid insult",
        "dumb": "dumb stupid insult",
        "dumbass": "stupid worthless insult",
        "stupid": "stupid insult",
        "moron": "moron stupid insult",
        "braindead": "brainless stupid insult",
        "retard": "extremely stupid offensive insult",
        "retarded": "extremely stupid offensive insult",
        "loser": "pathetic failure insult",
        "ass": "awful terrible bad",
        "trashcan": "worthless terrible insult",
        "washed": "no skill declined insult",
        "useless": "useless worthless insult",
        "ape": "brainless insult",
        "monkey": "brainless foolish insult",
        "l": "massive loss terrible failure",
        "L": "massive loss terrible failure",
    },
    # --- Toxic: blame / report / grief -------------------------------------
    "toxic_blame": {
        "report": "report this player toxic blaming",
        "reported": "reported toxic blaming",
        "reportable": "reportable behavior toxic blaming",
        "throw": "threw the game ruined it frustrated",
        "threw": "threw the game ruined it frustrated",
        "throwing": "throwing the game ruining it on purpose toxic",
        "thrower": "saboteur ruining the game toxic",
        "feed": "feeding dying on purpose ruining toxic",
        "feeder": "feeder ruining the game toxic",
        "feeding": "feeding dying on purpose ruining toxic",
        "fed": "fed the enemy ruined toxic",
        "int": "intentionally feeding ruining the game toxic",
        "inting": "intentionally feeding ruining toxic",
        "grief": "griefing ruining the game toxic",
        "griefer": "griefer ruining the game toxic",
        "ruin": "ruined the game toxic",
        "ruined": "ruined the game frustrated toxic",
        "uninstall": "you are terrible quit the game insult",
        "ff": "give up surrender frustrated",
        "bg": "bad game frustrated",
        "mute": "mute this toxic player",
        "muted": "muted the toxic player",
        "afk": "away from keyboard abandoning",
        "abandon": "abandon the game",
        "smurf": "smurf unfair higher skill account",
    },
    # --- Toxic: threats / severe ------------------------------------------
    "toxic_threat": {
        "kys": "kill yourself extreme toxic threat self harm",
        "die": "die hostile threat",
        "kill": "kill hostile",
    },
    # --- Toxic: profanity / obscene ---------------------------------------
    "toxic_obscene": {
        "fuck": "fuck angry profanity",
        "fucking": "fucking angry profanity",
        "fucked": "fucked angry profanity",
        "fck": "fuck angry profanity",
        "fk": "fuck angry profanity",
        "wtf": "what the fuck angry frustrated",
        "stfu": "shut the fuck up hostile toxic",
        "shit": "shit angry profanity",
        "shitty": "shitty terrible profanity",
        "bullshit": "bullshit unfair angry profanity",
        "bs": "bullshit unfair angry",
        "bitch": "bitch aggressive offensive insult",
        "bastard": "bastard offensive insult",
        "cunt": "cunt extremely offensive insult",
        "dick": "dick offensive insult",
        "suck": "awful terrible bad",
        "sucks": "awful terrible bad",
        "damn": "damn frustrated",
        "hell": "hell frustrated",
    },
}

# Flatten — later categories override earlier ones on key collision, so order
# matters. We resolve a couple of intentional collisions explicitly below.
DOTA_SLANG: dict[str, str] = {}
for _cat, _mapping in SLANG_BY_CATEGORY.items():
    DOTA_SLANG.update(_mapping)

# Explicit collision resolutions (keep the most informative meaning):
#   "damn" appears in both `sad` and `toxic_obscene` — keep the mild frustration.
DOTA_SLANG["damn"] = "damn frustrated"

# Token cleanup: strip surrounding punctuation, collapse repeated chars in the
# emote-style elongations (e.g. "noooob" -> "noob", "gggg" -> handled by exact).
_PUNCT_STRIP = re.compile(r"^[^\w]+|[^\w]+$")
_REPEAT = re.compile(r"(.)\1{2,}")  # 3+ repeats -> single (sheeesh -> sheesh? no)


def _normalize_token(tok: str) -> str:
    """Lowercase + strip leading/trailing punctuation for dictionary lookup."""
    return _PUNCT_STRIP.sub("", tok).lower()


def translate_slang(text: str) -> str:
    """Rewrite Dota 2 / gaming slang in ``text`` into plain descriptive English.

    Order of operations:
      1. Replace known multi-word phrases (``PHRASE_SLANG``).
      2. Replace single tokens (``DOTA_SLANG``), matching on a punctuation- and
         case-normalized form while preserving unknown tokens verbatim.

    Unknown words pass through unchanged, so real English chat ("you played
    really well") is left intact.

    Empty / NaN-ish input returns "" so downstream tokenizers never see None.
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    low = s.lower()
    # 1) Phrase pass — longest phrases first to avoid partial shadowing.
    #    Each matched phrase is swapped for a sentinel placeholder so its
    #    expansion is NOT re-expanded by the token pass below (that double-
    #    processing was producing garbage like "mocking mocking").
    placeholders: dict[str, str] = {}
    for i, phrase in enumerate(sorted(PHRASE_SLANG, key=len, reverse=True)):
        if phrase in low:
            sentinel = f"\x00{i}\x00"
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b")
            low = pattern.sub(f" {sentinel} ", low)
            placeholders[sentinel] = PHRASE_SLANG[phrase]
    # Work entirely in lowercase from here (pretrained uncased models lowercase
    # anyway, and chat casing carries little signal).

    # 2) Token pass.
    out: list[str] = []
    for tok in low.split():
        if tok in placeholders:  # restore a phrase expansion verbatim
            out.append(placeholders[tok])
            continue
        key = _normalize_token(tok)
        if key in DOTA_SLANG:
            out.append(DOTA_SLANG[key])
        elif key:
            out.append(tok)
        # drop pure-punctuation tokens
    translated = " ".join(out).strip()
    if not translated:
        # fall back to the de-sentinel'd lowercase text
        for sentinel, expansion in placeholders.items():
            low = low.replace(sentinel, expansion)
        return low.strip()
    return translated


def slang_coverage(texts) -> dict[str, float]:
    """Diagnostic: fraction of messages / tokens that hit the slang dictionary.

    Returns a dict with ``msg_hit_rate`` (share of messages with >=1 slang token)
    and ``token_hit_rate`` (share of all tokens that are slang).
    """
    n_msg = 0
    n_msg_hit = 0
    n_tok = 0
    n_tok_hit = 0
    phrase_keys = set(PHRASE_SLANG)
    for t in texts:
        s = str(t or "").strip().lower()
        if not s:
            continue
        n_msg += 1
        toks = s.split()
        hit_here = any(_normalize_token(tk) in DOTA_SLANG for tk in toks)
        hit_here = hit_here or any(p in s for p in phrase_keys)
        n_msg_hit += int(hit_here)
        for tk in toks:
            n_tok += 1
            n_tok_hit += int(_normalize_token(tk) in DOTA_SLANG)
    return {
        "n_messages": float(n_msg),
        "msg_hit_rate": (n_msg_hit / n_msg) if n_msg else 0.0,
        "token_hit_rate": (n_tok_hit / n_tok) if n_tok else 0.0,
        "dict_size": float(len(DOTA_SLANG)),
        "phrase_count": float(len(PHRASE_SLANG)),
    }
