"""AI-assisted labeling: 6 kategori (punct, insult, ez, ff, sport, obscene).

Run dari root repo:
    python scripts/label_batch.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ---- Helpers --------------------------------------------------------------

def norm_full(s: object) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


def norm_strict(s: object) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[!.?,~\s\"']+", "", s.lower())


def make_label(sentiment, tox_toxic=0, tox_severe=0, tox_obscene=0, tox_threat=0,
               tox_insult=0, tox_identity=0, is_jargon=0, is_ambig=0, notes=""):
    return dict(
        sentiment=sentiment,
        tox_toxic=tox_toxic, tox_severe_toxic=tox_severe, tox_obscene=tox_obscene,
        tox_threat=tox_threat, tox_insult=tox_insult, tox_identity_hate=tox_identity,
        is_dota_jargon=is_jargon, is_ambiguous=is_ambig, notes=notes,
    )


# ---- Category 1: Punctuation only -----------------------------------------

def cat_punct(s, time, duration, outcome):
    if not isinstance(s, str) or not s.strip():
        return None
    stripped = re.sub(r"[!?.,~\s]", "", s)
    if stripped == "" and re.search(r"[!?]", s):
        return make_label(
            "neutral", is_ambig=1,
            notes="reaksi punctuation-only (?, !, ??, !! dst), tone tidak jelas",
        )
    return None


# ---- Category 2: Toxic insults --------------------------------------------

INSULT_WORDS = {
    "noob", "noobs", "newb", "newbs", "noobie", "trash", "idiot", "idiots",
    "dumb", "stupid", "loser", "losers", "retard", "retards", "tard", "tards",
    "useless", "feeder", "feeders", "griefer", "griefers", "clown", "clowns",
}


def cat_insult(s, time, duration, outcome):
    n = norm_strict(s)
    if n in INSULT_WORDS:
        return make_label(
            "negative", tox_toxic=1, tox_insult=1,
            notes=f'penghinaan standalone "{n}"',
        )
    parts = norm_full(s).split()
    if 1 <= len(parts) <= 4:
        clean = [p.strip(",.!?\"'") for p in parts]
        hits = [p for p in clean if p in INSULT_WORDS]
        if hits:
            return make_label(
                "negative", tox_toxic=1, tox_insult=1,
                notes=f"penghinaan dalam frasa pendek (kata kunci: {hits[0]})",
            )
    return None


# ---- Category 3: ez/easy taunts -------------------------------------------

EZ_VARIANTS = {"ez", "ezz", "ezzz", "ezzzz", "eazy", "easy", "ezpz", "ezgg", "ggez"}


def cat_ez(s, time, duration, outcome):
    n = norm_strict(s)
    if n not in EZ_VARIANTS:
        return None
    if outcome == "win":
        return make_label(
            "negative", tox_toxic=1, tox_insult=1,
            notes=f'ez/easy taunt dari pengirim menang ("{n}"), taunting lawan',
        )
    if outcome == "loss":
        return make_label(
            "negative", is_ambig=1,
            notes=f'ez/easy dari pengirim kalah ("{n}"), kemungkinan sarcasm tentang situasi sendiri',
        )
    return make_label("negative", is_ambig=1, notes=f'ez/easy outcome unknown ("{n}")')


# ---- Category 4: ff/forfeit -----------------------------------------------

FF_VARIANTS = {"ff", "ggff", "ffgg", "ffs", "forfeit", "25ff", "30ff", "ff25", "ff30"}


def cat_ff(s, time, duration, outcome):
    n = norm_strict(s)
    if n not in FF_VARIANTS:
        return None
    if outcome == "loss":
        return make_label("negative", notes=f'ff/forfeit defeatism dari pengirim kalah ("{n}")')
    if outcome == "win":
        return make_label(
            "negative", tox_toxic=1, is_ambig=1,
            notes=f'ff dari pengirim menang ("{n}"), ajakan forfeit ke lawan (mild taunt)',
        )
    return make_label("negative", is_ambig=1, notes=f'ff outcome unknown ("{n}")')


# ---- Category 5: Sportsmanship --------------------------------------------

SPORT_EXACT = {
    "wp", "wps", "wpwp", "nice", "niice", "niiice", "nicee", "niceee",
    "nice play", "nice plays", "np", "gj", "goodgame", "good game",
    "well played", "nicely played", "good job", "nice one", "nice job",
    "gj guys", "wp guys", "gl", "glhf", "gl hf", "good luck", "have fun",
    "thx", "ty", "thanks", "thank you", "tysm", "tyvm", "gn", "gngn",
    "good night",
    # Extension
    "hf", "hf!", "hf gl", "gl hf boys",
    "g try", "go try", "good try", "gt", "gtgt", "g g",
    "gg gl", "gl next", "gl further", "gn next", "next game",
    "no worries", "nvm", "no problem", "no prob", "all good", "ok np", "k np",
    "close game", "close", "close one",
    "f",  # "F" to pay respects (gaming meme positive)
    "gfg",  # good fucking game (sportsmanship + mild profanity)
    "respect", "respects",
    "u2", "you too", "same to you", "me too", "same", "sama", "u 2", "y2",
    "ggl", "g3g3", "gygy",  # gg typo variants
    "grats", "congrats", "gz", "gz wp", "ggwp gz",
    # Indonesian positive
    "mantap", "mantul", "mantab", "mantaab", "anjay", "anjayy", "anjaay",
    "mabar", "yuk mabar", "ayok mabar", "gas mabar",
    "maaf", "yamaap", "yamap", "ya maaf", "ya maap", "ya map",
    "punten", "mksh", "mksih", "makasih", "makasihh", "thx ya", "thx ya bro",
    "sabar", "sabar bro", "sabar bg", "sabar bang", "tenang", "santai",
    "takpe", "takpe bro", "takpe bang", "ga papa", "gapapa", "no problemo",
    "full chill", "chill bro", "chill", "santai bro", "santai bg",
    "masuk terus", "gas terus", "lanjut bro",
    "sip", "sippp", "sipp",
}
SPORT_STRICT = {x.replace(" ", "") for x in SPORT_EXACT}


def cat_sport(s, time, duration, outcome):
    n = norm_full(s)
    if n in SPORT_EXACT or norm_strict(s) in SPORT_STRICT:
        return make_label("positive", notes=f'sportsmanship/sopan ("{n}")')
    return None


# ---- Category 6: Obscene/profanity ----------------------------------------

OBSCENE_STANDALONE = {
    "fuck", "fucking", "fucked", "fk", "fck", "fuk", "fak", "wtf", "wth", "omfg",
    "shit", "shitty", "sht", "crap", "damn", "damnit", "dammit", "goddamn", "gd",
    "bullshit", "bs", "fml", "jfc", "omg", "omgg", "omggg",
}
TARGET_TOKENS = {"you", "u", "ur", "your", "yall", "yourself"}


def cat_obscene(s, time, duration, outcome):
    n = norm_strict(s)
    if n in OBSCENE_STANDALONE:
        return make_label(
            "negative", tox_obscene=1, is_ambig=1,
            notes=f'profanity standalone ("{n}"), frustrasi tanpa target',
        )
    parts = norm_full(s).split()
    if 1 <= len(parts) <= 4:
        clean = [p.strip(",.!?\"'") for p in parts]
        if clean and clean[0] in OBSCENE_STANDALONE:
            if len(clean) >= 2 and clean[1] in TARGET_TOKENS:
                return make_label(
                    "negative", tox_toxic=1, tox_obscene=1, tox_insult=1,
                    notes=f'profanity + target person ("{" ".join(clean)}"), toxic insult',
                )
            return make_label(
                "negative", tox_obscene=1, is_ambig=1,
                notes=f'profanity dalam frasa pendek ("{" ".join(clean)}")',
            )
    return None


# ---- Category 9: Apology --------------------------------------------------
APOLOGY_EXACT = {
    "sry", "sryy", "sryyy", "srry", "sorry", "sorryy", "soryy", "sorri",
    "my bad", "mb", "mybad", "my fault", "mf", "apologies",
    "g sry", "sry g", "srry g", "sry guys", "sorry guys",
}
APOLOGY_STRICT = {x.replace(" ", "") for x in APOLOGY_EXACT}


def cat_apology(s, time, duration, outcome):
    n = norm_full(s)
    if n in APOLOGY_EXACT or norm_strict(s) in APOLOGY_STRICT:
        return make_label(
            "positive", notes=f'permintaan maaf / mengakui kesalahan ("{n}")',
        )
    return None


# ---- Category 10: Acknowledgment / agreement ------------------------------
ACK_EXACT = {
    "k", "kk", "kkk", "ok", "okk", "okay", "okayy",
    "y", "yy", "yes", "yess", "yep", "yea", "yea boi", "ya", "ye", "yeah",
    "n", "no", "nope", "nah", "nope nope",
    "rdy", "rdyy", "ready", "imready", "im ready", "rdy?", "ready?",
    "r", "p", "ts",  # short markers — sering "ready"/"pause"/etc
}
ACK_STRICT = {x.replace(" ", "") for x in ACK_EXACT}
ACK_AMBIG = {"r", "p", "k", "y", "n", "ts", "ya"}  # huruf tunggal ambigu


def cat_ack(s, time, duration, outcome):
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    if n_full in ACK_EXACT or n_strict in ACK_STRICT:
        is_ambig = 1 if n_full in ACK_AMBIG or n_strict in ACK_AMBIG else 0
        return make_label(
            "neutral", is_ambig=is_ambig,
            notes=f'acknowledgment/agreement ("{n_full}"), reaksi singkat',
        )
    return None


# ---- Category 11: Timer / wait calls --------------------------------------
TIMER_EXACT = {
    "sec", "secs", "second", "seconds", "1 sec", "2 sec", "5 sec", "10 sec",
    "30 sec", "30s", "60s", "1s", "2s", "5s", "10s",
    "wait", "w8", "w8t", "wait pls", "wait please",
    "1 min", "2 min", "3 min", "5 min", "min", "minute", "minutes",
    "soon", "almost", "1 sec pls", "sec pls",
}
TIMER_STRICT = {x.replace(" ", "") for x in TIMER_EXACT}


def cat_timer(s, time, duration, outcome):
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    if n_full in TIMER_EXACT or n_strict in TIMER_STRICT:
        return make_label(
            "neutral", notes=f'timer/wait call ("{n_full}"), permintaan jeda waktu',
        )
    return None


# ---- Category 12: Technical / connection issues ---------------------------
TECH_EXACT = {
    "lag", "lags", "lagg", "lagging", "lagged", "lagspike", "lag spike",
    "lag sry", "lag sorry", "lagg sry", "sec lag", "lag sec",
    "mic", "no mic", "mic check", "mic test", "mic?", "mic ok",
    "ping", "high ping", "ping issue", "-ping",
    "crash", "crashed", "client crash", "game crash",
    "pause", "paused", "pause pls", "pause please", "pause?", "unpause",
    "settings", "setting", "settings pls",
    "dc", "disconnect", "disconnected", "reconnect", "rc",
    "fps", "low fps", "lag fps",
    "loading", "load", "loaded",
    # Extension
    "black screen", "blackscreen", "packet loss", "packet", "packets",
    "headset", "mouse", "keyboard", "ds lag", "discord", "discord prob",
    "discord prob sry", "hang", "hanged", "hanging", "freeze", "frozen",
    "rec", "recording", "wc", "internet", "wifi", "router",
    "reset", "reboot", "restart", "ds", "issue", "issues",
    "log in", "login", "log out", "alt tab", "alt-tab",
    "delay", "delays", "delayed", "delaying", "input delay", "input lag",
    "stuttering", "stutter", "stutters", "freezing", "freezed",
}
TECH_STRICT = {x.replace(" ", "") for x in TECH_EXACT}


def cat_tech(s, time, duration, outcome):
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    if n_full in TECH_EXACT or n_strict in TECH_STRICT:
        return make_label(
            "neutral", notes=f'masalah teknis ("{n_full}"), bukan emosional',
        )
    return None


# ---- Category 13: XD/laughter (length-2 yang ketinggalan) -----------------
def cat_xd(s, time, duration, outcome):
    n = norm_strict(s)
    if re.fullmatch(r"x+d+", n):
        return make_label(
            "positive", is_ambig=1,
            notes=f'tawa "xd" variant ("{n}"), amusement / sarcasm ringan',
        )
    return None


# ---- Category 15: Mild emotional reactions --------------------------------
REACT_NEG = {"sad", "aw", "aww", "awww", "oh no", "ohno", "noo", "nooo", "noooo"}
REACT_NEUTRAL_LOSS = {"loss", "lose", "lost"}


def cat_react(s, time, duration, outcome):
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    if n_full in REACT_NEG or n_strict in REACT_NEG:
        return make_label(
            "negative", is_ambig=1,
            notes=f'reaksi negatif emosional ringan ("{n_full}")',
        )
    if n_full in REACT_NEUTRAL_LOSS or n_strict in REACT_NEUTRAL_LOSS:
        return make_label(
            "neutral", is_ambig=1,
            notes=f'akui kekalahan tanpa emosi kuat ("{n_full}")',
        )
    return None


# ---- Compound helpers -----------------------------------------------------
# Forward declarations: TOXIC_FOREIGN didefinisikan di bawah, kita rujuk
# secara nama (Python resolusi runtime).
TOXIC_BLOCKER_BASE = INSULT_WORDS | OBSCENE_STANDALONE


def has_toxic_blocker(s):
    parts = re.findall(r"[a-z]+", s.lower() if isinstance(s, str) else "")
    if any(p in TOXIC_BLOCKER_BASE for p in parts):
        return True
    if any(p in TOXIC_FOREIGN for p in parts):
        return True
    return False


# ---- Category 17-21: Compound matchers ------------------------------------
GG_PREFIX_TOKENS = {"gg", "ggwp", "ggs", "gege", "gw", "gwp"}
GG_PHONETIC = {"gee gee", "gee gee gee", "geegee", "gigi", "gigi gigi", "jiji",
               "jijiji", "gigigi", "geegeege"}
GG_PHONETIC_STRICT = {x.replace(" ", "") for x in GG_PHONETIC}


def cat_gg_compound(s, time, duration, outcome):
    parts = norm_full(s).split()
    if not parts or len(parts) > 6:
        return None
    first = parts[0].strip(",.!?\"'")
    if first not in GG_PREFIX_TOKENS:
        return None
    if has_toxic_blocker(s):
        return None
    if pd.isna(time) or pd.isna(duration) or duration <= 0:
        return make_label("positive", is_jargon=1, is_ambig=1,
                          notes=f'GG compound ("{norm_full(s)}"), timing tidak valid')
    ratio = time / duration
    if ratio >= 0.95:
        return make_label("positive", is_jargon=1,
                          notes=f'GG compound end-of-match ("{norm_full(s)}"), sportsmanship')
    if ratio >= 0.85:
        return make_label("positive", is_jargon=1, is_ambig=1,
                          notes=f'GG compound late-game ("{norm_full(s)}")')
    if ratio < 0:
        return make_label("neutral", is_jargon=1, is_ambig=1,
                          notes=f'GG compound pre-game ("{norm_full(s)}")')
    return make_label("positive", is_jargon=1, is_ambig=1,
                      notes=f'GG compound mid-game ("{norm_full(s)}"), tone ambigu')


def cat_gg_phonetic(s, time, duration, outcome):
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    if n_full in GG_PHONETIC or n_strict in GG_PHONETIC_STRICT:
        return make_label("positive", is_jargon=1,
                          notes=f'phonetic GG variant ("{n_full}")')
    return None


SPORT_PREFIX_TOKENS = {
    "gl", "hf", "wp", "gj", "np", "ty", "thx", "thanks", "tysm", "tyvm",
    "glhf", "gz", "gratz", "congrats", "grats", "respect", "respects",
}
SPORT_PHRASES_REGEX = re.compile(
    r"\b(?:good\s+game|good\s+job|well\s+played|nice\s+game|nice\s+play|nice\s+job|"
    r"good\s+luck|have\s+fun|good\s+try|no\s+worries|no\s+problem|all\s+good|"
    r"thanks?\s+for\s+(?:game|playing|the\s+game)|gl\s+(?:next|further|hf)|"
    r"gn\s+next|hf\s+gl|"
    # Indonesian sportsmanship/encouragement phrases
    r"anjay\s+mabar|yuk\s+mabar|ayok\s+mabar|gas\s+mabar|"
    r"ya\s+ma+p+|yama+p+|"
    r"masuk\s+terus|gas\s+terus|lanjut\s+bro|"
    r"full\s+chill|chill\s+bro|santai\s+bro|sabar\s+(?:bro|bang|bg)|tenang\s+bro|"
    r"takpe\s+(?:bro|bang|bg|gan)|"
    r"nice\s+\w+\s+bro|"
    r"mantap\s+\w+|mantap\s+kali"
    r")\b",
    re.IGNORECASE,
)


def cat_sport_compound(s, time, duration, outcome):
    if not isinstance(s, str):
        return None
    if has_toxic_blocker(s):
        return None
    if SPORT_PHRASES_REGEX.search(s):
        return make_label("positive",
                          notes=f'sportsmanship phrase ("{norm_full(s)}")')
    parts = norm_full(s).split()
    if not parts or len(parts) > 6:
        return None
    clean = [p.strip(",.!?\"'") for p in parts]
    hits = [p for p in clean if p in SPORT_PREFIX_TOKENS]
    if hits:
        return make_label("positive",
                          notes=f'sportsmanship compound (token: {hits[0]}, msg: "{norm_full(s)}")')
    return None


APOLOGY_TOKENS = {"sry", "sryy", "sryyy", "srry", "sorry", "sorryy", "soryy",
                  "mb", "mybad"}


def cat_apology_compound(s, time, duration, outcome):
    parts = norm_full(s).split()
    if not parts or len(parts) > 6:
        return None
    if has_toxic_blocker(s):
        return None
    clean = [p.strip(",.!?\"'") for p in parts]
    hits = [p for p in clean if p in APOLOGY_TOKENS]
    if hits:
        return make_label("positive", is_ambig=1,
                          notes=f'apology compound ("{norm_full(s)}")')
    return None


TECH_TOKENS_BROAD = {
    "lag", "lags", "lagg", "lagging", "crash", "crashed", "pause", "paused",
    "mic", "ping", "fps", "disconnect", "disconnected", "dc", "reconnect",
    "rc", "delay", "delays", "delayed", "freeze", "frozen", "stutter",
    "hang", "hanged", "loading", "load", "rec", "settings", "setting",
}


def cat_tech_compound(s, time, duration, outcome):
    parts = norm_full(s).split()
    if not parts or len(parts) > 6:
        return None
    if has_toxic_blocker(s):
        return None
    clean = [p.strip(",.!?\"'") for p in parts]
    hits = [p for p in clean if p in TECH_TOKENS_BROAD]
    if hits:
        return make_label("neutral",
                          notes=f'tech issue compound (token: {hits[0]}, msg: "{norm_full(s)}")')
    return None


# ---- Category 16: Foreign-language profanity (lolos filter EN) ------------
TOXIC_FOREIGN = {
    # === Indonesian profanity ===
    # Penis/vagina slang
    "kontol", "kontoll", "kontolll", "kntl", "k0ntol",
    "memek", "mmk", "mmek",
    "biji", "bijik",
    # Sex acts
    "ngentot", "ngentod", "ngentodt", "ngetot", "kentu", "ngewe",
    # Animal-based insults
    "anjing", "anjir", "anjr", "anjg", "njir", "anjeng", "anjeg", "anjink",
    "asu", "asuw", "asw", "asoo", "asoooo",
    "babi", "monyet", "monyong", "babiat", "bbi",
    # Asshole/bastard
    "bangsat", "bgst", "bgsd", "bnsdt", "bsndt", "bansat", "bjingan",
    "bajingan", "bjg", "jancok", "jancuk", "jncok", "cuk",
    # Idiot/stupid
    "goblok", "gblk", "goblg", "tolol", "tll", "pekok", "pkk",
    "bobrok", "bobok",
    # Shit
    "tai", "taik", "tahi", "thai",
    # Mouth/talk shit
    "bacot", "bcot", "bct", "bacotan", "ngebacot", "bacotin",
    "bacooot", "bacooootttt", "ngebacooottt", "ngebacotttt", "bcttt",
    "bacod", "bcod", "ngebacod", "bacodin",
    # Mother insults
    "ndasmu", "ndas", "kontolmu", "memekmu", "asumu",
    "pukimak", "puki", "pukima",
    # Mixed
    "bgsdt", "bsd",  # singkatan bangsat
    # === Russian (sering muncul di Dota international) ===
    "blyat", "blat", "blya", "suka", "pidor", "pidoras", "pizdec", "huy",
    "huyna", "ebal", "yebat", "nahuy",
    "loh", "lokh", "uebok", "uebki", "lohi", "huesos",
    "mudak", "mudaki", "duraki", "durak",
    # === Filipino ===
    "puta", "putangina", "putang ina", "tanga", "bobo", "ulol", "gago",
    "pi", "putik", "lintik", "tarantado",
    "kupal", "hayop", "siraulo", "leche", "salbahe", "kamukha",
    # === Spanish ===
    "maricon", "maricone", "marica", "puto", "pendejo", "cabron", "cabrón",
    "joder", "mierda", "carajo", "pinche", "huevon", "chinga", "chingada",
    # === German / Polish / French ===
    "scheisse", "scheise", "arschloch", "wichser",
    "kurwa", "kurwo", "skurwysyn", "jebac", "pierdole",
    "putain", "merde", "salope", "connard", "enculé", "encule",
    # === English additional identity slurs (obscene + identity_hate) ===
    "faggot", "fag", "nigger", "nigga", "chink", "gook", "spic", "kike",
    "tranny", "dyke",
    # === Other ===
    "sob",
}

# Subset yang juga perlu flag tox_identity_hate=1 (slur berbasis identitas)
TOXIC_IDENTITY_SLURS = {
    "maricon", "maricone", "marica",  # Spanish anti-LGBT
    "faggot", "fag", "tranny", "dyke",  # English anti-LGBT
    "nigger", "nigga", "chink", "gook", "spic", "kike",  # racial slurs
}

# Frasa multi-kata yang juga toxic (di-cek via substring/regex)
TOXIC_MULTIWORD_PHRASES = [
    re.compile(r"\banak\s+haram\b", re.IGNORECASE),  # Filipino bastard
    re.compile(r"\bputang\s*ina\b", re.IGNORECASE),  # Filipino strong
    re.compile(r"\bhijo\s+de\s+puta\b", re.IGNORECASE),  # Spanish son of bitch
    re.compile(r"\bpinche\s+puto\b", re.IGNORECASE),
    re.compile(r"\bmother\s*fucker\b|\bmotherfucker\b", re.IGNORECASE),
    re.compile(r"\bson\s+of\s+a?\s*bitch\b|\bsob\b", re.IGNORECASE),
]


def _make_toxic_label(hit, source, identity=False):
    return make_label(
        "negative",
        tox_toxic=1, tox_obscene=1, tox_insult=1,
        tox_identity=1 if identity else 0,
        notes=f'profanity bahasa lain ({source}): "{hit}"' + (
            " — identity slur" if identity else ""
        ),
    )


def cat_toxic_foreign(s, time, duration, outcome):
    if not isinstance(s, str):
        return None
    n = norm_strict(s)
    if n in TOXIC_FOREIGN:
        is_identity = n in TOXIC_IDENTITY_SLURS
        return _make_toxic_label(n, "lolos filter EN", identity=is_identity)
    parts = norm_full(s).split()
    if 1 <= len(parts) <= 12:
        clean = [p.strip(",.!?\"'") for p in parts]
        hits = [p for p in clean if p in TOXIC_FOREIGN]
        if hits:
            is_identity = any(h in TOXIC_IDENTITY_SLURS for h in hits)
            return _make_toxic_label(hits[0], "dalam pesan", identity=is_identity)
    # Regex-fuzzy untuk variasi typo / repetisi huruf (mis. NGENTOOOD)
    fuzzy_hit = _indo_toxic_regex_match(s)
    if fuzzy_hit:
        return _make_toxic_label(fuzzy_hit, "regex variant", identity=False)
    # Frasa multi-kata
    for rgx in TOXIC_MULTIWORD_PHRASES:
        m = rgx.search(s)
        if m:
            phrase = m.group(0).lower()
            return _make_toxic_label(phrase, "frasa multi-kata", identity=False)
    return None


# Regex untuk variasi typo/repetisi huruf — hati-hati supaya tidak false positive
# (mis. "tai" terlalu pendek dan ambigu — hanya match "tahi"/"taik" eksplisit).
_INDO_TOXIC_PATTERNS = [
    re.compile(r"\bng[ae]nt[o0]+[dt]+\b", re.IGNORECASE),
    re.compile(r"\bk[o0]nt[o0]l+\b", re.IGNORECASE),
    re.compile(r"\bm[ae]m[ae]k+\b", re.IGNORECASE),
    re.compile(r"\bg[o0]bl?[o0]k+\b", re.IGNORECASE),
    re.compile(r"\bt[o0]l[o0]l+\b", re.IGNORECASE),
    re.compile(r"\banjing+\b", re.IGNORECASE),
    re.compile(r"\banj[ir]+r?\b", re.IGNORECASE),
    re.compile(r"\bbangs?[ad]t+\b|\bbgs[dt]+\b", re.IGNORECASE),
    re.compile(r"\bj[ae]nc[ou]k+\b", re.IGNORECASE),
    re.compile(r"\bbac[oa]+[td]+\b", re.IGNORECASE),
    re.compile(r"\bngebac[oa]+[td]+\b", re.IGNORECASE),
    re.compile(r"\banj[ei]ng+\b|\banj[ei]g+\b", re.IGNORECASE),
    re.compile(r"\bp[uo]kim[ae]k+\b", re.IGNORECASE),
    re.compile(r"\bpekok+\b", re.IGNORECASE),
    re.compile(r"\btahi+\b|\btaik+\b", re.IGNORECASE),  # eksplisit, bukan "tak"
    re.compile(r"\bbabi+\b", re.IGNORECASE),
    re.compile(r"\bmonyet+\b|\bmonyong+\b", re.IGNORECASE),
]


def _indo_toxic_regex_match(s):
    for rgx in _INDO_TOXIC_PATTERNS:
        m = rgx.search(s)
        if m:
            return m.group(0)
    return None


# ---- Category 14: Sportsmanship REPETITION (glgl, hfhf, gngn) -------------
def cat_sport_repeat(s, time, duration, outcome):
    n = norm_strict(s)
    if re.fullmatch(r"(hf){2,6}", n):
        return make_label("positive", notes=f'sportsmanship "have fun" repetisi ("{n}")')
    if re.fullmatch(r"(gl){2,6}", n):
        return make_label("positive", notes=f'sportsmanship "good luck" repetisi ("{n}")')
    if re.fullmatch(r"(gn){2,6}", n):
        return make_label("positive", notes=f'sportsmanship "good night" repetisi ("{n}")')
    if re.fullmatch(r"(gj){2,6}", n):
        return make_label("positive", notes=f'sportsmanship "good job" repetisi ("{n}")')
    if re.fullmatch(r"(ty){2,6}|(thx){2,6}", n):
        return make_label("positive", notes=f'sportsmanship "thanks" repetisi ("{n}")')
    return None


# ---- Category 8: Strategic coordination calls -----------------------------
COORD_EXACT = {
    # Retreat
    "back", "b", "bb", "backk", "back back", "go back", "fall back", "retreat",
    # Missing / warning
    "miss", "mia", "ss", "ss mid", "ss top", "ss bot", "mid miss", "top miss",
    "bot miss", "missing", "mid missing", "top missing", "bot missing",
    "care", "careful", "beware", "watchout", "watch out",
    # Smoke
    "smoke", "smoke up", "smoking", "smoke now", "smoke?",
    # Push / defend
    "push", "pushing", "push mid", "push top", "push bot", "push now",
    "def", "defend", "defend base", "hold", "hold base", "farm",
    # Gank
    "gank", "ganking", "gnk", "gank mid", "gank top", "gank bot",
    # Roshan
    "rosh", "roshan", "aegis", "cheese", "rosh now", "rosh?",
    # Rune
    "rune", "runes", "top rune", "bot rune",
    # Tower / objective
    "tower", "tower mid", "tower top", "tower bot",
    "t1", "t2", "t3", "t4", "racks", "rax", "mid rax", "top rax", "bot rax",
    # Lane calls
    "mid", "top", "bot", "bottom", "jungle", "safelane", "offlane", "off",
    "safe", "mid lane", "top lane", "bot lane", "bottom lane",
    # Items (sering muncul)
    "bkb", "manta", "bf", "battlefury", "vlad", "pipe", "meka", "eul",
    "force", "glimmer", "lotus", "shard", "aghs", "agh", "aghanim", "midas",
    "octarine", "aether", "hex", "blink", "drum", "wand", "stick",
    # Status / cooldown
    "oom", "low", "low hp", "low mana", "no mana", "no ulti", "no ult",
    "ult", "ulti", "ulty", "have ult", "cd", "cooldown",
    # Buyback
    "buyback", "bo", "no bb", "have bb", "no buyback",
}
COORD_STRICT = {x.replace(" ", "") for x in COORD_EXACT}
# Subset yang punya makna non-Dota signifikan → mark is_ambiguous=1
COORD_AMBIG_TERMS = {
    "low", "back", "b", "bb", "safe", "off", "hold", "def", "bo", "farm",
}


def cat_coord(s, time, duration, outcome):
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    if n_full in COORD_EXACT or n_strict in COORD_STRICT:
        is_ambig = 1 if n_full in COORD_AMBIG_TERMS or n_strict in COORD_AMBIG_TERMS else 0
        return make_label(
            "neutral", is_jargon=1, is_ambig=is_ambig,
            notes=f'koordinasi taktis ("{n_full}"), Dota jargon strategis',
        )
    return None


# ---- Category 7: Initiation / coordination calls ---------------------------
# Single "g" = "go" initiation, bukan shorthand "gg". Plus go variants.
INIT_EXACT = {
    "g", "go", "goo", "gooo", "now", "init", "initiate", "engage",
    "go go", "go go go", "go now", "lets go", "let's go", "letsgo",
    "go!", "go?",  # punct sudah di-normalize tapi simpan sbg backup
}
INIT_STRICT = {x.replace(" ", "") for x in INIT_EXACT}


def cat_init(s, time, duration, outcome):
    n_full = norm_full(s)
    n_strict = norm_strict(s)
    if n_full in INIT_EXACT or n_strict in INIT_STRICT:
        # "g" lebih ambigu (bisa typo "gg" — meskipun jarang); lainnya cukup yakin.
        is_ambig = 1 if n_strict == "g" else 0
        return make_label(
            "neutral", is_jargon=1, is_ambig=is_ambig,
            notes=f'initiation/coordination call ("{n_full}"), Dota jargon untuk engage/attack',
        )
    return None


# Urutan match (pertama yang match dipakai). Specific dulu, generic terakhir.
CATEGORIES = [
    ("toxic_foreign", cat_toxic_foreign),  # cek pertama supaya tidak salah ke ack/coord
    ("insult", cat_insult),
    ("ez", cat_ez),
    ("ff", cat_ff),
    ("obscene", cat_obscene),
    ("init", cat_init),
    ("coord", cat_coord),
    ("apology", cat_apology),
    ("ack", cat_ack),
    ("timer", cat_timer),
    ("tech", cat_tech),
    ("xd", cat_xd),
    ("sport_rep", cat_sport_repeat),
    ("sport", cat_sport),
    # Compound matchers (lebih longgar — cek toxic blocker dulu)
    ("gg_phonetic", cat_gg_phonetic),
    ("gg_compound", cat_gg_compound),
    ("sport_compound", cat_sport_compound),
    ("apology_compound", cat_apology_compound),
    ("tech_compound", cat_tech_compound),
    ("react", cat_react),
    ("punct", cat_punct),
]


def main():
    sample_path = Path("data/gold/sample.csv")
    df = pd.read_csv(sample_path)
    print(f'Total: {len(df):,}  |  sudah dilabel: {(df["annotator_id"]=="AI").sum():,}')

    for col in ["annotator_id", "sentiment", "notes"]:
        df[col] = df[col].astype("object")
    for col in [
        "tox_toxic", "tox_severe_toxic", "tox_obscene", "tox_threat",
        "tox_insult", "tox_identity_hate", "is_dota_jargon", "is_ambiguous",
    ]:
        df[col] = df[col].astype("object")

    unlabeled_mask = df["annotator_id"].fillna("") == ""
    n_per_cat = {name: 0 for name, _ in CATEGORIES}

    for i in df[unlabeled_mask].index:
        key = df.at[i, "key"]
        time = df.at[i, "time"]
        duration = df.at[i, "duration"]
        outcome = df.at[i, "match_outcome_for_player"]
        for cat_name, fn in CATEGORIES:
            lbl = fn(key, time, duration, outcome)
            if lbl is not None:
                df.at[i, "annotator_id"] = "AI"
                for k, v in lbl.items():
                    df.at[i, k] = v
                n_per_cat[cat_name] += 1
                break

    df.to_csv(sample_path, index=False, encoding="utf-8")

    print("\n=== Hasil per kategori ===")
    for cat, n in n_per_cat.items():
        print(f"  {cat:10s}: {n:5,}")
    total_new = sum(n_per_cat.values())
    print(f'  {"TOTAL":10s}: {total_new:5,}')
    total_cumulative = (df["annotator_id"] == "AI").sum()
    print(f"\nCumulative dilabel: {total_cumulative:,}/{len(df):,} "
          f"({100*total_cumulative/len(df):.1f}%)")
    print(f"Sisa belum dilabel: {(len(df) - total_cumulative):,}")

    print("\n=== Distribusi sentiment (cumulative) ===")
    print(df.loc[df["annotator_id"] == "AI", "sentiment"].value_counts())
    print("\n=== Distribusi toxicity flags (cumulative) ===")
    for tcol in ["tox_toxic", "tox_obscene", "tox_insult"]:
        n = (df[tcol] == 1).sum()
        print(f"  {tcol}: {n}")

    print("\n=== Sample per kategori (3 baris) ===")
    cat_filter = {
        "toxic_foreign": "profanity bahasa lain",
        "insult": "penghinaan",
        "ez": "ez/easy",
        "ff": "forfeit|defeatism|forfeit ke lawan",
        "obscene": "profanity",
        "init": "initiation/coordination",
        "coord": "koordinasi taktis",
        "apology": "permintaan maaf",
        "ack": "acknowledgment/agreement",
        "timer": "timer/wait call",
        "tech": "masalah teknis",
        "xd": "tawa .xd",
        "sport_rep": "sportsmanship .*repetisi",
        "sport": "sportsmanship/sopan",
        "react": "reaksi negatif emosional|akui kekalahan",
        "gg_phonetic": "phonetic GG variant",
        "gg_compound": "GG compound",
        "sport_compound": "sportsmanship phrase|sportsmanship compound",
        "apology_compound": "apology compound",
        "tech_compound": "tech issue compound",
        "punct": "punctuation-only",
    }
    for cat_name, _ in CATEGORIES:
        if n_per_cat[cat_name] == 0:
            continue
        sub = df[df["notes"].astype(str).str.contains(cat_filter[cat_name], na=False, regex=True)]
        print(f"\n--- {cat_name.upper()} (n={n_per_cat[cat_name]}) ---")
        print(sub[["key", "match_outcome_for_player", "sentiment",
                   "tox_toxic", "tox_obscene", "tox_insult", "notes"]]
              .head(3).to_string(index=False, max_colwidth=70))


if __name__ == "__main__":
    main()
