"""Text preprocessing for Dota 2 in-game chat.

Public API:
    from src.preprocessing import translate_slang, DOTA_SLANG
"""

from .slang import (
    DOTA_SLANG,
    PHRASE_SLANG,
    SLANG_BY_CATEGORY,
    slang_coverage,
    translate_slang,
)

__all__ = [
    "DOTA_SLANG",
    "PHRASE_SLANG",
    "SLANG_BY_CATEGORY",
    "translate_slang",
    "slang_coverage",
]
