from __future__ import annotations

import re
import unicodedata

from .base import Chunker, Passage, Piece, pack

# Script blocks we actually see in this corpus.
_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Gujarati": (0x0A80, 0x0AFF),
}

# Devanagari and Gujarati terminate on danda; Latin sentences end on . ! ?
# Splitting Indic text on "." alone breaks on decimals and abbreviations far more
# often than it finds a real sentence end, so the rules are kept per script.
_INDIC_SPLIT = re.compile(r"(?<=[।॥])\s*|\n+")
_LATIN_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIXED_SPLIT = re.compile(r"(?<=[।॥])\s*|(?<=[.!?])\s+|\n+")


def detect_script(text: str) -> str:
    counts = {"Devanagari": 0, "Gujarati": 0, "Latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for name, (lo, hi) in _RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
                break
        else:
            if "LATIN" in unicodedata.name(ch, ""):
                counts["Latin"] += 1
    total = sum(counts.values())
    if not total:
        return "Unknown"
    script, n = max(counts.items(), key=lambda kv: kv[1])
    # Hinglish rows are common; call it mixed rather than forcing one rule set.
    return script if n / total >= 0.75 else "Mixed"


class IndicAwareChunker(Chunker):
    """Script-aware sentence splitting plus script recorded as chunk metadata.

    Two things a generic splitter gets wrong on this corpus: it never treats the
    danda as a terminator, and it applies Latin abbreviation rules to Devanagari.
    """

    name = "indic_aware"

    def split(self, p: Passage) -> list[Piece]:
        size = self.cfg.get("size", 400)
        overlap = self.cfg.get("overlap", 40)

        script = detect_script(p.text)
        if script in ("Devanagari", "Gujarati"):
            pattern = _INDIC_SPLIT
        elif script == "Latin":
            pattern = _LATIN_SPLIT
        else:
            pattern = _MIXED_SPLIT

        sents = [s.strip() for s in pattern.split(p.text) if s and s.strip()]
        if not sents:
            sents = [p.text]
        return [Piece(c) for c in pack(sents, size, overlap)]

    def chunk(self, p: Passage):
        chunks = super().chunk(p)
        script = detect_script(p.text)
        for c in chunks:
            c.meta["script"] = script
        return chunks
