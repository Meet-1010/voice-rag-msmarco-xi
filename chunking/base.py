"""Chunker interface and the tokenisation primitives every strategy shares.

Chunk sizes are measured in real e5 tokens, not words or characters. That matters:
multilingual-e5-small truncates at 512 tokens, and Devanagari/Gujarati text costs
roughly 2-3x more tokens per character than English under this tokenizer, so a
character-based budget silently produces chunks of wildly different real sizes
across languages.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache

MODEL_MAX_TOKENS = 512


@dataclass(slots=True)
class Passage:
    passage_id: str
    doc_id: str
    text: str
    lang: str
    query_id: int = -1
    is_selected: int = 0

    @classmethod
    def from_json(cls, d: dict) -> "Passage":
        return cls(d["passage_id"], d["doc_id"], d["text"], d["lang"],
                   d.get("query_id", -1), d.get("is_selected", 0))


@dataclass(slots=True)
class Piece:
    """One split of a passage. `context` is what the generator sees when it differs
    from what we index (parent-child)."""
    text: str
    context: str | None = None


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    passage_id: str
    doc_id: str
    text: str
    lang: str
    query_id: int = -1
    is_selected: int = 0
    context: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "passage_id": self.passage_id, "doc_id": self.doc_id,
            "text": self.text, "lang": self.lang, "query_id": self.query_id,
            "is_selected": self.is_selected,
            # Only carry context when it actually differs; it doubles index size otherwise.
            "context": self.context if self.context != self.text else "",
            "meta": self.meta,
        }


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained("intfloat/multilingual-e5-small")


def token_offsets(text: str) -> list[tuple[int, int]]:
    """Character spans of each token, so we can slice the original text losslessly."""
    return _tokenizer().encode(text, add_special_tokens=False).offsets


def n_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def slice_by_tokens(text: str, offsets: list[tuple[int, int]], lo: int, hi: int) -> str:
    """Rebuild a substring from a token range. Slicing the source beats decoding the
    token ids back, which loses the original spacing and normalises characters."""
    window = offsets[lo:hi]
    if not window:
        return ""
    return text[window[0][0]:window[-1][1]].strip()


# Danda and double danda terminate sentences in Devanagari and are also used in
# Gujarati prose; Latin punctuation still shows up in code-mixed rows.
_SENT_SPLIT = re.compile(r"(?<=[।॥.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    out = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return out or ([text.strip()] if text.strip() else [])


def pack(units: list[str], budget: int, overlap: int = 0, joiner: str = " ") -> list[str]:
    """Greedily pack text units into token-budgeted chunks, carrying `overlap` tokens
    of tail context into the next chunk."""
    if not units:
        return []
    sizes = [n_tokens(u) for u in units]
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0

    for unit, size in zip(units, sizes):
        # A single oversized unit cannot be packed; hard-split it on token boundaries.
        if size > budget:
            if cur:
                chunks.append(joiner.join(cur))
                cur, cur_n = [], 0
            offs = token_offsets(unit)
            for start in range(0, len(offs), budget):
                piece = slice_by_tokens(unit, offs, start, start + budget)
                if piece:
                    chunks.append(piece)
            continue
        if cur_n + size > budget and cur:
            chunks.append(joiner.join(cur))
            if overlap > 0:
                tail, tail_n = [], 0
                for u, s in zip(reversed(cur), reversed([n_tokens(c) for c in cur])):
                    if tail_n + s > overlap:
                        break
                    tail.insert(0, u)
                    tail_n += s
                cur, cur_n = tail, tail_n
            else:
                cur, cur_n = [], 0
        cur.append(unit)
        cur_n += size

    if cur:
        chunks.append(joiner.join(cur))
    return [c for c in (c.strip() for c in chunks) if c]


def enforce_limit(text: str, limit: int = MODEL_MAX_TOKENS) -> list[str]:
    """Guarantee no indexed chunk exceeds the encoder's context window.

    Anything longer is truncated by the encoder without raising, so the tail of the
    chunk silently stops being searchable. A handful of Hindi passages in this
    corpus are single sentences of 2000+ tokens and hit exactly that. Slicing on
    character offsets and re-tokenising can land a token or two over the boundary,
    hence the margin.
    """
    if len(text) <= limit:  # tokens never exceed characters, so this is a safe skip
        return [text]
    offs = token_offsets(text)
    if len(offs) <= limit:
        return [text]
    step = limit - 8
    return [s for s in (slice_by_tokens(text, offs, i, i + step)
                        for i in range(0, len(offs), step)) if s]


class Chunker(ABC):
    name = "base"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    @abstractmethod
    def split(self, p: Passage) -> list[Piece]:
        ...

    def chunk(self, p: Passage) -> list[Chunk]:
        out = []
        i = 0
        for piece in self.split(p):
            text = piece.text.strip()
            if not text:
                continue
            for part in enforce_limit(text):
                out.append(Chunk(
                    chunk_id=f"{self.name}:{p.passage_id}:{i}",
                    passage_id=p.passage_id, doc_id=p.doc_id, text=part, lang=p.lang,
                    query_id=p.query_id, is_selected=p.is_selected,
                    context=(piece.context or part).strip(),
                ))
                i += 1
        return out

    def chunk_all(self, passages) -> list[Chunk]:
        out: list[Chunk] = []
        for p in passages:
            out.extend(self.chunk(p))
        return out
