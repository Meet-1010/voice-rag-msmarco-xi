from __future__ import annotations

from .base import Chunk, Chunker, Passage, Piece
from .fixed import FixedChunker
from .indic_aware import IndicAwareChunker
from .parent_child import ParentChildChunker
from .passage_native import PassageNativeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sliding import SlidingChunker

STRATEGIES = {
    "fixed": FixedChunker,
    "sliding": SlidingChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "passage_native": PassageNativeChunker,
    "parent_child": ParentChildChunker,
    "indic_aware": IndicAwareChunker,
}

# Only this one needs to embed while it splits.
NEEDS_EMBEDDER = {"semantic"}


def build(name: str, cfg: dict, embedder=None) -> Chunker:
    if name not in STRATEGIES:
        raise KeyError(f"unknown chunking strategy {name!r}; have {sorted(STRATEGIES)}")
    params = (cfg.get("chunking") or {}).get(name, {})
    if name in NEEDS_EMBEDDER:
        return STRATEGIES[name](params, embedder=embedder)
    return STRATEGIES[name](params)


__all__ = ["STRATEGIES", "NEEDS_EMBEDDER", "build", "Chunk", "Chunker", "Passage", "Piece"]
