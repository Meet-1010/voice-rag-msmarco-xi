from __future__ import annotations

from .base import Chunker, Passage, Piece, slice_by_tokens, token_offsets


class FixedChunker(Chunker):
    """Hard token cuts at a fixed stride. The baseline the task explicitly warns
    against shipping alone, kept so the comparison table has an honest floor."""

    name = "fixed"

    def split(self, p: Passage) -> list[Piece]:
        size = self.cfg.get("size", 256)
        offs = token_offsets(p.text)
        if len(offs) <= size:
            return [Piece(p.text)]
        return [Piece(slice_by_tokens(p.text, offs, i, i + size))
                for i in range(0, len(offs), size)]
