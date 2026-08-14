from __future__ import annotations

from .base import Chunker, Passage, Piece, slice_by_tokens, token_offsets


class SlidingChunker(Chunker):
    """Overlapping token windows so an answer straddling a boundary still lands
    intact inside at least one chunk."""

    name = "sliding"

    def split(self, p: Passage) -> list[Piece]:
        size = self.cfg.get("size", 512)
        overlap = self.cfg.get("overlap", 64)
        stride = max(1, size - overlap)

        offs = token_offsets(p.text)
        if len(offs) <= size:
            return [Piece(p.text)]

        pieces = []
        for start in range(0, len(offs), stride):
            text = slice_by_tokens(p.text, offs, start, start + size)
            if text:
                pieces.append(Piece(text))
            # Trailing window already reached the end; another stride would only
            # re-emit a suffix we have covered.
            if start + size >= len(offs):
                break
        return pieces
