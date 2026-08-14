from __future__ import annotations

from .base import Chunker, Passage, Piece, n_tokens, pack, sentences, slice_by_tokens, token_offsets


class RecursiveChunker(Chunker):
    """Split on the strongest structural boundary that works, then fall back.

    Paragraph -> line -> sentence -> token window. The point is that we only reach
    for a weaker separator when the stronger one leaves a piece that still does not
    fit, so most chunks end on a boundary a human would also have picked.
    """

    name = "recursive"

    def split(self, p: Passage) -> list[Piece]:
        size = self.cfg.get("size", 384)
        overlap = self.cfg.get("overlap", 48)
        units = self._descend(p.text, size)
        return [Piece(c) for c in pack(units, size, overlap)]

    def _descend(self, text: str, budget: int, depth: int = 0) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if n_tokens(text) <= budget:
            return [text]

        if depth == 0:
            parts = [s for s in text.split("\n\n") if s.strip()]
        elif depth == 1:
            parts = [s for s in text.split("\n") if s.strip()]
        elif depth == 2:
            parts = sentences(text)
        else:
            # Nothing structural left to exploit; cut on token boundaries.
            offs = token_offsets(text)
            return [s for s in (slice_by_tokens(text, offs, i, i + budget)
                                for i in range(0, len(offs), budget)) if s]

        # The separator did not actually divide anything, so skip a level rather
        # than recursing on an identical string.
        if len(parts) <= 1:
            return self._descend(text, budget, depth + 1)

        out: list[str] = []
        for part in parts:
            out.extend(self._descend(part, budget, depth + 1))
        return out
