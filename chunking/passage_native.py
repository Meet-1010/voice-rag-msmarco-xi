from __future__ import annotations

from .base import MODEL_MAX_TOKENS, Chunker, Passage, Piece, slice_by_tokens, token_offsets


class PassageNativeChunker(Chunker):
    """Treat each MSMARCO passage as the atomic unit.

    The corpus is already human-curated into retrieval-sized passages, so any
    splitting we do is arguably damage. The one thing we must still handle is the
    encoder's 512-token ceiling: passages past it would be silently truncated,
    which loses the tail of the passage without any warning.
    """

    name = "passage_native"

    def split(self, p: Passage) -> list[Piece]:
        offs = token_offsets(p.text)
        if len(offs) <= MODEL_MAX_TOKENS:
            return [Piece(p.text)]
        return [Piece(slice_by_tokens(p.text, offs, i, i + MODEL_MAX_TOKENS))
                for i in range(0, len(offs), MODEL_MAX_TOKENS)]
