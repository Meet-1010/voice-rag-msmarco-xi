from __future__ import annotations

from .base import Chunker, Passage, Piece, pack, sentences


class ParentChildChunker(Chunker):
    """Retrieve on small precise units, generate on the whole passage.

    Small chunks embed sharply because a 180-token span is about one idea, so the
    query vector is not diluted by the rest of the passage. But a 180-token span is
    often too little for the model to answer from, so the chunk we match on and the
    text we hand the generator are deliberately different objects: `text` is the
    child, `context` is the parent passage.
    """

    name = "parent_child"

    def split(self, p: Passage) -> list[Piece]:
        size = self.cfg.get("child_size", 180)
        overlap = self.cfg.get("child_overlap", 30)
        children = pack(sentences(p.text), size, overlap)
        if not children:
            return [Piece(p.text, p.text)]
        return [Piece(c, p.text) for c in children]
