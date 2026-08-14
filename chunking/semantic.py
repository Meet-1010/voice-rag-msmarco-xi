from __future__ import annotations

import numpy as np

from .base import Chunker, Passage, Piece, n_tokens, sentences


class SemanticChunker(Chunker):
    """Cut where the topic actually moves.

    Embed each sentence, measure the distance to its neighbour, and break at the
    largest distances rather than at a fixed stride. The threshold is a percentile
    of the distances *within this passage*, not a global constant, because absolute
    cosine distance is not comparable across languages under a multilingual encoder.

    NOTE: this is by far the most expensive strategy to build, since it needs one
    embedding pass over every sentence before it can index anything.
    """

    name = "semantic"

    def __init__(self, cfg: dict | None = None, embedder=None):
        super().__init__(cfg)
        if embedder is None:
            raise ValueError("SemanticChunker needs an embedder")
        self.embedder = embedder

    def split(self, p: Passage) -> list[Piece]:
        max_size = self.cfg.get("max_size", 512)
        pct = self.cfg.get("breakpoint_percentile", 78)
        min_sents = self.cfg.get("min_sentences", 2)

        sents = sentences(p.text)
        if len(sents) < min_sents:
            return [Piece(p.text)]

        vecs = self.embedder.passages(sents)
        dist = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
        if dist.size == 0:
            return [Piece(p.text)]
        cut_at = float(np.percentile(dist, pct))

        pieces: list[str] = []
        cur = [sents[0]]
        cur_n = n_tokens(sents[0])
        for i, s in enumerate(sents[1:]):
            s_n = n_tokens(s)
            # Break on a genuine topic shift, or when we would overflow the encoder.
            if (dist[i] > cut_at or cur_n + s_n > max_size) and cur:
                pieces.append(" ".join(cur))
                cur, cur_n = [], 0
            cur.append(s)
            cur_n += s_n
        if cur:
            pieces.append(" ".join(cur))
        return [Piece(c) for c in pieces if c.strip()]

    def chunk_all(self, passages) -> list:
        """Batch every sentence in the corpus through the encoder in one pass.

        Per-passage embedding calls dominate build time otherwise; this is the
        difference between a few minutes and the better part of an hour.
        """
        passages = list(passages)
        per_passage = [sentences(p.text) for p in passages]
        flat = [s for sents in per_passage for s in sents]
        if not flat:
            return []
        all_vecs = self.embedder.passages(flat)

        out = []
        cursor = 0
        for p, sents in zip(passages, per_passage):
            vecs = all_vecs[cursor:cursor + len(sents)]
            cursor += len(sents)
            out.extend(self._chunk_with(p, sents, vecs))
        return out

    def _chunk_with(self, p: Passage, sents, vecs):
        cached = _Precomputed(sents, vecs)
        original, self.embedder = self.embedder, cached
        try:
            return self.chunk(p)
        finally:
            self.embedder = original


class _Precomputed:
    """Lets split() reuse vectors we already computed in the batch pass."""

    def __init__(self, sents, vecs):
        self._map = {s: v for s, v in zip(sents, vecs)}
        self._fallback = vecs

    def passages(self, texts):
        return np.asarray([self._map[t] for t in texts], dtype=np.float32)
