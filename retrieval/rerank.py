"""Optional cross-encoder rerank.

Off by default. Two honest reasons, both measured rather than assumed:

- It costs 40-70ms on CPU for 20 candidates, which is a third of the entire 200ms
  core budget for a gain that only shows up when fusion ordering was already close.
- ms-marco-MiniLM is English-only. Running it on Devanagari input produces
  confident nonsense, so we skip reranking entirely for non-English queries rather
  than silently degrading them. The multilingual alternative is 1.1GB and does not
  fit the free-tier footprint.

See the README latency table for both modes measured end to end.
"""
from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, cfg: dict):
        rc = cfg["retrieval"]["rerank"]
        self.enabled = rc.get("enabled", False)
        self.top_n = rc.get("top_n", 20)
        self.model_name = rc["model"]
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder(model_name=self.model_name)
            list(self._model.rerank("warmup", ["warmup passage"]))
        return self._model

    def applies_to(self, lang: str | None) -> bool:
        return self.enabled and lang == "en"

    def rerank(self, query: str, hits: list[dict], lang: str | None = None) -> list[dict]:
        if not self.applies_to(lang) or len(hits) < 2:
            return hits
        head, tail = hits[:self.top_n], hits[self.top_n:]
        scores = list(self._load().rerank(query, [h["text"] for h in head]))
        for h, s in zip(head, scores):
            h["rerank_score"] = float(s)
        head.sort(key=lambda h: -h["rerank_score"])
        return head + tail
