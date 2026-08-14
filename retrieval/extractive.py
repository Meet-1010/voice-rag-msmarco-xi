"""Answer without calling an LLM when the passage already contains the answer.

This is the main reason the pipeline lands inside the latency target rather than
merely near it. MSMARCO passages were selected by humans as answering their query,
so the answer span is usually present verbatim. Picking the right sentence costs
one embedding call over a handful of sentences - single-digit milliseconds - versus
several hundred for a generative round trip.

We only take this path when retrieval was confident AND one sentence clearly beats
the others. Otherwise we fall through to generation rather than guessing.
"""
from __future__ import annotations

import numpy as np

from chunking.base import sentences


class ExtractiveAnswerer:
    def __init__(self, cfg: dict, embedder):
        ec = cfg["extractive"]
        self.enabled = ec.get("enabled", True)
        self.min_score = ec["min_score"]
        self.max_chars = ec["max_span_chars"]
        self.min_span_similarity = ec.get("min_span_similarity", 0.80)
        self.embedder = embedder

    def try_answer(self, query: str, q_vec: np.ndarray, hits: list[dict]) -> dict | None:
        if not self.enabled or not hits:
            return None

        top = max(hits, key=lambda h: h.get("dense_score", 0.0))
        score = top.get("dense_score", 0.0)
        if score < self.min_score:
            return None

        # Prefer the parent passage when parent-child chunking is active: the child
        # is deliberately too small to be a self-contained answer.
        source = top.get("context") or top["text"]
        sents = sentences(source)
        if not sents:
            return None
        if len(sents) == 1:
            span, span_sim = sents[0], score
        else:
            vecs = self.embedder.passages(sents)
            sims = vecs @ q_vec
            best = int(np.argmax(sims))
            span_sim = float(sims[best])
            if span_sim < self.min_span_similarity:
                return None
            span = sents[best]
            # Carry the following sentence when it is nearly as relevant; MSMARCO
            # answers frequently straddle two sentences.
            if best + 1 < len(sents) and float(sims[best + 1]) >= span_sim - 0.04:
                joined = f"{span} {sents[best + 1]}"
                if len(joined) <= self.max_chars:
                    span = joined

        span = span.strip()
        if len(span) > self.max_chars:
            span = span[:self.max_chars].rsplit(" ", 1)[0] + "..."
        return {
            "answer": span,
            "confidence": round(min(1.0, (score + span_sim) / 2), 4),
            "hit": top,
            "span_similarity": round(span_sim, 4),
        }
