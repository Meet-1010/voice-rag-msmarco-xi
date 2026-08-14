"""Last gate: is the answer we are about to state actually supported by what we
retrieved?

Two cheap signals rather than an NLI model, because a DeBERTa-scale entailment
model costs more than the entire rest of the core budget:

- embedding similarity between answer and context, which catches an answer that
  drifted to a different topic
- token overlap, which catches fluent paraphrase that invented specifics. An
  answer full of numbers and names absent from the context scores low here even
  when it embeds close.

Neither alone is sufficient; an answer must clear both.
"""
from __future__ import annotations

import re

import numpy as np

from harness.schemas import ReasonCode
from index.bm25 import tokenize

_SENT = re.compile(r"(?<=[।॥.!?])\s+")
# Function words would inflate overlap regardless of whether the content matches.
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and", "or", "for",
    "on", "it", "that", "this", "with", "as", "by", "at", "from", "be", "has", "have",
    "है", "हैं", "का", "की", "के", "में", "और", "से", "को", "यह", "एक", "पर",
    "છે", "નો", "ની", "ના", "માં", "અને", "થી", "ને", "આ", "એક", "પર",
}


class GroundingGuard:
    def __init__(self, cfg: dict, embedder):
        gc = cfg["guardrails"]["grounding"]
        self.min_similarity = gc["min_similarity"]
        self.min_overlap = gc["min_token_overlap"]
        self.embedder = embedder

    @staticmethod
    def token_overlap(answer: str, context: str) -> float:
        a = {t for t in tokenize(answer) if t not in _STOP and len(t) > 1}
        c = {t for t in tokenize(context) if t not in _STOP and len(t) > 1}
        if not a:
            return 0.0
        return len(a & c) / len(a)

    def similarity(self, answer: str, context: str) -> float:
        vecs = self.embedder.passages([answer, context])
        return float(vecs[0] @ vecs[1])

    def check(self, answer: str, contexts: list[str]) -> tuple[ReasonCode | None, dict]:
        if not answer.strip():
            return ReasonCode.UNGROUNDED_OUTPUT, {"reason": "empty answer"}
        context = "\n".join(contexts)
        sim = self.similarity(answer, context)
        overlap = self.token_overlap(answer, context)
        metrics = {"similarity": round(sim, 4), "token_overlap": round(overlap, 4),
                   "min_similarity": self.min_similarity, "min_token_overlap": self.min_overlap}
        if sim < self.min_similarity or overlap < self.min_overlap:
            return ReasonCode.UNGROUNDED_OUTPUT, metrics
        return None, metrics

    def enforce_citations(self, answer: str, contexts: list[str],
                          per_sentence_overlap: float = 0.12) -> tuple[str, list[str]]:
        """Drop sentences that no retrieved passage supports.

        A partially-hallucinated answer usually has most sentences grounded and one
        invented. Stripping just that sentence is more useful than refusing the
        whole response.
        """
        sents = [s.strip() for s in _SENT.split(answer) if s.strip()]
        if len(sents) <= 1:
            return answer, []
        kept, dropped = [], []
        for s in sents:
            if any(self.token_overlap(s, c) >= per_sentence_overlap for c in contexts):
                kept.append(s)
            else:
                dropped.append(s)
        # Never strip everything; that turns a hedge into a blank response.
        return (" ".join(kept), dropped) if kept else (answer, [])
