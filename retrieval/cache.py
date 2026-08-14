"""Semantic answer cache.

An exact-string cache is nearly useless in front of speech input: two people asking
the same thing produce different transcripts, and the same person asking twice
produces different transcripts. Matching on the query embedding instead catches
paraphrases, and it is the single cheapest way to pull P50 down.
"""
from __future__ import annotations

import time

import numpy as np


class SemanticCache:
    def __init__(self, cfg: dict, dim: int):
        cc = cfg["cache"]
        self.enabled = cc.get("enabled", True)
        self.threshold = cc["similarity"]
        self.max_entries = cc["max_entries"]
        self.ttl = cc["ttl_seconds"]
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._entries: list[dict] = []
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, q_vec: np.ndarray, lang: str | None = None):
        if not self.enabled or not self._entries:
            self.misses += 1
            return None

        sims = self._vecs @ q_vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        entry = self._entries[best]

        if score < self.threshold:
            self.misses += 1
            return None
        if self.ttl and time.time() - entry["at"] > self.ttl:
            self._evict(best)
            self.misses += 1
            return None
        # A Hindi query must not be served an English cached answer even if the
        # multilingual encoder rates them near-identical.
        if lang and entry.get("lang") and entry["lang"] != lang:
            self.misses += 1
            return None

        self.hits += 1
        return {"similarity": score, "query": entry["query"], "payload": entry["payload"]}

    def put(self, q_vec: np.ndarray, query: str, payload: dict, lang: str | None = None) -> None:
        if not self.enabled:
            return
        if len(self._entries) >= self.max_entries:
            self._evict(0)  # oldest first; recency beats frequency for a demo workload
        self._entries.append({"query": query, "payload": payload, "lang": lang, "at": time.time()})
        self._vecs = np.vstack([self._vecs, q_vec.reshape(1, -1).astype(np.float32)])

    def _evict(self, i: int) -> None:
        self._entries.pop(i)
        self._vecs = np.delete(self._vecs, i, axis=0)

    def clear(self) -> None:
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)
        self._entries.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}
