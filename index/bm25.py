"""Sparse keyword index, partitioned by language.

Dense retrieval loses on rare literal tokens: product codes, numbers, names that
the encoder has never seen. BM25 catches exactly those, which is the whole reason
for fusing the two rather than picking one.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

# Keep Devanagari and Gujarati blocks as word characters; \w+ under ASCII rules
# would shred them into nothing.
_TOKEN = re.compile(r"[\wऀ-ॿ઀-૿]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """One BM25 per language. A single shared index would let English term
    statistics dominate the IDF table for the Indic languages."""

    def __init__(self):
        self.by_lang: dict[str, BM25Okapi] = {}
        self.ids: dict[str, list[str]] = {}

    def build(self, chunk_ids: list[str], texts: list[str], langs: list[str]) -> None:
        grouped: dict[str, tuple[list[str], list[list[str]]]] = {}
        for cid, text, lang in zip(chunk_ids, texts, langs):
            ids, docs = grouped.setdefault(lang, ([], []))
            ids.append(cid)
            docs.append(tokenize(text))
        for lang, (ids, docs) in grouped.items():
            self.by_lang[lang] = BM25Okapi(docs)
            self.ids[lang] = ids

    def search(self, query: str, lang: str, limit: int) -> list[tuple[str, float]]:
        bm = self.by_lang.get(lang)
        if bm is None:
            return []
        scores = bm.get_scores(tokenize(query))
        ids = self.ids[lang]
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]
        return [(ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"by_lang": self.by_lang, "ids": self.ids}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        obj = cls()
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        obj.by_lang, obj.ids = blob["by_lang"], blob["ids"]
        return obj
