"""Dense + BM25 fused with Reciprocal Rank Fusion.

RRF rather than score interpolation because cosine similarity and BM25 scores live
on incompatible scales; normalising them per query is fragile and needs
recalibration whenever the corpus changes. RRF only reads rank position, so it
sidesteps the problem entirely.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class HybridRetriever:
    def __init__(self, cfg: dict, store, bm25, embedder, root: Path):
        rc = cfg["retrieval"]
        self.k = rc["rrf_k"]
        self.top_k = rc["top_k"]
        self.candidates = rc["candidates"]
        self.w_dense = rc["dense_weight"]
        self.w_sparse = rc["sparse_weight"]
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.by_chunk_id = {}
        meta = root / ".artifacts" / "chunks.jsonl"
        if meta.exists():
            with meta.open(encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    self.by_chunk_id[d["chunk_id"]] = d

    def dense(self, q_vec: np.ndarray, lang: str | None, limit: int) -> list[dict]:
        return self.store.search(q_vec, limit, lang=lang)

    def sparse(self, query: str, lang: str, limit: int) -> list[dict]:
        out = []
        for cid, score in self.bm25.search(query, lang, limit):
            row = self.by_chunk_id.get(cid)
            if row:
                out.append({"score": score, **row})
        return out

    def search(self, query: str, q_vec: np.ndarray, lang: str | None = None,
               top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.top_k
        dense_hits = self.dense(q_vec, lang, self.candidates)
        sparse_hits = self.sparse(query, lang, self.candidates) if lang else []

        fused: dict[str, dict] = {}
        for hits, weight, tag in ((dense_hits, self.w_dense, "dense"),
                                  (sparse_hits, self.w_sparse, "sparse")):
            for rank, hit in enumerate(hits):
                cid = hit["chunk_id"]
                slot = fused.setdefault(cid, {**hit, "rrf": 0.0, "dense_score": 0.0, "sources": []})
                slot["rrf"] += weight / (self.k + rank + 1)
                slot["sources"].append(tag)
                if tag == "dense":
                    # Keep the raw cosine: the relevance guard thresholds on it, and
                    # an RRF score carries no absolute meaning to threshold against.
                    slot["dense_score"] = hit["score"]

        ranked = sorted(fused.values(), key=lambda h: -h["rrf"])[:top_k]
        for h in ranked:
            h["score"] = h["rrf"]
        return ranked
