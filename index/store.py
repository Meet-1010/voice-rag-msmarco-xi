"""Qdrant collection management, local (embedded) mode.

Embedded mode keeps everything in one process and one container, which is what
makes a single-container deployment possible - no separate server, no network hop
on the hot path.

The corpus is partitioned into one collection per language rather than one
collection with a `lang` payload filter. That is not a stylistic choice: embedded
Qdrant evaluates payload filters by walking points in Python, which measured
53ms per query against 5.7ms unfiltered on this index. Since every query already
knows its language and never wants cross-language hits, partitioning removes the
filter entirely. It also mirrors the BM25 index, which is per-language for
unrelated reasons (IDF statistics).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


class VectorStore:
    def __init__(self, cfg: dict, root: Path):
        sc = cfg["store"]
        self.base = sc["collection"]
        self.dim = cfg["embedder"]["dim"]
        self.hnsw = sc.get("hnsw", {})
        path = Path(sc["path"])
        self.path = path if path.is_absolute() else root / path
        self.client = QdrantClient(path=str(self.path))

    def collection(self, lang: str) -> str:
        return f"{self.base}_{lang}"

    def languages(self) -> list[str]:
        prefix = f"{self.base}_"
        return sorted(c.name[len(prefix):] for c in self.client.get_collections().collections
                      if c.name.startswith(prefix))

    def exists(self) -> bool:
        return bool(self.languages())

    def count(self) -> int:
        return sum(self.client.count(self.collection(l), exact=True).count
                   for l in self.languages())

    def recreate(self, langs) -> None:
        for lang in langs:
            name = self.collection(lang)
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.hnsw.get("m", 24),
                        ef_construct=self.hnsw.get("ef_construct", 128),
                    ),
                ),
            )
            self.client.create_payload_index(
                name, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD)

    def upsert(self, vecs: np.ndarray, payloads: list[dict], batch: int = 512) -> None:
        by_lang: dict[str, list[int]] = {}
        for i, p in enumerate(payloads):
            by_lang.setdefault(p["lang"], []).append(i)

        for lang, idxs in by_lang.items():
            name = self.collection(lang)
            for start in range(0, len(idxs), batch):
                window = idxs[start:start + batch]
                self.client.upsert(
                    collection_name=name,
                    points=models.Batch(
                        # Point ids are per-collection, so they restart at 0 for
                        # each language; chunk_id in the payload stays global.
                        ids=list(range(start, start + len(window))),
                        vectors=vecs[window].tolist(),
                        payloads=[payloads[i] for i in window],
                    ),
                    wait=True,
                )

    def search(self, vec: np.ndarray, limit: int, lang: str | None = None) -> list[dict]:
        langs = [lang] if lang else self.languages()
        out: list[dict] = []
        for l in langs:
            if not self.client.collection_exists(self.collection(l)):
                continue
            hits = self.client.query_points(
                collection_name=self.collection(l),
                query=vec.tolist(),
                limit=limit,
                with_payload=True,
                search_params=models.SearchParams(hnsw_ef=self.hnsw.get("ef_search", 64)),
            ).points
            out.extend({"score": float(h.score), **h.payload} for h in hits)
        if len(langs) > 1:
            out.sort(key=lambda h: -h["score"])
            out = out[:limit]
        return out
