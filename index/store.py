"""Qdrant collection management, local (embedded) mode.

Embedded mode keeps everything in one process and one container, which is what
makes a single-container Space deployment possible. No separate server to boot,
no network hop on the hot path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


class VectorStore:
    def __init__(self, cfg: dict, root: Path):
        sc = cfg["store"]
        self.collection = sc["collection"]
        self.dim = cfg["embedder"]["dim"]
        self.hnsw = sc.get("hnsw", {})
        path = Path(sc["path"])
        self.path = path if path.is_absolute() else root / path
        self.client = QdrantClient(path=str(self.path))

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    def recreate(self) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=self.hnsw.get("m", 24),
                    ef_construct=self.hnsw.get("ef_construct", 128),
                ),
            ),
        )
        # Language is a hard filter on every query, so it needs a payload index;
        # without one Qdrant falls back to a full scan of the filtered set.
        self.client.create_payload_index(
            self.collection, "lang", field_schema=models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(
            self.collection, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD)

    def upsert(self, ids: list[int], vecs: np.ndarray, payloads: list[dict], batch: int = 512) -> None:
        for i in range(0, len(ids), batch):
            self.client.upsert(
                collection_name=self.collection,
                points=models.Batch(
                    ids=ids[i:i + batch],
                    vectors=vecs[i:i + batch].tolist(),
                    payloads=payloads[i:i + batch],
                ),
                wait=False,
            )
        # One blocking call at the end rather than per batch; the intermediate
        # waits were most of the ingestion time.
        self.client.upsert(
            collection_name=self.collection,
            points=models.Batch(ids=ids[-1:], vectors=vecs[-1:].tolist(), payloads=payloads[-1:]),
            wait=True,
        )

    def search(self, vec: np.ndarray, limit: int, lang: str | None = None) -> list[dict]:
        flt = None
        if lang:
            flt = models.Filter(must=[models.FieldCondition(
                key="lang", match=models.MatchValue(value=lang))])
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vec.tolist(),
            limit=limit,
            query_filter=flt,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=self.hnsw.get("ef_search", 64)),
        ).points
        return [{"score": float(h.score), **h.payload} for h in hits]
