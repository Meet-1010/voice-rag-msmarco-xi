"""Exact dense search over a memory-mapped float32 matrix.

Why this exists rather than just calling Qdrant:

Embedded Qdrant scores candidates in single-threaded Python. Measured on the
60k-chunk index, that is 6.5ms on a dev machine but **126ms p50 on Cloud Run**,
and adding vCPUs does not help because the scan never leaves one thread. It was
by far the largest item in the latency budget.

The vectors are L2-normalised, so cosine similarity is a plain inner product and
the whole search is one `M @ q` matmul. numpy dispatches that to BLAS, which is
multi-threaded and cache-blocked: 0.43ms for 20k vectors locally, and it actually
scales with the cores Cloud Run gives us.

An ANN index (HNSW) would be the answer at millions of vectors. At 60k it would
trade recall for a speedup we do not need - exact search is already sub-millisecond
and returns the true top-k.

Qdrant remains the store of record: it owns the collections, the payloads and the
persistence. This is a read path built from it at index time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DenseIndex:
    """Per-language matrices, memory-mapped so startup does not pay for a copy."""

    def __init__(self, root: Path):
        self.dir = root / ".artifacts" / "dense"
        self.mats: dict[str, np.ndarray] = {}
        self.payloads: dict[str, list[dict]] = {}
        if not self.dir.exists():
            return
        for vec_path in sorted(self.dir.glob("vectors_*.npy")):
            lang = vec_path.stem.split("_", 1)[1]
            meta_path = self.dir / f"payloads_{lang}.jsonl"
            if not meta_path.exists():
                continue
            self.mats[lang] = np.load(vec_path, mmap_mode="r")
            with meta_path.open(encoding="utf-8") as fh:
                self.payloads[lang] = [json.loads(line) for line in fh]

    def available(self) -> bool:
        return bool(self.mats)

    def languages(self) -> list[str]:
        return sorted(self.mats)

    def count(self) -> int:
        return sum(m.shape[0] for m in self.mats.values())

    def search(self, vec: np.ndarray, limit: int, lang: str | None = None) -> list[dict]:
        langs = [lang] if lang and lang in self.mats else self.languages()
        out: list[dict] = []
        for l in langs:
            m = self.mats[l]
            scores = m @ vec
            k = min(limit, scores.shape[0])
            # argpartition is O(n) against a full O(n log n) sort; at 20k rows the
            # sort was costing more than the matmul it was ordering.
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            rows = self.payloads[l]
            out.extend({"score": float(scores[i]), **rows[i]} for i in top)
        if len(langs) > 1:
            out.sort(key=lambda h: -h["score"])
            out = out[:limit]
        return out

    @staticmethod
    def write(root: Path, vecs: np.ndarray, payloads: list[dict]) -> dict[str, int]:
        out_dir = root / ".artifacts" / "dense"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*"):
            f.unlink()

        by_lang: dict[str, list[int]] = {}
        for i, p in enumerate(payloads):
            by_lang.setdefault(p["lang"], []).append(i)

        counts = {}
        for lang, idxs in by_lang.items():
            arr = np.ascontiguousarray(vecs[idxs], dtype=np.float32)
            np.save(out_dir / f"vectors_{lang}.npy", arr)
            with (out_dir / f"payloads_{lang}.jsonl").open("w", encoding="utf-8") as fh:
                for i in idxs:
                    fh.write(json.dumps(payloads[i], ensure_ascii=False) + "\n")
            counts[lang] = len(idxs)
        return counts
