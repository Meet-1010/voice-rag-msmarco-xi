"""multilingual-e5-small on onnxruntime.

sentence-transformers would pull in torch, which costs ~2GB of image and a slower
CPU forward pass. e5-small publishes ONNX weights, so we register them with
fastembed instead and keep the whole runtime dependency footprint small enough
for a free Space.
"""
from __future__ import annotations

import os

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

_registered: set[str] = set()


def _register(cfg: dict) -> None:
    name = cfg["model"]
    if name in _registered:
        return
    known = {m["model"] for m in TextEmbedding.list_supported_models()}
    if name not in known:
        TextEmbedding.add_custom_model(
            model=name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=name),
            dim=cfg["dim"],
            model_file=cfg["onnx_file"],
        )
    _registered.add(name)


class Embedder:
    def __init__(self, cfg: dict):
        _register(cfg)
        self.dim = cfg["dim"]
        self.batch_size = cfg.get("batch_size", 64)
        self._q_prefix = cfg.get("query_prefix", "")
        self._p_prefix = cfg.get("passage_prefix", "")
        # Thread count has to match the vCPUs the host actually gives us.
        # Oversubscribing costs more in contention than it gains in parallelism,
        # and Cloud Run hands out 1-2 vCPU while a laptop has 8+.
        threads = int(os.getenv("EMBED_THREADS") or cfg.get("threads") or 4)
        self._model = TextEmbedding(
            model_name=cfg["model"],
            threads=threads,
            providers=["CPUExecutionProvider"],
        )
        self.warmup()

    def warmup(self) -> None:
        """First ONNX inference pays graph-init cost. Pay it before we start timing."""
        self._embed(["warmup"], self._q_prefix)

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [prefix + t for t in texts]
        vecs = list(self._model.embed(prefixed, batch_size=self.batch_size))
        return np.asarray(vecs, dtype=np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, self._q_prefix)

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, self._p_prefix)

    def query(self, text: str) -> np.ndarray:
        return self.queries([text])[0]
