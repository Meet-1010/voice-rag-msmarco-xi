"""Fail the image build if the shipped index is not usable.

Runs inside Docker. Catches the failure modes that would otherwise surface as a
broken cold start in front of a user: storage that does not load on this
platform, a dense matrix that disagrees with the manifest, or a missing sparse
index while sparse fusion is switched on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from index.dense import DenseIndex  # noqa: E402
from index.store import VectorStore  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    manifest = json.loads((ROOT / ".artifacts" / "manifest.json").read_text())
    expected = manifest["chunks"]

    store = VectorStore(cfg, ROOT)
    n_qdrant = store.count()
    if n_qdrant != expected:
        raise SystemExit(f"qdrant has {n_qdrant} points, manifest says {expected}")

    dense = DenseIndex(ROOT)
    if not dense.available():
        raise SystemExit("dense matrices missing - serving search would fall back to Qdrant")
    if dense.count() != expected:
        raise SystemExit(f"dense index has {dense.count()} vectors, manifest says {expected}")

    if cfg["retrieval"].get("sparse_weight", 0) > 0 and not (ROOT / ".artifacts" / "bm25.pkl").exists():
        raise SystemExit("sparse_weight > 0 but bm25.pkl is missing")

    print(f"index verified: {expected:,} chunks, langs={dense.languages()}, "
          f"strategy={manifest['strategy']}, search=numpy")


if __name__ == "__main__":
    main()
