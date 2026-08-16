"""One-time ingestion: corpus.jsonl -> chunks -> Qdrant + BM25 + chunk store."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunking.base import Passage  # noqa: E402
from chunking.registry import NEEDS_EMBEDDER, build as build_chunker  # noqa: E402
from index.bm25 import BM25Index  # noqa: E402
from index.embedder import Embedder  # noqa: E402
from index.store import VectorStore  # noqa: E402


def load_passages(path: Path, limit: int | None) -> list[Passage]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            out.append(Passage.from_json(json.loads(line)))
            if limit and len(out) >= limit:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strategy", default=None, help="override chunking.active")
    ap.add_argument("--corpus", default=None, help="corpus jsonl (default data/corpus.jsonl)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strategy = args.strategy or cfg["chunking"]["active"]
    data_dir = ROOT / cfg["corpus"]["out_dir"]

    corpus_path = Path(args.corpus) if args.corpus else data_dir / "corpus.jsonl"
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path
    passages = load_passages(corpus_path, args.limit)
    print(f"{len(passages):,} passages, strategy={strategy}")

    embedder = Embedder(cfg["embedder"])
    chunker = build_chunker(strategy, cfg, embedder=embedder if strategy in NEEDS_EMBEDDER else None)

    t0 = time.perf_counter()
    chunks = chunker.chunk_all(passages)
    print(f"chunked -> {len(chunks):,} chunks in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    vecs = embedder.passages([c.text for c in chunks])
    print(f"embedded in {time.perf_counter()-t0:.1f}s ({len(chunks)/(time.perf_counter()-t0):.0f} chunks/s)")

    store = VectorStore(cfg, ROOT)
    store.recreate(sorted({c.lang for c in chunks}))
    payloads = [{
        "chunk_id": c.chunk_id, "passage_id": c.passage_id, "doc_id": c.doc_id,
        "lang": c.lang, "text": c.text,
        # Only parent-child sets this; empty means "generate from text itself".
        "context": c.context if c.context != c.text else "",
        "query_id": c.query_id,
    } for c in chunks]

    t0 = time.perf_counter()
    store.upsert(vecs, payloads)
    print(f"upserted {store.count():,} points across {store.languages()} "
          f"in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    bm = BM25Index()
    bm.build([c.chunk_id for c in chunks], [c.text for c in chunks], [c.lang for c in chunks])
    bm.save(ROOT / ".artifacts" / "bm25.pkl")
    print(f"bm25 built in {time.perf_counter()-t0:.1f}s over {len(bm.by_lang)} languages")

    # BM25 returns chunk_ids, Qdrant returns payloads. Fusion needs one lookup table
    # that both sides can resolve against.
    meta_path = ROOT / ".artifacts" / "chunks.jsonl"
    with meta_path.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(chunks):
            fh.write(json.dumps({"point_id": i, **c.to_json()}, ensure_ascii=False) + "\n")

    manifest = {
        "strategy": strategy, "passages": len(passages), "chunks": len(chunks),
        "model": cfg["embedder"]["model"], "dim": cfg["embedder"]["dim"],
        "langs": sorted({c.lang for c in chunks}), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (ROOT / ".artifacts" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps(manifest))


if __name__ == "__main__":
    main()
