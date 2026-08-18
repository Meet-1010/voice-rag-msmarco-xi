"""One-time ingestion: corpus.jsonl -> chunks -> dense matrices (+ optional Qdrant, BM25).

Streams by language and writes vectors straight into a memmap. The previous
version held every chunk, its vector and its payload in memory at once; that is
fine at 60k and drove this machine 5GB into swap at 200k, where embedding
throughput collapsed from ~205/s to ~71/s. Peak memory here is one batch plus one
language's output matrix, so build time scales with corpus size instead of
falling off a cliff when RAM runs out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunking.base import Passage  # noqa: E402
from chunking.registry import NEEDS_EMBEDDER, build as build_chunker  # noqa: E402
from index.bm25 import BM25Index  # noqa: E402
from index.embedder import Embedder  # noqa: E402


def stream_passages(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                return
            yield Passage.from_json(json.loads(line))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strategy", default=None, help="override chunking.active")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--batch", type=int, default=2000, help="passages embedded per flush")
    ap.add_argument("--qdrant", action="store_true",
                    help="also populate Qdrant; excluded from the image, so off by default")
    ap.add_argument("--bm25", action="store_true", help="also build the sparse index")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strategy = args.strategy or cfg["chunking"]["active"]
    data_dir = ROOT / cfg["corpus"]["out_dir"]
    corpus_path = Path(args.corpus) if args.corpus else data_dir / "corpus.jsonl"
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path

    embedder = Embedder(cfg["embedder"])
    chunker = build_chunker(strategy, cfg, embedder=embedder if strategy in NEEDS_EMBEDDER else None)
    dim = cfg["embedder"]["dim"]

    out_dir = ROOT / ".artifacts" / "dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*"):
        f.unlink()

    # One pass to chunk and count per language, writing payloads as we go. Only
    # the counts are kept, so the chunk objects are freed batch by batch.
    print(f"pass 1/2: chunking {corpus_path.name} (strategy={strategy})", flush=True)
    t0 = time.perf_counter()
    counts: dict[str, int] = {}
    handles = {}
    for p in stream_passages(corpus_path, args.limit):
        for c in chunker.chunk(p):
            fh = handles.get(c.lang)
            if fh is None:
                fh = handles[c.lang] = (out_dir / f"payloads_{c.lang}.jsonl").open("w", encoding="utf-8")
            fh.write(json.dumps({
                "chunk_id": c.chunk_id, "passage_id": c.passage_id, "doc_id": c.doc_id,
                "lang": c.lang, "text": c.text,
                "context": c.context if c.context != c.text else "",
                "query_id": c.query_id,
            }, ensure_ascii=False) + "\n")
            counts[c.lang] = counts.get(c.lang, 0) + 1
    for fh in handles.values():
        fh.close()
    total = sum(counts.values())
    print(f"  {total:,} chunks across {sorted(counts)} in {time.perf_counter()-t0:.1f}s", flush=True)

    # Second pass embeds each language into a preallocated memmap, so the only
    # large allocation is the output matrix itself.
    print("pass 2/2: embedding", flush=True)
    t0 = time.perf_counter()
    done = 0
    for lang, n in sorted(counts.items()):
        mat = np.lib.format.open_memmap(out_dir / f"vectors_{lang}.npy", mode="w+",
                                        dtype=np.float32, shape=(n, dim))
        buf, at = [], 0
        with (out_dir / f"payloads_{lang}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                buf.append(json.loads(line)["text"])
                if len(buf) >= args.batch:
                    mat[at:at + len(buf)] = embedder.passages(buf)
                    at += len(buf); done += len(buf); buf = []
                    rate = done / max(1e-6, time.perf_counter() - t0)
                    print(f"  {done:,}/{total:,}  {rate:.0f}/s", flush=True)
        if buf:
            mat[at:at + len(buf)] = embedder.passages(buf)
            at += len(buf); done += len(buf)
        mat.flush()
        del mat
        print(f"  {lang}: {at:,} vectors", flush=True)
    print(f"embedded in {time.perf_counter()-t0:.1f}s ({done/(time.perf_counter()-t0):.0f}/s)")

    manifest = {
        "strategy": strategy, "passages": sum(1 for _ in stream_passages(corpus_path, args.limit)),
        "chunks": total, "model": cfg["embedder"]["model"], "dim": dim,
        "langs": sorted(counts), "per_lang": counts,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (ROOT / ".artifacts" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps({k: v for k, v in manifest.items() if k != "per_lang"}))

    if args.bm25:
        t0 = time.perf_counter()
        bm = BM25Index()
        ids, texts, langs = [], [], []
        for lang in sorted(counts):
            with (out_dir / f"payloads_{lang}.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    ids.append(d["chunk_id"]); texts.append(d["text"]); langs.append(lang)
        bm.build(ids, texts, langs)
        bm.save(ROOT / ".artifacts" / "bm25.pkl")
        print(f"bm25 built in {time.perf_counter()-t0:.1f}s")

    if args.qdrant:
        from index.store import VectorStore
        store = VectorStore(cfg, ROOT)
        store.recreate(sorted(counts))
        for lang in sorted(counts):
            vecs = np.load(out_dir / f"vectors_{lang}.npy", mmap_mode="r")
            payloads = [json.loads(l) for l in
                        (out_dir / f"payloads_{lang}.jsonl").open(encoding="utf-8")]
            store.upsert(np.asarray(vecs), payloads)
        print(f"qdrant: {store.count():,} points")


if __name__ == "__main__":
    main()
