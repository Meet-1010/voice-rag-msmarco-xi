"""Benchmark every chunking strategy against held-out queries with known positives.

Two methodology choices worth stating, because they are what make the numbers
comparable at all:

1. Scoring is at PASSAGE level, not chunk level. A strategy that emits six chunks
   per passage would otherwise fill the top-k window with fragments of the same
   document and post inflated recall. We retrieve chunks, map them back to their
   source passage, dedupe, and only then cut at k.

2. Search is exact (brute-force matmul), not HNSW. We are measuring the chunker
   here; letting approximate-index recall vary between runs would contaminate the
   comparison. The production path uses Qdrant/HNSW and is measured separately.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunking.base import Passage, n_tokens  # noqa: E402
from chunking.registry import NEEDS_EMBEDDER, STRATEGIES, build  # noqa: E402
from index.embedder import Embedder  # noqa: E402

K_VALUES = (1, 5, 10)
CHUNK_DEPTH = 100  # chunks pulled before passage dedupe; must exceed max k comfortably


def load_corpus(path: Path, limit: int | None) -> list[Passage]:
    """Sample evenly across languages.

    corpus.jsonl is written shard by shard, so a plain prefix of N lines is all
    English and Hindi and contains no Gujarati at all - which silently drops a
    third of the evaluation and makes the per-language columns meaningless.
    """
    by_lang: dict[str, list[Passage]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            p = Passage.from_json(json.loads(line))
            by_lang.setdefault(p.lang, []).append(p)
    if not limit:
        return [p for ps in by_lang.values() for p in ps]

    per_lang = max(1, limit // len(by_lang))
    out: list[Passage] = []
    for ps in by_lang.values():
        out.extend(ps[:per_lang])
    return out


def load_queries(path: Path, keep: set[str]) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            # Only queries whose positives all survived the corpus cap are scorable.
            if q["relevant"] and all(pid in keep for pid in q["relevant"]):
                out.append(q)
    return out


def evaluate_one(name, chunker, passages, queries, embedder, batch_hint=256):
    t0 = time.perf_counter()
    chunks = chunker.chunk_all(passages)
    build_s = time.perf_counter() - t0
    if not chunks:
        raise RuntimeError(f"{name} produced no chunks")

    t0 = time.perf_counter()
    mat = embedder.passages([c.text for c in chunks])
    embed_s = time.perf_counter() - t0

    chunk_pids = np.array([c.passage_id for c in chunks])
    chunk_langs = np.array([c.lang for c in chunks])
    lang_mask = {l: np.flatnonzero(chunk_langs == l) for l in np.unique(chunk_langs)}

    # Query embedding is timed separately from search so the table can attribute cost.
    q_texts = [q["query"] for q in queries]
    q_vecs = embedder.queries(q_texts)

    hits_at = {k: 0.0 for k in K_VALUES}
    per_lang = {}
    rr_total = 0.0
    latencies = []

    for q, qv in zip(queries, q_vecs):
        idx = lang_mask.get(q["lang"])
        if idx is None or idx.size == 0:
            continue
        t0 = time.perf_counter()
        scores = mat[idx] @ qv
        depth = min(CHUNK_DEPTH, scores.size)
        top = idx[np.argpartition(-scores, depth - 1)[:depth]]
        top = top[np.argsort(-(mat[top] @ qv))]

        ranked: list[str] = []
        seen = set()
        for pid in chunk_pids[top]:
            if pid not in seen:
                seen.add(pid)
                ranked.append(pid)
            if len(ranked) >= max(K_VALUES):
                break
        latencies.append((time.perf_counter() - t0) * 1000)

        rel = set(q["relevant"])
        for k in K_VALUES:
            got = len(rel & set(ranked[:k]))
            hits_at[k] += got / len(rel)
        rr = next((1.0 / (i + 1) for i, pid in enumerate(ranked[:10]) if pid in rel), 0.0)
        rr_total += rr
        slot = per_lang.setdefault(q["lang"], [0.0, 0])
        slot[0] += len(rel & set(ranked[:5])) / len(rel)
        slot[1] += 1

    n = len(latencies) or 1
    sizes = [n_tokens(c.text) for c in chunks[:4000]]  # sampled; tokenising 100k is slow
    return {
        "strategy": name,
        "chunks": len(chunks),
        "chunks_per_passage": len(chunks) / len(passages),
        "mean_tokens": float(np.mean(sizes)),
        "p95_tokens": float(np.percentile(sizes, 95)),
        "build_s": build_s,
        "embed_s": embed_s,
        "recall@1": hits_at[1] / n,
        "recall@5": hits_at[5] / n,
        "recall@10": hits_at[10] / n,
        "mrr@10": rr_total / n,
        "search_p50_ms": float(np.percentile(latencies, 50)),
        "search_p95_ms": float(np.percentile(latencies, 95)),
        "per_lang_recall5": {l: v[0] / v[1] for l, v in sorted(per_lang.items())},
        "index_mb": mat.nbytes / 1e6,
    }


def to_markdown(rows: list[dict], meta: dict) -> str:
    langs = sorted({l for r in rows for l in r["per_lang_recall5"]})
    head = ["Strategy", "Chunks", "Chk/Psg", "Mean tok", "P95 tok",
            "R@1", "R@5", "R@10", "MRR@10"]
    head += [f"R@5 {l}" for l in langs]
    head += ["Build s", "Embed s", "Search P50 ms", "Index MB"]

    best = {m: max(r[m] for r in rows) for m in ("recall@1", "recall@5", "recall@10", "mrr@10")}

    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in sorted(rows, key=lambda x: -x["recall@5"]):
        def fmt(metric):
            v = f"{r[metric]:.3f}"
            return f"**{v}**" if abs(r[metric] - best[metric]) < 1e-9 else v
        cells = [r["strategy"], f"{r['chunks']:,}", f"{r['chunks_per_passage']:.2f}",
                 f"{r['mean_tokens']:.0f}", f"{r['p95_tokens']:.0f}",
                 fmt("recall@1"), fmt("recall@5"), fmt("recall@10"), fmt("mrr@10")]
        cells += [f"{r['per_lang_recall5'].get(l, float('nan')):.3f}" for l in langs]
        cells += [f"{r['build_s']:.1f}", f"{r['embed_s']:.1f}",
                  f"{r['search_p50_ms']:.2f}", f"{r['index_mb']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")

    header = (
        "# Chunking strategy comparison\n\n"
        f"- Corpus: {meta['passages']:,} MSMARCO-XI passages ({', '.join(meta['langs'])})\n"
        f"- Held-out queries: {meta['queries']:,}, sampled with seed {meta['seed']}\n"
        f"- Encoder: `{meta['model']}` (ONNX, CPU)\n"
        "- Scoring is passage-level after deduping chunks back to their source passage.\n"
        "- Retrieval is exact inner-product search, so the numbers isolate the chunker\n"
        "  rather than mixing in approximate-index recall.\n"
        "- Queries are filtered to their own language, matching production behaviour.\n\n"
    )
    return header + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=15000, help="passages to index")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "chunking" / "comparison.md"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = ROOT / cfg["corpus"]["out_dir"]

    passages = load_corpus(data_dir / "corpus.jsonl", args.limit)
    keep = {p.passage_id for p in passages}
    pool = load_queries(data_dir / "queries.jsonl", keep)
    random.Random(args.seed).shuffle(pool)
    queries = pool[:args.queries]
    print(f"{len(passages):,} passages, {len(queries)} eval queries from {len(pool):,} scorable")

    embedder = Embedder(cfg["embedder"])
    names = args.only or list(STRATEGIES)
    rows = []
    for name in names:
        chunker = build(name, cfg, embedder=embedder if name in NEEDS_EMBEDDER else None)
        print(f"  running {name} ...", flush=True)
        row = evaluate_one(name, chunker, passages, queries, embedder)
        rows.append(row)
        print(f"    R@1={row['recall@1']:.3f} R@5={row['recall@5']:.3f} "
              f"MRR@10={row['mrr@10']:.3f} chunks={row['chunks']:,} "
              f"build={row['build_s']:.1f}s embed={row['embed_s']:.1f}s", flush=True)

    meta = {"passages": len(passages), "queries": len(queries), "seed": args.seed,
            "model": cfg["embedder"]["model"],
            "langs": sorted({p.lang for p in passages})}
    md = to_markdown(rows, meta)
    Path(args.out).write_text(md, encoding="utf-8")
    (Path(args.out).with_suffix(".json")).write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
