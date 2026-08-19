"""Pick guardrail thresholds from labelled data instead of guessing them.

Two independent sweeps:

RELEVANCE - separate in-corpus queries (positive) from out-of-corpus queries
(negative) by top dense retrieval score. Reported as a full ROC-style sweep so the
operating point is a visible choice with a stated cost, not a magic constant.

GROUNDING - separate answers paired with the passage they came from (grounded)
from the same answers paired with an unrelated passage (ungrounded), over both
embedding similarity and token overlap.

The default operating point targets a low false-accept rate: wrongly answering an
out-of-corpus question is a much worse failure for this system than wrongly
refusing an in-corpus one, and the task explicitly asks us to show the system
knows when not to answer.
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

from guardrails.grounding_guard import GroundingGuard  # noqa: E402
from index.bm25 import BM25Index  # noqa: E402
from index.embedder import Embedder  # noqa: E402
from index.store import VectorStore  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402

# Cost is asymmetric but not unboundedly so. Answering an out-of-corpus question
# is the worse failure, yet a guard that refuses half of all answerable questions
# is not "safe", it is broken. We cap false-accepts here and then balance.
TARGET_FPR = 0.10
# Flooring TPR was tried and reverted. At 200k the classes overlap badly enough in
# Gujarati (separation -0.021) that guaranteeing 65% acceptance pushed false
# accepts to 25%, and 3 of 5 out-of-corpus Gujarati questions started being
# answered - which breaks the requirement the guard exists to satisfy. Under-
# refusing is a correctness failure; over-refusing is a usability one, and the
# general-knowledge fallback already covers usability when Strict RAG is off.
# So the cap governs and TPR is what gives.
MIN_TPR = 0.0


def sweep(pos: np.ndarray, neg: np.ndarray, target_fpr: float = TARGET_FPR,
          min_tpr: float = MIN_TPR):
    """Return (chosen_threshold, rows).

    Prefer the FPR cap, but never at the cost of falling below min_tpr. Among the
    candidates, take the one maximising Youden's J - maximising TPR instead
    degenerates whenever the classes separate cleanly, returning the very bottom
    of the gap where the threshold filters almost nothing.
    """
    grid = np.unique(np.round(np.concatenate([pos, neg]), 3))
    rows = []
    for t in grid:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        rows.append({"threshold": float(t), "tpr": tpr, "fpr": fpr, "youden": tpr - fpr})

    pool = [r for r in rows if r["fpr"] <= target_fpr and r["tpr"] >= min_tpr]
    if not pool:                                  # cap and floor cannot both hold
        pool = [r for r in rows if r["tpr"] >= min_tpr] or rows
    best = max(pool, key=lambda r: (round(r["youden"], 4), -r["threshold"]))
    return best, rows


def calibrate_relevance(cfg, embedder, retriever, n_pos: int, seed: int, queries_file: Path):
    # Must be the query set matching the *indexed* corpus. Calibrating "in-corpus"
    # scores using queries whose passages were never indexed would measure misses
    # as if they were hits and drag the threshold far too low.
    queries = [json.loads(l) for l in queries_file.open(encoding="utf-8")]
    random.Random(seed).shuffle(queries)
    in_corpus = queries[:n_pos]
    ood_path = queries_file.parent / "ood_queries.json"
    ood = json.loads(ood_path.read_text(encoding="utf-8"))["queries"]

    def top_scores(items):
        scores, langs = [], []
        vecs = embedder.queries([i["query"] for i in items])
        for item, vec in zip(items, vecs):
            hits = retriever.search(item["query"], vec, lang=item["lang"], top_k=10)
            scores.append(max((h.get("dense_score", 0.0) for h in hits), default=0.0))
            langs.append(item["lang"])
        return np.array(scores, dtype=np.float32), np.array(langs)

    pos, pos_lang = top_scores(in_corpus)
    neg, neg_lang = top_scores(ood)
    best, rows = sweep(pos, neg)

    # A multilingual encoder does not produce comparable score distributions across
    # scripts: English in-corpus queries sit ~0.04 higher than Gujarati ones while
    # the out-of-corpus scores barely move. One global threshold therefore
    # over-refuses the lowest-resource language and under-refuses English.
    per_lang = {}
    for lang in sorted(set(pos_lang) | set(neg_lang)):
        p, n = pos[pos_lang == lang], neg[neg_lang == lang]
        if len(p) < 20 or len(n) < 5:
            continue
        lb, _ = sweep(p, n)
        # str() because these come out of a numpy array as np.str_, which yaml
        # refuses to serialise.
        per_lang[str(lang)] = {
            "threshold": lb["threshold"], "tpr": lb["tpr"], "fpr": lb["fpr"],
            "n_pos": int(len(p)), "n_neg": int(len(n)),
            "in_mean": float(p.mean()), "in_p10": float(np.percentile(p, 10)),
            "ood_mean": float(n.mean()), "ood_p90": float(np.percentile(n, 90)),
            "separation": float(np.percentile(p, 10) - np.percentile(n, 90)),
        }

    return {
        "n_in_corpus": len(pos), "n_out_of_corpus": len(neg),
        "in_corpus": {"mean": float(pos.mean()), "p05": float(np.percentile(pos, 5)),
                      "p50": float(np.percentile(pos, 50))},
        "out_of_corpus": {"mean": float(neg.mean()), "p95": float(np.percentile(neg, 95)),
                          "p50": float(np.percentile(neg, 50))},
        "chosen": best, "curve": rows, "per_lang": per_lang,
    }


def calibrate_grounding(cfg, embedder, n_pairs: int, seed: int,
                        queries_file: Path, corpus_file: Path):
    passages = {}
    with corpus_file.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            passages[d["passage_id"]] = d["text"]

    queries = [json.loads(l) for l in queries_file.open(encoding="utf-8")]
    queries = [q for q in queries if q.get("answer") and q["relevant"]
               and q["relevant"][0] in passages]
    rng = random.Random(seed)
    rng.shuffle(queries)
    queries = queries[:n_pairs]

    guard = GroundingGuard(cfg, embedder)
    all_pids = list(passages)

    grounded, ungrounded = [], []
    for q in queries:
        gold_ctx = passages[q["relevant"][0]]
        # Negative: the same answer against a passage it demonstrably did not come
        # from. This is what a hallucinated answer looks like to the guard.
        wrong_ctx = passages[rng.choice(all_pids)]
        for ctx, bucket in ((gold_ctx, grounded), (wrong_ctx, ungrounded)):
            bucket.append((guard.similarity(q["answer"], ctx),
                           guard.token_overlap(q["answer"], ctx)))

    g = np.array(grounded, dtype=np.float32)
    u = np.array(ungrounded, dtype=np.float32)
    sim_best, sim_rows = sweep(g[:, 0], u[:, 0], target_fpr=0.10)
    ovl_best, ovl_rows = sweep(g[:, 1], u[:, 1], target_fpr=0.10)
    return {
        "n_pairs": len(queries),
        "similarity": {"grounded_mean": float(g[:, 0].mean()),
                       "ungrounded_mean": float(u[:, 0].mean()),
                       "chosen": sim_best, "curve": sim_rows[:200]},
        "token_overlap": {"grounded_mean": float(g[:, 1].mean()),
                          "ungrounded_mean": float(u[:, 1].mean()),
                          "chosen": ovl_best, "curve": ovl_rows[:200]},
    }


def to_markdown(rel: dict, grd: dict) -> str:
    c = rel["chosen"]
    lines = [
        "# Guardrail threshold calibration\n",
        "Thresholds below are measured, not chosen by feel. Regenerate with "
        "`python guardrails/calibrate.py --write`.\n",
        "## Relevance guard (OUT_OF_CORPUS)\n",
        f"- In-corpus queries: {rel['n_in_corpus']}, out-of-corpus: {rel['n_out_of_corpus']}",
        f"- In-corpus top dense score: mean {rel['in_corpus']['mean']:.3f}, "
        f"p05 {rel['in_corpus']['p05']:.3f}",
        f"- Out-of-corpus top dense score: mean {rel['out_of_corpus']['mean']:.3f}, "
        f"p95 {rel['out_of_corpus']['p95']:.3f}",
        f"- **Chosen threshold {c['threshold']:.3f}** -> accepts "
        f"{c['tpr']*100:.1f}% of in-corpus queries, "
        f"{c['fpr']*100:.1f}% of out-of-corpus queries\n",
        "Operating point favours refusing over answering: a wrong answer to an "
        "out-of-corpus question is worse than an unnecessary refusal.\n",
        "| Threshold | Accepts in-corpus (TPR) | Accepts out-of-corpus (FPR) |",
        "|---|---|---|",
    ]
    curve = rel["curve"]
    step = max(1, len(curve) // 12)
    for r in curve[::step]:
        mark = " **<- chosen**" if abs(r["threshold"] - c["threshold"]) < 1e-9 else ""
        lines.append(f"| {r['threshold']:.3f} | {r['tpr']*100:.1f}% | {r['fpr']*100:.1f}%{mark} |")

    if rel.get("per_lang"):
        lines += [
            "\n### Per-language thresholds\n",
            "multilingual-e5 scores are not comparable across scripts. In-corpus "
            "English queries sit well above the out-of-corpus band; Gujarati barely "
            "clears it. A single global threshold over-refuses the lowest-resource "
            "language, so each gets its own operating point.\n",
            "| Lang | In-corpus mean | In p10 | OOD mean | OOD p90 | Separation | Threshold | TPR | FPR |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for lg, v in rel["per_lang"].items():
            lines.append(
                f"| {lg} | {v['in_mean']:.3f} | {v['in_p10']:.3f} | {v['ood_mean']:.3f} | "
                f"{v['ood_p90']:.3f} | {v['separation']:+.3f} | **{v['threshold']:.3f}** | "
                f"{v['tpr']*100:.1f}% | {v['fpr']*100:.1f}% |")
        lines.append(
            "\n`Separation` is in-corpus p10 minus out-of-corpus p90: positive means the "
            "two distributions are cleanly apart at those quantiles. It is comfortably "
            "positive for English and negative for Gujarati, which is the honest limit of "
            "this guard on the lowest-resource language rather than something to paper over.\n")

    s, o = grd["similarity"], grd["token_overlap"]
    lines += [
        "\n## Grounding guard (UNGROUNDED_OUTPUT)\n",
        f"Built from {grd['n_pairs']} answer/passage pairs. Positives pair each gold "
        "answer with the passage it came from; negatives pair the same answer with a "
        "random unrelated passage.\n",
        f"- Embedding similarity: grounded mean {s['grounded_mean']:.3f} vs "
        f"ungrounded {s['ungrounded_mean']:.3f} -> **threshold "
        f"{s['chosen']['threshold']:.3f}** (keeps {s['chosen']['tpr']*100:.1f}% of "
        f"grounded, admits {s['chosen']['fpr']*100:.1f}% of ungrounded)",
        f"- Token overlap: grounded mean {o['grounded_mean']:.3f} vs ungrounded "
        f"{o['ungrounded_mean']:.3f} -> **threshold {o['chosen']['threshold']:.3f}** "
        f"(keeps {o['chosen']['tpr']*100:.1f}%, admits {o['chosen']['fpr']*100:.1f}%)\n",
        "An answer must clear both. They fail differently: similarity catches an "
        "answer that wandered off topic, overlap catches fluent text that invented "
        "specifics the passage never contained.\n",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--in-corpus", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--queries-file", default=None,
                    help="query set matching the indexed corpus (default data/queries.jsonl)")
    ap.add_argument("--corpus-file", default=None)
    ap.add_argument("--write", action="store_true", help="write chosen thresholds into config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = ROOT / cfg["corpus"]["out_dir"]
    queries_file = Path(args.queries_file) if args.queries_file else data_dir / "queries.jsonl"
    corpus_file = Path(args.corpus_file) if args.corpus_file else data_dir / "corpus.jsonl"
    for p in (queries_file, corpus_file):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    embedder = Embedder(cfg["embedder"])
    store = VectorStore(cfg, ROOT)
    from index.dense import DenseIndex
    if not DenseIndex(ROOT).available() and not store.exists():
        raise SystemExit("no index found - run index/build_index.py first")
    bm_path = ROOT / ".artifacts" / "bm25.pkl"
    bm25 = BM25Index.load(bm_path) if bm_path.exists() else BM25Index()
    retriever = HybridRetriever(cfg, store, bm25, embedder, ROOT)

    print(f"calibrating relevance guard over {queries_file.name} ...", flush=True)
    rel = calibrate_relevance(cfg, embedder, retriever, args.in_corpus, args.seed, queries_file)
    print(f"  chosen {rel['chosen']['threshold']:.3f} "
          f"(tpr {rel['chosen']['tpr']:.3f}, fpr {rel['chosen']['fpr']:.3f})")

    print("calibrating grounding guard ...", flush=True)
    grd = calibrate_grounding(cfg, embedder, args.pairs, args.seed, queries_file, corpus_file)
    print(f"  similarity {grd['similarity']['chosen']['threshold']:.3f}, "
          f"overlap {grd['token_overlap']['chosen']['threshold']:.3f}")

    out_dir = ROOT / "guardrails"
    (out_dir / "calibration.json").write_text(
        json.dumps({"relevance": rel, "grounding": grd}, indent=2), encoding="utf-8")
    (out_dir / "calibration.md").write_text(to_markdown(rel, grd), encoding="utf-8")
    print(f"wrote {out_dir/'calibration.md'}")

    if args.write:
        # Written to a sidecar rather than edited into config.yaml: the values are
        # generated output, and string-patching a hand-commented YAML file is how
        # you eventually corrupt it.
        out = {
            "_generated_by": "guardrails/calibrate.py",
            "_generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "_index": {"queries": queries_file.name, "corpus": corpus_file.name},
            "relevance": {
                "default": round(rel["chosen"]["threshold"], 3),
                "per_lang": {l: round(v["threshold"], 3) for l, v in rel["per_lang"].items()},
            },
            "grounding": {
                "min_similarity": round(grd["similarity"]["chosen"]["threshold"], 3),
                "min_token_overlap": round(grd["token_overlap"]["chosen"]["threshold"], 3),
            },
        }
        (out_dir / "thresholds.yaml").write_text(
            yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote {out_dir/'thresholds.yaml'} (loaded at startup, overrides config.yaml)")


if __name__ == "__main__":
    main()
