"""End-to-end latency benchmark.

Reports P50/P70/P90/P95/P99/P100 for the core pipeline and end to end, broken down
by stage and segmented by which path served the request. Run in two modes:

  cold - cache disabled, every query pays full retrieval. This is the honest
         worst case and the number to quote if only one is quoted.
  warm - cache enabled with realistic repetition, which is what a live demo
         actually looks like.

Both are reported. Quoting only the warm number is the classic way to make a RAG
pipeline look faster than it is.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.orchestrator import Orchestrator  # noqa: E402

PCTS = [50, 70, 90, 95, 99, 100]
WARMUP = 10


def pct(values: list[float]) -> dict:
    # n and mean are always present, including for the empty case: callers branch
    # on out["n"] and a missing key there is a crash rather than a zero.
    if not values:
        return {**{f"p{p}": None for p in PCTS}, "mean": None, "n": 0}
    arr = np.array(values, dtype=np.float64)
    out = {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in PCTS}
    out["mean"] = round(float(arr.mean()), 2)
    out["n"] = len(values)
    return out


def load_queries(n: int, seed: int, repeat_rate: float = 0.0,
                 path: Path | None = None) -> list[dict]:
    # Must match the indexed corpus, otherwise most queries miss and the benchmark
    # measures the refusal path rather than the retrieval path.
    path = path or ROOT / "data" / "queries.jsonl"
    pool = [json.loads(l) for l in path.open(encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(pool)
    picked = pool[:n]
    if repeat_rate > 0:
        # Real voice traffic repeats: the same question asked twice, or two people
        # asking the same thing. Without repetition a semantic cache is untestable.
        n_rep = int(n * repeat_rate)
        picked = picked[:n - n_rep] + [rng.choice(picked[:n - n_rep]) for _ in range(n_rep)]
        rng.shuffle(picked)
    return picked


def run(orc: Orchestrator, queries: list[dict], use_cache: bool, delay: float = 0.0) -> list[dict]:
    rows = []
    for q in queries:
        # Groq's free tier is rate limited per minute. Firing requests back to back
        # trips the circuit breaker a third of the way in, after which every
        # remaining request degrades to extractive and the measured generate
        # latency describes the breaker rather than the model.
        if delay:
            time.sleep(delay)
        t0 = time.perf_counter()
        ans = orc.ask(q["query"], lang=q["lang"], use_cache=use_cache)
        wall = (time.perf_counter() - t0) * 1000
        rows.append({
            "query": q["query"], "lang": q["lang"], "path": ans.path.value,
            "refused": ans.refused, "reason": ans.reason_code.value if ans.reason_code else None,
            "core_ms": ans.timings.core_ms, "total_ms": ans.timings.total_ms,
            "wall_ms": round(wall, 3), "stages": ans.timings.stages,
            "within_budget": ans.timings.within_budget,
        })
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    core = [r["core_ms"] for r in rows]
    total = [r["total_ms"] for r in rows]
    by_path: dict[str, list[float]] = {}
    for r in rows:
        by_path.setdefault(r["path"], []).append(r["core_ms"])

    stage_names = sorted({s for r in rows for s in r["stages"]})
    stages = {s: pct([r["stages"][s] for r in rows if s in r["stages"]]) for s in stage_names}

    return {
        "label": label,
        "n": len(rows),
        "core": pct(core),
        "total": pct(total),
        # Reported apart from core: this is someone else's network, and folding it
        # into the headline number is what makes latency claims dishonest.
        # Only rows the LLM actually answered count - a request that tripped the
        # breaker records a ~0ms generate span and would drag the percentile down.
        "generate": pct([r["stages"]["generate"] for r in rows
                         if r["path"] == "generative" and "generate" in r["stages"]]),
        "within_budget_pct": round(100 * sum(r["within_budget"] for r in rows) / max(1, len(rows)), 2),
        "by_path": {k: {**pct(v), "share_pct": round(100 * len(v) / len(rows), 1)}
                    for k, v in sorted(by_path.items())},
        "by_stage": stages,
        "refused": sum(r["refused"] for r in rows),
        "by_lang": {l: pct([r["core_ms"] for r in rows if r["lang"] == l])
                    for l in sorted({r["lang"] for r in rows})},
    }


def chart(summaries: list[dict], all_rows: dict[str, list[dict]], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable, skipping chart")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    labels = [f"P{p}" for p in PCTS]
    width = 0.38
    x = np.arange(len(PCTS))
    for i, s in enumerate(summaries):
        vals = [s["core"][f"p{p}"] or 0 for p in PCTS]
        ax.bar(x + i * width, vals, width, label=s["label"])
    ax.axhline(200, color="crimson", ls="--", lw=1.4, label="200ms target")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("core latency (ms)")
    ax.set_title("Core pipeline latency percentiles")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25)

    ax = axes[1]
    cold = summaries[0]
    stages = [(k, v["p50"]) for k, v in cold["by_stage"].items() if v["p50"]]
    stages.sort(key=lambda kv: kv[1])
    ax.barh([k for k, _ in stages], [v for _, v in stages], color="#5b8cff")
    ax.set_xlabel("P50 (ms)")
    ax.set_title(f"Per-stage P50 — {cold['label']}")
    ax.grid(axis="x", alpha=.25)

    ax = axes[2]
    for label, rows in all_rows.items():
        ax.hist([r["core_ms"] for r in rows], bins=40, alpha=.6, label=label)
    ax.axvline(200, color="crimson", ls="--", lw=1.4)
    ax.set_xlabel("core latency (ms)")
    ax.set_ylabel("queries")
    ax.set_title("Distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def to_markdown(summaries: list[dict]) -> str:
    lines = ["# Latency benchmark\n"]
    s0 = summaries[0]
    lines += [
        f"Measured over {s0['n']} MSMARCO-XI validation queries per mode, "
        f"{WARMUP} warm-up runs discarded. Core = embed -> retrieve -> guards -> answer, "
        "which is the span the 200ms requirement is scoped to.\n",
        "## Headline\n",
        "| Mode | P50 | P70 | P90 | P95 | P99 | P100 | Within 200ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        c = s["core"]
        lines.append(f"| {s['label']} | {c['p50']} | {c['p70']} | {c['p90']} | "
                     f"{c['p95']} | {c['p99']} | {c['p100']} | {s['within_budget_pct']}% |")

    gen = next((s for s in summaries if s.get("generate", {}).get("n")), None)
    if gen:
        g = gen["generate"]
        lines += [
            f"\nThe LLM call itself, excluded from core above and measured over "
            f"{g['n']} forced-generative requests: P50 **{g['p50']}ms**, P95 "
            f"**{g['p95']}ms**, P100 **{g['p100']}ms**. This is a third-party "
            "network round trip and no local optimisation reduces it, which is why "
            "the pipeline routes around it whenever retrieval is confident.\n"]

    for s in summaries:
        lines += [f"\n## {s['label']} — by answer path\n",
                  "| Path | Share | P50 | P70 | P95 | P100 |", "|---|---|---|---|---|---|"]
        for path, v in s["by_path"].items():
            lines.append(f"| {path} | {v['share_pct']}% | {v['p50']} | {v['p70']} | "
                         f"{v['p95']} | {v['p100']} |")
        lines += [f"\n### {s['label']} — by stage\n",
                  "| Stage | P50 | P70 | P95 | P100 |", "|---|---|---|---|---|"]
        for st, v in sorted(s["by_stage"].items(), key=lambda kv: -(kv[1]["p50"] or 0)):
            lines.append(f"| {st} | {v['p50']} | {v['p70']} | {v['p95']} | {v['p100']} |")
        lines += [f"\n### {s['label']} — by language\n",
                  "| Lang | P50 | P95 | n |", "|---|---|---|---|"]
        for lg, v in s["by_lang"].items():
            lines.append(f"| {lg} | {v['p50']} | {v['p95']} | {v['n']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--queries", type=int, default=220)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--repeat-rate", type=float, default=0.30)
    ap.add_argument("--queries-file", default=None)
    # The extractive fast-path serves most in-corpus queries, so a normal run
    # never exercises generation and the table would silently have no LLM row.
    # This forces it so the generative cost is measured rather than assumed.
    ap.add_argument("--generative", type=int, default=40,
                    help="queries to run with the extractive path disabled (0 to skip)")
    ap.add_argument("--generative-delay", type=float, default=2.2,
                    help="seconds between generative calls; Groq free tier is ~30/min")
    ap.add_argument("--out", default=str(ROOT / "bench"))
    args = ap.parse_args()

    qfile = Path(args.queries_file) if args.queries_file else ROOT / "data" / "queries.jsonl"
    if not qfile.exists():
        raise SystemExit(f"missing {qfile}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    orc = Orchestrator()
    print("providers:", orc.providers.status())
    print(f"index: {orc.store.count():,} points")

    warm_q = load_queries(WARMUP, args.seed + 999, path=qfile)
    print(f"warming up ({WARMUP} runs, discarded) ...")
    run(orc, warm_q, use_cache=False)
    orc.cache.clear()

    print(f"cold run: {args.queries} queries, cache disabled ...")
    cold_rows = run(orc, load_queries(args.queries, args.seed, path=qfile), use_cache=False)

    orc.cache.clear()
    print(f"warm run: {args.queries} queries, cache on, {args.repeat_rate:.0%} repeats ...")
    warm_rows = run(orc, load_queries(args.queries, args.seed, args.repeat_rate, path=qfile),
                    use_cache=True)

    summaries = [summarize(cold_rows, "cold (no cache)"), summarize(warm_rows, "warm (cache on)")]
    gen_rows: list[dict] = []
    if args.generative and orc.providers.any_available():
        print(f"generative run: {args.generative} queries, extractive path disabled, "
              f"{args.generative_delay}s apart ...")
        orc.cache.clear()
        orc.extractive.enabled = False
        for b in orc.providers.breakers.values():
            b.record_success()  # clear any breaker tripped by the earlier runs
        gen_rows = run(orc, load_queries(args.generative, args.seed + 7, path=qfile),
                       use_cache=False, delay=args.generative_delay)
        orc.extractive.enabled = True
        summaries.append(summarize(gen_rows, "generative (LLM forced)"))
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "queries_per_mode": args.queries, "warmup_discarded": WARMUP,
        "budget_ms": orc.budgets["core_total"],
        "providers": orc.providers.status(),
        "cache_stats": orc.cache.stats(),
        "summaries": summaries,
        "rows": {"cold": cold_rows, "warm": warm_rows},
    }
    payload["rows"]["generative"] = gen_rows
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = to_markdown(summaries)
    (out_dir / "results.md").write_text(md, encoding="utf-8")
    series = {"cold": cold_rows, "warm": warm_rows}
    if gen_rows:
        series["generative"] = gen_rows
    chart(summaries, series, out_dir / "latency.png")
    print("\n" + md)


if __name__ == "__main__":
    main()
