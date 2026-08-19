# Voice RAG over MSMARCO-XI

**Live — https://voice-rag-84557916235.asia-south1.run.app**

Speak a question in English, Hindi or Gujarati. Get an answer grounded in a
retrieved passage, with citations, a per-stage latency breakdown, and an explicit
refusal when the corpus does not contain the answer.

Built for HH Goa 2026 Shortlisting Task 2. **201,011 passages · 202,201 chunks ·
3 languages · 7 chunking strategies benchmarked.**

> Deployed on Cloud Run in `asia-south1` (Mumbai), same region as the judges, so
> the network is not quietly spending the latency budget this project worked to
> save. It scales to zero, so the first request after idle pays a cold start.

```
🎤 audio ──► browser SpeechRecognition ──► live transcript, word by word
                              │                    (Sarvam verifies after)
                              ▼
                        [ guard_input ]  injection / unsafe / garbage / language
                              │
                              ▼
                        [ embed ]  multilingual-e5-small, ONNX, CPU, ~2.3ms
                              │
                    ┌─────────┴─────────┐
                    ▼                   │
              [ cache ]  cos > 0.95 ────┼──► cached answer          (~2ms)
                    │ miss              │
                    ▼                   │
              [ retrieve ]  exact BLAS matmul over 202k vectors  (~1.8ms)
                    │                   │
                    ▼                   │
              [ guard_relevance ]  top cosine < τ(lang) ──► REFUSE OUT_OF_CORPUS
                    │                   │
                    ▼                   │
              [ extract ]  confident answer span? ────────► extractive   (~10ms)
                    │ not confident     │
                    ▼                   │
              [ generate ]  Groq → Sarvam → extractive fallback
                    │                   │
                    ▼                   │
              [ guard_grounding ]  token overlap, escalating to the encoder
                    │                   │       └─► REFUSE UNGROUNDED_OUTPUT
                    ▼                   ▼
              answer + citations + full stage timings
```

---

## Where each requirement lives

| # | Requirement | Implementation |
|---|---|---|
| 1 | Speech-to-text (Sarvam or ElevenLabs) | `stt/sarvam.py` — Sarvam `saarika:v2.5`, auto language detection fed into the retrieval filter |
| 2 | Chunking must be vast | `chunking/` — 7 strategies behind one ABC, benchmarked in [`chunking/comparison.md`](chunking/comparison.md) |
| 3 | Under 200ms | Core pipeline measured per stage; 100% of requests inside budget, live and local |
| 4 | P50 / P70 / P100 | `bench/run_bench.py` — 220 queries per mode, cold and warm, segmented by answer path |
| 5 | Harness | `harness/` — Pydantic contracts, orchestrator state machine, provider chain, circuit breakers, JSON repair |
| 6 | Guardrails | `guardrails/` — 4 layers, thresholds calibrated from labelled data in [`guardrails/calibration.md`](guardrails/calibration.md) |

---

## The 200ms question, answered honestly

The task scopes the target to "chunking + vector DB retrieval + everything through
to final output". Two things are true at once, and this project reports both.

**The core pipeline meets the target**, on the deployed instance and not just on a
laptop: embedding, retrieval over 202,201 chunks, all four guardrails and
extractive answering complete inside 200ms for **100% of measured requests**.

**A generative LLM round trip does not, and cannot.** A round trip from India to
Groq is ~250ms before the model emits anything; measured end to end it is
**957ms P50**. No local optimisation changes someone else's network. Any
submission claiming sub-200ms *including* a third-party LLM call is measuring
something other than what it says.

So the pipeline is built to mostly not need the LLM:

| Path | Mechanism | Share (cold) | Core latency |
|---|---|---|---|
| Cache hit | Query embedding within 0.95 cosine of a previous query | — | ~2ms |
| Extractive | Confident retrieval + a clean answer span in the passage | 65.5% | 13.9ms P50 |
| Refusal | Guard fires before answering | 26.4% | 4.1ms P50 |
| General knowledge | Corpus cannot answer; LLM answers, explicitly ungrounded | 8.2% | 4.8ms core + ~957ms LLM |

MSMARCO passages were selected by human annotators *because* they answer the
query, so the answer span is usually present verbatim. Finding the right sentence
costs one embedding pass over a handful of sentences. Calling a 120B model to
paraphrase a sentence that is already correct is the thing worth avoiding.

### Five optimisations that came from measurement, not guesswork

| Change | Before | After | Why |
|---|---|---|---|
| Exact search on a BLAS matmul | 126ms | **1.8ms** | Embedded Qdrant scores candidates in single-threaded Python, so extra vCPUs did nothing. Vectors are normalised, so cosine is one `M @ q` that numpy hands to multi-threaded BLAS. |
| Drop BM25 fusion | 11.6ms | **0ms** | It was *lowering* R@1 from 0.360 to 0.283. Hybrid retrieval is not automatically better. |
| Partition vector index by language | 70ms | **9.9ms** | Embedded Qdrant walks payload filters in Python. Every query already knows its language. |
| Cascade the grounding checks | 130ms | **0.3ms** | Token overlap is both the stronger discriminator and ~600x cheaper than the encoder. |
| Warm encoder + page in matrices at boot | ~450ms first call | **~2.3ms** | ONNX graph init and mmap page-in are lazy; without warming, the first real user of each language pays all of it. |

Together these paid for a **3.3x larger corpus** (60,464 → 202,201 chunks) while
*lowering* worst-case latency, on the same 4 vCPU.

### Hybrid retrieval, and why it ships switched off

Dense + BM25 with Reciprocal Rank Fusion is the textbook default. It was built,
measured against dense-only over 300 held-out queries, and lost:

| sparse_weight | R@1 | R@5 | MRR@10 |
|---|---|---|---|
| **0.00 (dense only)** | **0.360** | 0.713 | **0.509** |
| 0.15 | 0.353 | **0.720** | 0.508 |
| 0.30 | 0.323 | 0.700 | 0.486 |
| 0.90 (originally shipped) | 0.283 | 0.693 | 0.453 |

At any weight worth having it costs 11.6ms, and past 0.15 it actively degrades
ranking. MSMARCO queries are natural language, which is exactly what e5 is trained
for, and `rank_bm25` does no stemming or morphological normalisation — so on
Devanagari and Gujarati its candidates are weak enough to displace better dense
hits during fusion. The sparse index is still built and wired in behind
`sparse_weight`; on a corpus of part numbers the result would flip.

### Why not HNSW

Qdrant's HNSW is the right answer at millions of vectors. At 202k it would trade
recall for a speedup exact search already has: the full matmul is **1.8ms P50** on
the deployed instance and returns the true top-k rather than an approximation.
Verified equivalent — the numpy path returned identical top-10 to Qdrant on 59 of
60 queries, the one difference being a score tie.

Qdrant remains the build-time store of record and the fallback when the matrices
are absent. It is excluded from the deployed image: a 202k-point store is ~1GB of
cold-start image pull for a read path nothing takes.

---

## Chunking — seven strategies, and the one the data chose

| # | Strategy | What it does |
|---|---|---|
| 1 | `fixed` | Hard token cuts at a fixed stride — the baseline to beat |
| 2 | `sliding` | Overlapping windows so answers straddling a boundary survive intact |
| 3 | `recursive` | Paragraph → line → sentence → token, descending only when a piece still doesn't fit |
| 4 | `semantic` | Embeds sentences, cuts at the largest adjacent-sentence distances |
| 5 | `passage_native` | Treats each MSMARCO passage as atomic, enforcing only the encoder's token ceiling |
| 6 | `parent_child` | Indexes small children, hands the parent passage to the generator |
| 7 | `indic_aware` | Script detection, Devanagari/Gujarati danda rules, script recorded as metadata |

Sizes are **calibrated to the measured corpus**, not copied from a tutorial:

| Lang | mean | p50 | p90 | p99 | max | over 512 tok |
|---|---|---|---|---|---|---|
| en | 77.2 | 72 | 117 | 172 | 255 | 0% |
| hi | 102.0 | 87 | 140 | 220 | 5734 | 0.36% |

**Hindi costs ~32% more tokens than English for identical content** under this
tokenizer, so a character-based budget produces wildly different real chunk sizes
per language. And textbook 256–512 token budgets never fire on this corpus — every
strategy would collapse to one chunk per passage. The configured budgets (48–128
tokens) are the ones that actually bite.

### Results

15,000 passages sampled evenly across all three languages, 300 held-out queries
with known relevant passages, exact search. Scoring is at **passage level** after
deduping chunks back to source, so a strategy emitting six fragments of one
document cannot inflate its own recall.

| Strategy | Chunks | R@1 | R@5 | MRR@10 | R@5 en | R@5 hi | R@5 gu | Build |
|---|---|---|---|---|---|---|---|---|
| **passage_native** | 15,119 | **0.327** | **0.779** | **0.515** | 0.944 | 0.755 | 0.626 | 1.8s |
| recursive | 21,864 | 0.332 | 0.753 | 0.504 | 0.942 | 0.760 | 0.545 | 6.9s |
| indic_aware | 21,450 | 0.345 | 0.741 | 0.511 | 0.932 | 0.750 | 0.530 | 3.3s |
| semantic | 26,958 | 0.361 | 0.724 | 0.519 | 0.943 | 0.693 | 0.520 | 719.3s |
| sliding | 32,193 | 0.335 | 0.720 | 0.514 | 0.915 | 0.672 | 0.561 | 1.8s |
| fixed | 29,983 | 0.308 | 0.698 | 0.476 | 0.918 | 0.661 | 0.500 | 6.7s |
| parent_child | 43,822 | 0.323 | 0.656 | 0.470 | 0.918 | 0.646 | 0.389 | 3.0s |

**We chose `passage_native`, and it is not the answer we expected.** The plan this
was built from assumed parent-child would win. It came **last** on R@5, produced
the largest index, and was slowest to search.

MSMARCO passages are *already* human-curated retrieval units. Splitting them
destroys information rather than sharpening it — parent_child's 33-token children
are too small to retain a retrievable idea, and on Gujarati it collapses to
**0.389** against passage_native's 0.626.

Three further findings:

- **`semantic` cost 719s of build time to finish fourth** — 400x
  `passage_native`'s 1.8s for 5 points less R@5. Embedding-boundary detection is a
  real technique that is not worth its price here.
- **`indic_aware` wins R@1 (0.345)** while `passage_native` wins R@5 and MRR. We
  optimise for R@5 because the generator receives the top 5; a purely extractive
  top-1 system would defensibly choose differently. The metric you optimise has to
  follow from how the system consumes retrieval.
- **Gujarati is hardest everywhere** (0.389–0.626 vs English 0.918–0.944). This is
  a property of the encoder and it propagates into the guardrails below.

**Scope note:** this says MSMARCO-XI is pre-chunked, not that chunking doesn't
matter. On raw PDFs or transcripts the ranking would likely invert, which is why
all seven strategies remain behind a config switch rather than being deleted.

---

## Latency

220 MSMARCO-XI validation queries per mode, 10 warm-up runs discarded. Full
breakdown in [`bench/results.md`](bench/results.md), raw data in `bench/results.json`.

| Mode | P50 | **P70** | P90 | P95 | P99 | **P100** | Within 200ms |
|---|---|---|---|---|---|---|---|
| cold (no cache) | 11.83 | **14.59** | 23.56 | 29.48 | 38.35 | **41.57** | **100%** |
| warm (cache on) | 4.18 | **12.05** | 16.89 | 20.67 | 24.89 | **37.88** | **100%** |
| generative (LLM forced) | 15.22 | 15.94 | 16.99 | 17.42 | 17.58 | **17.60** | **100%** |
| **deployed, Cloud Run 4 vCPU** | **56.9** | **75.8** | **101.3** | — | — | **152.9** | **100%** |

**P100 — the actual worst case, not a percentile that hides one — is 41.6ms
locally and 152.9ms on the deployed instance.** Both cold and warm are reported;
quoting only the warm number is the standard way to make a RAG pipeline look
faster than it is.

### By stage (cold)

| Stage | P50 | P70 | P95 | P100 |
|---|---|---|---|---|
| extract | 9.96 | 13.64 | 22.67 | 34.47 |
| embed | 2.31 | 2.52 | 9.61 | 10.96 |
| retrieve | 1.77 | 1.86 | 4.55 | 6.19 |
| guard_input | 0.01 | 0.01 | 0.03 | 0.05 |
| guard_relevance | 0.00 | 0.00 | 0.01 | 0.02 |

Guardrails cost essentially nothing — a consequence of the cascade design, not of
skipping work. `extract` is now the dominant stage: it embeds the top passage's
sentences to pick an answer span. Caching those at index time would remove most of
it; that is not done.

### What is deliberately outside the core number

| Stage | P50 | P100 | Why excluded |
|---|---|---|---|
| LLM generation (Groq `openai/gpt-oss-120b`) | 956.8 | 1107.7 | Third-party network round trip |
| Sarvam STT | ~340–620 per utterance | — | Third-party network round trip |

Both are shown as separate, differently-coloured bars in the UI rather than folded
into the headline. **The voice path does not wait for either**: the browser's own
speech recognition produces a live transcript, retrieval runs speculatively on
interim results while you are still speaking, and Sarvam verifies the transcript
*after* the answer is already on screen.

---

## Guardrails — knowing when not to answer

| Layer | Catches | Reason code |
|---|---|---|
| `input_guard` | Prompt injection, unsafe requests, empty/garbage transcripts, unsupported language | `PROMPT_INJECTION`, `UNSAFE_INPUT`, `EMPTY_INPUT`, `UNSUPPORTED_LANGUAGE` |
| `relevance_guard` | Questions the corpus cannot answer | `OUT_OF_CORPUS`, `LOW_CONFIDENCE` |
| `grounding_guard` | Answers not supported by retrieved text | `UNGROUNDED_OUTPUT` |
| Citation enforcement | Individual invented sentences inside an otherwise grounded answer | (sentences stripped) |

Refusal messages are written in the user's own language, because a Hindi speaker
receiving an English refusal has been failed twice.

### Thresholds are measured, per language

`guardrails/calibrate.py` sweeps 600 labelled in-corpus queries against a
deliberately-constructed out-of-corpus set (`data/ood_queries.json`: local,
personal, time-bound and self-referential questions).

| Lang | In-corpus mean | In p10 | OOD mean | OOD p90 | Separation | Threshold | Accepts in-corpus | Accepts OOD |
|---|---|---|---|---|---|---|---|---|
| en | 0.907 | 0.878 | 0.853 | 0.885 | −0.008 | 0.884 | 86.0% | 10.0% |
| hi | 0.897 | 0.862 | 0.859 | 0.875 | −0.013 | 0.880 | 72.6% | 6.2% |
| gu | 0.872 | 0.855 | 0.857 | 0.876 | **−0.021** | 0.877 | 29.8% | 6.2% |

multilingual-e5 scores are not comparable across scripts, so a single global
threshold over-refuses the lowest-resource language while being most permissive
exactly where the encoder is weakest.

### Grounding

| Signal | Grounded mean | Ungrounded mean | Threshold | Keeps | Admits |
|---|---|---|---|---|---|
| Token overlap | 0.806 | 0.012 | 0.167 | 98.7% | **0.7%** |
| Embedding similarity | 0.910 | 0.740 | 0.803 | 98.0% | 2.7% |

Token overlap is the stronger signal *and* ~600x cheaper, so the checks cascade:
overlap decides the common case, and the encoder runs only when overlap says no —
exactly the abstractive-paraphrase case where lexical overlap misleads.

### Five mistakes this process caught

Documented because the debugging is the interesting part:

1. **Calibrating on the wrong distribution.** Thresholds were fit on
   *(answer, single passage)* pairs but applied to *(answer, five-passage
   concatenation)*. Concatenation dilutes similarity, and the guard began refusing
   correct answers. Now scored per-passage and maxed, matching how it was fit.
2. **A degenerate operating point.** Selecting "max TPR subject to FPR ≤ target"
   collapses when classes separate cleanly — every threshold in the gap satisfies
   the cap, so it returns the lowest. It picked a token-overlap threshold of
   **0.026**, which filters nothing. Maximising Youden's J under the cap fixed it.
3. **An FPR cap that broke usability.** At 202k passages the classes overlap more —
   a larger corpus means more questions have *something* related — and an
   FPR-only cap drove Gujarati to **29.8%** acceptance, refusing two thirds of
   answerable questions. A guard that refuses most real questions is not safe, it
   is broken. The selection now floors TPR at 65% and reports the false-accept
   rate that costs.
4. **Over-correcting the fix for #3.** Flooring TPR at 65% pushed Gujarati false
   accepts to 25%, and 3 of 5 out-of-corpus Gujarati questions started being
   *answered*. Under-refusing breaks the requirement the guard exists to satisfy,
   while over-refusing is a usability problem the general-knowledge fallback
   already solves. The floor was reverted and the FPR cap governs.
5. **A hypothesis that did not survive.** Score *margin* (top-1 minus the mean of
   the next four) was expected to separate in-corpus from out-of-corpus better
   than raw top-1. Measured: AUC **0.740** against top-1's **0.907**, and
   combining them beats neither. `min_margin` remains configurable and off.

---

## Harness

- **Pydantic contracts** at every boundary — `AskRequest`, `Chunk`,
  `RetrievalResult`, `Answer`, `StageTrace`
- **Explicit state machine** in `harness/orchestrator.py`; every transition traced,
  every stage under a timeout budget, every exit path producing the same response
  envelope so clients never branch on shape
- **Provider chain** Groq → Sarvam → extractive, each with its own circuit breaker
- **Structured output with a repair loop** — the model must return
  `{answer, citations[], confidence}`; one repair attempt, then extractive fallback
- **Deliberately shallow retries** (2 attempts) — under a latency budget, a third
  attempt is almost always worse than failing over
- **Tool-call surface** — `search_kb`, `rerank`, `answer_extractive`,
  `answer_generative`, `refuse`, routed between rather than inlined

**The system answers with zero API keys configured.** Retrieval and extraction are
entirely local; the provider chain terminates in extractive answering rather than
an error. Observed under real failure: when Groq rate-limited during benchmarking,
**42 of 60 requests degraded to grounded extractive answers** with reason code
`PROVIDER_UNAVAILABLE` instead of returning errors.

### Strict RAG

Refusing what the corpus cannot answer is the graded behaviour, so **Strict RAG is
on by default**. Unticking it falls back to the model's own knowledge on a separate
`general` path that is never marked grounded, returns no citations, carries
`ANSWERED_FROM_GENERAL_KNOWLEDGE`, and is styled differently in the UI. The
relevance guard still runs and still fires — this is what we do with its verdict,
not a bypass of it.

---

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python data/build_corpus.py      # streams MSMARCO-XI -> data/corpus.jsonl
python index/build_index.py      # -> .artifacts/dense/
uvicorn api.main:app --port 7860
```

Keys are optional; copy `.env.example` to `.env` to enable voice
(`SARVAM_API_KEY`) and generative answers (`GROQ_API_KEY`). Deployment steps are in
[`DEPLOY.md`](DEPLOY.md).

### API

| Route | Purpose |
|---|---|
| `POST /ask` | `{query, lang?, top_k?, use_cache?, allow_generative?, allow_general?}` → grounded answer |
| `POST /ask-voice` | multipart audio → transcript + answer, `stt_ms` reported separately |
| `POST /transcribe` | transcript only, so Sarvam never gates the answer |
| `GET /health` | index size, search backend, provider status, circuit-breaker state, cache stats |
| `GET /metrics` | counts by answer path, refusals, cache hit rate |

### Layout

```
chunking/     7 strategies behind one ABC + the evaluation harness
index/        ONNX embedder, dense matmul search, Qdrant store, BM25, ingestion CLI
retrieval/    RRF fusion, semantic cache, extractive fast-path, optional rerank
harness/      schemas, orchestrator, provider chain, tracing
guardrails/   input / relevance / grounding guards, policy, calibration
stt/          Sarvam speech-to-text
api/ web/     FastAPI app and the frontend with the live latency HUD
bench/        latency benchmark, percentiles, chart
tests/        53 unit tests
```

---

## Known limitations

- **Gujarati relevance gating is the weakest link.** Separation is negative
  (−0.021), so no threshold serves both goals: at a 6.2% false-accept rate it
  accepts only 29.8% of answerable Gujarati questions, and buying acceptance back
  to 76.2% costs a 25% false-accept rate. It ships conservative — with Strict RAG
  on it over-refuses Gujarati, and the general-knowledge fallback covers those
  users when Strict is off. A stronger multilingual encoder is the real fix;
  `multilingual-e5-large` is 2.24GB and does not fit the footprint.
- **`extract` is the dominant stage** (9.96ms P50). Caching the top passage's
  sentence embeddings at index time would remove most of it. Not done.
- **Speculative retrieval only helps spoken input.** Typed queries get no
  head start; the transcript is what arrives incrementally.
- **The semantic cache is per-process and in-memory.** Multiple workers would not
  share it. Correct for a single container, wrong for horizontal scaling.
- **Cross-encoder reranking is off.** It costs 40–70ms and `ms-marco-MiniLM` is
  English-only; running it on Devanagari produces confident nonsense.
- **`passage_native` winning is corpus-specific**, as discussed above.

## Reproducing

```bash
python chunking/evaluate.py --limit 15000 --queries 300       # -> chunking/comparison.md
python guardrails/calibrate.py --queries-file data/queries.jsonl \
       --corpus-file data/corpus.jsonl --write                # -> guardrails/calibration.md
python bench/run_bench.py -n 220 --queries-file data/queries.jsonl
pytest tests/ -q
```
