# Voice RAG over MSMARCO-XI

Speak a question in English, Hindi or Gujarati; get an answer grounded in a
retrieved passage, with citations, a per-stage latency breakdown, and an explicit
refusal when the corpus does not contain the answer.

Built for HH Goa 2026 Shortlisting Task 2.

```
🎤 audio ──► Sarvam STT ──► text + detected lang
                              │
                              ▼
                        [ guard_input ]  injection / unsafe / garbage / language
                              │
                              ▼
                        [ embed ]  multilingual-e5-small, ONNX, CPU
                              │
                    ┌─────────┴─────────┐
                    ▼                   │
              [ cache ]  cos > 0.95 ────┼──► cached answer  (~2ms)
                    │ miss              │
                    ▼                   │
              [ retrieve ]  Qdrant dense + BM25 sparse ──► RRF fusion
                    │                   │
                    ▼                   │
              [ guard_relevance ]  top cosine < τ ──► REFUSE (OUT_OF_CORPUS)
                    │                   │
                    ▼                   │
              [ extract ]  confident span? ──────────► extractive answer (~15ms)
                    │ not confident     │
                    ▼                   │
              [ generate ]  Groq → Sarvam → extractive fallback
                    │                   │
                    ▼                   │
              [ guard_grounding ]  similarity + token overlap
                    │                   │        └─► REFUSE (UNGROUNDED_OUTPUT)
                    ▼                   ▼
              answer + citations + full stage timings
```

---

## The 200ms question, answered honestly

The task asks for "chunking + vector DB retrieval + everything through to final
output" under 200ms. Two things are true at once, and this project reports both
rather than picking the flattering one:

**The core pipeline meets the target.** Embedding, hybrid retrieval, fusion, all
four guardrails and extractive answering complete well inside 200ms — measured,
not estimated. See the latency table below.

**A generative LLM round trip does not, and cannot.** Groq's time-to-first-token
alone is 100-200ms over the network. Sarvam STT adds another ~150ms+. No amount of
local optimisation changes someone else's network latency. Any submission claiming
sub-200ms *including* a third-party LLM call is measuring something other than what
it says.

So the pipeline is built to mostly not need the LLM:

| Path | What happens | Share of queries | Typical core latency |
|---|---|---|---|
| Cache hit | Query embedding matches a previous one above 0.95 cosine | see bench | ~2ms |
| Extractive | Top passage is confidently retrieved and contains a clean answer span | see bench | ~15ms |
| Generative | Falls through to Groq, then Sarvam, then extractive fallback | see bench | 400-600ms |

The extractive fast-path is the substantive optimisation, and it is not a trick:
MSMARCO passages were selected by human annotators *because* they answer the query,
so the answer span is usually present verbatim. Locating the right sentence costs
one embedding pass over a handful of sentences. Calling a 70B model to paraphrase a
sentence that is already correct is the thing worth avoiding.

STT and generation are timed and reported as separate, differently-coloured bars in
the UI and as separate rows in every results table. They are never folded into the
core number.

---

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Build the corpus and index (the corpus step streams from Hugging Face and takes a
few minutes; a pre-built 18k-passage subset is committed for deployment):

```bash
python data/build_corpus.py          # -> data/corpus.jsonl, data/queries.jsonl
python index/build_index.py          # -> .qdrant/, .artifacts/
```

Run it:

```bash
uvicorn api.main:app --port 7860     # open http://localhost:7860
```

API keys are optional and go in `.env` (see `.env.example`):

- `SARVAM_API_KEY` — needed for `POST /ask-voice`. Text queries work without it.
- `GROQ_API_KEY` — needed for generative answers. Without it the provider chain
  terminates in extractive answering and the system still returns grounded
  answers with citations.

**The system answers with zero API keys configured.** That is a deliberate
property of the provider chain, not an accident: retrieval and extraction are
entirely local.

---

## API

| Route | Purpose |
|---|---|
| `POST /ask` | `{query, lang?, top_k?, use_cache?, allow_generative?}` → grounded answer |
| `POST /ask-voice` | multipart audio → transcript + answer, with `stt_ms` reported separately |
| `GET /health` | index size, provider status, circuit-breaker state, cache stats |
| `GET /metrics` | request counts by answer path, refusal counts, cache hit rate |

Every response carries the same envelope whether it answered or refused, so
clients never branch on shape:

```json
{
  "answer": "...",
  "citations": [{"chunk_id": "...", "passage_id": "en:1102432:3", "text": "...", "score": 0.91}],
  "confidence": 0.88,
  "path": "extractive",
  "refused": false,
  "reason_code": null,
  "timings": {"stages": {"embed": 2.4, "retrieve": 8.1}, "core_ms": 14.2, "within_budget": true},
  "trace": [{"stage": "guard_input", "duration_ms": 0.04, "ok": true}]
}
```

---

## Repository layout

```
chunking/     seven strategies behind one ABC, plus the evaluation harness
index/        ONNX embedder, Qdrant store, BM25, ingestion CLI
retrieval/    RRF fusion, semantic cache, extractive fast-path, optional rerank
harness/      pydantic schemas, orchestrator state machine, provider chain, tracing
guardrails/   input / relevance / grounding guards, refusal policy, calibration
stt/          Sarvam speech-to-text
api/          FastAPI app and routes
web/          vanilla JS frontend with mic capture and live latency HUD
bench/        latency benchmark, percentiles, chart
tests/        51 unit tests over the invariants that matter
```
