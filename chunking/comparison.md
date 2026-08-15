# Chunking strategy comparison

- Corpus: 12,000 MSMARCO-XI passages (en, gu, hi)
- Held-out queries: 300, sampled with seed 13
- Encoder: `intfloat/multilingual-e5-small` (ONNX, CPU)
- Scoring is passage-level after deduping chunks back to their source passage.
- Retrieval is exact inner-product search, so the numbers isolate the chunker
  rather than mixing in approximate-index recall.
- Queries are filtered to their own language, matching production behaviour.

| Strategy | Chunks | Chk/Psg | Mean tok | P95 tok | R@1 | R@5 | R@10 | MRR@10 | R@5 en | R@5 gu | R@5 hi | Build s | Embed s | Search P50 ms | Index MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| passage_native | 12,074 | 1.01 | 78 | 137 | 0.324 | **0.801** | **0.893** | 0.521 | 0.945 | 0.713 | 0.748 | 1.4 | 401.7 | 0.35 | 19 |
| indic_aware | 17,076 | 1.42 | 66 | 93 | 0.323 | 0.766 | 0.853 | 0.507 | 0.914 | 0.612 | 0.766 | 2.5 | 195.6 | 0.57 | 26 |
| recursive | 17,406 | 1.45 | 66 | 93 | 0.318 | 0.754 | 0.853 | 0.508 | 0.914 | 0.564 | 0.775 | 4.7 | 189.1 | 0.51 | 27 |
| sliding | 25,707 | 2.14 | 50 | 64 | **0.344** | 0.739 | 0.882 | **0.524** | 0.875 | 0.601 | 0.739 | 1.4 | 181.9 | 0.76 | 39 |
| semantic | 21,741 | 1.81 | 44 | 91 | 0.337 | 0.719 | 0.840 | 0.501 | 0.909 | 0.511 | 0.729 | 462.0 | 396.1 | 0.70 | 33 |
| fixed | 23,988 | 2.00 | 45 | 64 | 0.298 | 0.704 | 0.797 | 0.475 | 0.857 | 0.537 | 0.711 | 1.8 | 128.7 | 0.78 | 37 |
| parent_child | 35,092 | 2.92 | 33 | 48 | 0.288 | 0.646 | 0.724 | 0.444 | 0.864 | 0.351 | 0.706 | 2.4 | 189.6 | 1.09 | 54 |
