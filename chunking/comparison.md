# Chunking strategy comparison

- Corpus: 15,000 MSMARCO-XI passages (en, gu, hi)
- Held-out queries: 300, sampled with seed 13
- Encoder: `intfloat/multilingual-e5-small` (ONNX, CPU)
- Scoring is passage-level after deduping chunks back to their source passage.
- Retrieval is exact inner-product search, so the numbers isolate the chunker
  rather than mixing in approximate-index recall.
- Queries are filtered to their own language, matching production behaviour.

| Strategy | Chunks | Chk/Psg | Mean tok | P95 tok | R@1 | R@5 | R@10 | MRR@10 | R@5 en | R@5 gu | R@5 hi | Build s | Embed s | Search P50 ms | Index MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| passage_native | 15,119 | 1.01 | 78 | 137 | 0.327 | **0.779** | **0.857** | 0.515 | 0.944 | 0.626 | 0.755 | 1.8 | 589.4 | 0.54 | 23 |
| recursive | 21,864 | 1.46 | 66 | 93 | 0.332 | 0.753 | 0.815 | 0.504 | 0.942 | 0.545 | 0.760 | 6.9 | 282.7 | 0.84 | 34 |
| indic_aware | 21,450 | 1.43 | 66 | 93 | 0.345 | 0.741 | 0.820 | 0.511 | 0.932 | 0.530 | 0.750 | 3.3 | 292.6 | 1.12 | 33 |
| semantic | 26,958 | 1.80 | 44 | 91 | **0.361** | 0.724 | 0.825 | **0.519** | 0.943 | 0.520 | 0.693 | 719.3 | 584.0 | 1.03 | 41 |
| sliding | 32,193 | 2.15 | 50 | 64 | 0.335 | 0.720 | 0.855 | 0.514 | 0.915 | 0.561 | 0.672 | 1.8 | 272.2 | 1.54 | 49 |
| fixed | 29,983 | 2.00 | 45 | 64 | 0.308 | 0.698 | 0.780 | 0.476 | 0.918 | 0.500 | 0.661 | 6.7 | 480.5 | 1.47 | 46 |
| parent_child | 43,822 | 2.92 | 33 | 48 | 0.323 | 0.656 | 0.735 | 0.470 | 0.918 | 0.389 | 0.646 | 3.0 | 267.9 | 1.87 | 67 |
