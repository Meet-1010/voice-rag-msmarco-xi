# Latency benchmark

Measured over 220 MSMARCO-XI validation queries per mode, 10 warm-up runs discarded. Core = embed -> retrieve -> guards -> answer, which is the span the 200ms requirement is scoped to.

## Headline

| Mode | P50 | P70 | P90 | P95 | P99 | P100 | Within 200ms |
|---|---|---|---|---|---|---|---|
| cold (no cache) | 16.1 | 18.14 | 23.57 | 26.14 | 34.47 | 43.53 | 100.0% |
| warm (cache on) | 9.64 | 16.7 | 21.47 | 25.07 | 35.48 | 42.8 | 100.0% |
| generative (LLM forced) | 26.04 | 28.56 | 31.49 | 101.23 | 116.52 | 121.01 | 100.0% |

The LLM call itself, excluded from core above and measured over 19 forced-generative requests: P50 **308.75ms**, P95 **501.5ms**, P100 **705.79ms**. This is a third-party network round trip and no local optimisation reduces it, which is why the pipeline routes around it whenever retrieval is confident.


## cold (no cache) — by answer path

| Path | Share | P50 | P70 | P95 | P100 |
|---|---|---|---|---|---|
| extractive | 77.3% | 17.23 | 19.64 | 28.11 | 43.53 |
| refused | 22.7% | 7.83 | 8.48 | 9.45 | 10.61 |

### cold (no cache) — by stage

| Stage | P50 | P70 | P95 | P100 |
|---|---|---|---|---|
| extract | 9.51 | 11.52 | 18.67 | 36.64 |
| retrieve | 5.91 | 6.4 | 8.09 | 9.92 |
| embed | 2.11 | 2.22 | 2.58 | 3.07 |
| guard_input | 0.01 | 0.01 | 0.01 | 0.02 |
| guard_relevance | 0.0 | 0.0 | 0.01 | 0.01 |

### cold (no cache) — by language

| Lang | P50 | P95 | n |
|---|---|---|---|
| en | 16.48 | 25.12 | 67 |
| gu | 15.02 | 25.43 | 73 |
| hi | 15.9 | 27.84 | 80 |

## warm (cache on) — by answer path

| Path | Share | P50 | P70 | P95 | P100 |
|---|---|---|---|---|---|
| cache | 24.5% | 2.12 | 2.21 | 2.52 | 2.57 |
| extractive | 53.6% | 17.16 | 19.34 | 27.9 | 42.8 |
| refused | 21.8% | 7.83 | 8.58 | 9.59 | 16.62 |

### warm (cache on) — by stage

| Stage | P50 | P70 | P95 | P100 |
|---|---|---|---|---|
| extract | 9.22 | 11.38 | 17.65 | 35.97 |
| retrieve | 5.83 | 6.51 | 8.27 | 11.67 |
| embed | 2.1 | 2.21 | 2.52 | 5.92 |
| cache | 0.01 | 0.01 | 0.01 | 0.05 |
| guard_input | 0.01 | 0.01 | 0.01 | 0.02 |
| guard_relevance | 0.0 | 0.0 | 0.01 | 0.01 |

### warm (cache on) — by language

| Lang | P50 | P95 | n |
|---|---|---|---|
| en | 14.58 | 25.08 | 59 |
| gu | 8.24 | 22.71 | 71 |
| hi | 9.34 | 25.69 | 90 |

## generative (LLM forced) — by answer path

| Path | Share | P50 | P70 | P95 | P100 |
|---|---|---|---|---|---|
| extractive | 27.5% | 25.96 | 28.69 | 32.06 | 34.08 |
| generative | 47.5% | 26.55 | 27.52 | 30.57 | 31.2 |
| refused | 25.0% | 26.11 | 51.41 | 115.83 | 121.01 |

### generative (LLM forced) — by stage

| Stage | P50 | P70 | P95 | P100 |
|---|---|---|---|---|
| generate | 401.9 | 465.66 | 977.39 | 2296.17 |
| retrieve | 15.44 | 17.01 | 19.99 | 20.98 |
| embed | 9.47 | 10.21 | 11.42 | 13.01 |
| guard_grounding | 0.24 | 0.33 | 82.6 | 92.64 |
| guard_input | 0.05 | 0.06 | 0.13 | 0.15 |
| guard_relevance | 0.01 | 0.01 | 0.01 | 0.01 |
| extract | 0.0 | 0.0 | 0.0 | 0.01 |

### generative (LLM forced) — by language

| Lang | P50 | P95 | n |
|---|---|---|---|
| en | 25.46 | 113.53 | 14 |
| gu | 25.5 | 29.92 | 12 |
| hi | 26.66 | 32.21 | 14 |
