# Latency benchmark

Measured over 220 MSMARCO-XI validation queries per mode, 10 warm-up runs discarded. Core = embed -> retrieve -> guards -> answer, which is the span the 200ms requirement is scoped to.

## Headline

| Mode | P50 | P70 | P90 | P95 | P99 | P100 | Within 200ms |
|---|---|---|---|---|---|---|---|
| cold (no cache) | 25.74 | 31.52 | 40.23 | 45.7 | 59.1 | 79.71 | 100.0% |
| warm (cache on) | 13.63 | 24.57 | 32.18 | 37.13 | 49.25 | 62.45 | 100.0% |

## cold (no cache) — by answer path

| Path | Share | P50 | P70 | P95 | P100 |
|---|---|---|---|---|---|
| extractive | 77.3% | 28.79 | 33.49 | 48.29 | 79.71 |
| refused | 22.7% | 12.37 | 13.43 | 16.3 | 19.38 |

### cold (no cache) — by stage

| Stage | P50 | P70 | P95 | P100 |
|---|---|---|---|---|
| extract | 15.82 | 19.78 | 32.88 | 69.0 |
| retrieve | 9.0 | 10.55 | 14.22 | 18.47 |
| embed | 3.29 | 3.6 | 4.32 | 5.31 |
| guard_input | 0.01 | 0.01 | 0.02 | 0.07 |
| guard_relevance | 0.01 | 0.01 | 0.01 | 0.02 |

### cold (no cache) — by language

| Lang | P50 | P95 | n |
|---|---|---|---|
| en | 26.67 | 41.51 | 67 |
| gu | 25.38 | 44.06 | 73 |
| hi | 25.57 | 54.67 | 80 |

## warm (cache on) — by answer path

| Path | Share | P50 | P70 | P95 | P100 |
|---|---|---|---|---|---|
| cache | 24.5% | 3.13 | 3.39 | 3.64 | 4.21 |
| extractive | 53.6% | 25.2 | 29.54 | 42.61 | 62.45 |
| refused | 21.8% | 11.22 | 12.33 | 13.67 | 14.7 |

### warm (cache on) — by stage

| Stage | P50 | P70 | P95 | P100 |
|---|---|---|---|---|
| extract | 13.99 | 17.08 | 28.25 | 53.63 |
| retrieve | 8.43 | 9.25 | 11.6 | 16.47 |
| embed | 3.0 | 3.2 | 3.65 | 4.17 |
| cache | 0.01 | 0.01 | 0.02 | 2.65 |
| guard_input | 0.01 | 0.01 | 0.02 | 0.04 |
| guard_relevance | 0.0 | 0.01 | 0.01 | 0.01 |

### warm (cache on) — by language

| Lang | P50 | P95 | n |
|---|---|---|---|
| en | 22.3 | 35.26 | 59 |
| gu | 12.22 | 36.59 | 71 |
| hi | 13.04 | 41.18 | 90 |
