# Guardrail threshold calibration

Thresholds below are measured, not chosen by feel. Regenerate with `python guardrails/calibrate.py --write`.

## Relevance guard (OUT_OF_CORPUS)

- In-corpus queries: 600, out-of-corpus: 52
- In-corpus top dense score: mean 0.893, p05 0.854
- Out-of-corpus top dense score: mean 0.847, p95 0.879
- **Chosen threshold 0.871** -> accepts 76.5% of in-corpus queries, 7.7% of out-of-corpus queries

Operating point favours refusing over answering: a wrong answer to an out-of-corpus question is worse than an unnecessary refusal.

| Threshold | Accepts in-corpus (TPR) | Accepts out-of-corpus (FPR) |
|---|---|---|
| 0.778 | 100.0% | 100.0% |
| 0.834 | 99.3% | 78.8% |
| 0.846 | 98.3% | 50.0% |
| 0.857 | 92.3% | 26.9% |
| 0.867 | 80.3% | 11.5% |
| 0.877 | 68.7% | 5.8% |
| 0.887 | 57.7% | 3.8% |
| 0.897 | 45.2% | 3.8% |
| 0.907 | 32.0% | 0.0% |
| 0.917 | 20.7% | 0.0% |
| 0.927 | 10.5% | 0.0% |
| 0.937 | 3.7% | 0.0% |
| 0.949 | 0.8% | 0.0% |

### Per-language thresholds

multilingual-e5 scores are not comparable across scripts. In-corpus English queries sit well above the out-of-corpus band; Gujarati barely clears it. A single global threshold over-refuses the lowest-resource language, so each gets its own operating point.

| Lang | In-corpus mean | In p10 | OOD mean | OOD p90 | Separation | Threshold | TPR | FPR |
|---|---|---|---|---|---|---|---|---|
| en | 0.909 | 0.881 | 0.844 | 0.868 | +0.012 | **0.867** | 94.2% | 10.0% |
| gu | 0.874 | 0.856 | 0.848 | 0.865 | -0.009 | **0.871** | 52.3% | 6.2% |
| hi | 0.896 | 0.860 | 0.851 | 0.865 | -0.005 | **0.871** | 83.0% | 6.2% |

`Separation` is in-corpus p10 minus out-of-corpus p90: positive means the two distributions are cleanly apart at those quantiles. It is comfortably positive for English and negative for Gujarati, which is the honest limit of this guard on the lowest-resource language rather than something to paper over.


## Grounding guard (UNGROUNDED_OUTPUT)

Built from 300 answer/passage pairs. Positives pair each gold answer with the passage it came from; negatives pair the same answer with a random unrelated passage.

- Embedding similarity: grounded mean 0.905 vs ungrounded 0.746 -> **threshold 0.804** (keeps 94.3% of grounded, admits 2.7% of ungrounded)
- Token overlap: grounded mean 0.809 vs ungrounded 0.018 -> **threshold 0.188** (keeps 96.7%, admits 2.3%)

An answer must clear both. They fail differently: similarity catches an answer that wandered off topic, overlap catches fluent text that invented specifics the passage never contained.
