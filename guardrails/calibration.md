# Guardrail threshold calibration

Thresholds below are measured, not chosen by feel. Regenerate with `python guardrails/calibrate.py --write`.

## Relevance guard (OUT_OF_CORPUS)

- In-corpus queries: 600, out-of-corpus: 52
- In-corpus top dense score: mean 0.891, p05 0.846
- Out-of-corpus top dense score: mean 0.842, p95 0.878
- **Chosen threshold 0.867** -> accepts 76.5% of in-corpus queries, 9.6% of out-of-corpus queries

Operating point favours refusing over answering: a wrong answer to an out-of-corpus question is worse than an unnecessary refusal.

| Threshold | Accepts in-corpus (TPR) | Accepts out-of-corpus (FPR) |
|---|---|---|
| 0.767 | 100.0% | 100.0% |
| 0.828 | 99.7% | 78.8% |
| 0.841 | 96.8% | 51.9% |
| 0.852 | 91.0% | 30.8% |
| 0.863 | 80.7% | 15.4% |
| 0.874 | 68.5% | 7.7% |
| 0.885 | 55.5% | 3.8% |
| 0.896 | 44.0% | 3.8% |
| 0.907 | 30.8% | 0.0% |
| 0.918 | 21.3% | 0.0% |
| 0.929 | 11.5% | 0.0% |
| 0.940 | 4.2% | 0.0% |
| 0.956 | 0.5% | 0.0% |

### Per-language thresholds

multilingual-e5 scores are not comparable across scripts. In-corpus English queries sit well above the out-of-corpus band; Gujarati barely clears it. A single global threshold over-refuses the lowest-resource language, so each gets its own operating point.

| Lang | In-corpus mean | In p10 | OOD mean | OOD p90 | Separation | Threshold | TPR | FPR |
|---|---|---|---|---|---|---|---|---|
| en | 0.910 | 0.880 | 0.838 | 0.868 | +0.011 | **0.867** | 96.4% | 10.0% |
| gu | 0.869 | 0.844 | 0.844 | 0.862 | -0.017 | **0.864** | 56.8% | 6.2% |
| hi | 0.896 | 0.858 | 0.846 | 0.863 | -0.005 | **0.871** | 83.6% | 6.2% |

`Separation` is in-corpus p10 minus out-of-corpus p90: positive means the two distributions are cleanly apart at those quantiles. It is comfortably positive for English and negative for Gujarati, which is the honest limit of this guard on the lowest-resource language rather than something to paper over.


## Grounding guard (UNGROUNDED_OUTPUT)

Built from 300 answer/passage pairs. Positives pair each gold answer with the passage it came from; negatives pair the same answer with a random unrelated passage.

- Embedding similarity: grounded mean 0.888 vs ungrounded 0.747 -> **threshold 0.794** (keeps 89.0% of grounded, admits 3.3% of ungrounded)
- Token overlap: grounded mean 0.778 vs ungrounded 0.010 -> **threshold 0.231** (keeps 91.7%, admits 0.7%)

An answer must clear both. They fail differently: similarity catches an answer that wandered off topic, overlap catches fluent text that invented specifics the passage never contained.
