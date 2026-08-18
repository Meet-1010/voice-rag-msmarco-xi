# Guardrail threshold calibration

Thresholds below are measured, not chosen by feel. Regenerate with `python guardrails/calibrate.py --write`.

## Relevance guard (OUT_OF_CORPUS)

- In-corpus queries: 600, out-of-corpus: 52
- In-corpus top dense score: mean 0.893, p05 0.854
- Out-of-corpus top dense score: mean 0.856, p95 0.898
- **Chosen threshold 0.864** -> accepts 85.0% of in-corpus queries, 25.0% of out-of-corpus queries

Operating point favours refusing over answering: a wrong answer to an out-of-corpus question is worse than an unnecessary refusal.

| Threshold | Accepts in-corpus (TPR) | Accepts out-of-corpus (FPR) |
|---|---|---|
| 0.782 | 100.0% | 100.0% |
| 0.840 | 99.2% | 73.1% |
| 0.849 | 97.7% | 55.8% |
| 0.858 | 91.7% | 40.4% |
| 0.867 | 81.0% | 25.0% |
| 0.876 | 68.8% | 15.4% |
| 0.885 | 58.8% | 7.7% |
| 0.894 | 48.8% | 7.7% |
| 0.903 | 37.2% | 3.8% |
| 0.912 | 27.7% | 3.8% |
| 0.921 | 15.5% | 1.9% |
| 0.930 | 8.5% | 1.9% |
| 0.939 | 2.8% | 1.9% |
| 0.959 | 0.0% | 0.0% |

### Per-language thresholds

multilingual-e5 scores are not comparable across scripts. In-corpus English queries sit well above the out-of-corpus band; Gujarati barely clears it. A single global threshold over-refuses the lowest-resource language, so each gets its own operating point.

| Lang | In-corpus mean | In p10 | OOD mean | OOD p90 | Separation | Threshold | TPR | FPR |
|---|---|---|---|---|---|---|---|---|
| en | 0.907 | 0.878 | 0.853 | 0.885 | -0.008 | **0.884** | 86.0% | 10.0% |
| gu | 0.872 | 0.855 | 0.857 | 0.876 | -0.021 | **0.862** | 76.2% | 25.0% |
| hi | 0.897 | 0.862 | 0.859 | 0.875 | -0.013 | **0.880** | 72.6% | 6.2% |

`Separation` is in-corpus p10 minus out-of-corpus p90: positive means the two distributions are cleanly apart at those quantiles. It is comfortably positive for English and negative for Gujarati, which is the honest limit of this guard on the lowest-resource language rather than something to paper over.


## Grounding guard (UNGROUNDED_OUTPUT)

Built from 300 answer/passage pairs. Positives pair each gold answer with the passage it came from; negatives pair the same answer with a random unrelated passage.

- Embedding similarity: grounded mean 0.910 vs ungrounded 0.740 -> **threshold 0.803** (keeps 98.0% of grounded, admits 2.7% of ungrounded)
- Token overlap: grounded mean 0.806 vs ungrounded 0.012 -> **threshold 0.167** (keeps 98.7%, admits 0.7%)

An answer must clear both. They fail differently: similarity catches an answer that wandered off topic, overlap catches fluent text that invented specifics the passage never contained.
