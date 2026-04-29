# Eval Compare Report

Generated: 2026-04-29T05:05:04Z
Dataset: smoke-eval
Split: validation
Quantization: Q4_0
Model: synthetic-smoke
Status: pass

## Summary

| Metric | Value |
| --- | ---: |
| Records | 3 |
| HolyC accuracy | 1.0000 |
| llama.cpp accuracy | 1.0000 |
| Accuracy delta | 0.0000 |
| HolyC macro F1 | 1.0000 |
| llama.cpp macro F1 | 1.0000 |
| Macro F1 delta | 0.0000 |
| Mean gold NLL delta | -0.1845 |
| Agreement | 1.0000 |

## Paired Correctness

| Both correct | Both wrong | HolyC only correct | llama.cpp only correct | Discordant | McNemar p-value | Method |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 0 | 0 | 0 | 0 | 1.000000 | exact_binomial_two_sided |

## Score Calibration

| Engine | Score coverage | Mean confidence | Accuracy when scored | Brier score | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| HolyC | 1/3 (0.3333) | 0.9993 | 1.0000 | 0.0000 | 0.0007 |
| llama.cpp | 1/3 (0.3333) | 0.8310 | 1.0000 | 0.0432 | 0.1690 |

## Score NLL

| Engine | Score coverage | Mean gold NLL | Median gold NLL | Perplexity | Mean correct NLL | Mean wrong NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HolyC | 1/3 (0.3333) | 0.0007 | 0.0007 | 1.0007 | 0.0007 | 0.0000 |
| llama.cpp | 1/3 (0.3333) | 0.1852 | 0.1852 | 1.2034 | 0.1852 | 0.0000 |

## Score Ranking

| Engine | Score coverage | Top-1 | Top-2 | Top-3 | Mean gold rank | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HolyC | 1/3 (0.3333) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| llama.cpp | 1/3 (0.3333) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Score Margins

| Engine | Score coverage | Mean margin | Median margin | P10 margin | Min margin | Mean correct | Mean wrong | Low-margin rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HolyC | 1/3 (0.3333) | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.0000 | 0 <= 0.1000 (0.0000) |
| llama.cpp | 1/3 (0.3333) | 0.7185 | 0.7185 | 0.7185 | 0.7185 | 0.7185 | 0.0000 | 0 <= 0.1000 (0.0000) |

## Score Ties

| Engine | Score coverage | Top-score tied rows | Tie rate | Max tied choices |
| --- | ---: | ---: | ---: | ---: |
| HolyC | 1/3 (0.3333) | 0 | 0.0000 | 1 |
| llama.cpp | 1/3 (0.3333) | 0 | 0.0000 | 1 |

## Dataset Breakdown

| Dataset | Split | Records | HolyC accuracy | llama.cpp accuracy | Accuracy delta | Agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| arc-smoke | validation | 1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| hellaswag-smoke | validation | 1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| truthfulqa-smoke | validation | 1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |

## Confidence Intervals

| Metric | Point | Lower | Upper | Confidence | Method |
| --- | ---: | ---: | ---: | ---: | --- |
| holyc_accuracy | 1.0000 | 0.4385 | 1.0000 | 0.95 | wilson |
| llama_accuracy | 1.0000 | 0.4385 | 1.0000 | 0.95 | wilson |
| agreement | 1.0000 | 0.4385 | 1.0000 | 0.95 | wilson |

## Quality Gates

No quality gate regressions.

## Per-Answer F1

| Answer index | HolyC support | HolyC F1 | llama.cpp support | llama.cpp F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 1.0000 | 3 | 1.0000 |
| 1 | 0 | 0.0000 | 0 | 0.0000 |
| 2 | 0 | 0.0000 | 0 | 0.0000 |
| 3 | 0 | 0.0000 | 0 | 0.0000 |

## Confusion Matrices

Rows are gold answer indexes; columns are predicted answer indexes.

### HolyC

| Gold \ Pred | 0 | 1 | 2 | 3 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |

### llama.cpp

| Gold \ Pred | 0 | 1 | 2 | 3 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |

## Disagreements

No prediction disagreements.
