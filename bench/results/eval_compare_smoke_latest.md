# Eval Compare Report

Generated: 2026-04-27T22:45:13Z
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
| Agreement | 1.0000 |

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
