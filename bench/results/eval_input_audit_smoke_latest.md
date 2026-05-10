# Eval Input Audit

Generated: 2026-04-30T02:54:43Z
Status: pass
Dataset: smoke-eval
Split: validation
Model: synthetic-smoke
Quantization: Q4_0

## Summary

| Metric | Value |
| --- | ---: |
| Gold records | 3 |
| HolyC valid predictions | 3 |
| llama.cpp valid predictions | 3 |
| Errors | 0 |
| Warnings | 0 |

## Gold Distribution

| Answer histogram | Choice counts | Majority | Majority % |
| --- | --- | --- | ---: |
| {"0": 3} | {"4": 3} | 0 | 100.00 |

Choice gates: min=4, max=4

## Prediction Distribution

| Engine | Histogram | Majority | Majority % |
| --- | --- | --- | ---: |
| holyc | {"0": 3} | 0 | 100.00 |
| llama | {"0": 3} | 0 | 100.00 |

## Score Coverage

| Engine | Scored predictions | Coverage % | Top-score ties | Tie % | Score lengths |
| --- | ---: | ---: | ---: | ---: | --- |
| holyc | 1/3 | 33.33 | 0 | 0.00 | {"4": 1} |
| llama | 1/3 | 33.33 | 0 | 0.00 | {"4": 1} |

## Score Margins

| Engine | Min margin | Low margins | Low margin % |
| --- | ---: | ---: | ---: |
| holyc | 8 | 0 | 0.00 |
| llama | 2 | 0 | 0.00 |

## Input Hash Parity

| Engine | Prompt matches | Prompt missing | Prompt mismatches | Choices matches | Choices missing | Choices mismatches | Input matches | Input missing | Input mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| holyc | 0 | 3 | 0 | 0 | 3 | 0 | 0 | 3 | 0 |
| llama | 0 | 3 | 0 | 0 | 3 | 0 | 0 | 3 | 0 |

## Issues

No input issues found.
