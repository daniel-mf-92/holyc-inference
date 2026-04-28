# Eval Input Audit

Generated: 2026-04-28T12:25:45Z
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

## Prediction Distribution

| Engine | Histogram | Majority | Majority % |
| --- | --- | --- | ---: |
| holyc | {"0": 3} | 0 | 100.00 |
| llama | {"0": 3} | 0 | 100.00 |

## Score Coverage

| Engine | Scored predictions | Coverage % | Score lengths |
| --- | ---: | ---: | --- |
| holyc | 1/3 | 33.33 | {"4": 1} |
| llama | 1/3 | 33.33 | {"4": 1} |

## Issues

No input issues found.
