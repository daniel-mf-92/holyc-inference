# Perplexity Compare Report

Generated: 2026-04-28T20:28:10Z
Dataset: smoke-eval
Split: validation
Quantization: Q4_0
Model: synthetic-smoke
Status: pass

## Summary

| Metric | HolyC | llama.cpp | Delta |
| --- | ---: | ---: | ---: |
| Records | 3 | 3 | - |
| Tokens | 11 | 11 | - |
| NLL/token | 0.361818 | 0.356364 | 0.005454 |
| Perplexity | 1.435938 | 1.428127 | 0.007811 |
| Perplexity ratio | 1.005469 | 1.000000 | - |
| Token count mismatches | 0 | - | - |
| Mean abs record NLL delta | 0.005833 | - | - |
| Median abs record NLL delta | 0.007500 | - | - |
| P95 abs record NLL delta | 0.010000 | - | - |
| P95 signed record NLL delta | 0.010000 | - | - |
| Max abs record NLL delta | 0.010000 | - | - |

## Quality Gates

No quality gate regressions.

## Largest NLL Deltas

| ID | HolyC NLL/token | llama.cpp NLL/token | Delta |
| --- | ---: | ---: | ---: |
| smoke-arc-1 | 0.350000 | 0.340000 | 0.010000 |
| smoke-truthfulqa-1 | 0.420000 | 0.412500 | 0.007500 |
| smoke-hellaswag-1 | 0.312500 | 0.312500 | 0.000000 |
