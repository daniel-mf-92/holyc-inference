# Eval Efficiency Frontier

- Status: pass
- Scorecard rows: 4
- Frontier rows: 3
- Dominated rows: 1
- Memory-aware: True
- Findings: 0

| cohort | model | quantization | quality | speed | max memory bytes | frontier | dominated by | quality gap | speed gap | memory gap bytes |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| smoke-eval/validation | synthetic-smoke | Q2_K | 0.9 | 120.0 | 60000000 | True |  |  |  |  |
| smoke-eval/validation | synthetic-smoke | Q4_0 | 0.92 | 180.0 | 64000000 | True |  |  |  |  |
| smoke-eval/validation | synthetic-smoke | Q8_0 | 0.98 | 130.0 | 72000000 | True |  |  |  |  |
| smoke-eval/validation | synthetic-smoke | Q3_0 | 0.91 | 150.0 | 68000000 | False | synthetic-smoke:Q4_0:smoke-eval/validation | 0.010000000000000009 | 30.0 | 4000000 |
