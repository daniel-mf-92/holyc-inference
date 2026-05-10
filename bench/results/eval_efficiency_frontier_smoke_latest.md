# Eval Efficiency Frontier

- Status: pass
- Scorecard rows: 3
- Frontier rows: 3
- Dominated rows: 0
- Memory-aware: True
- Findings: 0

| cohort | model | quantization | quality | speed | max memory bytes | frontier | dominated by |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| smoke-eval/validation | synthetic-smoke | Q2_K | 0.9 | 120.0 | 60000000 | True |  |
| smoke-eval/validation | synthetic-smoke | Q4_0 | 0.92 | 180.0 | 64000000 | True |  |
| smoke-eval/validation | synthetic-smoke | Q8_0 | 0.98 | 130.0 | 72000000 | True |  |
