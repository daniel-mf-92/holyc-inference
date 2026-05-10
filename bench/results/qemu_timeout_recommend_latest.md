# QEMU Timeout Recommendations

Status: **pass**

| Groups | Samples | Findings | Max recommended timeout |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 0 | 30 |

| Benchmark | Profile | Model | Quantization | Samples | P95 wall s | Current timeout s | Recommended timeout s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| qemu_prompt | ci-airgap-smoke | synthetic-smoke | Q4_0 | 4 | 0.687427 | 5 | 30 |
