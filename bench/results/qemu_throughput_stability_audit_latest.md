# QEMU Throughput Stability Audit

Status: pass
Rows: 4
Groups: 2
Findings: 0

## Groups

| Profile | Model | Quantization | Prompt | Samples | Wall tok/s min | Wall tok/s median | Wall tok/s CV |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | smoke-code | 2 | 291.910432 | 330.941997 | 0.117941 |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | smoke-short | 2 | 101.893629 | 118.564353 | 0.140605 |

## Findings

No throughput stability findings.
