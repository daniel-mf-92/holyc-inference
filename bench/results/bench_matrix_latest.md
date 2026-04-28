# Benchmark Matrix

Generated: 2026-04-28T03:52:04Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Variability gates: suite CV <= -, prompt CV <= 0.1

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt suite | Runs | Warmups | Median tok/s | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | a0126ad2c367 | pass | 2 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 4 | 2 | 160.000 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | a0126ad2c367 | pass | 2 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 4 | 2 | 160.000 | 67207168 | 0 |
