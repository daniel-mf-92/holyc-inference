# Benchmark Matrix

Generated: 2026-04-28T06:20:15Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Variability gates: suite CV <= 0.1, prompt CV <= 0.1

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt suite | Command SHA256 | Runs | Warmups | Median tok/s | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | d1ee8f0158c3 | pass | 2 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | d1ee8f0158c3 | pass | 2 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 67207168 | 0 |
