# Benchmark Matrix

Generated: 2026-04-28T10:56:14Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Variability gates: suite CV <= 0.1, prompt CV <= 0.1

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt suite | Command SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | c8e896a45950 | pass | 2 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 544.084 | 12400.000 | -69.840 | 6250.000 | 1884.995 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | c8e896a45950 | pass | 2 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 242.242 | 12400.000 | -33.471 | 6250.000 | 4158.052 | 67207168 | 0 |
