# Benchmark Matrix

Generated: 2026-04-28T14:49:09Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Variability gates: suite CV <= 0.1, prompt CV <= 0.1

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Prompt suite | Command SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | fd1f1247258a | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 624.759 | 12400.000 | -74.390 | 46164.000 | 74.438 | 589824 | 6250.000 | 1600.646 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | fd1f1247258a | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 811.458 | 12400.000 | -79.859 | 41405.000 | 86.604 | 442368 | 6250.000 | 1258.781 | 67207168 | 0 |
