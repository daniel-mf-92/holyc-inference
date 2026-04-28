# Benchmark Matrix

Generated: 2026-04-28T15:20:32Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Variability gates: suite CV <= -, prompt CV <= 0.1

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Prompt suite | Command SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Host child tok/CPU s | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | ff0cb4429e89 | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 673.500 | 12400.000 | -75.534 | 48881.000 | 80.989 | 825.772 | 573440 | 6250.000 | 1529.135 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | ff0cb4429e89 | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 484.800 | 12400.000 | -66.044 | 55121.500 | 68.727 | 726.658 | 557056 | 6250.000 | 2122.271 | 67207168 | 0 |
