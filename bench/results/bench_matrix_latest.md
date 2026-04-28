# Benchmark Matrix

Generated: 2026-04-28T20:07:33Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Variability gates: suite CV <= -, prompt CV <= 0.1

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Total tokens | Total elapsed us | Prompt suite | Command SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Host child tok/CPU s | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | 5025377e127c | pass | 2 | 120 | 49-71 | 160 | 1000000 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 733.992 | 12400.000 | -77.412 | 47377.000 | 84.116 | 853.532 | 589824 | 6250.000 | 1411.740 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | 5025377e127c | pass | 2 | 120 | 49-71 | 160 | 1000000 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 696.294 | 12400.000 | -75.571 | 49428.500 | 85.122 | 815.952 | 540672 | 6250.000 | 1526.792 | 67207168 | 0 |
