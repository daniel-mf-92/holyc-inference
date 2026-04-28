# Benchmark Matrix

Generated: 2026-04-28T14:39:40Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Variability gates: suite CV <= -, prompt CV <= 0.1

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Prompt suite | Command SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | 9bcc18706a99 | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 221.100 | 12400.000 | 9.400 | 58104.000 | 37.987 | 15663104 | 6250.000 | 6837.516 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | 9bcc18706a99 | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 4 | 2 | 160.000 | 352.587 | 12400.000 | -40.490 | 50411.500 | 48.128 | 14090240 | 6250.000 | 3719.365 | 67207168 | 0 |
