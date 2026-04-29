# Benchmark Matrix

Generated: 2026-04-29T09:44:56Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Coverage gates: expect cells = 2
Variability gates: suite CV <= -, prompt CV <= 0.1, suite IQR <= -, prompt IQR <= -

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Total tokens | Total elapsed us | Prompt suite | Command SHA256 | Launch plan SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Host child tok/CPU s | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | b397ba037624 | pass | 2 | 120 | 49-71 | 160 | 1000000 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 0e6e0a7f4f79bd3404fa242861cd90d1b5e7ec08fce24797dc36bddcb8535a86 | 4 | 2 | 160.000 | 610.277 | 12400.000 | -72.662 | 57410.000 | 87.546 | 696.540 | 589824 | 6250.000 | 1708.635 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | b397ba037624 | pass | 2 | 120 | 49-71 | 160 | 1000000 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 0e6e0a7f4f79bd3404fa242861cd90d1b5e7ec08fce24797dc36bddcb8535a86 | 4 | 2 | 160.000 | 741.727 | 12400.000 | -78.213 | 43943.000 | 86.770 | 855.818 | 458752 | 6250.000 | 1361.703 | 67207168 | 0 |
