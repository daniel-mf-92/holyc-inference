# Benchmark Matrix

Generated: 2026-04-28T17:27:56Z
Status: pass
Matrix: synthetic-command-hash-smoke
Cells: 2
Variability gates: suite CV <= -, prompt CV <= -

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Prompt suite | Command SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Host child tok/CPU s | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | 750b69927e30 | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 1fe7be915be8056f3459da4bb9d1300701c8c673df80addccc4b4a822b33b661 | 4 | 2 | 160.000 | 768.926 | 12400.000 | -77.122 | 41948.000 | 78.615 | 962.432 | 458752 | 6250.000 | 1429.880 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | 750b69927e30 | pass | 2 | 120 | 49-71 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | 1fe7be915be8056f3459da4bb9d1300701c8c673df80addccc4b4a822b33b661 | 4 | 2 | 160.000 | 704.570 | 12400.000 | -76.575 | 42801.500 | 76.805 | 919.302 | 13336576 | 6250.000 | 1464.083 | 67207168 | 0 |
