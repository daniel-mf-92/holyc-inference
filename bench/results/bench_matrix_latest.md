# Benchmark Matrix

Generated: 2026-05-02T01:14:28Z
Status: pass
Matrix: synthetic-smoke-matrix
Cells: 2
Coverage gates: expect cells = 2
Variability gates: suite CV <= -, prompt CV <= 0.1, suite IQR <= -, prompt IQR <= -

## Cells

| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Total tokens | Total elapsed us | Prompt suite | Command SHA256 | Launch plan SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Host child tok/CPU s | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ci-airgap-smoke | synthetic-smoke | Q4_0 | a70776642a09 | pass | 2 | 120 | 49-71 | 160 | 1000000 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 1984f18af8d221fea7e13bfa2403bd87662cb633f728095187b698a10b4f99c5 | 4 | 2 | 160.000 | 75.477 | 12400.000 | 116.879 | 54027.000 | 10.062 | 749.746 | 18432000 | 6250.000 | 13554.917 | 67207168 | 0 |
| ci-airgap-smoke | synthetic-smoke | Q8_0 | a70776642a09 | pass | 2 | 120 | 49-71 | 160 | 1000000 | 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a | b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d | 1984f18af8d221fea7e13bfa2403bd87662cb633f728095187b698a10b4f99c5 | 4 | 2 | 160.000 | 81.997 | 12400.000 | 95.952 | 46376.500 | 9.785 | 859.568 | 18366464 | 6250.000 | 12246.995 | 67207168 | 0 |
