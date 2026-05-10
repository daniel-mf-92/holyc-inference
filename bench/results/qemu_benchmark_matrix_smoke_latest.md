# QEMU Benchmark Matrix

- Status: pass
- Builds: 2
- Prompts: 2
- Launches: 12
- Air-gap OK builds: 2
- Findings: 0

| Build | Profile | Model | Quantization | Launches | Air-gap | Command |
| --- | --- | --- | --- | ---: | --- | --- |
| baseline | ci-smoke | synthetic | Q4_0 | 6 | True | 1b66b1774970811cc1821725b7d13833f8ee332796f2ed2f38bd4fc042a673b2 |
| candidate | ci-smoke | synthetic | Q4_0 | 6 | True | 6e28e0a66332d2ed7c2454ee729cd66483ad777ca9a3a2714eec9f27b4a1688a |
