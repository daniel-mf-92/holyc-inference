# QEMU Matrix Budget Audit

- Status: pass
- Sources: 1
- Builds: 2
- Launches: 6
- Prompt bytes: 108
- Expected tokens: 48
- Findings: 0

| Build | Profile | Model | Quantization | Launches | Prompt bytes | Expected tokens | Missing expected tokens | Air-gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | ci-smoke | synthetic | Q4_0 | 3 | 54 | 24 | 0 | True |
| candidate | ci-smoke | synthetic | Q4_0 | 3 | 54 | 24 | 0 | True |
