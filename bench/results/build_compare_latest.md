# Build Benchmark Compare

Generated: 2026-04-28T06:56:33Z
Status: pass
Baseline: base
Builds: base, head
Throughput regressions: 0
P05 throughput regressions: 0
Wall throughput regressions: 0
TTFT regressions: 0
Memory regressions: 0
Coverage violations: 0
Prompt-suite drift: 0

## Deltas

| Candidate | Prompt key | Base tok/s | Candidate tok/s | Tok/s delta % | Base P05 tok/s | Candidate P05 tok/s | P05 tok/s delta % | Base wall tok/s | Candidate wall tok/s | Wall tok/s delta % | Base elapsed us | Candidate elapsed us | Elapsed delta % | Base TTFT us | Candidate TTFT us | TTFT delta % | Base memory bytes | Candidate memory bytes | Memory delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head | qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 160.000 | 160.000 | 0.000 | 160.000 | 160.000 | 0.000 | 881.025 | 881.025 | 0.000 | 300000.000 | 300000.000 | 0.000 | 12400.000 | 12400.000 | 0.000 | 67207168 | 67207168 | 0.000 |
| head | qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 160.000 | 160.000 | 0.000 | 160.000 | 160.000 | 0.000 | 574.001 | 574.001 | 0.000 | 200000.000 | 200000.000 | 0.000 | 11600.000 | 11600.000 | 0.000 | 67174400 | 67174400 | 0.000 |
