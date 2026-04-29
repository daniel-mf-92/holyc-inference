# Build Benchmark Compare

Generated: 2026-04-29T02:56:47Z
Status: pass
Baseline: base
Builds: base, head
Throughput regressions: 0
P05 throughput regressions: 0
Wall throughput regressions: 0
P05 wall throughput regressions: 0
TTFT regressions: 0
Token latency regressions: 0
Host child CPU/RSS regressions: 0
Serial output regressions: 0
Memory regressions: 0
Coverage violations: 0
Prompt-suite drift: 0
Command drift: 0

## Deltas

| Candidate | Prompt key | Base tok/s | Candidate tok/s | Tok/s delta % | Base P05 tok/s | Candidate P05 tok/s | P05 tok/s delta % | Base wall tok/s | Candidate wall tok/s | Wall tok/s delta % | Base P05 wall tok/s | Candidate P05 wall tok/s | P05 wall tok/s delta % | Base elapsed us | Candidate elapsed us | Elapsed delta % | Base TTFT us | Candidate TTFT us | TTFT delta % | Base us/token | Candidate us/token | us/token delta % | Base wall us/token | Candidate wall us/token | Wall us/token delta % | Base child CPU us | Candidate child CPU us | Child CPU us delta % | Base child CPU % | Candidate child CPU % | Child CPU % delta % | Base child tok/CPU s | Candidate child tok/CPU s | Child tok/CPU s delta % | Base child RSS bytes | Candidate child RSS bytes | Child RSS delta % | Base serial bytes | Candidate serial bytes | Serial delta % | Base memory bytes | Candidate memory bytes | Memory delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head | qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 160.000 | 160.000 | 0.000 | 160.000 | 160.000 | 0.000 | 639.949 | 639.949 | 0.000 | 567.820 | 567.820 | 0.000 | 300000.000 | 300000.000 | 0.000 | 12400.000 | 12400.000 | 0.000 | 6250.000 | 6250.000 | 0.000 | 1562.625 | 1562.625 | 0.000 | 51272.000 | 51272.000 | 0.000 | 68.357 | 68.357 | 0.000 | 936.183 | 936.183 | 0.000 | 655360 | 655360 | 0.000 | 110 | 110 | 0.000 | 67207168 | 67207168 | 0.000 |
| head | qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 160.000 | 160.000 | 0.000 | 160.000 | 160.000 | 0.000 | 590.351 | 590.351 | 0.000 | 576.928 | 576.928 | 0.000 | 200000.000 | 200000.000 | 0.000 | 11600.000 | 11600.000 | 0.000 | 6250.000 | 6250.000 | 0.000 | 1693.906 | 1693.906 | 0.000 | 45624.000 | 45624.000 | 0.000 | 84.774 | 84.774 | 0.000 | 701.385 | 701.385 | 0.000 | 10354688 | 10354688 | 0.000 | 110 | 110 | 0.000 | 67174400 | 67174400 | 0.000 |
