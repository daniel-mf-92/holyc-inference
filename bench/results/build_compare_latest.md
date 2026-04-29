# Build Benchmark Compare

Generated: 2026-04-29T03:13:43Z
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
| head | qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-code | 160.000 | 160.000 | 0.000 | 160.000 | 160.000 | 0.000 | 501.356 | 639.949 | 27.644 | 388.925 | 567.820 | 45.998 | 300000.000 | 300000.000 | 0.000 | 12400.000 | 12400.000 | 0.000 | 6250.000 | 6250.000 | 0.000 | 2126.625 | 1562.625 | -26.521 | 63473.500 | 51272.000 | -19.223 | 65.432 | 68.357 | 4.470 | 758.301 | 936.183 | 23.458 | 589824 | 655360 | 11.111 | - | 110 | - | 67207168 | 67207168 | 0.000 |
| head | qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/smoke-short | 160.000 | 160.000 | 0.000 | 160.000 | 160.000 | 0.000 | 449.220 | 590.351 | 31.417 | 401.349 | 576.928 | 43.747 | 200000.000 | 200000.000 | 0.000 | 11600.000 | 11600.000 | 0.000 | 6250.000 | 6250.000 | 0.000 | 2257.734 | 1693.906 | -24.973 | 53165.500 | 45624.000 | -14.185 | 73.680 | 84.774 | 15.058 | 608.999 | 701.385 | 15.170 | 524288 | 10354688 | 1875.000 | - | 110 | - | 67174400 | 67174400 | 0.000 |
