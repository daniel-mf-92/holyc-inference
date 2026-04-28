# QEMU Prompt Benchmark

Generated: 2026-04-28T06:20:14Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d
Warmup runs: 2
Runs: 4

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 240 | 160 | 1000000 | -174385.500 | -68.958 | 12000.000 | 12400.000 | 160.000 | 160.000 | 524.051 | 856.432 | 6250.000 | 6250.000 | 1940.141 | 4105.914 | 67207168 |

| tok/s stdev | tok/s CV % |
| ---: | ---: |
| 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Min tok/s | Median tok/s | tok/s stdev | tok/s CV % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 48.000 | 300000.000 | -232838.500 | -77.613 | 12400.000 | 12400.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 747.251 | 887.626 | 6250.000 | 6250.000 | 1399.198 | 1662.045 | 67207168 |
| smoke-short | 49 | 2 | 2 | 32.000 | 200000.000 | -93868.500 | -46.934 | 11600.000 | 11600.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 340.918 | 445.233 | 6250.000 | 6250.000 | 3316.609 | 4331.430 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | /Users/danielmatthews-ferrero/Documents/worktrees/holyc-gpt55/bench/fixtures/qemu_synthetic_bench.py |
