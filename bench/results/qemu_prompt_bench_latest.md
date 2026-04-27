# QEMU Prompt Benchmark

Generated: 2026-04-27T23:40:33Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Warmup runs: 0
Runs: 6

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median tok/s | P95 tok/s | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 360 | 240 | 1500000 | 160.000 | 160.000 | 67207168 |

| tok/s stdev | tok/s CV % |
| ---: | ---: |
| 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Min tok/s | Median tok/s | tok/s stdev | tok/s CV % | Max tok/s | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 48 | 300000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 67207168 |
| smoke-short | 49 | 3 | 3 | 32 | 200000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
