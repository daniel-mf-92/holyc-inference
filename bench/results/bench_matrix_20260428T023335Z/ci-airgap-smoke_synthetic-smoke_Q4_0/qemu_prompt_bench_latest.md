# QEMU Prompt Benchmark

Generated: 2026-04-28T02:33:36Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Warmup runs: 2
Runs: 4

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median TTFT us | P95 TTFT us | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 240 | 160 | 1000000 | 12000.000 | 12400.000 | 160.000 | 160.000 | 723.550 | 867.293 | 67207168 |

| tok/s stdev | tok/s CV % |
| ---: | ---: |
| 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median TTFT us | P95 TTFT us | Min tok/s | Median tok/s | tok/s stdev | tok/s CV % | Max tok/s | Median wall tok/s | P95 wall tok/s | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 48.000 | 300000.000 | 12400.000 | 12400.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 810.315 | 883.573 | 67207168 |
| smoke-short | 49 | 2 | 2 | 32.000 | 200000.000 | 11600.000 | 11600.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 653.354 | 711.699 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | /Users/danielmatthews-ferrero/Documents/worktrees/holyc-gpt55/bench/fixtures/qemu_synthetic_bench.py |
