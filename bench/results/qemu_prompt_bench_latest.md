# QEMU Prompt Benchmark

Generated: 2026-04-28T05:32:08Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Warmup runs: 2
Runs: 4

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 240 | 160 | 1000000 | -170173.000 | -68.058 | 12000.000 | 12400.000 | 160.000 | 160.000 | 500.911 | 560.289 | 6250.000 | 6250.000 | 1996.370 | 2001.092 | 67207168 |

| tok/s stdev | tok/s CV % |
| ---: | ---: |
| 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Min tok/s | Median tok/s | tok/s stdev | tok/s CV % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 48.000 | 300000.000 | -210110.500 | -70.037 | 12400.000 | 12400.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 536.198 | 567.172 | 6250.000 | 6250.000 | 1872.698 | 1980.876 | 67207168 |
| smoke-short | 49 | 2 | 2 | 32.000 | 200000.000 | -135981.500 | -67.991 | 11600.000 | 11600.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 499.856 | 500.021 | 6250.000 | 6250.000 | 2000.578 | 2001.239 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | ./bench/fixtures/qemu_synthetic_bench.py |
