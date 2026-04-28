# QEMU Prompt Benchmark

Generated: 2026-04-28T04:53:36Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Warmup runs: 0
Runs: 6

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 360 | 240 | 1500000 | -189511.000 | -74.782 | 12000.000 | 12400.000 | 160.000 | 160.000 | 661.659 | 857.962 | 6250.000 | 6250.000 | 1576.130 | 2371.867 | 67207168 |

| tok/s stdev | tok/s CV % |
| ---: | ---: |
| 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Min tok/s | Median tok/s | tok/s stdev | tok/s CV % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 48 | 300000 | -241560 | -80.520 | 12400 | 12400.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 821.355 | 865.283 | 6250.000 | 6250.000 | 1217.500 | 1252.694 | 67207168 |
| smoke-short | 49 | 3 | 3 | 32 | 200000 | -131850 | -65.925 | 11600 | 11600.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 469.552 | 521.725 | 6250.000 | 6250.000 | 2129.688 | 2420.303 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
