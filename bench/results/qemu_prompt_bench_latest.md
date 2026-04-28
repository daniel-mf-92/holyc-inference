# QEMU Prompt Benchmark

Generated: 2026-04-28T05:17:23Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Warmup runs: 0
Runs: 6

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 360 | 240 | 1500000 | -162496.000 | -62.598 | 12000.000 | 12400.000 | 160.000 | 160.000 | 476.886 | 858.063 | 6250.000 | 6250.000 | 2337.615 | 3140.992 | 67207168 |

| tok/s stdev | tok/s CV % |
| ---: | ---: |
| 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Min tok/s | Median tok/s | tok/s stdev | tok/s CV % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 48 | 300000 | -241403 | -80.468 | 12400 | 12400.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 819.155 | 865.845 | 6250.000 | 6250.000 | 1220.771 | 1550.865 | 67207168 |
| smoke-short | 49 | 3 | 3 | 32 | 200000 | -100029 | -50.014 | 11600 | 11600.000 | 160.000 | 160.000 | 0.000 | 0.000 | 160.000 | 320.093 | 323.490 | 6250.000 | 6250.000 | 3124.094 | 3144.372 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
