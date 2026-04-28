# QEMU Prompt Benchmark

Generated: 2026-04-28T12:42:20Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: e415146c052b9f0baa23706ffde97548f2374f6601337823b4d810c39bed48e7
Launch budget: 6
Total launches: 6
Warmup runs: 0
Runs: 6

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 360 | 240 | 1500000 | -179884.000 | -72.873 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 607.241 | 1203.785 | 6250.000 | 6250.000 | 1695.453 | 6319.703 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s P05-P95 spread % |
| ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | P05-P95 spread % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 48 | 300000 | -259776 | -86.592 | 12400 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 1193.317 | 1205.878 | 6250.000 | 6250.000 | 838.000 | 1868.219 | 67207168 |
| smoke-short | 49 | 3 | 3 | 32 | 200000 | -15726 | -7.863 | 11600 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 173.654 | 656.471 | 6250.000 | 6250.000 | 5758.562 | 6431.931 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
