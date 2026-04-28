# QEMU Prompt Benchmark

Generated: 2026-04-28T23:13:20Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: cd63fce9b104ed8c38566a736230f0f586e388a4959cc554564571547acc6e6a
Launch budget: 8
Total launches: 8
Warmup runs: 2
Runs: 6

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| /tmp/holyc-qemu-prompt-bench-smoke.img | True | 34 | 826b43bb0768a6f60e93d9a1bcc5e9aaef01185f91efc7c5df30b6c7c90349c4 | 0 |

## Suite Summary

| Prompts | Runs | OK | Failed | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 0 | 0 | 0 | 360 | 240 | 1500000 | -176560.000 | -74.038 | 46797.000 | 81.321 | 777.603 | 15187968 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 376.156 | 626.054 | 1008.640 | 6250.000 | 6250.000 | 1622.625 | 2709.742 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 51.125 | 101.027 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 0 | 0 | 0 | 48 | 300000 | -243383 | -81.128 | 44385 | 78.395 | 1081.446 | 15187968 | 12400 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 510.639 | 847.802 | 1040.807 | 34.741 | 62.534 | 6250.000 | 6250.000 | 1179.521 | 2019.990 | 67207168 |
| smoke-short | 49 | 3 | 3 | 0 | 0 | 0 | 32 | 200000 | -141590 | -70.795 | 49209 | 84.248 | 650.288 | 573440 | 11600 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 364.220 | 547.851 | 688.616 | 32.896 | 59.212 | 6250.000 | 6250.000 | 1825.312 | 2800.209 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
