# QEMU Prompt Benchmark

Generated: 2026-04-28T14:28:10Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: e415146c052b9f0baa23706ffde97548f2374f6601337823b4d810c39bed48e7
Launch budget: -
Total launches: 6
Warmup runs: 0
Runs: 6

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 360 | 240 | 1500000 | -47853.000 | -19.283 | 55606.500 | 28.966 | 715.801 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 198.888 | 399.644 | 6250.000 | 6250.000 | 5044.828 | 23967.445 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s P05-P95 spread % |
| ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | P05-P95 spread % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 48 | 300000 | -71856 | -23.952 | 54772 | 24.881 | 876.360 | 12400 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 210.393 | 428.713 | 6250.000 | 6250.000 | 4753.000 | 11373.850 | 67207168 |
| smoke-short | 49 | 3 | 3 | 32 | 200000 | -29227 | -14.614 | 56441 | 33.050 | 566.964 | 11600 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 187.383 | 234.435 | 6250.000 | 6250.000 | 5336.656 | 25661.750 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
