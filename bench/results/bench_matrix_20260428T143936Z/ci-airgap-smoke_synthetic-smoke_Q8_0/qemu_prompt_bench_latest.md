# QEMU Prompt Benchmark

Generated: 2026-04-28T14:39:40Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d
Launch budget: -
Total launches: 6
Warmup runs: 2
Runs: 4

## Suite Summary

| Prompts | Runs | OK | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 240 | 160 | 1000000 | -86726.000 | -40.490 | 50411.500 | 48.128 | 807.182 | 14090240 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 352.587 | 622.943 | 6250.000 | 6250.000 | 3719.365 | 9477.297 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s P05-P95 spread % |
| ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | P05-P95 spread % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 48.000 | 300000.000 | 76928.500 | 25.643 | 54929.000 | 15.943 | 873.879 | 14090240 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 139.534 | 176.649 | 6250.000 | 6250.000 | 7852.677 | 9941.474 | 67207168 |
| smoke-short | 49 | 2 | 2 | 32.000 | 200000.000 | -144502.000 | -72.251 | 44565.500 | 80.812 | 718.988 | 458752 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 582.367 | 634.536 | 6250.000 | 6250.000 | 1734.312 | 1889.675 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | /Users/danielmatthews-ferrero/Documents/worktrees/holyc-gpt55/bench/fixtures/qemu_synthetic_bench.py |
