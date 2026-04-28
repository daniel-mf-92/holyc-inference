# QEMU Prompt Benchmark

Generated: 2026-04-28T20:07:33Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: b38e17f8fe6b79397ba9c4cb6dc68ad1d24ed501d479195500b1d8eaf615640d
Launch budget: -
Total launches: 6
Warmup runs: 2
Runs: 4

## Suite Summary

| Prompts | Runs | OK | Failed | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 0 | 0 | 0 | 240 | 160 | 1000000 | -191904.000 | -75.571 | 49428.500 | 85.122 | 815.952 | 540672 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 696.294 | 999.639 | 6250.000 | 6250.000 | 1526.792 | 2389.891 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s P05-P95 spread % |
| ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | P05-P95 spread % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 0 | 0 | 0 | 48.000 | 300000.000 | -248829.000 | -82.943 | 44288.500 | 86.597 | 1090.234 | 540672 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 944.581 | 1015.370 | 6250.000 | 6250.000 | 1066.062 | 1145.956 | 67207168 |
| smoke-short | 49 | 2 | 2 | 0 | 0 | 0 | 32.000 | 200000.000 | -129995.000 | -64.998 | 54433.500 | 78.510 | 589.998 | 458752 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 465.224 | 520.518 | 6250.000 | 6250.000 | 2187.656 | 2447.672 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | /Users/danielmatthews-ferrero/Documents/worktrees/holyc-gpt55/bench/fixtures/qemu_synthetic_bench.py |
