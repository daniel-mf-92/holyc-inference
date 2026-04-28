# QEMU Prompt Benchmark

Generated: 2026-04-28T22:38:35Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: e415146c052b9f0baa23706ffde97548f2374f6601337823b4d810c39bed48e7
Launch budget: -
Total launches: 6
Warmup runs: 0
Runs: 6

## Suite Summary

| Prompts | Runs | OK | Failed | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 0 | 0 | 0 | 360 | 240 | 1500000 | -206002.500 | -81.906 | 38124.500 | 86.742 | 1033.002 | 11911168 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 693.934 | 901.130 | 1106.258 | 6250.000 | 6250.000 | 1130.870 | 1442.438 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s P05-P95 spread % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 45.756 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall P05-P95 spread % | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 0 | 0 | 0 | 48 | 300000 | -256440 | -85.480 | 38427 | 87.282 | 1249.122 | 11911168 | 12400 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 1032.127 | 1101.928 | 1107.124 | 6.806 | 6250.000 | 6250.000 | 907.500 | 969.337 | 67207168 |
| smoke-short | 49 | 3 | 3 | 0 | 0 | 0 | 32 | 200000 | -156269 | -78.135 | 36934 | 84.457 | 866.410 | 589824 | 11600 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 686.371 | 731.746 | 773.274 | 11.876 | 6250.000 | 6250.000 | 1366.594 | 1457.606 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
