# QEMU Prompt Benchmark

Generated: 2026-04-28T14:49:09Z
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
| 2 | 4 | 4 | 240 | 160 | 1000000 | -201108.000 | -79.859 | 41405.000 | 86.604 | 949.661 | 442368 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 811.458 | 979.762 | 6250.000 | 6250.000 | 1258.781 | 1445.331 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s P05-P95 spread % |
| ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | P05-P95 spread % | Max tok/s | Median wall tok/s | P95 wall tok/s | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 48.000 | 300000.000 | -249893.000 | -83.298 | 42872.000 | 85.596 | 1119.976 | 442368 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 958.878 | 985.729 | 6250.000 | 6250.000 | 1043.896 | 1073.127 | 67207168 |
| smoke-short | 49 | 2 | 2 | 32.000 | 200000.000 | -153804.000 | -76.902 | 40367.000 | 87.384 | 792.784 | 442368 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 160.000 | 692.703 | 693.755 | 6250.000 | 6250.000 | 1443.625 | 1445.819 | 67174400 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | /Users/danielmatthews-ferrero/Documents/worktrees/holyc-gpt55/bench/fixtures/qemu_synthetic_bench.py |
