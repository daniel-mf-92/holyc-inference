# QEMU Prompt Benchmark

Generated: 2026-04-29T04:14:10Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 16276fe5218629c6d18b6c6bd2d299bf0826c2ddd15ff0ce3871037bda7f4a6b
Launch plan SHA256: 0e6e0a7f4f79bd3404fa242861cd90d1b5e7ec08fce24797dc36bddcb8535a86
Launch budget: 6
Total launches: 6
Warmup runs: 2
Runs: 4

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| bench/fixtures/synthetic-temple.img | False | - | - | 0 |

## Suite Summary

| Prompts | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 0 | 100.000 | 0 | 0 | 240 | 160 | 1000000 | -175916.500 | -68.170 | 47065.000 | 66.614 | 859.463 | 13336576 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 121.409 | 570.623 | 937.149 | 240.833 | 853.779 | 0.665 | 6250.000 | 6250.000 | 1989.396 | 11456.570 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 90.766 | 142.956 |

| Serial output bytes total | Serial output bytes max |
| ---: | ---: |
| 440 | 110 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 0 | 100.000 | 0 | 0 | 48.000 | 300000.000 | -243914.500 | -81.305 | 43877.500 | 79.067 | 1094.563 | 573440 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 777.522 | 867.312 | 957.102 | 11.503 | 20.705 | 236.667 | 1282.899 | 0.676 | 6250.000 | 6250.000 | 1168.448 | 1289.414 | 67207168 |
| smoke-short | 49 | 2 | 2 | 0 | 100.000 | 0 | 0 | 32.000 | 200000.000 | 50912.500 | 25.456 | 50902.000 | 36.618 | 629.348 | 13336576 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 91.728 | 225.294 | 358.860 | 65.873 | 118.571 | 245.000 | 344.981 | 0.653 | 6250.000 | 6250.000 | 7841.016 | 12489.586 | 67174400 |

## Prompt Serial Output

| Prompt | Serial output bytes total | Serial output bytes max |
| --- | ---: | ---: |
| smoke-code | 220 | 110 |
| smoke-short | 220 | 110 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
