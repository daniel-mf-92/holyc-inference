# QEMU Prompt Benchmark

Generated: 2026-04-29T10:33:12Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: e415146c052b9f0baa23706ffde97548f2374f6601337823b4d810c39bed48e7
Launch plan SHA256: 7bf1257265af0839c236008631b4062821e27fef1c51977fd318fdf878f2db14
Launch budget: -
Total launches: 2
Warmup runs: 0
Runs: 2

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| /tmp/TempleOS.synthetic.img | False | - | - | 0 |

## Suite Summary

| Prompts | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 2 | 0 | 100.000 | 0 | 0 | 120 | 80 | 500000 | 712134.000 | 375.546 | 61777.500 | 39.767 | 697.041 | 17547264 | 12000.000 | 12360.000 | 160.000 | 160.000 | 160.000 | 52.587 | 370.865 | 689.142 | 240.833 | 549.019 | 0.665 | 6250.000 | 6250.000 | 29721.625 | 55228.862 | 67207168 | 1749674.667 | 2099200.000 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 95.356 | 171.641 |

| Serial output bytes total | Serial output bytes max |
| ---: | ---: |
| 430 | 215 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 1 | 1 | 0 | 100.000 | 0 | 0 | 48 | 300000 | -233748 | -77.916 | 50073 | 75.580 | 958.600 | 573440 | 12400 | 12400.000 | 160.000 | 160.000 | 160.000 | - | - | 0.000 | 0.000 | 160.000 | 724.506 | 724.506 | 724.506 | 0.000 | 0.000 | 236.667 | 1071.666 | 0.676 | 6250.000 | 6250.000 | 1380.250 | 1380.250 | 67207168 | 1400149.333 | 1400149.333 |
| smoke-short | 49 | 1 | 1 | 0 | 100.000 | 0 | 0 | 32 | 200000 | 1658016 | 829.008 | 73482 | 3.955 | 435.481 | 17547264 | 11600 | 11600.000 | 160.000 | 160.000 | 160.000 | - | - | 0.000 | 0.000 | 160.000 | 17.223 | 17.223 | 17.223 | 0.000 | 0.000 | 245.000 | 26.372 | 0.653 | 6250.000 | 6250.000 | 58063.000 | 58063.000 | 67174400 | 2099200.000 | 2099200.000 |

## Prompt Serial Output

| Prompt | Serial output bytes total | Serial output bytes max |
| --- | ---: | ---: |
| smoke-code | 215 | 215 |
| smoke-short | 215 | 215 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
