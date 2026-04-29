# QEMU Prompt Benchmark

Generated: 2026-04-29T06:29:42Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 237d2ed6d810297657001950f1ad01f0ff59d2b8e1011629c4500ef508be9b7a
Launch plan SHA256: 7bf1257265af0839c236008631b4062821e27fef1c51977fd318fdf878f2db14
Launch budget: 2
Total launches: 2
Warmup runs: 0
Runs: 2

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| bench/fixtures/airgap-smoke.img | False | - | - | 0 |

## Suite Summary

| Prompts | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 2 | 0 | 100.000 | 0 | 0 | 120 | 80 | 500000 | -178314.500 | -69.829 | 60680.000 | 84.658 | 667.085 | 589824 | 12000.000 | 12360.000 | 160.000 | 160.000 | 160.000 | 438.937 | 565.075 | 691.214 | 240.833 | 846.906 | 0.665 | 6250.000 | 6250.000 | 1885.677 | 2306.605 | 67207168 | 1749674.667 | 2099200.000 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 24.803 | 44.645 |

| Serial output bytes total | Serial output bytes max |
| ---: | ---: |
| 390 | 195 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 1 | 1 | 0 | 100.000 | 0 | 0 | 48 | 300000 | -231937 | -77.312 | 57760 | 84.863 | 831.025 | 458752 | 12400 | 12400.000 | 160.000 | 160.000 | 160.000 | - | - | 0.000 | 0.000 | 160.000 | 705.229 | 705.229 | 705.229 | 0.000 | 0.000 | 236.667 | 1043.151 | 0.676 | 6250.000 | 6250.000 | 1417.979 | 1417.979 | 67207168 | 1400149.333 | 1400149.333 |
| smoke-short | 49 | 1 | 1 | 0 | 100.000 | 0 | 0 | 32 | 200000 | -124692 | -62.346 | 63600 | 84.453 | 503.145 | 589824 | 11600 | 11600.000 | 160.000 | 160.000 | 160.000 | - | - | 0.000 | 0.000 | 160.000 | 424.922 | 424.922 | 424.922 | 0.000 | 0.000 | 245.000 | 650.661 | 0.653 | 6250.000 | 6250.000 | 2353.375 | 2353.375 | 67174400 | 2099200.000 | 2099200.000 |

## Prompt Serial Output

| Prompt | Serial output bytes total | Serial output bytes max |
| --- | ---: | ---: |
| smoke-code | 195 | 195 |
| smoke-short | 195 | 195 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
