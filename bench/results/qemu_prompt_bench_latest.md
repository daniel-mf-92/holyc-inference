# QEMU Prompt Benchmark

Generated: 2026-04-29T16:10:25Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 633b78c56d2f2d5c6739b10ca9b48d66a6bb83328887ac7aceabb4ab48fd1c96
Launch plan SHA256: 1984f18af8d221fea7e13bfa2403bd87662cb633f728095187b698a10b4f99c5
Launch budget: 6
Prompt count floor: 2
Total launches: 6
Warmup runs: 2
Runs: 4

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| /tmp/TempleOS.synthetic.img | False | - | - | 0 |

## Suite Summary

| Prompts | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 0 | 100.000 | 0 | 0 | 240 | 160 | 1000000 | 1152038.000 | 464.532 | 61849.500 | 4.765 | 631.647 | 18497536 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 16.202 | 28.373 | 320.411 | 240.833 | 42.683 | 0.665 | 6250.000 | 6250.000 | 35283.276 | 65247.395 | 67207168 | 1749674.667 | 2099200.000 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 319.997 | 1072.185 |

| Serial output bytes total | Serial output bytes max |
| ---: | ---: |
| 860 | 215 |

| Guest prompt SHA records | Guest prompt SHA matches | Guest prompt SHA mismatches | Guest prompt byte records | Guest prompt byte matches | Guest prompt byte mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | 4 | 4 | 0 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 0 | 100.000 | 0 | 0 | 48.000 | 300000.000 | 583473.500 | 194.491 | 61454.000 | 24.025 | 785.525 | 18497536 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 46.431 | 200.544 | 354.658 | 85.386 | 153.695 | 236.667 | 296.638 | 0.676 | 6250.000 | 6250.000 | 18405.698 | 32550.051 | 67207168 | 1400149.333 | 1400149.333 |
| smoke-short | 49 | 2 | 2 | 0 | 100.000 | 0 | 0 | 32.000 | 200000.000 | 1508401.500 | 754.201 | 61849.500 | 4.072 | 518.070 | 17367040 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 14.880 | 20.829 | 26.778 | 31.735 | 57.123 | 245.000 | 31.894 | 0.653 | 6250.000 | 6250.000 | 53387.547 | 68635.923 | 67174400 | 2099200.000 | 2099200.000 |

## Prompt Serial Output

| Prompt | Serial output bytes total | Serial output bytes max |
| --- | ---: | ---: |
| smoke-code | 430 | 215 |
| smoke-short | 430 | 215 |

## Phase Summary

| Phase | Launches | Prompts | OK | Failed | OK % | Timed out | Nonzero exit | Total tokens | Median tok/s | Median wall tok/s | Max memory bytes | Serial output bytes total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup | 2 | 2 | 2 | 0 | 100.000 | 0 | 0 | 80 | 160.000 | 20.545 | 67207168 | 430 |
| measured | 4 | 2 | 4 | 0 | 100.000 | 0 | 0 | 160 | 160.000 | 28.373 | 67207168 | 860 |
| all | 6 | 2 | 6 | 0 | 100.000 | 0 | 0 | 240 | 160.000 | 28.373 | 67207168 | 1290 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
