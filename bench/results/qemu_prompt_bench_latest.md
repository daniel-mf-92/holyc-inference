# QEMU Prompt Benchmark

Generated: 2026-04-29T20:24:17Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 633b78c56d2f2d5c6739b10ca9b48d66a6bb83328887ac7aceabb4ab48fd1c96
Command air-gap OK: True
Launch plan SHA256: 1984f18af8d221fea7e13bfa2403bd87662cb633f728095187b698a10b4f99c5
Expected launch sequence SHA256: b2700d55880665cd3be86418a939e5ca949f7df4e2ecc60705a1b3909c9f1f0d
Observed launch sequence SHA256: b2700d55880665cd3be86418a939e5ca949f7df4e2ecc60705a1b3909c9f1f0d
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

| Prompts | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Total wall elapsed us | Median wall elapsed us | P95 wall elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 0 | 100.000 | 0 | 0 | 240 | 160 | 1000000 | 473720 | 84269.000 | 226343.700 | -165731.000 | -65.193 | 55541.000 | 66.265 | 736.642 | 589824 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 168.012 | 471.437 | 831.720 | 240.833 | 707.672 | 0.665 | 6250.000 | 6250.000 | 2175.448 | 7038.938 | 67207168 | 1749674.667 | 2099200.000 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 63.706 | 140.784 |

| Serial output bytes total | Serial output bytes max | Serial output lines total | Serial output lines max |
| ---: | ---: | ---: | ---: |
| 860 | 215 | 4 | 1 |

| Exit class OK | Exit class timeout | Exit class launch error | Exit class nonzero exit |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0 | 0 |

| Guest prompt SHA records | Guest prompt SHA matches | Guest prompt SHA mismatches | Guest prompt byte records | Guest prompt byte matches | Guest prompt byte mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | 4 | 4 | 0 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median wall elapsed us | P95 wall elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 0 | 100.000 | 0 | 0 | 48.000 | 300000.000 | 71170.000 | 86252.200 | -228830.000 | -76.277 | 48544.500 | 71.335 | 991.429 | 589824 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 562.714 | 714.030 | 865.345 | 23.546 | 42.384 | 236.667 | 1056.169 | 0.676 | 6250.000 | 6250.000 | 1482.708 | 1796.921 | 67207168 | 1400149.333 | 1400149.333 |
| smoke-short | 49 | 2 | 2 | 0 | 100.000 | 0 | 0 | 32.000 | 200000.000 | 165690.000 | 242262.000 | -34310.000 | -17.155 | 63522.000 | 50.596 | 505.289 | 573440 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 141.075 | 262.290 | 383.505 | 51.349 | 92.428 | 245.000 | 401.632 | 0.653 | 6250.000 | 6250.000 | 5177.812 | 7570.688 | 67174400 | 2099200.000 | 2099200.000 |

## Slowest Prompts

| Rank | Prompt | Median wall us/token | Median wall tok/s | Median tokens | Median wall elapsed us | P95 TTFT us | Max memory bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-short | 5177.812 | 262.290 | 32.000 | 165690.000 | 11600.000 | 67174400 |
| 2 | smoke-code | 1482.708 | 714.030 | 48.000 | 71170.000 | 12400.000 | 67207168 |

## Prompt Variability

| Rank | Prompt | Wall tok/s IQR % | Wall P05-P95 spread % | tok/s CV % | Median wall tok/s | Median tokens | P95 TTFT us |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-short | 51.349 | 92.428 | 0.000 | 262.290 | 32.000 | 11600.000 |
| 2 | smoke-code | 23.546 | 42.384 | 0.000 | 714.030 | 48.000 | 12400.000 |

## Prompt Efficiency

| Rank | Prompt | Median wall prompt bytes/s | Median tokens/prompt byte | Median wall tok/s | Median tokens | Median wall elapsed us | P95 TTFT us |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-code | 1056.169 | 0.676 | 714.030 | 48.000 | 71170.000 | 12400.000 |
| 2 | smoke-short | 401.632 | 0.653 | 262.290 | 32.000 | 165690.000 | 11600.000 |

## Prompt Serial Output

| Prompt | Serial output bytes total | Serial output bytes max | Serial output lines total | Serial output lines max |
| --- | ---: | ---: | ---: | ---: |
| smoke-code | 430 | 215 | 2 | 1 |
| smoke-short | 430 | 215 | 2 | 1 |

## Launch Sequence Integrity

| Expected launches | Observed launches | Matched | Mismatched | Missing | Extra | Match |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 6 | 6 | 0 | 0 | 0 | True |


## Phase Summary

| Phase | Launches | Prompts | OK | Failed | OK % | Timed out | Nonzero exit | Exit class OK | Exit class timeout | Exit class launch error | Exit class nonzero exit | Total tokens | Total wall elapsed us | Median wall elapsed us | Median tok/s | Median wall tok/s | Max memory bytes | Serial output bytes total | Serial output lines total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup | 2 | 2 | 2 | 0 | 100.000 | 0 | 0 | 2 | 0 | 0 | 0 | 80 | 821854 | 410927.000 | 160.000 | 121.856 | 67207168 | 430 | 2 |
| measured | 4 | 2 | 4 | 0 | 100.000 | 0 | 0 | 4 | 0 | 0 | 0 | 160 | 473720 | 84269.000 | 160.000 | 471.437 | 67207168 | 860 | 4 |
| all | 6 | 2 | 6 | 0 | 100.000 | 0 | 0 | 6 | 0 | 0 | 0 | 240 | 1295574 | 169349.000 | 160.000 | 292.040 | 67207168 | 1290 | 6 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
