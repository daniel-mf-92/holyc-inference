# QEMU Prompt Benchmark

Generated: 2026-05-01T23:04:12Z
Artifact schema: qemu-prompt-bench/v1
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
| 2 | 4 | 4 | 0 | 100.000 | 0 | 0 | 240 | 160 | 1000000 | 2427611 | 586442.500 | 687427.050 | 336442.500 | 142.873 | 58830.500 | 9.743 | 677.841 | 18415616 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 47.180 | 67.857 | 85.748 | 240.833 | 101.838 | 0.665 | 6250.000 | 6250.000 | 15179.536 | 21315.769 | 67207168 | 1749674.667 | 2099200.000 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 40.827 | 56.837 |

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
| smoke-code | 71 | 2 | 2 | 0 | 100.000 | 0 | 0 | 48.000 | 300000.000 | 578396.500 | 601605.250 | 278396.500 | 92.799 | 56394.500 | 9.743 | 854.282 | 18415616 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 79.817 | 83.153 | 86.490 | 4.458 | 8.025 | 236.667 | 122.998 | 0.676 | 6250.000 | 6250.000 | 12049.927 | 12533.443 | 67207168 | 1400149.333 | 1400149.333 |
| smoke-short | 49 | 2 | 2 | 0 | 100.000 | 0 | 0 | 32.000 | 200000.000 | 635409.000 | 695446.200 | 435409.000 | 217.704 | 62530.000 | 10.029 | 514.637 | 18399232 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 46.111 | 50.923 | 55.734 | 10.498 | 18.897 | 245.000 | 77.975 | 0.653 | 6250.000 | 6250.000 | 19856.531 | 21732.694 | 67174400 | 2099200.000 | 2099200.000 |

## Slowest Prompts

| Rank | Prompt | Median wall us/token | Median wall tok/s | Median tokens | Median wall elapsed us | P95 TTFT us | Max memory bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-short | 19856.531 | 50.923 | 32.000 | 635409.000 | 11600.000 | 67174400 |
| 2 | smoke-code | 12049.927 | 83.153 | 48.000 | 578396.500 | 12400.000 | 67207168 |

## Prompt Variability

| Rank | Prompt | Wall tok/s IQR % | Wall P05-P95 spread % | tok/s CV % | Median wall tok/s | Median tokens | P95 TTFT us |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-short | 10.498 | 18.897 | 0.000 | 50.923 | 32.000 | 11600.000 |
| 2 | smoke-code | 4.458 | 8.025 | 0.000 | 83.153 | 48.000 | 12400.000 |

## Prompt Efficiency

| Rank | Prompt | Median wall prompt bytes/s | Median tokens/prompt byte | Median wall tok/s | Median tokens | Median wall elapsed us | P95 TTFT us |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-code | 122.998 | 0.676 | 83.153 | 48.000 | 578396.500 | 12400.000 |
| 2 | smoke-short | 77.975 | 0.653 | 50.923 | 32.000 | 635409.000 | 11600.000 |

## Prompt Serial Output Ranking

| Rank | Prompt | Serial output bytes total | Serial output bytes max | Serial output lines total | Serial output lines max | Runs | OK | Failed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-code | 430 | 215 | 2 | 1 | 2 | 2 | 0 |
| 2 | smoke-short | 430 | 215 | 2 | 1 | 2 | 2 | 0 |

## Prompt Failure Ranking

| Rank | Prompt | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Launch errors | Serial output bytes total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-code | 2 | 2 | 0 | 100.000 | 0 | 0 | 0 | 430 |
| 2 | smoke-short | 2 | 2 | 0 | 100.000 | 0 | 0 | 0 | 430 |

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
| warmup | 2 | 2 | 2 | 0 | 100.000 | 0 | 0 | 2 | 0 | 0 | 0 | 80 | 1150621 | 575310.500 | 160.000 | 68.838 | 67207168 | 430 | 2 |
| measured | 4 | 2 | 4 | 0 | 100.000 | 0 | 0 | 4 | 0 | 0 | 0 | 160 | 2427611 | 586442.500 | 160.000 | 67.857 | 67207168 | 860 | 4 |
| all | 6 | 2 | 6 | 0 | 100.000 | 0 | 0 | 6 | 0 | 0 | 0 | 240 | 3578232 | 586442.500 | 160.000 | 68.838 | 67207168 | 1290 | 6 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
