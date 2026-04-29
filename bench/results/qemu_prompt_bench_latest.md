# QEMU Prompt Benchmark

Generated: 2026-04-29T19:59:49Z
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
| 2 | 4 | 4 | 0 | 100.000 | 0 | 0 | 240 | 160 | 1000000 | 241360 | 59552.000 | 67102.300 | -188872.000 | -75.071 | 48028.500 | 81.022 | 806.751 | 589824 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 511.542 | 647.771 | 849.560 | 240.833 | 973.413 | 0.665 | 6250.000 | 6250.000 | 1558.089 | 1961.362 | 67207168 | 1749674.667 | 2099200.000 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 28.888 | 52.182 |

| Serial output bytes total | Serial output bytes max | Serial output lines total | Serial output lines max |
| ---: | ---: | ---: | ---: |
| 860 | 215 | 4 | 1 |

| Guest prompt SHA records | Guest prompt SHA matches | Guest prompt SHA mismatches | Guest prompt byte records | Guest prompt byte matches | Guest prompt byte mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | 4 | 4 | 0 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median wall elapsed us | P95 wall elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes | Median memory bytes/token | Max memory bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 2 | 2 | 0 | 100.000 | 0 | 0 | 48.000 | 300000.000 | 61261.000 | 66979.600 | -238739.000 | -79.580 | 49296.500 | 80.970 | 975.626 | 573440 | 12400.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 718.117 | 792.054 | 865.990 | 10.372 | 18.670 | 236.667 | 1171.579 | 0.676 | 6250.000 | 6250.000 | 1276.271 | 1395.408 | 67207168 | 1400149.333 | 1400149.333 |
| smoke-short | 49 | 2 | 2 | 0 | 100.000 | 0 | 0 | 32.000 | 200000.000 | 59419.000 | 63719.200 | -140581.000 | -70.291 | 47962.500 | 81.110 | 667.471 | 589824 | 11600.000 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 502.824 | 542.053 | 581.282 | 8.041 | 14.474 | 245.000 | 830.019 | 0.653 | 6250.000 | 6250.000 | 1856.844 | 1991.225 | 67174400 | 2099200.000 | 2099200.000 |

## Slowest Prompts

| Rank | Prompt | Median wall us/token | Median wall tok/s | Median tokens | Median wall elapsed us | P95 TTFT us | Max memory bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | smoke-short | 1856.844 | 542.053 | 32.000 | 59419.000 | 11600.000 | 67174400 |
| 2 | smoke-code | 1276.271 | 792.054 | 48.000 | 61261.000 | 12400.000 | 67207168 |

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

| Phase | Launches | Prompts | OK | Failed | OK % | Timed out | Nonzero exit | Total tokens | Total wall elapsed us | Median wall elapsed us | Median tok/s | Median wall tok/s | Max memory bytes | Serial output bytes total | Serial output lines total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup | 2 | 2 | 2 | 0 | 100.000 | 0 | 0 | 80 | 584158 | 292079.000 | 160.000 | 301.263 | 67207168 | 430 | 2 |
| measured | 4 | 2 | 4 | 0 | 100.000 | 0 | 0 | 160 | 241360 | 59552.000 | 160.000 | 647.771 | 67207168 | 860 | 4 |
| all | 6 | 2 | 6 | 0 | 100.000 | 0 | 0 | 240 | 825518 | 65906.000 | 160.000 | 561.755 | 67207168 | 1290 | 6 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
