# QEMU Prompt Benchmark

Generated: 2026-04-29T03:23:37Z
Status: pass
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: e415146c052b9f0baa23706ffde97548f2374f6601337823b4d810c39bed48e7
Launch plan SHA256: 87148377012250dda7f1be10ab8c73ac784dd582df50f95013f6da0406443d6f
Launch budget: -
Total launches: 6
Warmup runs: 0
Runs: 6

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| /tmp/TempleOS.synthetic.img | False | - | - | 0 |

## Suite Summary

| Prompts | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Measured prompt bytes | Total tokens | Total elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | P05 tok/s | Median tok/s | P95 tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6 | 6 | 0 | 100.000 | 0 | 0 | 360 | 240 | 1500000 | -196031.500 | -77.763 | 47656.000 | 81.567 | 888.540 | 11436032 | 12000.000 | 12400.000 | 160.000 | 160.000 | 160.000 | 110.042 | 735.209 | 969.359 | 240.833 | 1103.846 | 0.665 | 6250.000 | 6250.000 | 1389.823 | 13155.812 | 67207168 |

| tok/s stdev | tok/s CV % | tok/s IQR % | tok/s P05-P95 spread % | Wall tok/s IQR % | Wall tok/s P05-P95 spread % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.000 | 0.000 | 72.069 | 116.881 |

| Serial output bytes total | Serial output bytes max |
| ---: | ---: |
| 660 | 110 |

## Prompt Summary

| Prompt | Prompt bytes | Runs | OK | Failed | OK % | Timed out | Nonzero exit | Median tokens | Median elapsed us | Median host overhead us | Median host overhead % | Median host child CPU us | Median host child CPU % | Median host child tok/CPU-s | Max host child RSS bytes | Median TTFT us | P95 TTFT us | Min tok/s | P05 tok/s | Median tok/s | tok/s stdev | tok/s CV % | tok/s IQR % | P05-P95 spread % | Max tok/s | P05 wall tok/s | Median wall tok/s | P95 wall tok/s | Wall tok/s IQR % | Wall P05-P95 spread % | Median prompt bytes/s | Median wall prompt bytes/s | Median tokens/prompt byte | Median us/token | P95 us/token | Median wall us/token | P95 wall us/token | Max memory bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-code | 71 | 3 | 3 | 0 | 100.000 | 0 | 0 | 48 | 300000 | -246083 | -82.028 | 46320 | 85.910 | 1036.269 | 573440 | 12400 | 12400.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 847.386 | 890.257 | 985.180 | 8.599 | 15.478 | 236.667 | 1316.839 | 0.676 | 6250.000 | 6250.000 | 1123.271 | 1180.421 | 67207168 |
| smoke-short | 49 | 3 | 3 | 0 | 100.000 | 0 | 0 | 32 | 200000 | -74703 | -37.352 | 51639 | 41.213 | 619.687 | 11436032 | 11600 | 11600.000 | 160.000 | 160.000 | 160.000 | 0.000 | 0.000 | 0.000 | 0.000 | 160.000 | 80.972 | 255.393 | 590.555 | 110.849 | 199.529 | 245.000 | 391.071 | 0.653 | 6250.000 | 6250.000 | 3915.531 | 15003.869 | 67174400 |

## Prompt Serial Output

| Prompt | Serial output bytes total | Serial output bytes max |
| --- | ---: | ---: |
| smoke-code | 330 | 110 |
| smoke-short | 330 | 110 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
