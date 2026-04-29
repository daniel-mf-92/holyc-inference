# QEMU Prompt Benchmark Dry Run

Generated: 2026-04-29T19:36:20Z
Status: planned
Profile: ci-airgap-smoke
Model: synthetic-smoke
Quantization: Q4_0
Commit: d3daa15c824c
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 633b78c56d2f2d5c6739b10ca9b48d66a6bb83328887ac7aceabb4ab48fd1c96
Command air-gap OK: True
Launch plan SHA256: 1984f18af8d221fea7e13bfa2403bd87662cb633f728095187b698a10b4f99c5
Expected launch sequence SHA256: b2700d55880665cd3be86418a939e5ca949f7df4e2ecc60705a1b3909c9f1f0d
Prompt count: 2
Prompt count floor: 2
Warmup launches: 2
Measured launches: 4
Total launches: 6

## Command

```text
bench/fixtures/qemu_synthetic_bench.py -nic none -serial stdio -display none -drive file=/tmp/TempleOS.synthetic.img,format=raw,if=ide -m 256M
```

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| /tmp/TempleOS.synthetic.img | False | - | - | 0 |

## Launch Plan

| Launch | Phase | Prompt index | Prompt | Iteration | Prompt bytes | Prompt SHA256 |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 1 | warmup | 1 | smoke-short | 1 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 2 | warmup | 2 | smoke-code | 1 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |
| 3 | measured | 1 | smoke-short | 1 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 4 | measured | 1 | smoke-short | 2 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 5 | measured | 2 | smoke-code | 1 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |
| 6 | measured | 2 | smoke-code | 2 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
