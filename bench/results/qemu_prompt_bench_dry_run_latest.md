# QEMU Prompt Benchmark Dry Run

Generated: 2026-04-28T23:13:32Z
Status: planned
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: cd63fce9b104ed8c38566a736230f0f586e388a4959cc554564571547acc6e6a
Launch plan SHA256: 9bf69bc6a6f70cc016dc9f5eb260948a2c5781fc504d397f36ac0e6992b91fff
Prompt count: 2
Warmup launches: 2
Measured launches: 6
Total launches: 8

## Command

```text
bench/fixtures/qemu_synthetic_bench.py -nic none -serial stdio -display none -drive file=/tmp/holyc-qemu-prompt-bench-smoke.img,format=raw,if=ide -m 512M
```

## Inputs

| Image | Exists | Size bytes | SHA256 | QEMU args files |
| --- | --- | ---: | --- | ---: |
| /tmp/holyc-qemu-prompt-bench-smoke.img | True | 34 | 826b43bb0768a6f60e93d9a1bcc5e9aaef01185f91efc7c5df30b6c7c90349c4 | 0 |

## Launch Plan

| Launch | Phase | Prompt | Iteration | Prompt bytes | Prompt SHA256 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | warmup | smoke-short | 1 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 2 | warmup | smoke-code | 1 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |
| 3 | measured | smoke-short | 1 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 4 | measured | smoke-short | 2 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 5 | measured | smoke-short | 3 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 6 | measured | smoke-code | 1 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |
| 7 | measured | smoke-code | 2 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |
| 8 | measured | smoke-code | 3 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
