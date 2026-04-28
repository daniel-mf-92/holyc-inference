# QEMU Prompt Benchmark Dry Run

Generated: 2026-04-28T22:30:47Z
Status: planned
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 72742a3c184093a30f9923158e9d8f9f87376e61ba1ff2368bef36920abe9755
Launch plan SHA256: b223f9a0e6d1633add356e01e86a8af33c00349530d3d8d9eba2b81b68013712
Prompt count: 2
Warmup launches: 2
Measured launches: 4
Total launches: 6

## Command

```text
qemu-system-x86_64 -nic none -serial stdio -display none -drive file=/tmp/templeos.img,format=raw,if=ide -nic none -serial stdio
```

## Launch Plan

| Launch | Phase | Prompt | Iteration | Prompt bytes | Prompt SHA256 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | warmup | smoke-short | 1 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 2 | warmup | smoke-code | 1 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |
| 3 | measured | smoke-short | 1 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 4 | measured | smoke-short | 2 | 49 | 8b1a97085df1b00a3d726b492997598fd4bd640080566602f9de113f29a4e7d4 |
| 5 | measured | smoke-code | 1 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |
| 6 | measured | smoke-code | 2 | 71 | 587c04a07a905d3852e39d6ac74a8db16c707622e7db66c642ee841caaab9091 |

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | QEMU emulator version 10.2.2 |
