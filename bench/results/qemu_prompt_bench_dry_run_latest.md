# QEMU Prompt Benchmark Dry Run

Generated: 2026-04-28T22:22:13Z
Status: planned
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 1f08e65115fc8d318ef7726804d26085b1c556e5cee8743249e52ecae3581d74
Prompt count: 2
Warmup launches: 2
Measured launches: 4
Total launches: 6

## Command

```text
qemu-system-x86_64 -nic none -serial stdio -display none -drive file=/tmp/TempleOS.synthetic.img,format=raw,if=ide
```

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | qemu-system-x86_64 |
