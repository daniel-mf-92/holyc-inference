# QEMU Prompt Benchmark Dry Run

Generated: 2026-04-28T12:41:08Z
Status: planned
Prompt suite: 68fc621f9f3916e73aa05b83ba0fa8da9f3cffad22a1c29f5acf8980d8dd743a
Command SHA256: 633b78c56d2f2d5c6739b10ca9b48d66a6bb83328887ac7aceabb4ab48fd1c96
Prompt count: 2
Warmup launches: 2
Measured launches: 4
Total launches: 6

## Command

```text
bench/fixtures/qemu_synthetic_bench.py -nic none -serial stdio -display none -drive file=/tmp/TempleOS.synthetic.img,format=raw,if=ide -m 256M
```

## Environment

| Platform | Machine | Python | CPU count | QEMU |
| --- | --- | --- | ---: | --- |
| macOS-26.2-arm64-arm-64bit-Mach-O | arm64 | 3.14.3 | 10 | bench/fixtures/qemu_synthetic_bench.py |
