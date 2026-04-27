# Benchmark and Evaluation Host Tools

This directory is for host-side benchmarking, evaluation, dataset, and quantization
validation infrastructure around the HolyC inference engine. Tools here must keep
TempleOS air-gapped; any QEMU command added under this tree must pass `-nic none`.

## Quantization Audit

`quant_audit.py` checks two host-side invariants:

- HolyC quantization sources do not contain runtime float types, float literals, or
  common float math helper calls after comments and strings are stripped.
- Raw Q4_0/Q8_0 block streams have valid block sizes and finite fp16 scales.

Example:

```bash
python3 bench/quant_audit.py --source-root src/quant --output bench/results/quant_audit_latest.json
```

Raw block streams can be checked with:

```bash
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin
```
