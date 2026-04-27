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

## QEMU Prompt Benchmark

`qemu_prompt_bench.py` runs the TempleOS image once per prompt, captures serial
output, extracts `BENCH_RESULT` JSON or key/value metrics, and writes normalized
records to `bench/results/`.

The runner always injects `-nic none` and rejects conflicting QEMU networking
arguments.

Example dry run:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --qemu-arg=-m --qemu-arg=512M \
  --dry-run
```

Expected guest serial output can be JSON:

```text
BENCH_RESULT: {"tokens": 64, "elapsed_us": 16000000, "tok_per_s": 4.0}
```

or key/value text:

```text
tokens=64 elapsed_us=16000000 tok_per_s_milli=4000
```

## Perf Regression Dashboard

`perf_regression.py` scans host-side benchmark result files and writes dashboards
to `bench/dashboards/`. It accepts JSON, JSONL, and CSV records with `tok_per_s`
or `tok_per_s_milli`, plus optional memory fields such as `memory_bytes` or
`max_rss_bytes`.

Example:

```bash
python3 bench/perf_regression.py --input bench/results --output-dir bench/dashboards
```

CI can fail on throughput or memory regressions with:

```bash
python3 bench/perf_regression.py --fail-on-regression
```
