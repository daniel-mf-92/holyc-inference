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

`qemu_prompt_bench.py` launches an air-gapped QEMU guest once per prompt, captures
serial output, extracts token timing records, and writes normalized JSON to
`bench/results/`. The runner always injects `-nic none` and rejects conflicting
network flags such as `-netdev` or virtual NIC devices.

Prompt files can be JSON, JSONL, or plain text split with `---`. Guest output may
include a JSON line such as:

```text
BENCH_RESULT: {"tokens": 128, "elapsed_us": 500000, "tok_per_s": 256.0}
```

Example:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --profile secure-local \
  --quantization Q4_0 \
  -- -m 512M
```

Validate the final QEMU command without launching:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --dry-run
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
