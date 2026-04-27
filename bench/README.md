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

## Offline Eval Dataset Packer

`dataset_pack.py` converts local JSONL multiple-choice evaluation rows into a
deterministic HolyC-loadable binary plus a provenance manifest. It accepts a
normalized schema as well as HellaSwag-, ARC-, and TruthfulQA-shaped rows. It is
offline-only; place source data on disk first and document provenance.

Example:

```bash
python3 bench/dataset_pack.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --dataset smoke-eval \
  --split validation
```

## QEMU Prompt Benchmark

`qemu_prompt_bench.py` launches an air-gapped QEMU guest once per prompt, captures
serial output, extracts token timing records, and writes normalized JSON to
`bench/results/`. The runner always injects `-nic none` and rejects conflicting
network flags such as `-netdev` or virtual NIC devices.

Use `--repeat N` to run every prompt multiple times. Reports include raw per-run
records plus per-prompt medians and min/max tok/s in both JSON and Markdown.

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
  --repeat 5 \
  -- -m 512M
```

Validate the final QEMU command without launching:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --dry-run
```

## HolyC vs llama.cpp Eval Compare

`eval_compare.py` compares offline multiple-choice predictions from HolyC and
llama.cpp against the same local gold JSONL dataset. It aligns by record id,
supports prediction indexes, labels, exact choice text, or score arrays, and
writes JSON plus Markdown reports to `bench/results/`.

Example:

```bash
python3 bench/eval_compare.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --output-stem eval_compare_smoke_latest
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
