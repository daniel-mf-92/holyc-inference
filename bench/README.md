# Benchmark and Evaluation Host Tools

This directory is for host-side benchmarking, evaluation, dataset, and quantization
validation infrastructure around the HolyC inference engine. Tools here must keep
TempleOS air-gapped; any QEMU command added under this tree must pass `-nic none`.

## Quantization Audit

`quant_audit.py` checks two host-side invariants:

- HolyC quantization sources do not contain runtime float types, float literals, or
  common float math helper calls after comments and strings are stripped.
- Raw Q4_0/Q8_0 block streams have valid block sizes, optional expected
  block/element counts, finite fp16 scales, fp16-to-Q16 scale ranges, optional
  Q16 scale magnitude limits, quant ranges, and quant histograms.

Example:

```bash
python3 bench/quant_audit.py \
  --source-root src/quant \
  --output bench/results/quant_audit_latest.json \
  --markdown bench/results/quant_audit_latest.md
```

Raw block streams can be checked with:

```bash
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --expect-elements 4096
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --expect-blocks 128
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --max-abs-scale-q16 1048576
```

Mixed-format audits can validate Q4_0 and Q8_0 streams in one report:

```bash
python3 bench/quant_audit.py \
  --q4-block-file path/to/q4_blocks.bin \
  --q8-block-file path/to/q8_blocks.bin
```

## Offline Eval Dataset Packer

`dataset_curate.py` prepares deterministic, local-only evaluation subsets before
packing. It accepts the same normalized, HellaSwag-, ARC-, and TruthfulQA-shaped
JSONL rows as the packer, writes normalized JSONL plus a provenance manifest,
and can optionally write the `.hceval` binary in the same run. It never fetches
remote datasets; stage source data on disk first.

Example:

```bash
python3 bench/dataset_curate.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/smoke_curated.jsonl \
  --manifest bench/results/datasets/smoke_curated.manifest.json \
  --source-name smoke-eval \
  --source-version synthetic \
  --source-license synthetic-smoke \
  --max-records 3 \
  --balance-answer-index \
  --pack-output bench/results/datasets/smoke_curated.hceval \
  --pack-manifest bench/results/datasets/smoke_curated.hceval.manifest.json
```

When `--max-records` trims a larger local source, `--balance-answer-index`
round-robins through answer labels before applying the final stable output sort.
This keeps compact multiple-choice subsets from accidentally overrepresenting a
single answer index.

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

`hceval_inspect.py` independently parses `.hceval` binaries, validates record
bounds, verifies source/binary hashes against a companion manifest, and writes
JSON or Markdown inspection reports:

```bash
python3 bench/hceval_inspect.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --output bench/results/datasets/smoke_eval.inspect.json \
  --markdown bench/results/datasets/smoke_eval.inspect.md
```

`dataset_index.py` scans curated manifests, packed `.hceval` manifests, and
inspection reports, verifies local hashes/provenance fields where possible, and
writes JSON/Markdown/CSV rollups:

```bash
python3 bench/dataset_index.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --fail-on-findings
```

## QEMU Prompt Benchmark

`prompt_audit.py` validates benchmark prompt files before a guest run. It
checks prompt ID uniqueness, duplicate prompt text, byte/line stats, optional
minimum prompt count and maximum prompt byte limits, then writes a stable suite
hash so benchmark artifacts can name the exact prompt set they used.

Example:

```bash
python3 bench/prompt_audit.py \
  --prompts bench/prompts/smoke.jsonl \
  --output bench/results/prompt_audit_smoke_latest.json \
  --markdown bench/results/prompt_audit_smoke_latest.md \
  --min-prompts 2 \
  --max-prompt-bytes 1024
```

`qemu_prompt_bench.py` launches an air-gapped QEMU guest once per prompt, captures
serial output, extracts token timing records, and writes normalized JSON to
`bench/results/`. The runner always injects `-nic none` and rejects conflicting
network flags such as `-netdev` or virtual NIC devices, including legacy QEMU
NIC models such as e1000, ne2k, pcnet, rtl8139, usb-net, virtio-net, and vmxnet.

Use `--warmup N` to launch each prompt before measurement without mixing those
runs into throughput dashboards, and `--repeat N` to run every prompt multiple
times. Reports include separate warmup records, raw measured per-run records,
an overall suite summary, and per-prompt medians and min/max tok/s in JSON and
Markdown. The suite summary includes measured prompt count, run count, total
tokens, total elapsed time, median/P95 tok/s, tok/s standard deviation,
coefficient of variation, and max memory. The runner also writes a deterministic
prompt-suite SHA256 matching `prompt_audit.py`, plus
`qemu_prompt_bench_latest.csv` with one row per measured run for CI artifact
upload, spreadsheets, and simple shell comparisons.

Prompt files can be JSON, JSONL, or plain text split with `---`. Guest output may
include a JSON line such as:

```text
BENCH_RESULT: {"tokens": 128, "elapsed_us": 500000, "tok_per_s": 256.0}
```

Memory telemetry is optional. The runner normalizes `memory_bytes`,
`max_rss_bytes`, `rss_bytes`, `peak_memory_bytes`, plus `_kib`, `_kb`, `_mib`,
and `_mb` variants into `memory_bytes` so the perf regression dashboard can
track peak memory alongside tok/s.

Use `--max-suite-cv-pct` and `--max-prompt-cv-pct` to fail noisy benchmark runs
when measured tok/s coefficient of variation exceeds a CI threshold. Gate
findings are written into the JSON and Markdown reports as
`variability_findings`.

Example:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --profile secure-local \
  --quantization Q4_0 \
  --warmup 1 \
  --repeat 5 \
  --max-prompt-cv-pct 5 \
  -- -m 512M
```

Validate the final QEMU command without launching:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --dry-run
```

Refresh the committed smoke report without booting a guest:

```bash
python3 bench/qemu_prompt_bench.py \
  --image /tmp/TempleOS.synthetic.img \
  --prompts bench/prompts/smoke.jsonl \
  --qemu-bin bench/fixtures/qemu_synthetic_bench.py \
  --profile synthetic-airgap-smoke \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --repeat 3
```

## Benchmark Matrix

`bench_matrix.py` runs `qemu_prompt_bench.py` across a local JSON matrix of
profiles, models, and quantization formats. Each cell writes an isolated
`qemu_prompt_bench_latest.json` under `bench/results/bench_matrix_*`, while the
matrix summary is written to `bench/results/bench_matrix_latest.json`, `.md`,
and `.csv`. QEMU launches still flow through the air-gap guard that injects
`-nic none` and rejects NIC/network arguments. Each matrix cell records the
prompt-suite SHA256 from its underlying prompt benchmark report so matrix
comparisons can reject accidental prompt drift.

Example:

```bash
python3 bench/bench_matrix.py \
  --matrix bench/fixtures/bench_matrix_smoke.json \
  --output-dir bench/results
```

Validate the expanded matrix without launching QEMU:

```bash
python3 bench/bench_matrix.py \
  --matrix bench/fixtures/bench_matrix_smoke.json \
  --dry-run
```

`bench_result_index.py` scans existing QEMU prompt and matrix JSON reports,
rolls their tok/s, memory, prompt-suite, and run-count metadata into a single
JSON/Markdown/CSV index, and checks each recorded QEMU command for explicit
`-nic none` air-gap compliance. It also reports prompt-suite drift when
comparable profile/model/quantization artifacts carry different non-empty suite
hashes, which catches accidental prompt changes before throughput numbers are
compared. The drift findings are also written to
`bench_result_index_prompt_suite_drift_latest.csv` for CI upload or spreadsheet
review. It never launches QEMU.

Example:

```bash
python3 bench/bench_result_index.py \
  --input bench/results \
  --output-dir bench/results \
  --fail-on-airgap \
  --fail-on-drift
```

`bench_artifact_manifest.py` builds on the same indexer and writes a
deterministic latest-artifact manifest for CI upload/download consumers. It
records SHA256 and byte size for each benchmark JSON, retains compact history,
selects the newest artifact for each profile/model/quantization/prompt-suite
key, and keeps the same recorded-command air-gap checks.

```bash
python3 bench/bench_artifact_manifest.py \
  --input bench/results \
  --output-dir bench/results \
  --fail-on-airgap
```

## Perf Regression Dashboard

`perf_regression.py` scans JSON, JSONL, and CSV benchmark artifacts, groups
results by benchmark/profile/model/quantization/prompt plus commit, and writes
tok/s, memory, and sample-coverage dashboards under `bench/dashboards/`.

Example CI gate:

```bash
python3 bench/perf_regression.py \
  --input bench/results \
  --output-dir bench/dashboards \
  --min-records-per-point 3 \
  --fail-on-regression
```

`--min-records-per-point` fails the dashboard when any benchmark key/commit
point has fewer samples than required. This catches partial matrix uploads and
single-run artifacts before noisy throughput medians are accepted.
The dashboard also writes `perf_regression_junit_latest.xml` so CI systems can
surface throughput regressions and sample-coverage failures as test failures.

## Build Benchmark Compare

`build_compare.py` compares multiple `qemu_prompt_bench.py` JSON reports by
prompt/profile/model/quantization and writes per-build throughput and elapsed
time deltas to `bench/results/` as JSON, Markdown, CSV, and JUnit XML. It also
compares max memory bytes when benchmark reports include memory telemetry. Use
`--fail-on-regression` with `--max-tok-regression-pct` to gate median tok/s
drops in CI without launching QEMU, and add `--max-memory-growth-pct` to gate
peak memory growth.

Example:

```bash
python3 bench/build_compare.py \
  --input base=bench/results/qemu_prompt_bench_base.json \
  --input head=bench/results/qemu_prompt_bench_latest.json \
  --baseline base \
  --fail-on-regression \
  --max-tok-regression-pct 5 \
  --max-memory-growth-pct 10
```

## HolyC vs llama.cpp Eval Compare

`eval_compare.py` compares offline multiple-choice predictions from HolyC and
llama.cpp against the same local gold JSONL dataset. It aligns by record id,
supports prediction indexes, labels, exact choice text, or score arrays, and
writes JSON, Markdown, and per-record CSV reports to `bench/results/`. Reports
include accuracy, agreement, macro-F1, per-answer F1, and confusion matrices
for each engine.

Before comparing, `eval_input_audit.py` can gate apples-to-apples inputs. It
checks gold/prediction record coverage, duplicates, invalid prediction indexes,
dataset/split metadata, and optional model/quantization metadata drift. The
audit writes JSON and Markdown reports and exits non-zero when it finds errors:

```bash
python3 bench/eval_input_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --output-stem eval_input_audit_smoke_latest
```

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

`perplexity_compare.py` compares offline token logprob or aggregate NLL outputs
from HolyC and llama.cpp. It aligns rows by record id, computes token-weighted
NLL/token and perplexity, writes JSON, Markdown, and per-record CSV reports, and
fails on token-count mismatches unless `--allow-token-count-mismatch` is passed.

Example:

```bash
python3 bench/perplexity_compare.py \
  --holyc bench/eval/samples/holyc_smoke_logprobs.jsonl \
  --llama bench/eval/samples/llama_smoke_logprobs.jsonl \
  --dataset smoke-eval \
  --split validation \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --output-stem perplexity_compare_smoke_latest
```

## Perf Regression Dashboard

`perf_regression.py` scans host-side benchmark result files and writes dashboards
to `bench/dashboards/`. It accepts JSON, JSONL, and CSV records with `tok_per_s`
or `tok_per_s_milli`, plus optional memory fields such as `memory_bytes` or
`max_rss_bytes`. Regression checks compare commit-level aggregates, so repeated
runs and duplicate latest/stamped result files are collapsed by benchmark key and
commit before the latest distinct commits are compared. Outputs include JSON,
Markdown, JUnit XML, commit-point CSV, and regression CSV artifacts.

Example:

```bash
python3 bench/perf_regression.py --input bench/results --output-dir bench/dashboards
```

CI can fail on throughput or memory regressions with:

```bash
python3 bench/perf_regression.py --fail-on-regression
```

Pin an explicit baseline/candidate pair when CI provides known SHAs:

```bash
python3 bench/perf_regression.py \
  --baseline-commit "$BASE_SHA" \
  --candidate-commit "$GITHUB_SHA" \
  --fail-on-regression
```

`airgap_audit.py` scans benchmark artifacts, including benchmark-matrix cells,
for recorded QEMU commands and fails if any QEMU-like command is missing
`-nic none` or includes networking flags/devices. It writes JSON, and can also
emit Markdown and CSV reports for CI artifacts:

```bash
python3 bench/airgap_audit.py \
  --input bench/results \
  --output bench/results/airgap_audit_latest.json \
  --markdown bench/results/airgap_audit_latest.md \
  --csv bench/results/airgap_audit_latest.csv
```

The `bench-perf-regression` GitHub Actions workflow runs a stdlib-only smoke
gate:

```bash
python3 bench/perf_ci_smoke.py
```

It scans committed benchmark results plus `bench/fixtures/perf_regression/`,
writes dashboards into a temporary directory, audits QEMU benchmark artifacts
for explicit air-gap settings, and fails if regressions or unsafe guest
networking settings are found.
