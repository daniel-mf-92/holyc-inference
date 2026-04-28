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
  Q16 scale magnitude limits, quant ranges, quant histograms, and optional
  packing-distribution gates for distinct quant values and saturated payloads.

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
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --min-used-quant-values 8 --max-saturation-pct 25
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
Manifests include UTF-8 prompt/choice/record byte statistics, and optional size
gates can fail packing before writing oversized artifacts.

Example:

```bash
python3 bench/dataset_pack.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --dataset smoke-eval \
  --split validation \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024 \
  --max-record-payload-bytes 8192
```

`hceval_inspect.py` independently parses `.hceval` binaries, validates record
bounds, verifies source/binary hashes against a companion manifest, and writes
JSON or Markdown inspection reports:
It can re-apply the same byte-size gates to already packed binaries.

```bash
python3 bench/hceval_inspect.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --output bench/results/datasets/smoke_eval.inspect.json \
  --markdown bench/results/datasets/smoke_eval.inspect.md \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024
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

`dataset_leak_audit.py` checks local eval JSONL before packing for duplicate
record IDs, normalized prompt reuse across splits, repeated prompt+choice
payloads across splits, and conflicting answers for the same payload. It never
fetches datasets and can fail CI when split leakage is found:

```bash
python3 bench/dataset_leak_audit.py \
  --input bench/results/datasets/smoke_curated.jsonl \
  --output bench/results/datasets/dataset_leak_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_leak_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_leak_audit_smoke_latest.csv \
  --fail-on-leaks
```

## Offline Eval Comparator

`eval_compare.py` compares local HolyC and llama.cpp multiple-choice predictions
against the same gold JSONL and writes JSON, Markdown, CSV, and JUnit XML
reports. Optional quality gates can fail CI when HolyC accuracy, engine
agreement, or accuracy delta versus llama.cpp falls outside configured bounds.
Prediction score vectors are treated as choice-aligned logits/logprobs: every
score must be finite and the score count must match the gold choice count.

```bash
python3 bench/eval_compare.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --min-holyc-accuracy 0.95 \
  --min-agreement 0.95 \
  --max-accuracy-drop 0.02 \
  --fail-on-regression
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
prompt bytes launched, total tokens, total elapsed time, median/P95 tok/s, tok/s
standard deviation, coefficient of variation, and max memory. Per-run and
per-prompt reports include UTF-8 prompt byte counts so benchmark changes can be
separated from prompt-suite size drift. Optional first-token latency telemetry is
normalized from `ttft_us`, `time_to_first_token_us`, `first_token_us`, and their
`_ms` or `_s` variants into `ttft_us`; suite and per-prompt reports include
median and P95 TTFT when present. The runner also writes a deterministic
prompt-suite SHA256 matching `prompt_audit.py`, plus
`qemu_prompt_bench_latest.csv` with one row per measured run for CI artifact
upload, spreadsheets, and simple shell comparisons. JSON and Markdown reports
also include host provenance for reproducibility: platform, machine, Python
version, CPU count, QEMU binary/path, and QEMU version when discoverable.

Prompt files can be JSON, JSONL, or plain text split with `---`. Guest output may
include a JSON line such as:

```text
BENCH_RESULT: {"tokens": 128, "elapsed_us": 500000, "ttft_us": 42000, "tok_per_s": 256.0}
```

Memory telemetry is optional. The runner normalizes `memory_bytes`,
`max_rss_bytes`, `rss_bytes`, `peak_memory_bytes`, plus `_kib`, `_kb`, `_mib`,
and `_mb` variants into `memory_bytes` so the perf regression dashboard can
track peak memory alongside tok/s.

Each measured run also records `wall_tok_per_s`, derived from host wall-clock
elapsed time and emitted token count. This is reported next to guest telemetry
in JSON, Markdown, and CSV so suspicious guest-side timing can be compared
against the host-observed launch duration.

Use `--max-suite-cv-pct` and `--max-prompt-cv-pct` to fail noisy benchmark runs
when measured tok/s coefficient of variation exceeds a CI threshold. Gate
findings are written into the JSON and Markdown reports as
`variability_findings`. The runner also writes
`qemu_prompt_bench_junit_latest.xml` so CI can surface failed prompt launches
and variability gate failures directly from the benchmark job.

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

Dry-runs also write `qemu_prompt_bench_dry_run_latest.json` and `.md` under the
selected output directory. These artifacts record the exact `-nic none` command,
prompt-suite hash, warmup count, repeat count, and planned launch totals for CI
review without booting a guest, plus the same host/QEMU provenance fields used
by measured benchmark reports.

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
`.csv`, and `bench_matrix_junit_latest.xml`. QEMU launches still flow through
the air-gap guard that injects `-nic none` and rejects NIC/network arguments.
Each matrix cell records the
prompt-suite SHA256 from its underlying prompt benchmark report so matrix
comparisons can reject accidental prompt drift.
Use `max_suite_cv_pct` and `max_prompt_cv_pct` in the matrix JSON, or the
matching CLI flags, to pass tok/s variability gates through to every cell. A
cell that fails a variability gate still writes its prompt-benchmark report and
is preserved in the matrix summary as a failed cell with a findings count.

Example:

```bash
python3 bench/bench_matrix.py \
  --matrix bench/fixtures/bench_matrix_smoke.json \
  --output-dir bench/results \
  --max-prompt-cv-pct 5
```

Validate the expanded matrix without launching QEMU:

```bash
python3 bench/bench_matrix.py \
  --matrix bench/fixtures/bench_matrix_smoke.json \
  --dry-run
```

`bench_result_index.py` scans existing QEMU prompt and matrix JSON reports,
rolls their tok/s, memory, prompt-suite, and run-count metadata into a single
JSON/Markdown/CSV/JUnit XML index, and checks each recorded QEMU command for
explicit `-nic none` air-gap compliance. It also reports prompt-suite drift when
comparable profile/model/quantization artifacts carry different non-empty suite
hashes, which catches accidental prompt changes before throughput numbers are
compared. The drift findings are also written to
`bench_result_index_prompt_suite_drift_latest.csv`, and
`bench_result_index_junit_latest.xml` exposes artifact failures, air-gap
violations, and prompt drift as CI test failures. It never launches QEMU.

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
  --max-tok-cv-pct 7.5 \
  --fail-on-regression
```

`--min-records-per-point` fails the dashboard when any benchmark key/commit
point has fewer samples than required. This catches partial matrix uploads and
single-run artifacts before noisy throughput medians are accepted.
`--max-tok-cv-pct` fails the dashboard when repeated tok/s samples inside a
benchmark key/commit point are too variable to trust as a baseline.
The dashboard also writes `perf_regression_junit_latest.xml` so CI systems can
surface throughput regressions, sample-coverage failures, and variability
failures as test failures.

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
for each engine. Accuracy and agreement summaries also include stdlib-only
Wilson confidence intervals; use `--confidence-level` to select 0.80, 0.90,
0.95, 0.98, or 0.99.

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
Markdown, JUnit XML, commit-point CSV, regression CSV, sample-coverage CSV, and
tok/s variability CSV artifacts.

Example:

```bash
python3 bench/perf_regression.py --input bench/results --output-dir bench/dashboards
```

CI can fail on throughput or memory regressions with:

```bash
python3 bench/perf_regression.py --max-tok-cv-pct 7.5 --fail-on-regression
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
emit Markdown, CSV, and JUnit XML reports for CI artifacts:

```bash
python3 bench/airgap_audit.py \
  --input bench/results \
  --output bench/results/airgap_audit_latest.json \
  --markdown bench/results/airgap_audit_latest.md \
  --csv bench/results/airgap_audit_latest.csv \
  --junit bench/results/airgap_audit_junit_latest.xml
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
