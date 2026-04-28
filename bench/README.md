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
  Q16 scale magnitude limits, zero-scale/nonzero-payload counts, quant ranges,
  quant histograms, and optional packing-distribution gates for distinct quant
  values, saturated payloads, non-canonical zero-scale blocks, and zero or
  subnormal fp16 scale fields.

Example:

```bash
python3 bench/quant_audit.py \
  --source-root src/quant \
  --output bench/results/quant_audit_latest.json \
  --markdown bench/results/quant_audit_latest.md \
  --csv bench/results/quant_audit_latest.csv \
  --junit bench/results/quant_audit_junit_latest.xml
```

Raw block streams can be checked with:

```bash
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --expect-elements 4096
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --expect-blocks 128
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --max-abs-scale-q16 1048576
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --min-used-quant-values 8 --max-saturation-pct 25
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --fail-zero-scale-nonzero-blocks
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --fail-zero-scales --fail-subnormal-scales
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
  --min-choices 4 \
  --max-choices 4 \
  --max-records-per-provenance 1 \
  --max-records 3 \
  --balance-answer-index \
  --pack-output bench/results/datasets/smoke_curated.hceval \
  --pack-manifest bench/results/datasets/smoke_curated.hceval.manifest.json
```

When `--max-records` trims a larger local source, `--balance-answer-index`
round-robins through answer labels before applying the final stable output sort.
This keeps compact multiple-choice subsets from accidentally overrepresenting a
single answer index.
Use `--min-choices` and `--max-choices` when a run needs homogeneous choice
counts across HellaSwag-, ARC-, TruthfulQA-, and normalized local rows.
Use `--max-records-per-provenance` when local source files combine multiple
staged shards and each shard should contribute at most a fixed number of rows
before the final global sample is selected.

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
JSON, Markdown, or JUnit XML inspection reports:
It can re-apply the same byte-size gates to already packed binaries.

```bash
python3 bench/hceval_inspect.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --output bench/results/datasets/smoke_eval.inspect.json \
  --markdown bench/results/datasets/smoke_eval.inspect.md \
  --junit bench/results/datasets/smoke_eval.inspect.junit.xml \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024
```

`dataset_index.py` scans curated manifests, packed `.hceval` manifests, and
inspection reports, verifies local hashes/provenance fields where possible, and
writes JSON/Markdown/CSV/JUnit XML rollups:
Relative artifact paths are resolved from the current working directory first,
then from the manifest/report directory so archived dataset bundles remain
self-validating after they are moved.

```bash
python3 bench/dataset_index.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --fail-on-findings
```

`dataset_ci_smoke.py` is a stdlib-only CI gate for the offline dataset pipeline.
It curates the synthetic sample, packs and inspects the `.hceval` binary, runs
the split-leakage audit, indexes the generated artifacts, and checks that a
known leaky fixture is rejected:

```bash
python3 bench/dataset_ci_smoke.py
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
  --junit bench/results/datasets/dataset_leak_audit_smoke_latest_junit.xml \
  --fail-on-leaks
```

`dataset_provenance_audit.py` checks curated JSONL manifests for source/license
metadata, source and output hashes, selected IDs, dataset/split counts, and
answer histograms. Reports include overall and per-dataset answer-majority
telemetry; `--max-majority-answer-pct` and
`--max-dataset-majority-answer-pct` can fail CI when curated subsets are label
skewed.

```bash
python3 bench/dataset_provenance_audit.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --max-dataset-majority-answer-pct 80 \
  --fail-on-findings
```

## Offline Eval Comparator

`eval_compare.py` compares local HolyC and llama.cpp multiple-choice predictions
against the same gold JSONL and writes JSON, Markdown, per-record CSV,
per-dataset/split breakdown CSV, confusion-matrix CSV, and JUnit XML reports.
Optional quality gates can fail CI when HolyC accuracy, engine agreement, or
accuracy delta versus llama.cpp falls outside configured bounds.
Prediction score vectors are treated as choice-aligned logits/logprobs: every
score must be finite and the score count must match the gold choice count.
`eval_input_audit.py` also records per-engine prediction histograms and can fail
early with `--max-majority-prediction-pct` when either engine collapses onto one
answer index before quality metrics are computed.
It also records per-engine score-vector coverage and can fail early with
`--min-score-coverage-pct` when calibration/ranking evals require logprob-style
choice scores from both engines.
When score vectors are present, reports include per-row confidence/margin plus
score coverage, mean confidence, Brier score, and expected calibration error.
Reports also rank the gold answer within each score vector and summarize top-1,
top-2, top-3, mean gold rank, and mean reciprocal rank for each engine.
Reports also include paired correctness counts and an exact two-sided McNemar
binomial p-value so HolyC-vs-llama quality deltas can be interpreted as paired
eval outcomes on the same records.

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
hash so benchmark artifacts can name the exact prompt set they used. It can
also write CSV and JUnit XML artifacts so CI can upload prompt stats and fail
directly on prompt-suite errors.

Example:

```bash
python3 bench/prompt_audit.py \
  --prompts bench/prompts/smoke.jsonl \
  --output bench/results/prompt_audit_smoke_latest.json \
  --markdown bench/results/prompt_audit_smoke_latest.md \
  --csv bench/results/prompt_audit_smoke_latest.csv \
  --junit bench/results/prompt_audit_smoke_latest_junit.xml \
  --min-prompts 2 \
  --max-prompt-bytes 1024
```

`qemu_prompt_bench.py` launches an air-gapped QEMU guest once per prompt, captures
serial output, extracts token timing records, and writes normalized JSON to
`bench/results/`. The runner always injects `-nic none` and rejects conflicting
network flags such as `-netdev` or virtual NIC devices, including legacy QEMU
NIC models such as e1000, ne2k, pcnet, rtl8139, usb-net, virtio-net, and vmxnet.
Extra QEMU options can be passed one token at a time with `--qemu-arg`, after
`--`, or from local `--qemu-args-file` files. Argument files are offline-only:
`.json` files must contain a string array, while other files use shell-style
tokenization with `#` comments. Loaded file tokens go through the same air-gap
network rejection before any dry run or guest launch.

Use `--warmup N` to launch each prompt before measurement without mixing those
runs into throughput dashboards, and `--repeat N` to run every prompt multiple
times. Use `--max-launches N` to fail before booting QEMU when
`prompts * (warmup + repeat)` would exceed the expected launch budget. Reports
include separate warmup records, raw measured per-run records, an overall suite
summary, and per-prompt medians, min/max tok/s, P05/P95 tok/s, and
P05-to-P95 spread percentages in JSON and
Markdown. The suite summary includes measured prompt count, run count, total
prompt bytes launched, total tokens, total elapsed time, P05/median/P95 tok/s,
tok/s standard deviation, coefficient of variation, P05-to-P95 spread
percentage, and max memory. Per-run and per-prompt reports include UTF-8 prompt
byte counts so benchmark changes can be
separated from prompt-suite size drift. Optional first-token latency telemetry is
normalized from `ttft_us`, `time_to_first_token_us`, `first_token_us`, and their
`_ms` or `_s` variants into `ttft_us`; suite and per-prompt reports include
median and P95 TTFT when present. The runner also writes a deterministic
prompt-suite SHA256 matching `prompt_audit.py`, plus
`qemu_prompt_bench_latest.csv` with one row per measured run for CI artifact
upload, spreadsheets, and simple shell comparisons. JSON and Markdown reports
also include host provenance for reproducibility: platform, machine, Python
version, CPU count, QEMU binary/path, QEMU version when discoverable, and a
stable SHA256 fingerprint of the QEMU command line.

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
elapsed time. Reports now include `host_overhead_us` and `host_overhead_pct`,
computed as host wall elapsed time minus guest-reported elapsed time, so QEMU
launch/serial/host orchestration overhead can be tracked separately from guest
decode telemetry. This is reported next to guest telemetry in JSON, Markdown,
and CSV so suspicious guest-side timing can be compared against the
host-observed launch duration. On Unix hosts, the runner also records
`host_child_user_cpu_us`, `host_child_system_cpu_us`, `host_child_cpu_us`, and
`host_child_cpu_pct` from child-process resource usage around each QEMU launch;
suite and prompt summaries include median child CPU time and utilization so CPU
saturation can be distinguished from guest decode timing drift. The same
artifacts also include derived
`us_per_token` and `wall_us_per_token` latency metrics, plus median/P95 latency
rollups, so dashboards can compare either throughput or per-token decode cost
without reprocessing raw elapsed times.

Use `--max-suite-cv-pct` and `--max-prompt-cv-pct` to fail noisy benchmark runs
when measured tok/s coefficient of variation exceeds a CI threshold. Gate
findings are written into the JSON and Markdown reports as
`variability_findings`. The runner also writes
`qemu_prompt_bench_junit_latest.xml` so CI can surface failed prompt launches
and variability gate failures directly from the benchmark job.
Use `--require-tokens`, `--require-tok-per-s`, `--require-memory`,
`--require-ttft-us`, `--min-tokens`, `--min-tok-per-s`,
`--min-wall-tok-per-s`, `--max-memory-bytes`, `--max-ttft-us`,
`--max-host-overhead-us`, and `--max-host-overhead-pct` to fail measured runs
that omit required telemetry, produce too little work for a trustworthy
throughput sample, exceed a host-observed latency, memory, or orchestration
overhead budget, or exceed a first-token latency budget. Telemetry gate failures
are written as `telemetry_findings` in JSON/Markdown and as
`benchmark_telemetry` failures in the JUnit report.

Example:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --profile secure-local \
  --quantization Q4_0 \
  --warmup 1 \
  --repeat 5 \
  --max-launches 30 \
  --max-prompt-cv-pct 5 \
  --require-tokens \
  --require-tok-per-s \
  --require-ttft-us \
  --min-tokens 16 \
  --min-wall-tok-per-s 10 \
  --max-memory-bytes 536870912 \
  --max-ttft-us 1000000 \
  --max-host-overhead-pct 25 \
  --qemu-args-file bench/fixtures/local-qemu.args \
  -- -m 512M
```

Validate the final QEMU command without launching:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --max-launches 10 \
  --dry-run
```

Dry-runs also write `qemu_prompt_bench_dry_run_latest.json` and `.md` under the
selected output directory. These artifacts record the exact `-nic none` command,
the command SHA256 fingerprint, prompt-suite hash, warmup count, repeat count,
configured launch budget, and planned launch totals for CI review without
booting a guest, plus the same host/QEMU provenance fields used by measured
benchmark reports.

`qemu_source_audit.py` statically scans host-side docs/config/shell-like files
for literal `qemu-system*` launch snippets and applies the same air-gap command
rules. This catches unsafe copied commands before they become benchmark scripts
or operator runbooks. Raw QEMU examples must keep `-nic none` explicit:

```bash
qemu-system-x86_64 \
  -nic none \
  -m 512M \
  -drive file=/tmp/TempleOS.img,format=raw,if=ide
```

Run the source audit with JSON, Markdown, CSV, and JUnit output:

```bash
python3 bench/qemu_source_audit.py \
  --output bench/results/qemu_source_audit_latest.json \
  --markdown bench/results/qemu_source_audit_latest.md \
  --csv bench/results/qemu_source_audit_latest.csv \
  --junit bench/results/qemu_source_audit_junit_latest.xml
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
`.csv`, and `bench_matrix_junit_latest.xml`. QEMU launches still flow through
the air-gap guard that injects `-nic none` and rejects NIC/network arguments.
Matrix-level and per-axis QEMU options may be declared with `qemu_args`,
`qemu_args_file`, or `qemu_args_files`; file paths are resolved relative to the
matrix JSON file and are validated before cells are launched. Matrix summaries
carry each cell's command SHA256 fingerprint so CI artifacts can detect QEMU
argument drift independently from prompt-suite drift.
Each matrix cell records the
prompt-suite SHA256 from its underlying prompt benchmark report so matrix
comparisons can reject accidental prompt drift.
Matrix rollups also preserve each cell's guest tok/s, host wall-clock tok/s,
P95 TTFT, median host-overhead percentage, and guest/wall us-per-token latency
from the underlying prompt benchmark report.
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
rolls their tok/s, wall-clock tok/s, TTFT, host-overhead, per-token latency,
memory, prompt-suite, and run-count metadata into a single
JSON/Markdown/CSV/JUnit XML index, and checks each recorded QEMU command for
explicit `-nic none` air-gap compliance. It also reports prompt-suite drift when
comparable profile/model/quantization artifacts carry different non-empty suite
hashes, and command drift when comparable artifacts carry different
`command_sha256` values. These drift checks catch accidental prompt or launch
changes before throughput numbers are compared. The index marks artifacts with
missing required telemetry, such as zero measured runs or absent median tok/s,
as failures so empty or malformed benchmark reports do not enter CI dashboards
as valid data. The drift findings are also written to
`bench_result_index_prompt_suite_drift_latest.csv` and
`bench_result_index_command_drift_latest.csv`, and
`bench_result_index_junit_latest.xml` exposes artifact failures, air-gap
violations, missing telemetry, inconsistent commit metadata, inconsistent
command hash metadata, prompt drift, and command drift as CI test failures. It
never launches QEMU. The index also records per-artifact commit metadata and can
optionally fail when benchmark artifacts were produced from a different commit
than the current checkout. It can also enforce freshness with
`--max-artifact-age-hours`, marking artifacts stale when their `generated_at`
timestamp is too old. Use `--fail-on-airgap`, `--fail-on-telemetry`,
`--fail-on-commit-metadata`, `--fail-on-command-hash-metadata`,
`--fail-on-drift`, `--fail-on-command-drift`, and `--fail-on-stale-artifact` to
gate those failure classes independently.

Example:

```bash
python3 bench/bench_result_index.py \
  --input bench/results \
  --output-dir bench/results \
  --fail-on-airgap \
  --fail-on-drift \
  --fail-on-command-drift
```

For a single CI job output directory that should only contain artifacts from
the current checkout, add:

```bash
python3 bench/bench_result_index.py \
  --input bench/results/current-job \
  --output-dir bench/results/current-job \
  --fail-on-stale-commit \
  --max-artifact-age-hours 6 \
  --fail-on-stale-artifact
```

`bench_artifact_manifest.py` builds on the same indexer and writes a
deterministic latest-artifact manifest for CI upload/download consumers. It
records SHA256 and byte size for each benchmark JSON, retains compact history,
selects the newest artifact for each profile/model/quantization/prompt-suite
key, writes both latest-key and full-history CSV exports, keeps the same
recorded-command air-gap, command SHA256, and commit
metadata checks, and writes `bench_artifact_manifest_junit_latest.xml` so CI can
surface failed artifacts, air-gap violations, missing telemetry, stale
artifacts, inconsistent command hashes, inconsistent commit metadata, and empty
manifests directly. Empty manifests are marked failed so missing benchmark
uploads do not pass silently. For current-job manifests,
`--fail-on-stale-commit` returns non-zero when any artifact was produced from a
different commit than the current checkout. `--max-artifact-age-hours` records
artifact freshness status, and `--fail-on-stale-artifact` returns non-zero when
any artifact exceeds that age. Use `--fail-on-airgap`, `--fail-on-telemetry`,
`--fail-on-command-hash-metadata`, and `--fail-on-commit-metadata` to gate those
failure classes independently.

```bash
python3 bench/bench_artifact_manifest.py \
  --input bench/results \
  --output-dir bench/results \
  --fail-on-stale-commit \
  --max-artifact-age-hours 6 \
  --fail-on-stale-artifact \
  --fail-on-airgap \
  --fail-on-telemetry \
  --fail-on-command-hash-metadata
```

## Perf Regression Dashboard

`perf_regression.py` scans JSON, JSONL, and CSV benchmark artifacts, groups
results by benchmark/profile/model/quantization/prompt plus commit, and writes
guest tok/s, host wall-clock tok/s, QEMU host overhead percentage, memory,
first-token latency, and sample-coverage dashboards under `bench/dashboards/`.
The dashboard also emits
`perf_regression_comparisons_latest.csv` so CI can archive baseline-vs-candidate
metric deltas even when no regression threshold is crossed.

Example CI gate:

```bash
python3 bench/perf_regression.py \
  --input bench/results \
  --output-dir bench/dashboards \
  --min-records-per-point 3 \
  --min-commits-per-key 2 \
  --max-tok-cv-pct 7.5 \
  --p05-tok-regression-pct 7.5 \
  --wall-tok-regression-pct 7.5 \
  --p95-ttft-regression-pct 15 \
  --host-overhead-regression-pct 25 \
  --require-tok-per-s \
  --require-wall-tok-per-s \
  --require-ttft-us \
  --require-host-overhead-pct \
  --require-memory \
  --fail-on-regression
```

`--min-records-per-point` fails the dashboard when any benchmark key/commit
point has fewer samples than required. This catches partial matrix uploads and
single-run artifacts before noisy throughput medians are accepted.
`--min-commits-per-key` fails when a benchmark key has fewer distinct commits
than required. This catches head-only perf uploads where there is no baseline to
compare against.
`--max-tok-cv-pct` fails the dashboard when repeated tok/s samples inside a
benchmark key/commit point are too variable to trust as a baseline.
`--p05-tok-regression-pct` optionally gates low-tail guest tok/s drops, which
catches slow individual runs that median throughput can hide.
`--wall-tok-regression-pct` optionally gates host-observed wall-clock tok/s
drops, which is useful when guest-side timing looks suspicious.
`--ttft-regression-pct` and `--p95-ttft-regression-pct` optionally gate median
and tail first-token latency growth. `--host-overhead-regression-pct`
optionally gates increases in QEMU host overhead. `--require-tok-per-s`,
`--require-wall-tok-per-s`, `--require-ttft-us`,
`--require-host-overhead-pct`, and `--require-memory` fail the dashboard when
any benchmark key/commit point has zero samples for that telemetry field. This
catches malformed or partially uploaded artifacts before CI treats missing
metrics as merely non-comparable.
Prompt-suite hashes from QEMU benchmark reports are carried into commit points;
the dashboard fails when comparable benchmark/profile/model/quantization/prompt
records contain multiple non-empty prompt-suite hashes, preventing accidental
throughput comparisons across different prompt sets.
The dashboard also writes `perf_regression_junit_latest.xml` so CI systems can
surface throughput regressions, sample-coverage failures, commit-coverage
failures, comparison-coverage failures, prompt-suite drift, and variability
failures as test failures. When `--baseline-commit` or `--candidate-commit` is
provided, every benchmark key must contain the requested commit before the
dashboard passes; missing explicit comparison commits are written to
`perf_regression_comparison_coverage_violations_latest.csv`.

## Build Benchmark Compare

`build_compare.py` compares multiple `qemu_prompt_bench.py` JSON reports by
prompt/profile/model/quantization and writes per-build throughput and elapsed
time deltas to `bench/results/` as JSON, Markdown, CSV, and JUnit XML. It also
compares P05 guest tok/s, median and P05 host wall-clock tok/s, first-token
latency, and max memory bytes when benchmark reports include that telemetry. Use
`--fail-on-regression` with `--max-tok-regression-pct` to gate median guest
tok/s drops in CI without launching QEMU, add `--max-p05-tok-regression-pct` to
gate low-tail guest tok/s drops, add `--max-wall-tok-regression-pct` to gate
host-observed tok/s drops, add `--max-p05-wall-tok-regression-pct` to gate
low-tail host-observed tok/s drops, add `--max-ttft-growth-pct` to gate
first-token latency growth, and add `--max-memory-growth-pct` to gate peak memory growth. Use
`--min-ok-runs-per-build` with `--fail-on-coverage` to reject comparisons where
the baseline or candidate build has too few successful runs for a prompt key.
Coverage violations are written to
`build_compare_coverage_violations_latest.csv` and surfaced in the JUnit report.
Prompt-suite SHA256s are carried through from QEMU prompt benchmark reports;
`--fail-on-prompt-suite-drift` rejects comparable build pairs whose prompt-suite
hashes differ, with details written to
`build_compare_prompt_suite_drift_latest.csv` and the JUnit report.

Example:

```bash
python3 bench/build_compare.py \
  --input base=bench/results/qemu_prompt_bench_base.json \
  --input head=bench/results/qemu_prompt_bench_latest.json \
  --baseline base \
  --fail-on-regression \
  --max-tok-regression-pct 5 \
  --max-p05-tok-regression-pct 10 \
  --max-wall-tok-regression-pct 5 \
  --max-p05-wall-tok-regression-pct 10 \
  --max-ttft-growth-pct 10 \
  --max-memory-growth-pct 10 \
  --min-ok-runs-per-build 3 \
  --fail-on-coverage
```

## HolyC vs llama.cpp Eval Compare

`eval_compare.py` compares offline multiple-choice predictions from HolyC and
llama.cpp against the same local gold JSONL dataset. It aligns by record id,
supports prediction indexes, labels, exact choice text, or score arrays, and
writes JSON, Markdown, per-record CSV, per-dataset/split breakdown CSV,
confusion-matrix CSV, and JUnit XML reports to `bench/results/`. Reports include
accuracy, agreement, macro-F1, per-answer F1, per-dataset/split breakdowns, and
confusion matrices for each engine. Accuracy and agreement summaries also
include stdlib-only Wilson confidence intervals; use `--confidence-level` to
select 0.80, 0.90, 0.95, 0.98, or 0.99. Add `--gate-dataset-breakdowns` to apply
the same quality gates to each dataset/split bucket, which prevents mixed eval
suites from hiding small-subset regressions behind healthy aggregate scores.

Before comparing, `eval_input_audit.py` can gate apples-to-apples inputs. It
checks gold/prediction record coverage, duplicates, invalid prediction indexes,
dataset/split metadata, optional model/quantization metadata drift, and gold
answer distribution. Use `--max-majority-gold-answer-pct` to fail early when a
gold file is too label-skewed for a useful paired comparison. The audit writes
JSON, Markdown, CSV, and JUnit XML reports and exits non-zero when it finds
errors:

```bash
python3 bench/eval_input_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --max-majority-gold-answer-pct 100 \
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
  --gate-dataset-breakdowns \
  --output-stem eval_compare_smoke_latest
```

`perplexity_compare.py` compares offline token logprob or aggregate NLL outputs
from HolyC and llama.cpp. It aligns rows by record id, computes token-weighted
NLL/token and perplexity, writes JSON, Markdown, per-record CSV, and JUnit XML
reports, and fails on token-count mismatches unless
`--allow-token-count-mismatch` is passed. Optional quality gates can fail CI
when aggregate NLL drift, HolyC/llama.cpp perplexity ratio, or per-record NLL
delta exceed configured bounds.

Example:

```bash
python3 bench/perplexity_compare.py \
  --holyc bench/eval/samples/holyc_smoke_logprobs.jsonl \
  --llama bench/eval/samples/llama_smoke_logprobs.jsonl \
  --dataset smoke-eval \
  --split validation \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --max-nll-delta 0.02 \
  --max-perplexity-ratio 1.05 \
  --max-record-nll-delta 0.10 \
  --fail-on-regression \
  --output-stem perplexity_compare_smoke_latest
```

## Perf Regression Dashboard

`perf_regression.py` scans host-side benchmark result files and writes dashboards
to `bench/dashboards/`. It accepts JSON, JSONL, and CSV records with `tok_per_s`
or `tok_per_s_milli`, optional `wall_tok_per_s` or `wall_tok_per_s_milli`,
optional first-token latency fields such as `ttft_us` or `ttft_ms`, optional
`host_overhead_pct`, plus memory fields such as `memory_bytes` or
`max_rss_bytes`. Regression checks compare
commit-level aggregates, so repeated runs and duplicate latest/stamped result
files are collapsed by benchmark key and commit before the latest distinct
commits are compared. Outputs include JSON, Markdown, JUnit XML, commit-point
CSV, baseline/candidate comparison CSV, regression CSV, sample-coverage CSV,
commit-coverage CSV, comparison-coverage CSV, prompt-suite drift CSV,
telemetry-coverage CSV, and tok/s variability CSV artifacts.

Example:

```bash
python3 bench/perf_regression.py --input bench/results --output-dir bench/dashboards
```

CI can fail on median throughput, low-tail throughput, host wall-clock
throughput, median or P95 first-token latency, QEMU host overhead, or memory
regressions with:

```bash
python3 bench/perf_regression.py \
  --max-tok-cv-pct 7.5 \
  --p05-tok-regression-pct 7.5 \
  --wall-tok-regression-pct 7.5 \
  --ttft-regression-pct 15 \
  --p95-ttft-regression-pct 15 \
  --host-overhead-regression-pct 25 \
  --require-ttft-us \
  --require-host-overhead-pct \
  --fail-on-regression
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
