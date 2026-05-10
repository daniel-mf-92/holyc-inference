# Benchmark and Evaluation Host Tools

This directory is for host-side benchmarking, evaluation, dataset, and quantization
validation infrastructure around the HolyC inference engine. Tools here must keep
TempleOS air-gapped; any QEMU command added under this tree must pass `-nic none`.
Benchmark artifacts should not add legacy `-net none`; the launcher injects
`-nic none`, and downstream audits treat legacy `-net` flags as drift. QEMU TLS
credential options are also rejected by the launcher and source/artifact audits.
Do not include `-nic none` in reusable launcher arg fragments; `qemu_prompt_bench.py`
owns that injection so final benchmark commands contain exactly one NIC disablement.
Reusable QEMU argument fragments are also gated against any embedded `-nic none`
entries so final launch commands keep exactly one explicit launcher-owned NIC
disablement.

## Quantization Audit

`quant_audit.py` checks two host-side invariants:

- HolyC quantization sources do not contain runtime float types, float literals, or
  common float math helper calls after comments and strings are stripped.
- Raw Q4_0/Q8_0 block streams have valid block sizes, optional expected
  block/element counts, finite fp16 scales, fp16-to-Q16 scale ranges, optional
  Q16 scale magnitude limits, fp16 scale exponent histograms and optional
  exponent range gates, fp16 scale sign counts and optional negative-scale
  gates, zero-scale/nonzero-payload counts, quant ranges,
  quant histograms, signed quant payload coverage, expected-element tail padding, and optional
  packing-distribution gates for distinct quant values, saturated payloads,
  duplicate complete blocks, identical block runs, repeated fp16 scale fields, non-canonical zero-scale
  blocks, nonzero padding quants, minimum negative/positive quant payload counts,
  signed quant payload balance, zero-quant payload percentage gates,
  Q4_0 low/high nibble-lane diversity, and
  zero or subnormal fp16 scale fields.

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
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --min-scale-exponent -12 --max-scale-exponent 8
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --min-used-quant-values 8 --max-saturation-pct 25
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --fail-zero-scale-nonzero-blocks
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --fail-zero-scales --fail-subnormal-scales
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --fail-negative-scales
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --expect-elements 4095 --fail-nonzero-padding-quants
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --max-duplicate-block-pct 5 --max-identical-block-run 2
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --min-scale-used-values 16 --max-duplicate-scale-pct 75 --max-identical-scale-run 8
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --min-quant-negative-count 1 --min-quant-positive-count 1
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --max-quant-sign-balance-delta 1024
python3 bench/quant_audit.py --format q4_0 --block-file path/to/blocks.bin --min-q4-nibble-lane-used-quant-values 8
python3 bench/quant_audit.py --format q8_0 --block-file path/to/blocks.bin --max-zero-quant-pct 95
```

Mixed-format audits can validate Q4_0 and Q8_0 streams in one report:

```bash
python3 bench/quant_audit.py \
  --q4-block-file path/to/q4_blocks.bin \
  --q8-block-file path/to/q8_blocks.bin
```

`quant_audit_ci_smoke.py` creates temporary Q4_0/Q8_0 block fixtures and checks
the raw-block, scale-exponent, scale-sign, duplicate-block/run, signed
quant-payload coverage and balance, repeated-scale, zero-quant percentage,
Q4_0 nibble-lane diversity, Markdown, CSV, and JUnit paths:

```bash
python3 bench/quant_audit_ci_smoke.py
```

`quant_block_compare.py` compares two raw Q4_0/Q8_0 block streams for
deterministic packing. By default, any scale or quant payload mismatch fails the
run; use `--allow-mismatches` only when collecting telemetry without gating.

```bash
python3 bench/quant_block_compare.py \
  --format q4_0 \
  --reference bench/results/reference.q4_0 \
  --candidate bench/results/candidate.q4_0 \
  --output bench/results/quant_block_compare_latest.json \
  --csv bench/results/quant_block_compare_latest.csv \
  --markdown bench/results/quant_block_compare_latest.md \
  --junit bench/results/quant_block_compare_latest_junit.xml
```

`quant_manifest_audit.py` checks saved quant block manifests against local
Q4_0/Q8_0 artifacts. It verifies artifact existence, SHA-256, byte counts, block
counts, and element counts without launching QEMU.

```bash
python3 bench/quant_manifest_audit.py \
  --manifest bench/results/quant_blocks.manifest.json \
  --root bench/results \
  --output bench/results/quant_manifest_audit_latest.json \
  --csv bench/results/quant_manifest_audit_latest.csv \
  --markdown bench/results/quant_manifest_audit_latest.md \
  --junit bench/results/quant_manifest_audit_latest_junit.xml \
  --fail-on-findings
```

Its smoke gate builds temporary Q4_0/Q8_0 block fixtures and exercises passing
and failing manifest metadata paths:

```bash
python3 bench/quant_manifest_audit_ci_smoke.py
```

## Perf SLO Audit

`perf_slo_audit.py` checks existing benchmark JSON/JSONL/CSV artifacts against
absolute CI gates for token throughput, latency, memory, prompt failures, and
minimum measured-row coverage. It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/perf_slo_audit.py bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/dashboards \
  --output-stem perf_slo_audit_latest \
  --min-rows 4 \
  --require-success \
  --max-failure-pct 0 \
  --min-tok-per-s 100 \
  --min-wall-tok-per-s 100 \
  --max-ttft-us 20000 \
  --max-memory-bytes 80000000
```

The tool writes JSON, Markdown, CSV findings, and JUnit outputs. Its smoke gate
builds temporary benchmark artifacts and exercises both passing and failing SLO
paths:

```bash
python3 bench/perf_slo_audit_ci_smoke.py
```

`qemu_exit_rate_audit.py` aggregates saved QEMU prompt benchmark rows by
profile/model/quantization/phase and gates failure, timeout, nonzero-exit, and
launch-error percentages. It reads artifacts only and does not launch QEMU.

```bash
python3 bench/qemu_exit_rate_audit.py bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_exit_rate_audit_latest \
  --min-rows 4 \
  --max-failure-pct 0 \
  --max-timeout-pct 0 \
  --max-nonzero-exit-pct 0 \
  --max-launch-error-pct 0
```

Its smoke gate covers passing and failing exit-rate thresholds:

```bash
python3 bench/qemu_exit_rate_audit_ci_smoke.py
```

`dashboard_freshness_audit.py` checks saved dashboard JSON artifacts for
missing, stale, or future `generated_at` timestamps. It is host-side only and
does not launch QEMU.

```bash
python3 bench/dashboard_freshness_audit.py bench/dashboards \
  --output-dir bench/dashboards \
  --output-stem dashboard_freshness_audit_latest \
  --max-age-hours 96 \
  --min-dashboards 1
```

Its smoke gate covers fresh, stale, and future-dated dashboard artifacts:

```bash
python3 bench/dashboard_freshness_audit_ci_smoke.py
```

`qemu_quant_coverage_audit.py` checks saved QEMU prompt benchmark artifacts for
required quantization coverage across a profile/model result set. By default it
requires both Q4_0 and Q8_0 rows and can gate rows, successful rows, and unique
prompts per quantization. It reads artifacts only and does not launch QEMU.

```bash
python3 bench/qemu_quant_coverage_audit.py bench/results/bench_matrix_20260428T150609Z \
  --output-dir bench/results \
  --output-stem qemu_quant_coverage_audit_latest \
  --min-rows-per-quant 1 \
  --min-ok-rows-per-quant 1 \
  --min-prompts-per-quant 1 \
  --require-airgap-command
```

Its smoke gate covers passing Q4_0/Q8_0 coverage plus missing-quantization and
low-success-count failures:

```bash
python3 bench/qemu_quant_coverage_audit_ci_smoke.py
```

`qemu_quant_pairing_audit.py` checks saved QEMU prompt benchmark artifacts for
per-prompt Q4_0/Q8_0 pairing. It groups rows by profile, model, prompt, phase,
iteration, and commit so matrix comparisons cannot silently compare different
prompt/build rows. It reads artifacts only and does not launch QEMU.

```bash
python3 bench/qemu_quant_pairing_audit.py bench/results/bench_matrix_20260428T200733Z \
  --output-dir bench/results \
  --output-stem qemu_quant_pairing_audit_latest \
  --require-success
```

`qemu_host_filesystem_share_audit.py` checks saved QEMU benchmark artifacts for
forbidden host filesystem passthrough flags such as `-virtfs`, `-fsdev`,
virtio/9p filesystem devices, and SMB host-share markers. It reads artifacts
only and does not launch QEMU.

```bash
python3 bench/qemu_host_filesystem_share_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_host_filesystem_share_audit_latest
```

Its smoke gate covers a clean air-gapped command and rejected host filesystem
sharing flags:

```bash
python3 bench/qemu_host_filesystem_share_audit_ci_smoke.py
```

Its smoke gate covers passing pairs, missing quantization pairs, failed pair
members, and the older `returncode=0` success fallback:

```bash
python3 bench/qemu_quant_pairing_audit_ci_smoke.py
```

`qemu_prompt_id_audit.py` checks saved QEMU prompt benchmark artifacts for
non-empty prompt identities, SHA-256 prompt hashes, and one-to-one prompt/hash
mappings so replay and quantization comparisons cannot silently mix prompts.
It reads artifacts only and does not launch QEMU.

```bash
python3 bench/qemu_prompt_id_audit.py bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_prompt_id_audit_latest \
  --min-rows 1
```

Its smoke gate covers passing prompt identities, prompt/hash drift, prompt/hash
collisions, missing prompts, malformed hashes, and result sidecars:

```bash
python3 bench/qemu_prompt_id_audit_ci_smoke.py
```

`qemu_prompt_source_audit.py` checks saved QEMU prompt benchmark rows against
the artifact's declared local prompt suite source. It validates prompt IDs,
prompt SHA-256 values, prompt byte counts, and optionally expected-token fields.
It reads artifacts and prompt files only and does not launch QEMU.

```bash
python3 bench/qemu_prompt_source_audit.py bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_prompt_source_audit_latest \
  --require-expected-tokens
```

Its smoke gate covers passing row/source parity plus prompt-hash drift and
unknown prompt IDs:

```bash
python3 bench/qemu_prompt_source_audit_ci_smoke.py
```

`qemu_exit_class_audit.py` checks row-level consistency between `returncode`,
`timed_out`, `exit_class`, `failure_reason`, and success telemetry in saved
benchmark artifacts. It reads artifacts only and does not launch QEMU.

```bash
python3 bench/qemu_exit_class_audit.py bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_exit_class_audit_latest \
  --require-success-telemetry \
  --require-failure-reason
```

Its smoke gate covers passing, mismatched exit-class, stale failure-reason, and
missing success-telemetry cases:

```bash
python3 bench/qemu_exit_class_audit_ci_smoke.py
```

`eval_score_sparsity_audit.py` checks local HolyC and llama.cpp scored
prediction JSONL files for degenerate score vectors, including too few nonzero
or unique scores and excessive zero-score percentage. It is host-side only and
does not launch QEMU.

```bash
python3 bench/eval_score_sparsity_audit.py \
  --holyc bench/eval/samples/holyc_smoke_scored_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_scored_predictions.jsonl \
  --output-dir bench/results \
  --output-stem eval_score_sparsity_audit_latest \
  --min-nonzero-scores-per-record 2 \
  --min-unique-scores-per-record 2 \
  --max-zero-score-pct 75
```

Its smoke gate covers passing dense score vectors and failing sparse/constant
vectors:

```bash
python3 bench/eval_score_sparsity_audit_ci_smoke.py
```

`eval_score_order_audit.py` checks local scored prediction artifacts for
declared-prediction vs score-vector ordering drift. It validates that each
declared prediction index matches the top score, flags out-of-range predictions
and ambiguous top-score ties, and can require every row to carry both fields.
It is host-side only and does not launch QEMU.

```bash
python3 bench/eval_score_order_audit.py \
  --predictions holyc=bench/eval/samples/holyc_smoke_scored_predictions.jsonl \
  --predictions llama=bench/eval/samples/llama_smoke_scored_predictions.jsonl \
  --output-dir bench/results \
  --output-stem eval_score_order_audit_latest \
  --require-both \
  --min-checked-records 3
```

Its smoke gate covers matching rows plus prediction/top-score mismatches and
top-score ties:

```bash
python3 bench/eval_score_order_audit_ci_smoke.py
```

`qemu_environment_audit.py` checks host environment provenance in saved QEMU
benchmark artifacts. Use `--require-row-command-provenance` to require every
benchmark row to carry command hash and air-gap metadata derived from its saved
command.

```bash
python3 bench/qemu_environment_audit.py bench/results/qemu_prompt_bench_latest.json \
  --require-qemu-path \
  --require-qemu-version \
  --require-row-command-provenance
```

`qemu_latest_alias_audit.py` checks saved QEMU `*_latest.json` aliases against
the newest stamped sibling artifact in the same directory. It flags stale alias
payloads, `generated_at` drift, missing stamped siblings, and invalid JSON while
reading artifacts only.

```bash
python3 bench/qemu_latest_alias_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_latest_alias_audit_latest
```

## Eval Score Parity Audit

`eval_score_parity_audit.py` checks HolyC and llama.cpp prediction streams
against the same local gold dataset before `eval_compare.py` consumes them. It
verifies record-id coverage and paired score-vector presence/shape parity, with
optional gates requiring scores on every paired row and bounding top-score ties.

```bash
python3 bench/eval_score_parity_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_scored_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_scored_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --require-scores \
  --min-score-parity-pct 100 \
  --max-top-score-tie-pct 0 \
  --output-dir bench/results \
  --output-stem eval_score_parity_audit_smoke_latest
```

The tool writes JSON, Markdown, findings CSV, per-record pair CSV, and JUnit
outputs. Its smoke gate exercises passing score parity plus missing-score,
missing-row, extra-row, and tied-top-score failures:

```bash
python3 bench/eval_score_parity_audit_ci_smoke.py
```

## Eval Score Delta Audit

`eval_score_delta_audit.py` checks paired HolyC and llama.cpp score vectors
against the same local gold dataset and gates absolute score drift before
quality reports consume the predictions. It reports pair coverage, top-index
match rate, max/mean absolute deltas, top-score deltas, and gold-choice score
deltas.

```bash
python3 bench/eval_score_delta_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_scored_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_scored_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --min-pair-coverage-pct 100 \
  --min-top-index-match-pct 100 \
  --max-abs-delta 0.25 \
  --max-mean-abs-delta 0.125 \
  --max-top-score-abs-delta 0.25 \
  --output-dir bench/results \
  --output-stem eval_score_delta_audit_smoke_latest
```

The smoke gate exercises the local scored eval fixtures and refreshes JSON,
Markdown, CSV, findings CSV, and JUnit sidecars:

```bash
python3 bench/eval_score_delta_audit_ci_smoke.py
```

## Eval Prediction Coverage Audit

`eval_prediction_coverage_audit.py` checks raw HolyC and llama.cpp prediction
artifacts against a local gold JSONL dataset before comparison. It reports
global and per dataset/split coverage, flags missing/extra/duplicate prediction
ids, and writes JSON, Markdown, CSV, findings CSV, and JUnit outputs.

```bash
python3 bench/eval_prediction_coverage_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --min-coverage-pct 100 \
  --min-slice-coverage-pct 100 \
  --output-dir bench/results \
  --output-stem eval_prediction_coverage_audit_smoke_latest
```

The smoke gate refreshes the sample report artifacts:

```bash
python3 bench/eval_prediction_coverage_audit_ci_smoke.py
```

## Eval Prompt Hash Audit

`eval_prompt_hash_audit.py` checks HolyC and llama.cpp prediction artifacts
against the local gold dataset's `prompt_sha256`, `choices_sha256`, and
`input_sha256` fingerprints. Use `--require-hashes` to fail stale or incomplete
prediction streams before `eval_compare.py` computes quality metrics.

```bash
python3 bench/eval_prompt_hash_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_hashed_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_hashed_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --require-hashes \
  --min-hashed-rows 6 \
  --output-dir bench/results \
  --output-stem eval_prompt_hash_audit_smoke_latest
```

The smoke gate covers matching hashes, missing hash metadata, stale input hashes,
and extra/missing prediction rows:

```bash
python3 bench/eval_prompt_hash_audit_ci_smoke.py
```

## Eval Choice Map Audit

`eval_choice_map_audit.py` checks HolyC and llama.cpp prediction label mapping
against the same local gold dataset before comparison. It reports raw answer
formats (`index`, `alpha`, `choice_text`, or `scores_only`), validates each
normalized prediction against the gold choice count, detects duplicate gold
choice text, and can gate mixed formats or HolyC/llama format drift.

```bash
python3 bench/eval_choice_map_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --min-valid-pct 100 \
  --require-engine-format-parity \
  --output-dir bench/results \
  --output-stem eval_choice_map_audit_smoke_latest
```

The smoke gate exercises the local smoke eval inputs:

```bash
python3 bench/eval_choice_map_audit_ci_smoke.py
```

## Eval Pairing Audit

`eval_pairing_audit.py` checks HolyC and llama.cpp prediction streams before
comparison. It verifies record pairing, optional row order, prediction presence,
score-vector shape, and matching top-level or nested `metadata` identity fields
including model/tokenizer hashes and prompt-template/input hashes.

```bash
python3 bench/eval_pairing_audit.py \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --min-records 3 \
  --require-same-order \
  --require-predictions \
  --fail-on-findings
```

The smoke gate covers passing pair streams plus ordering, counterpart, top-level
metadata, and nested identity-metadata drift:

```bash
python3 bench/eval_pairing_audit_ci_smoke.py
```

## Eval Identity Audit

`eval_identity_audit.py` checks HolyC and llama.cpp prediction artifacts for
apples-to-apples model identity metadata before quality metrics consume them.
It summarizes per-row `model`, `model_sha256`, `tokenizer_sha256`,
`quantization`, and `prompt_template_sha256` values, can require complete
identity metadata, and can gate cross-engine equality for selected keys.

```bash
python3 bench/eval_identity_audit.py \
  bench/eval/samples/holyc_smoke_identity_predictions.jsonl \
  bench/eval/samples/llama_smoke_identity_predictions.jsonl \
  --require-identity \
  --compare-key model_sha256 \
  --compare-key tokenizer_sha256 \
  --compare-key quantization \
  --compare-key prompt_template_sha256 \
  --output-dir bench/results \
  --output-stem eval_identity_audit_smoke_latest
```

The smoke gate exercises passing identity parity plus cross-engine model-hash
drift:

```bash
python3 bench/eval_identity_audit_ci_smoke.py
```

## Eval Artifact Identity Audit

`eval_artifact_identity_audit.py` is a stricter paired-row identity gate for
HolyC-vs-llama prediction streams. It accepts common hash aliases such as
`weights_sha256`/`gguf_sha256`, requires stable model/tokenizer/quantization
metadata by default, and fails when paired records disagree.

```bash
python3 bench/eval_artifact_identity_audit.py \
  --holyc bench/eval/samples/holyc_smoke_identity_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_identity_predictions.jsonl \
  --output-dir bench/results \
  --output-stem eval_artifact_identity_audit_latest \
  --fail-on-findings
```

The smoke gate covers a passing paired stream, model-hash mismatch, and missing
tokenizer-hash metadata:

```bash
python3 bench/eval_artifact_identity_audit_ci_smoke.py
```

## Eval Error Overlap Audit

`eval_error_overlap_audit.py` checks local HolyC and llama.cpp prediction
streams against the same gold set, then reports shared misses, engine-unique
misses, and error-set Jaccard overlap. This helps separate broad dataset
difficulty from engine-specific regressions before interpreting eval deltas.

```bash
python3 bench/eval_error_overlap_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --min-paired-records 3 \
  --min-error-jaccard 0.25 \
  --max-holyc-unique-error-excess 1 \
  --output-dir bench/results \
  --output-stem eval_error_overlap_audit_smoke_latest
```

The smoke gate exercises shared/unique error accounting plus a failing Jaccard
gate:

```bash
python3 bench/eval_error_overlap_audit_ci_smoke.py
```

## Eval Top-k Overlap Audit

`eval_topk_overlap_audit.py` checks scored HolyC and llama.cpp multiple-choice
prediction streams for paired top-k ranking overlap. It reports pair coverage,
top-k exact-match rate, top-1 disagreement, average Jaccard overlap, and whether
the gold answer is inside each engine's top-k set.

```bash
python3 bench/eval_topk_overlap_audit.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_scored_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_scored_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --top-k 2 \
  --min-pair-coverage-pct 100 \
  --min-topk-exact-match-pct 100 \
  --min-avg-jaccard 1 \
  --max-top1-disagree-pct 0 \
  --output-dir bench/results \
  --output-stem eval_topk_overlap_audit_smoke_latest
```

The smoke gate exercises passing top-k overlap plus failing top-1/top-k drift:

```bash
python3 bench/eval_topk_overlap_audit_ci_smoke.py
```

## QEMU Resource Coverage Audit

`qemu_resource_coverage_audit.py` checks saved QEMU benchmark artifacts for
resource telemetry coverage before perf dashboards or regressions consume them.
It verifies measured OK rows include host child RSS, child CPU, token/CPU, guest
memory, and memory-per-token telemetry. It is host-side only and does not launch
QEMU.

Example:

```bash
python3 bench/qemu_resource_coverage_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_resource_coverage_audit_latest \
  --min-rows 4
```

The tool writes JSON, Markdown, CSV records, CSV findings, and JUnit outputs.
Its smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_resource_coverage_audit_ci_smoke.py
```

## QEMU Prompt Length Bucket Audit

`qemu_prompt_length_bucket_audit.py` checks saved QEMU prompt benchmark
artifacts for prompt-size coverage. It buckets measured rows by `prompt_bytes`
and reports per-bucket success counts, unique prompt counts, token totals, p50
wall tok/s, low-tail wall tok/s, and p95 first-token latency. This keeps
benchmark comparisons from silently relying on only one prompt length class.

```bash
python3 bench/qemu_prompt_length_bucket_audit.py bench/results \
  --require-buckets \
  --min-successful-samples-per-bucket 1 \
  --min-prompts-per-bucket 1 \
  --max-failure-pct 25 \
  --output-dir bench/results \
  --output-stem qemu_prompt_length_bucket_audit_latest
```

Custom prompt byte buckets use `name:min:max`; leave `max` empty for the final
open-ended bucket:

```bash
python3 bench/qemu_prompt_length_bucket_audit.py bench/results \
  --bucket short:0:255 \
  --bucket medium:256:1023 \
  --bucket long:1024:
```

The smoke gate builds temporary passing and failing artifacts and verifies JSON,
Markdown, CSV, findings CSV, and JUnit outputs:

```bash
python3 bench/qemu_prompt_length_bucket_audit_ci_smoke.py
```

## QEMU CPU Accounting Audit

`qemu_cpu_accounting_audit.py` checks saved QEMU benchmark artifacts for host
child CPU telemetry consistency before perf dashboards consume them. It verifies
child CPU microseconds equal user plus system CPU, CPU percentage matches wall
time, and token-per-CPU-second metrics match token counts. It is host-side only
and does not launch QEMU.

Example:

```bash
python3 bench/qemu_cpu_accounting_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_cpu_accounting_audit_latest \
  --min-rows 4 \
  --require-cpu-metrics
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_cpu_accounting_audit_ci_smoke.py
```

## QEMU Memory Accounting Audit

`qemu_memory_accounting_audit.py` checks saved QEMU benchmark artifacts for
memory telemetry consistency before perf dashboards consume them. It verifies
guest memory bytes/token and host child peak RSS bytes/token accounting, and can
optionally require guest-reported memory to stay within host child peak RSS. It
is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_memory_accounting_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_memory_accounting_audit_latest \
  --min-rows 4 \
  --require-memory-bytes \
  --require-host-rss \
  --require-guest-memory-within-host-rss \
  --max-host-rss-over-guest-ratio 4
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_memory_accounting_audit_ci_smoke.py
```

## QEMU Serial Accounting Audit

`qemu_serial_accounting_audit.py` checks saved QEMU benchmark artifacts for
serial output telemetry consistency before perf dashboards consume them. It
verifies stdout/stderr byte counters sum to `serial_output_bytes`, and verifies
stdout/stderr line counters when those component line counters are present. It
is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_serial_accounting_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_serial_accounting_audit_latest \
  --min-rows 4 \
  --require-metrics
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_serial_accounting_audit_ci_smoke.py
```

## QEMU Serial Payload Audit

`qemu_serial_payload_audit.py` checks saved QEMU benchmark artifacts for
BENCH_RESULT extraction integrity. It verifies OK rows still contain a captured
serial payload and that payload fields match normalized row metrics for tokens,
elapsed time, TTFT, memory, prompt bytes, and prompt SHA. It is host-side only
and does not launch QEMU.

Example:

```bash
python3 bench/qemu_serial_payload_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_serial_payload_audit_latest \
  --min-rows 4
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts, then
refreshes the latest serial payload audit artifacts:

```bash
python3 bench/qemu_serial_payload_audit_ci_smoke.py
```

## QEMU Timeout Margin Audit

`qemu_timeout_margin_audit.py` checks saved QEMU benchmark artifacts for timeout
headroom before perf dashboards consume them. It verifies timeout telemetry is
present, `wall_timeout_pct` matches wall elapsed time over timeout budget, and
OK rows do not run too close to their timeout limit. Timeout rows are also
checked for near-budget wall time so early exits cannot be mislabeled as guest
timeouts. It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_timeout_margin_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_timeout_margin_audit_latest \
  --min-rows 4 \
  --max-ok-timeout-pct 90 \
  --min-timeout-timeout-pct 90
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_timeout_margin_audit_ci_smoke.py
```

## QEMU Timeout Recommendations

`qemu_timeout_recommend.py` reads saved QEMU prompt benchmark artifacts and
groups successful measured rows by benchmark/profile/model/quantization to
recommend launch timeout budgets from P95 wall time plus configurable headroom.
It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_timeout_recommend.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_timeout_recommend_latest \
  --min-samples 2 \
  --require-timeout-telemetry \
  --min-current-timeout-headroom-pct 50
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and low-headroom benchmark artifacts and
refreshes latest recommendation sidecars when a local latest QEMU benchmark
result exists:

```bash
python3 bench/qemu_timeout_recommend_ci_smoke.py
```

## QEMU Host Overhead Audit

`qemu_host_overhead_audit.py` checks saved QEMU benchmark artifacts for
host-side timing overhead accounting before perf dashboards consume them. It
verifies `host_overhead_us` and `host_overhead_pct` match wall elapsed time
minus guest-reported elapsed time, reports max/median/p95 overhead percentages,
and provides an optional negative-overhead gate for non-synthetic runs. It is
host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_host_overhead_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_host_overhead_audit_latest \
  --min-rows 4 \
  --max-ok-host-overhead-pct 50
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_host_overhead_audit_ci_smoke.py
```

## QEMU TTFT Audit

`qemu_ttft_audit.py` checks saved QEMU prompt benchmark artifacts for
time-to-first-token telemetry before dashboards compare prompt responsiveness.
It verifies measured OK rows include non-negative `ttft_us`, that TTFT does not
exceed guest or wall elapsed time, and reports min/median/p95/max TTFT plus the
maximum TTFT share of guest elapsed time. It is host-side only and does not
launch QEMU.

Example:

```bash
python3 bench/qemu_ttft_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_ttft_audit_latest \
  --min-rows 4 \
  --max-ttft-elapsed-pct 100
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_ttft_audit_ci_smoke.py
```

## QEMU Latency Distribution Audit

`qemu_latency_distribution_audit.py` checks saved QEMU benchmark artifacts for
latency distribution telemetry before perf dashboards consume them. It groups
measured OK rows by profile, model, quantization, and prompt, then reports p50
and p95 wall latency, TTFT, wall us/token, and wall tok/s. It is host-side only
and does not launch QEMU.

Example:

```bash
python3 bench/qemu_latency_distribution_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_latency_distribution_audit_latest \
  --min-rows 4 \
  --min-samples-per-group 2 \
  --max-p95-wall-us-per-token 10000 \
  --min-p50-wall-tok-per-s 100
```

The tool writes JSON, Markdown, CSV group rows, CSV sample rows, CSV findings,
and JUnit outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_latency_distribution_audit_ci_smoke.py
```

## QEMU Token Accounting Audit

`qemu_token_accounting_audit.py` checks saved QEMU benchmark artifacts for
token-derived metric consistency before benchmark dashboards consume them. It
verifies tok/s, us/token, prompt-byte ratios, memory/token ratios, and optional
expected-token contracts from measured OK rows. With
`--require-expected-tokens-match`, the audit fails both stale match flags and
honestly recorded token-count mismatches. It is host-side only and does not
launch QEMU.

Example:

```bash
python3 bench/qemu_token_accounting_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_token_accounting_audit_latest \
  --min-rows 4 \
  --require-expected-tokens \
  --require-expected-tokens-match
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_token_accounting_audit_ci_smoke.py
```

## QEMU Timing Consistency Audit

`qemu_timing_consistency_audit.py` checks saved QEMU benchmark artifacts for
derived timing metric consistency before result dashboards consume them. It
verifies elapsed/wall rates, us/token ratios, host overhead math, timeout
percentages, TTFT bounds, and child CPU totals. It is host-side only and does
not launch QEMU.

Example:

```bash
python3 bench/qemu_timing_consistency_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_timing_consistency_audit_latest \
  --measured-only
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_timing_consistency_audit_ci_smoke.py
```

## QEMU Prompt Echo Audit

`qemu_prompt_echo_audit.py` checks saved QEMU benchmark artifacts for host/guest
prompt identity parity. It verifies measured OK rows have matching host prompt
bytes, guest-reported prompt bytes, host prompt SHA-256, guest-reported prompt
SHA-256, and match flags before throughput or eval reports trust a guest run.
It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_prompt_echo_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_prompt_echo_audit_latest \
  --min-rows 4
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_prompt_echo_audit_ci_smoke.py
```

## QEMU Seed Audit

`qemu_seed_audit.py` checks saved QEMU prompt benchmark artifacts for deterministic
seed metadata before repeatability audits compare outputs across runs. It
requires measured OK rows to carry an integer `seed`, `rng_seed`, or
`sampler_seed`, rejects negative or malformed seeds, and can gate seed drift
within the same profile/model/quantization/prompt/iteration/commit group. It is
host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_seed_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_seed_audit_latest \
  --min-rows 4
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_seed_audit_ci_smoke.py
```

## QEMU Output Determinism Audit

`qemu_output_determinism_audit.py` checks saved QEMU prompt benchmark artifacts
for repeated-run output drift. It groups measured OK rows by profile, model,
quantization, prompt, commit, and seed, then verifies each group has enough
repeats plus stable generated output hashes and token counts. It is host-side
only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_output_determinism_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_output_determinism_audit_latest \
  --require-output-hash \
  --require-tokens \
  --min-repeats 2
```

The tool writes JSON, Markdown, CSV rows, CSV findings, and JUnit outputs. Its
smoke gate builds temporary passing and failing benchmark artifacts:

```bash
python3 bench/qemu_output_determinism_audit_ci_smoke.py
```

## QEMU Throughput Stability Audit

`qemu_throughput_stability_audit.py` checks saved QEMU benchmark artifacts for
per-prompt wall tok/s floors and variability before perf dashboards consume
them. It groups measured OK rows by profile, model, quantization, and prompt,
then reports min/mean/median/max wall tok/s plus coefficient of variation. It is
host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_throughput_stability_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_throughput_stability_audit_latest \
  --min-rows 4 \
  --min-samples-per-group 2 \
  --min-wall-tok-per-s 100 \
  --max-wall-tok-per-s-cv 0.10
```

The tool writes JSON, Markdown, CSV group rows, CSV sample rows, CSV findings,
and JUnit outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_throughput_stability_audit_ci_smoke.py
```

## QEMU Phase Sequence Audit

`qemu_phase_sequence_audit.py` checks saved QEMU benchmark artifacts for expected
warmup/measured phase structure before benchmark dashboards consume them. It
verifies per-prompt warmup and measured row coverage, rejects warmups recorded
after measured rows, duplicate per-phase iterations, unknown phases, and measured
rows that did not complete successfully when requested. It is host-side only and
does not launch QEMU.

Example:

```bash
python3 bench/qemu_phase_sequence_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_phase_sequence_audit_latest \
  --min-rows 4 \
  --min-warmups-per-group 1 \
  --min-measured-per-group 2 \
  --require-measured-ok
```

The tool writes JSON, Markdown, CSV group rows, CSV run rows, CSV findings, and
JUnit outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_phase_sequence_audit_ci_smoke.py
```

## QEMU Launch Order Audit

`qemu_launch_order_audit.py` checks saved QEMU benchmark artifacts for launch
index integrity before dashboards or regression gates consume them. It verifies
unique contiguous launch indices, planned launch counts, warmups before measured
runs, row timestamps, and optional interval-overlap checks with a default
tolerance for second-resolution timestamps. It is host-side only and does not
launch QEMU.

Example:

```bash
python3 bench/qemu_launch_order_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_launch_order_audit_latest
```

The tool writes JSON, Markdown, CSV artifact rows, CSV launch rows, CSV
findings, and JUnit outputs. Its smoke gate builds temporary passing and failing
benchmark artifacts:

```bash
python3 bench/qemu_launch_order_audit_ci_smoke.py
```

## QEMU Failure Taxonomy Audit

`qemu_failure_audit.py` checks saved QEMU benchmark artifacts for consistent
failure accounting before dashboards or regression gates consume them. It
validates known `exit_class` values, timeout flag parity, failure reasons,
return-code consistency, OK-row timing/token metrics, and optional aggregate
failure/timeout percentage gates. It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_failure_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_failure_audit_latest \
  --min-rows 4 \
  --max-failure-pct 0 \
  --max-timeout-pct 0
```

The tool writes JSON, Markdown, CSV exit-class summaries, CSV row details, CSV
findings, and JUnit outputs. Its smoke gate builds temporary passing and failing
benchmark artifacts:

```bash
python3 bench/qemu_failure_audit_ci_smoke.py
```

## QEMU Launch Integrity Audit

`qemu_launch_integrity_audit.py` checks saved QEMU benchmark artifacts for
launch-plan integrity before dashboards or regression gates consume them. It
recomputes launch-plan hashes, expected/observed launch sequence hashes, and
stored launch-sequence integrity metadata when present. Legacy artifacts without
launch-plan telemetry are tolerated unless `--require-launch-plan` is set. It
is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_launch_integrity_audit.py \
  bench/results \
  --output-dir bench/results \
  --output-stem qemu_launch_integrity_audit_latest \
  --require-match
```

The tool writes JSON, Markdown, CSV artifact rows, CSV findings, and JUnit
outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_launch_integrity_audit_ci_smoke.py
```

## QEMU Artifact Budget Audit

`qemu_artifact_budget_audit.py` checks saved QEMU benchmark artifacts for
bounded file sizes, captured serial output, stdout/stderr tails, and failure
reason payloads before dashboards consume them. It is host-side only and does
not launch QEMU. It also verifies stdout/stderr byte counters are consistent
with retained tails so diagnostics are not silently dropped or overstated.

Example:

```bash
python3 bench/qemu_artifact_budget_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_artifact_budget_audit_latest \
  --max-file-bytes 2000000 \
  --max-serial-output-bytes 131072 \
  --max-stdout-tail-bytes 4096
```

The tool writes JSON, Markdown, CSV artifact rows, CSV findings, and JUnit
outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_artifact_budget_audit_ci_smoke.py
```

## QEMU Artifact Reference Audit

`qemu_artifact_reference_audit.py` checks saved QEMU benchmark artifacts for
remote URLs, network shares, scp-style remote paths, and QEMU command arrays
that drift from the air-gap contract. It is host-side only and does not launch
QEMU.

Example:

```bash
python3 bench/qemu_artifact_reference_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_artifact_reference_audit_latest \
  --min-artifacts 1
```

The tool writes JSON, Markdown, CSV artifact rows, CSV findings, and JUnit
outputs. Its smoke gate builds a temporary synthetic benchmark artifact and
checks both passing local references and failing remote/network references:

```bash
python3 bench/qemu_artifact_reference_audit_ci_smoke.py
```

## QEMU Image Reference Audit

`qemu_image_reference_audit.py` checks saved QEMU benchmark artifacts for drift
between declared `image.path` metadata and disk image references recorded in
QEMU command arrays. It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_image_reference_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_image_reference_audit_latest \
  --require-drive-reference \
  --require-single-drive-path
```

The tool writes JSON, Markdown, CSV artifact rows, CSV findings, and JUnit
outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_image_reference_audit_ci_smoke.py
```

`qemu_block_device_policy_audit.py` checks saved QEMU benchmark command
telemetry for canonical local raw IDE image drives. It rejects remote block
transports, `-blockdev` graphs, non-raw/non-IDE drive options, and extra legacy
disk media options:

```bash
python3 bench/qemu_block_device_policy_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_block_device_policy_audit_latest
```

The tool writes JSON, Markdown, CSV artifact rows, CSV findings, and JUnit
outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_block_device_policy_audit_ci_smoke.py
```

## QEMU Launch Profile Audit

`qemu_launch_profile_audit.py` checks saved QEMU benchmark artifacts for drift
between the top-level launch command and warmup/measured command arrays. It
compares executable, machine, CPU, accelerator, and memory settings, and can
require specific fields such as `-m`. Use `--fail-on-cross-artifact-drift`
when a matrix must compare artifacts with the same executable, machine, CPU,
accelerator, and memory profile. It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_launch_profile_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_launch_profile_audit_latest \
  --require-memory \
  --fail-on-cross-artifact-drift
```

The tool writes JSON, Markdown, CSV profile rows, CSV findings, and JUnit
outputs. Its smoke gate builds temporary passing and failing benchmark
artifacts:

```bash
python3 bench/qemu_launch_profile_audit_ci_smoke.py
```

## QEMU Command Fingerprint Audit

`qemu_command_fingerprint_audit.py` checks saved QEMU benchmark artifacts for
recomputable command SHA256s, explicit `-nic none`, legacy `-net none` drift,
row-vs-top-level command hash consistency, and stale row-level air-gap metadata.
It is host-side only and does not launch QEMU. Use
`--require-single-command-hash` when a benchmark artifact must prove all
measured and warmup rows used exactly the same launch command. Use
`--require-row-airgap-metadata` when every row must carry recomputable
`command_airgap_*` metadata.

Example:

```bash
python3 bench/qemu_command_fingerprint_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_command_fingerprint_audit_latest \
  --require-top-command \
  --require-single-command-hash \
  --require-row-airgap-metadata
```

The tool writes JSON, Markdown, CSV row records, CSV findings, and JUnit
outputs. Its smoke gate exercises passing hashes plus hash drift, air-gap
violations, stale air-gap metadata, row-command drift, and multi-command
rejection:

```bash
python3 bench/qemu_command_fingerprint_audit_ci_smoke.py
```

## QEMU Replay Manifest

`qemu_replay_manifest.py` builds a replay manifest from saved
`qemu_prompt_bench.py` JSON artifacts. It captures the benchmark argv, prompt
suite hash, launch-plan hashes, measured-row counts, and provenance needed to
replay a run while auditing that each recorded command remains air-gapped with
explicit `-nic none`. It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/qemu_replay_manifest.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_replay_manifest_latest
```

The tool writes JSON, Markdown, CSV replay rows, JSONL argv records, CSV
findings, and JUnit outputs. Its smoke gate builds a temporary passing replay
artifact:

```bash
python3 bench/qemu_replay_manifest_ci_smoke.py
```

## QEMU Replay Manifest Audit

`qemu_replay_manifest_audit.py` checks exported replay manifests and argv
sidecars for schema parity, recomputed command hashes, sidecar hash drift,
source artifact presence, replay provenance fields, `qemu_bin`/argv parity, and
explicit `-nic none` air-gap metadata. It is host-side only and does not launch
QEMU.

Example:

```bash
python3 bench/qemu_replay_manifest_audit.py \
  bench/results/qemu_replay_manifest_latest.json \
  --output-dir bench/results \
  --output-stem qemu_replay_manifest_audit_latest
```

The tool writes JSON, Markdown, CSV manifest rows, CSV findings, and JUnit
outputs. Its smoke gate builds a temporary replay manifest and audits it:

```bash
python3 bench/qemu_replay_manifest_audit_ci_smoke.py
```

## QEMU Input Provenance Audit

`qemu_input_provenance_audit.py` checks saved QEMU benchmark artifacts for
prompt-suite, image, and QEMU args-file provenance drift. It recomputes local
prompt-suite hashes when the recorded prompt source is present, verifies
recorded file metadata shape, and can gate live file size/SHA drift without
launching QEMU or touching the TempleOS guest.

Example:

```bash
python3 bench/qemu_input_provenance_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_input_provenance_audit_latest \
  --require-live-inputs \
  --require-file-sha256
```

The tool writes JSON, Markdown, CSV records, CSV findings, and JUnit outputs.
Its smoke gate builds temporary prompt/image/args fixtures and exercises both
passing and drift-detection paths:

```bash
python3 bench/qemu_input_provenance_audit_ci_smoke.py
```

## Airgap Audit

`airgap_audit.py` checks saved benchmark command artifacts for the TempleOS
air-gap policy. It reads existing JSON/JSONL/CSV results, verifies every saved
QEMU command has explicit `-nic none`, rejects legacy `-net none` drift and
network devices/backends, and cross-checks recorded command air-gap telemetry.
It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/airgap_audit.py bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem airgap_audit_latest \
  --min-commands 4
```

The tool writes JSON, Markdown, CSV findings, and JUnit outputs. Its smoke gate
builds temporary passing/failing command artifacts:

```bash
python3 bench/airgap_audit_ci_smoke.py
```

## Dashboard Digest

`dashboard_digest.py` summarizes existing dashboard JSON artifacts into a
single CI-friendly status digest. It is host-side only and does not launch QEMU.

Example:

```bash
python3 bench/dashboard_digest.py \
  bench/dashboards/perf_regression_latest.json \
  bench/dashboards/perf_slo_audit_latest.json \
  bench/dashboards/bench_trend_export_latest.json \
  --output-dir bench/dashboards \
  --output-stem dashboard_digest_latest \
  --min-dashboards 3 \
  --fail-on-missing \
  --fail-on-fail-status
```

The tool writes JSON, Markdown, CSV, and JUnit outputs. Its smoke gate builds
temporary dashboard artifacts and verifies aggregate status/finding counts:

```bash
python3 bench/dashboard_digest_ci_smoke.py
```

## Dashboard Sidecar Audit

`dashboard_sidecar_audit.py` verifies that dashboard JSON artifacts have CSV,
Markdown, and JUnit sidecars for CI upload and review. It accepts both exact
`*_latest.csv` sidecars and metric-specific `*_*_latest.csv` exports. It is
host-side only and does not launch QEMU.

Example:

```bash
python3 bench/dashboard_sidecar_audit.py \
  bench/dashboards/perf_regression_latest.json \
  bench/dashboards/perf_slo_audit_latest.json \
  bench/dashboards/bench_trend_export_latest.json \
  --output-dir bench/dashboards \
  --output-stem dashboard_sidecar_audit_latest \
  --min-dashboards 3
```

The tool writes JSON, Markdown, CSV, and JUnit outputs. Its smoke gate builds
temporary passing and failing dashboard artifacts:

```bash
python3 bench/dashboard_sidecar_audit_ci_smoke.py
```

## Offline Eval Dataset Packer

`dataset_schema_audit.py` validates local eval JSONL before curation or
packing. It normalizes the same HellaSwag-, ARC-, TruthfulQA-, and normalized
row shapes accepted by the packer, reports dataset/split counts, answer and
choice histograms, byte telemetry, duplicate IDs, and optional provenance or
loader-size and answer-label coverage gate findings. It never fetches remote
datasets.

```bash
python3 bench/dataset_schema_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_schema_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_schema_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_schema_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_schema_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_schema_audit_smoke_latest_junit.xml \
  --require-provenance \
  --min-choices 4 \
  --max-choices 4 \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024 \
  --max-record-payload-bytes 8192 \
  --min-answer-labels 1 \
  --min-dataset-split-answer-labels 1 \
  --fail-on-duplicate-ids \
  --fail-on-duplicate-payloads \
  --fail-on-conflicting-payload-answers \
  --fail-on-findings
```

Use `--record-csv` to emit per-record normalized telemetry for loader-bound
checks, including prompt bytes, total/max choice bytes, record payload bytes,
answer index, and stable normalized prompt+choices payload hashes.
Use `--min-answer-labels` and `--min-dataset-split-answer-labels` when a subset
must exercise at least N answer indexes overall or in every dataset/split group.

`dataset_content_hash_audit.py` verifies optional row-level prompt, choices,
and combined input SHA-256 metadata against normalized eval JSONL content. Use
`--require-all-hashes` in CI for promoted curated slices; without it, the audit
still emits canonical hashes for rows that do not yet carry hash metadata.

```bash
python3 bench/dataset_content_hash_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output-dir bench/results/datasets \
  --output-stem dataset_content_hash_audit_latest
python3 bench/dataset_content_hash_audit_ci_smoke.py
```

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
  --require-dataset-split arc-smoke:validation \
  --max-records 3 \
  --balance-answer-index \
  --dedupe-within-split-payloads \
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
Use repeatable `--require-dataset-split DATASET:SPLIT` when a curated subset
must retain at least one selected record for specific eval slices after
filters, caps, deduplication, and sampling.
Use `--dedupe-within-split-payloads` to collapse repeated normalized
dataset/split/prompt/choices/answer rows before caps and sampling. If duplicate
within-split prompt/choices payloads disagree on the answer index, curation
fails instead of silently choosing one label.

`dataset_select.py` is the smaller deterministic selector for already-staged
JSONL rows. It can now write a selected-record CSV sidecar for quick review of
record IDs, answer indexes, byte budgets, source rows, payload hashes, and
stable rank hashes:

```bash
python3 bench/dataset_select.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/smoke_selected.jsonl \
  --manifest bench/results/datasets/smoke_selected.manifest.json \
  --csv bench/results/datasets/smoke_selected.csv \
  --max-records-per-slice 1 \
  --balance-answer \
  --fail-on-findings
```

`dataset_manifest_audit.py` validates curated JSONL provenance manifests before
publishing packed eval artifacts. It checks local source and output digests,
derives actual curated row counts plus dataset/split coverage from JSONL, and
verifies optional `.hceval` pack manifests preserve source digest, record count,
record ID order, and binary digest.

Example:

```bash
python3 bench/dataset_manifest_audit.py \
  --manifest bench/results/datasets/smoke_curated.manifest.json \
  --pack-manifest bench/results/datasets/smoke_curated.hceval.manifest.json \
  --root . \
  --output bench/results/datasets/dataset_manifest_audit_smoke_latest.json \
  --csv bench/results/datasets/dataset_manifest_audit_smoke_latest.csv \
  --markdown bench/results/datasets/dataset_manifest_audit_smoke_latest.md \
  --junit bench/results/datasets/dataset_manifest_audit_smoke_latest_junit.xml \
  --require-pack-manifest \
  --fail-on-findings
```

`dataset_contamination_audit.py` checks mixed local eval suites for
cross-dataset contamination before packing. It normalizes rows through the
packer schema, then flags normalized prompt reuse, prompt+choice payload reuse,
and conflicting answer indexes across dataset families:

```bash
python3 bench/dataset_contamination_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_contamination_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_contamination_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_contamination_audit_smoke_latest.csv \
  --junit bench/results/datasets/dataset_contamination_audit_smoke_latest_junit.xml \
  --fail-on-contamination
```

`dataset_prompt_choice_overlap_audit.py` checks local eval JSONL for prompt
templates that already contain answer or distractor choice text. It normalizes
rows through the packer schema, writes per-record overlap telemetry, and can
fail only answer leaks or any prompt/choice overlap before packing:

```bash
python3 bench/dataset_prompt_choice_overlap_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_prompt_choice_overlap_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_prompt_choice_overlap_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_prompt_choice_overlap_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_prompt_choice_overlap_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_prompt_choice_overlap_audit_smoke_latest_junit.xml \
  --fail-on-answer-overlap
```

`dataset_mix_audit.py` checks that curated local eval suites are not dominated
by one dataset, split, or dataset/split bucket before packing. It writes
aggregate distribution CSVs plus optional per-record telemetry so each bucket
can be traced back to source rows:

```bash
python3 bench/dataset_mix_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_mix_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_mix_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_mix_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_mix_audit_smoke_records_latest.csv \
  --findings-csv bench/results/datasets/dataset_mix_audit_smoke_latest_findings.csv \
  --junit bench/results/datasets/dataset_mix_audit_smoke_latest_junit.xml \
  --min-datasets 3 \
  --max-dataset-pct 34 \
  --max-dataset-split-pct 34
```

`dataset_provenance_balance_audit.py` checks that local eval JSONL rows carry
non-empty provenance/source strings and that no single staged source dominates
the overall subset or a dataset/split bucket. It writes aggregate distribution
CSVs plus optional per-record provenance telemetry for review before packing:

```bash
python3 bench/dataset_provenance_balance_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_provenance_balance_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_provenance_balance_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_provenance_balance_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_provenance_balance_audit_smoke_records_latest.csv \
  --findings-csv bench/results/datasets/dataset_provenance_balance_audit_smoke_latest_findings.csv \
  --junit bench/results/datasets/dataset_provenance_balance_audit_smoke_latest_junit.xml \
  --require-provenance \
  --min-provenance-sources 3 \
  --max-provenance-pct 34
```

`dataset_id_audit.py` checks local eval JSONL record IDs before packing. It
normalizes rows through the packer schema, then gates missing explicit IDs,
overlong IDs, format mismatches, duplicate raw IDs, and duplicate
dataset/split-scoped IDs:

```bash
python3 bench/dataset_id_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_id_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_id_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_id_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_id_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_id_audit_smoke_latest_junit.xml \
  --require-explicit-id \
  --max-record-id-bytes 64 \
  --id-pattern '[a-z0-9-]+' \
  --fail-duplicate-record-ids \
  --fail-duplicate-dataset-split-record-ids \
  --fail-on-findings
```

`dataset_text_audit.py` checks normalized prompt and choice text for
loader-hostile content before curation or packing. It gates blank text,
disallowed C0 control characters, Unicode replacement markers, prompt/choice
byte budgets, line-byte budgets, and raw choice label prefixes such as `A.` or
`1)`, then writes optional per-field telemetry:

```bash
python3 bench/dataset_text_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_text_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_text_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_text_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_text_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_text_audit_smoke_latest_junit.xml \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024 \
  --max-line-bytes 4096 \
  --fail-on-control-chars \
  --fail-on-replacement-chars \
  --fail-on-blank-text \
  --fail-on-choice-label-prefixes \
  --fail-on-findings
```

`dataset_choice_length_audit.py` checks individual multiple-choice records for
answer-length cue artifacts before packing. It reports choice byte spans,
answer-to-distractor ratios, unique longest/shortest answers, and writes
optional per-record length telemetry:

```bash
python3 bench/dataset_choice_length_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_choice_length_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_choice_length_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_choice_length_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_choice_length_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_choice_length_audit_smoke_latest_junit.xml \
  --max-choice-byte-span 128 \
  --max-answer-delta-bytes 128 \
  --max-answer-to-mean-other-ratio 8.0 \
  --min-answer-to-mean-other-ratio 0.125 \
  --fail-on-findings
```

`dataset_choice_similarity_audit.py` checks individual multiple-choice records
for duplicate or near-duplicate choices after case/spacing/punctuation
normalization. It writes per-record and per-choice-pair telemetry so ambiguous
curated examples can be rejected before packing:

```bash
python3 bench/dataset_choice_similarity_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_choice_similarity_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_choice_similarity_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_choice_similarity_audit_smoke_latest.csv \
  --pair-csv bench/results/datasets/dataset_choice_similarity_audit_smoke_pairs_latest.csv \
  --findings-csv bench/results/datasets/dataset_choice_similarity_audit_smoke_latest_findings.csv \
  --junit bench/results/datasets/dataset_choice_similarity_audit_smoke_latest_junit.xml \
  --min-unique-choices 4 \
  --max-pair-similarity 0.95 \
  --fail-duplicate-normalized
```

`dataset_answer_bias_audit.py` checks curated multiple-choice subsets for
answer-length artifacts before packing. It reports whether correct answers are
overrepresented as the longest or shortest option overall and per dataset/split,
then writes optional per-record answer/distractor byte telemetry:

```bash
python3 bench/dataset_answer_bias_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_answer_bias_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_answer_bias_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_answer_bias_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_answer_bias_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_answer_bias_audit_smoke_latest_junit.xml \
  --max-answer-longest-pct 100 \
  --max-answer-shortest-pct 100 \
  --min-mean-answer-distractor-ratio 0.01 \
  --max-mean-answer-distractor-ratio 100 \
  --check-dataset-splits \
  --fail-on-findings
```

`dataset_answer_position_audit.py` checks curated multiple-choice subsets for
correct-answer position concentration. It reports answer-index histograms,
dominant answer positions, and distinct answer-position coverage overall and
per dataset/split:

```bash
python3 bench/dataset_answer_position_audit.py \
  bench/datasets/samples/smoke_eval.jsonl \
  --output-dir bench/results/datasets \
  --output-stem dataset_answer_position_audit_smoke_latest \
  --min-records 3 \
  --max-dominant-answer-pct 100
```

`dataset_pack.py` converts local JSONL multiple-choice evaluation rows into a
deterministic HolyC-loadable binary plus a provenance manifest. It accepts a
normalized schema as well as HellaSwag-, ARC-, and TruthfulQA-shaped rows. It is
offline-only; place source data on disk first and document provenance.
Manifests include UTF-8 prompt/choice/record byte statistics, choice-count
histograms/stats, and optional size gates can fail packing before writing
oversized artifacts. Manifests also include `record_spans` with each record's
binary offset, length, and payload bytes for loader-bound audits without
changing the `.hceval` bytes, plus binary-recoverable `record_fingerprints` for
prompt, choices, prompt+choices input, and answer payload drift checks. The
`binary_layout` section summarizes fixed header, metadata, record header,
payload, choice length-prefix, body, and total binary byte counts for loader
buffer planning.
For HellaSwag-shaped rows, `ctx` is preferred when present; locally staged rows
that only carry `ctx_a`/`ctx_b` are normalized by joining those context parts
before packing.

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
It can re-apply the same byte-size gates to already packed binaries and verifies
manifest choice-count telemetry, `record_spans`, and `binary_layout` when
present. Add `--csv` to emit a flat record-span table for loader offset diffs.

```bash
python3 bench/hceval_inspect.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --output bench/results/datasets/smoke_eval.inspect.json \
  --markdown bench/results/datasets/smoke_eval.inspect.md \
  --csv bench/results/datasets/smoke_eval.inspect.csv \
  --fingerprints-csv bench/results/datasets/smoke_eval.inspect.fingerprints.csv \
  --junit bench/results/datasets/smoke_eval.inspect.junit.xml \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024
```

`hceval_metadata_audit.py` checks packed `.hceval` metadata without launching
QEMU. It verifies the canonical compact metadata bytes produced by
`dataset_pack.py`, expected metadata keys, non-empty dataset/split fields,
format/version values, and header/metadata/parsed record-count consistency:

```bash
python3 bench/hceval_metadata_audit.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --output bench/results/datasets/hceval_metadata_audit_latest.json \
  --markdown bench/results/datasets/hceval_metadata_audit_latest.md \
  --csv bench/results/datasets/hceval_metadata_audit_latest.csv \
  --findings-csv bench/results/datasets/hceval_metadata_audit_latest_findings.csv \
  --junit bench/results/datasets/hceval_metadata_audit_latest_junit.xml \
  --fail-on-findings
```

`hceval_budget_audit.py` scans existing `.hceval` artifacts and gates suite-level
binary layout budgets without unpacking data through QEMU. It can require
companion manifests, enforce minimum/maximum record counts, and cap binary,
metadata, body, record-header, choice-length-prefix, aggregate prompt/choice/
provenance arena, and per-record payload bytes across a directory:

```bash
python3 bench/hceval_budget_audit.py bench/results/datasets \
  --output-dir bench/results/datasets \
  --output-stem hceval_budget_audit_latest \
  --require-manifest \
  --max-binary-bytes 1048576 \
  --max-metadata-bytes 4096 \
  --max-record-header-bytes 65536 \
  --max-choice-length-prefix-bytes 65536 \
  --max-total-prompt-bytes 262144 \
  --max-total-choice-bytes 262144 \
  --max-total-provenance-bytes 65536 \
  --max-record-payload-bytes 8192
```

`hceval_choice_semantics_audit.py` scans packed `.hceval` files for semantic
choice hazards that can survive structural packing: duplicate normalized
choices, answer aliases, and candidate choice text already present in the
prompt. It stays host-side and emits JSON, Markdown, CSV, findings CSV, and
JUnit artifacts:

```bash
python3 bench/hceval_choice_semantics_audit.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --output bench/results/datasets/hceval_choice_semantics_audit_smoke_latest.json \
  --markdown bench/results/datasets/hceval_choice_semantics_audit_smoke_latest.md \
  --csv bench/results/datasets/hceval_choice_semantics_audit_smoke_latest.csv \
  --findings-csv bench/results/datasets/hceval_choice_semantics_audit_smoke_latest_findings.csv \
  --junit bench/results/datasets/hceval_choice_semantics_audit_smoke_latest_junit.xml \
  --fail-on-findings
```

`hceval_record_identity_audit.py` scans packed `.hceval` files for record ID
collisions and duplicate prompt/choice/answer payloads after binary packing, so
curated subsets cannot silently carry repeated eval rows into HolyC or
llama.cpp comparisons:

```bash
python3 bench/hceval_record_identity_audit.py \
  --input bench/results/datasets/smoke_curated.hceval \
  --output bench/results/datasets/hceval_record_identity_audit_smoke_latest.json \
  --markdown bench/results/datasets/hceval_record_identity_audit_smoke_latest.md \
  --csv bench/results/datasets/hceval_record_identity_audit_smoke_latest.csv \
  --artifacts-csv bench/results/datasets/hceval_record_identity_audit_smoke_artifacts_latest.csv \
  --findings-csv bench/results/datasets/hceval_record_identity_audit_smoke_findings_latest.csv \
  --junit bench/results/datasets/hceval_record_identity_audit_smoke_latest_junit.xml \
  --fail-on-findings
```

`hceval_export.py` converts a packed `.hceval` file back into normalized JSONL
for `eval_compare.py` gold inputs and `eval_input_audit.py` hash-parity checks.
Pass the pack manifest when exporting mixed-dataset packs so per-record
dataset/split provenance is restored. Rows include `prompt_sha256`,
`choices_sha256`, and `input_sha256` by default so HolyC and llama.cpp
prediction files can prove they scored the same prompt and choice payloads.

```bash
python3 bench/hceval_export.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --output bench/results/datasets/smoke_eval.export.jsonl \
  --manifest bench/results/datasets/smoke_eval.export.manifest.json \
  --pack-manifest bench/results/datasets/smoke_eval.manifest.json
```

`hceval_export_roundtrip_audit.py` checks that a packed `.hceval` artifact can
be exported through the normalized JSONL path and repacked to the same source
digest, binary digest, record fingerprints, and binary layout. Pass the pack
manifest for mixed-dataset packs so per-record dataset/split metadata is
restored before the repack.

```bash
python3 bench/hceval_export_roundtrip_audit.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --pack-manifest bench/results/datasets/smoke_eval.manifest.json \
  --output bench/results/datasets/hceval_export_roundtrip_audit_smoke_latest.json \
  --markdown bench/results/datasets/hceval_export_roundtrip_audit_smoke_latest.md \
  --csv bench/results/datasets/hceval_export_roundtrip_audit_smoke_latest.csv \
  --junit bench/results/datasets/hceval_export_roundtrip_audit_smoke_latest_junit.xml \
  --fail-on-findings
```

`dataset_fingerprint.py` writes stable prompt, choice, input, answer, and full
payload hashes for local eval JSONL rows. `dataset_fingerprint_diff.py` compares
two fingerprint reports and emits JSON, Markdown, CSV, and JUnit artifacts so CI
can flag unreviewed eval-set drift before packed `.hceval` files are promoted.
Use the opt-in gates to reject added/removed rows, prompt/choice changes, answer
changes, metadata changes, or any fingerprint drift:

```bash
python3 bench/dataset_fingerprint.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_fingerprint_smoke_latest.json \
  --jsonl bench/results/datasets/dataset_fingerprint_smoke_latest.jsonl \
  --csv bench/results/datasets/dataset_fingerprint_smoke_latest.csv \
  --markdown bench/results/datasets/dataset_fingerprint_smoke_latest.md \
  --junit bench/results/datasets/dataset_fingerprint_smoke_latest_junit.xml \
  --fail-on-duplicate-ids \
  --fail-on-conflicting-input-answers \
  --fail-on-findings

python3 bench/dataset_fingerprint_diff.py \
  --baseline bench/results/datasets/dataset_fingerprint_smoke_latest.json \
  --candidate bench/results/datasets/dataset_fingerprint_smoke_latest.json \
  --output bench/results/datasets/dataset_fingerprint_diff_smoke_latest.json \
  --csv bench/results/datasets/dataset_fingerprint_diff_smoke_latest.csv \
  --findings-csv bench/results/datasets/dataset_fingerprint_diff_smoke_findings_latest.csv \
  --markdown bench/results/datasets/dataset_fingerprint_diff_smoke_latest.md \
  --junit bench/results/datasets/dataset_fingerprint_diff_smoke_latest_junit.xml \
  --fail-on-any-change \
  --fail-on-findings

python3 bench/dataset_fingerprint_diff_ci_smoke.py
```

`dataset_slice_manifest.py` emits deterministic dataset/split coverage
manifests for local eval JSONL files before curation, packing, or comparison.
It records per-slice answer histograms, byte totals, stable slice hashes, and
optional gates for required slices or minimum records per slice:

```bash
python3 bench/dataset_slice_manifest.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_slice_manifest_smoke_latest.json \
  --csv bench/results/datasets/dataset_slice_manifest_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_slice_manifest_smoke_records_latest.csv \
  --markdown bench/results/datasets/dataset_slice_manifest_smoke_latest.md \
  --junit bench/results/datasets/dataset_slice_manifest_smoke_latest_junit.xml \
  --require-slice arc-smoke:validation \
  --require-slice hellaswag-smoke:validation \
  --require-slice truthfulqa-smoke:validation \
  --min-total-slices 3 \
  --min-records-per-slice 1 \
  --fail-on-findings

python3 bench/dataset_slice_manifest_ci_smoke.py
```

`dataset_index.py` scans curated manifests, packed `.hceval` manifests, and
inspection reports, verifies local hashes/provenance fields where possible, and
writes JSON/Markdown/CSV/JUnit XML rollups:
Relative artifact paths are resolved from the current working directory first,
then from the manifest/report directory so archived dataset bundles remain
self-validating after they are moved. Use `--require-dataset-split DATASET:SPLIT`
with `--fail-on-dataset-split-coverage` when CI should prove promoted dataset
artifacts include required eval slices, such as `arc:validation`, before they
are consumed by HolyC-vs-llama comparisons.

```bash
python3 bench/dataset_index.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --require-dataset-split smoke-eval:validation \
  --fail-on-dataset-split-coverage \
  --fail-on-findings
```

The focused smoke gate checks passing artifact-type and dataset/split coverage,
then verifies both missing-artifact and missing-slice failures:

```bash
python3 bench/dataset_index_ci_smoke.py
```

`dataset_ci_smoke.py` is a stdlib-only CI gate for the offline dataset pipeline.
It curates the synthetic sample, packs and inspects the `.hceval` binary, runs
the split-leakage audit, indexes the generated artifacts, and checks that a
known leaky fixture is rejected:

```bash
python3 bench/dataset_ci_smoke.py
```

`bench_artifact_manifest.py` builds latest/history manifests for benchmark
artifacts, including memory-per-token and serial-output telemetry when present,
and can gate air-gap status, telemetry, commit metadata, command hashes,
artifact freshness, per-key history coverage, minimum measured runs/tokens,
matching dry-run launch plans, duplicate key/timestamp artifacts, and comparable host/QEMU environment stability. Use
`--fail-on-environment-drift` when CI should reject throughput comparisons whose
profile/model/quantization/prompt-suite/command/launch-plan keys span multiple
environment fingerprints. `bench_artifact_manifest_ci_smoke.py` is a
stdlib-only CI gate for this path; it builds synthetic QEMU benchmark reports,
verifies the latest/history JSON, Markdown, CSV, and JUnit outputs, and checks
that stale artifacts and a NIC-enabled QEMU command are rejected:

```bash
python3 bench/bench_artifact_manifest_ci_smoke.py
```

`qemu_result_retention_audit.py` checks QEMU prompt benchmark latest JSON
aliases against their immutable timestamped siblings. It derives the expected
history filename from each artifact's `generated_at`, verifies the timestamped
file exists, and checks that the latest alias has the same SHA256 as that
history artifact. It writes JSON, Markdown, record CSV, findings CSV, and JUnit
outputs. It is host-side only and does not launch QEMU:

```bash
python3 bench/qemu_result_retention_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_result_retention_audit_latest \
  --min-latest 2 \
  --min-history-per-latest 1
```

The stdlib-only smoke gate exercises passing, missing-history, and
latest/history mismatch cases:

```bash
python3 bench/qemu_result_retention_audit_ci_smoke.py
```

`qemu_timestamp_audit.py` checks saved QEMU prompt benchmark JSON artifacts for
timestamp hygiene before dashboards or regression gates consume them. It
validates artifact `generated_at` parseability, optional timestamped filename
stamps, row timestamp parseability, row timestamp monotonicity, future skew, and
row timestamps that appear too far before or after artifact generation. It is
host-side only and does not launch QEMU:

```bash
python3 bench/qemu_timestamp_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_timestamp_audit_latest \
  --max-row-before-generated-at-seconds 3600 \
  --require-rows
```

The stdlib-only smoke gate exercises passing chronology plus filename-stamp,
row-regression, stale-row, and row-skew failure cases:

```bash
python3 bench/qemu_timestamp_audit_ci_smoke.py
```

`bench_result_index.py` also records measured-run dry-run coverage. A measured
`qemu_prompt` artifact is covered when the index finds a `qemu_prompt_dry_run`
artifact with the same profile, model, quantization, prompt-suite hash, command
hash, launch-plan hash, and environment hash. Use `--fail-on-missing-dry-run`
when CI should require a reviewed dry-run launch plan beside every measured
QEMU prompt benchmark artifact:

```bash
python3 bench/bench_result_index.py \
  --input bench/results \
  --output-dir bench/results \
  --fail-on-airgap \
  --fail-on-command-hash-metadata \
  --fail-on-missing-dry-run
```

`bench_trend_export.py` turns existing benchmark JSON artifacts into compact
dashboard trend files without launching QEMU. It groups comparable
profile/model/quantization/prompt-suite points, records latest-vs-previous
throughput and memory deltas, a machine-readable JSON `summary` block for
dashboard digests, and writes JSON, Markdown, CSV, point-history CSV,
recent-window CSV, drift CSV, findings CSV, and JUnit XML under
`bench/dashboards/`. Optional threshold gates fail CI when latest guest tok/s or
host wall-clock tok/s falls below an absolute floor, when latest guest tok/s,
host wall-clock tok/s, or max-memory trends regress beyond the configured
percentage, when a comparable trend key has too little retained history for a
regression decision, or when command, launch-plan, or host/QEMU environment
hashes drift inside a comparable trend key. Use `--window-points` to tune the
per-key recent-window rows written to `bench_trend_export_windows_latest.csv`;
window gates compare the first retained point in that recent window against the
latest point. Structured gate findings are also written to
`bench_trend_export_findings_latest.csv` for dashboard filtering by gate, key,
metric, value, and threshold.
Recent-window rows also include guest tok/s, wall tok/s, and max-memory
coefficient-of-variation telemetry. Use `--max-window-tok-cv-pct`,
`--max-window-wall-tok-cv-pct`, and `--max-window-memory-cv-pct` when CI should
reject unstable retained windows even if the latest-vs-previous point does not
cross a regression threshold.

```bash
python3 bench/bench_trend_export.py \
  --input bench/results \
  --output-dir bench/dashboards \
  --max-points-per-key 25 \
  --window-points 5 \
  --fail-on-empty \
  --fail-on-airgap \
  --fail-on-telemetry \
  --min-points-per-key 2 \
  --min-latest-tok-per-s 25 \
  --min-latest-wall-tok-per-s 20 \
  --fail-on-tok-regression-pct 5 \
  --fail-on-wall-tok-regression-pct 5 \
  --fail-on-memory-growth-pct 10 \
  --fail-on-window-tok-regression-pct 5 \
  --fail-on-window-wall-tok-regression-pct 5 \
  --fail-on-window-memory-growth-pct 10 \
  --max-window-tok-cv-pct 8 \
  --max-window-wall-tok-cv-pct 8 \
  --max-window-memory-cv-pct 8
python3 bench/bench_trend_export_ci_smoke.py
```

Add `--fail-on-command-drift`, `--fail-on-launch-plan-drift`, and
`--fail-on-environment-drift` when the retained trend window should only compare
runs with identical launch and host/QEMU fingerprints.

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
python3 bench/dataset_leak_audit_ci_smoke.py
```

`dataset_provenance_audit.py` checks curated JSONL manifests for source/license
metadata, source and output hashes, selected IDs, dataset/split counts, and
answer histograms. Reports include provenance/source contribution counts plus
overall, per-dataset, per-split, and per-dataset/split answer-majority telemetry;
the sidecar `dataset_provenance_audit_records_latest.csv` adds per-record
provenance percentages, byte budgets, and stable prompt+choices hashes for
review before packing.
`--max-majority-answer-pct`,
`--max-provenance-pct`, `--max-dataset-majority-answer-pct`, and
`--max-split-majority-answer-pct`,
`--max-dataset-split-majority-answer-pct` can fail CI when curated subsets are
label skewed or dominated by one local source/provenance string. Use repeatable
`--allow-license` and `--deny-license` gates to enforce an exact local license
or usage-note policy before promoting a curated subset. Source URL policy can
also be pinned offline with repeatable `--allow-source-url-scheme`,
`--deny-source-url-scheme`, `--allow-source-url-host`, and
`--deny-source-url-host`. Path policy can be pinned with repeatable
`--allow-source-url-path-prefix` and `--deny-source-url-path-prefix`; the audit
parses manifest strings only and never fetches them.

```bash
python3 bench/dataset_provenance_audit.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --max-provenance-pct 80 \
  --max-dataset-majority-answer-pct 80 \
  --max-split-majority-answer-pct 80 \
  --max-dataset-split-majority-answer-pct 90 \
  --allow-license synthetic-smoke \
  --allow-source-url-scheme https \
  --allow-source-url-path-prefix /datasets/ \
  --fail-on-findings
```

The provenance audit smoke creates a local curated fixture, verifies the
JSON/Markdown/CSV/JUnit outputs, and checks license plus source-url policy
failure paths without fetching any remote data:

```bash
python3 bench/dataset_provenance_audit_ci_smoke.py
```

## Offline Eval Workload Estimate

`eval_workload_estimate.py` projects local multiple-choice eval JSONL files
before QEMU runs. It estimates prompt, choice, scored-token, launch, and wall
time budgets by dataset/split and per record, with row-level gates to catch one
oversized prompt or choice set before it dominates an air-gapped benchmark job.

```bash
python3 bench/eval_workload_estimate.py bench/fixtures/eval_workload_estimate/smoke.jsonl \
  --output-dir bench/results \
  --output-stem eval_workload_estimate_latest \
  --tok-per-s 100 \
  --qemu-launch-overhead-s 0.25 \
  --max-record-scored-tokens 4096 \
  --max-record-launches 8 \
  --max-choices-per-record 8 \
  --max-scored-tokens 65536
```

CI smoke coverage for pass/fail budget paths:

```bash
python3 bench/eval_workload_estimate_ci_smoke.py
```

## Offline Eval Comparator

`eval_compare.py` compares local HolyC and llama.cpp multiple-choice predictions
against the same gold JSONL and writes JSON, Markdown, per-record CSV,
per-dataset/split breakdown CSV, confusion-matrix CSV, calibration-bin CSV, and
score-NLL CSV, score-rank CSV, score-tie CSV, engine-disagreement CSV, and JUnit XML reports.
Optional quality gates can fail CI when HolyC accuracy, engine agreement,
accuracy delta versus llama.cpp, or a paired exact McNemar loss falls outside
configured bounds.
Prediction score vectors are treated as choice-aligned logits/logprobs: every
score must be finite and the score count must match the gold choice count.
`eval_input_audit.py` also records per-engine prediction histograms and can fail
early with `--max-majority-prediction-pct` when either engine collapses onto one
answer index before quality metrics are computed.
It also records per-engine score-vector coverage and can fail early with
`--min-score-coverage-pct` when calibration/ranking evals require logprob-style
choice scores from both engines.
For scored prediction rows, it also records predicted-vs-runner-up score
margins and can fail early with `--min-top-score-margin` when either engine
emits under-separated best choices before the comparator runs.
When score vectors are present, reports include per-row confidence/margin plus
score coverage, mean confidence, Brier score, expected calibration error, mean
gold-answer negative log likelihood, and choice-set perplexity.
Reports also rank the gold answer within each score vector and summarize top-1,
top-2, top-3, mean gold rank, and mean reciprocal rank for each engine.
Reports also count top-score ties in scored rows, and
`--max-holyc-score-tie-rate` can fail CI when HolyC emits ambiguous tied best
choices too often.
Reports also include paired correctness counts and an exact two-sided McNemar
binomial p-value so HolyC-vs-llama quality deltas can be interpreted as paired
eval outcomes on the same records. `--max-mcnemar-loss-p` optionally fails CI
only when llama.cpp has more discordant wins than HolyC and the paired p-value
is at or below the configured threshold.
`eval_compare_ci_smoke.py` is a stdlib-only smoke gate for the comparator; it
checks report artifacts, paired metrics, CSV/JUnit outputs, and an intentional
score-coverage regression failure without launching QEMU.

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
  --max-mcnemar-loss-p 0.05 \
  --min-holyc-nll-coverage 0.95 \
  --max-holyc-nll-delta 0.05 \
  --max-holyc-score-tie-rate 0.01 \
  --fail-on-regression

python3 bench/eval_compare_ci_smoke.py
```

`eval_length_bucket_report.py` groups the same local gold/prediction inputs by
normalized prompt byte length. It reports per-bucket HolyC accuracy, llama.cpp
accuracy, agreement, paired outcome counts, CSV/Markdown/JUnit sidecars, and
optional gates for sparse buckets or HolyC accuracy loss on long-context slices.

```bash
python3 bench/eval_length_bucket_report.py \
  --gold bench/datasets/samples/smoke_eval.jsonl \
  --holyc bench/eval/samples/holyc_smoke_scored_predictions.jsonl \
  --llama bench/eval/samples/llama_smoke_scored_predictions.jsonl \
  --dataset smoke-eval \
  --split validation \
  --bucket-edges 128,256,512 \
  --min-records-per-bucket 1 \
  --min-holyc-accuracy 0.95 \
  --max-holyc-accuracy-loss 0.05

python3 bench/eval_length_bucket_report_ci_smoke.py
```

`eval_margin_audit.py` gates existing `eval_compare.py` JSON reports for scored
multiple-choice margin health. It checks score coverage, scored-row counts,
mean/p10/min top-1 margins, low-margin rates, and HolyC margin loss versus
llama.cpp. It is host-side only and reads saved artifacts.

```bash
python3 bench/eval_margin_audit.py \
  bench/results/eval_compare_smoke_latest.json \
  --output-dir bench/results \
  --output-stem eval_margin_audit_smoke_latest \
  --min-score-coverage 0.95 \
  --min-mean-margin 0.20 \
  --min-p10-margin 0.05 \
  --max-holyc-mean-margin-loss 0.05 \
  --fail-on-findings

python3 bench/eval_margin_audit_ci_smoke.py
```

`eval_suite_summary.py` aggregates existing `eval_compare.py` JSON reports into
a suite-level CI artifact. It is useful when multiple datasets, splits,
quantizations, or model builds are compared separately but CI needs one summary
for total record coverage, weighted HolyC accuracy, weighted engine agreement,
failed reports, per-report regressions, and required dataset/split/model/quant
coverage.

```bash
python3 bench/eval_suite_summary.py \
  bench/results/eval_compare_smoke_latest.json \
  --output-dir bench/results \
  --output-stem eval_suite_summary_smoke_latest \
  --min-reports 1 \
  --min-records 3 \
  --min-holyc-accuracy 0.95 \
  --min-agreement 0.95 \
  --require-dataset smoke-eval \
  --require-split validation \
  --require-model synthetic-smoke \
  --require-quantization Q4_0 \
  --fail-on-failed-reports \
  --fail-on-regressions

python3 bench/eval_suite_summary_ci_smoke.py
```

`eval_result_index.py` indexes existing `eval_compare.py` reports and
`eval_suite_summary.py` outputs into one CI/dashboard rollup. It extracts
artifact type, status, dataset/split/model/quant metadata, record counts,
weighted HolyC accuracy, HolyC-vs-llama agreement, regressions, and result
hashes without rerunning either engine.

```bash
python3 bench/eval_result_index.py \
  bench/results \
  --output-dir bench/results \
  --output-stem eval_result_index_latest \
  --min-artifacts 1 \
  --min-records 3 \
  --require-dataset smoke-eval \
  --require-quantization Q8_0 \
  --min-holyc-accuracy 0.95 \
  --min-agreement 0.95 \
  --fail-on-failed \
  --fail-on-regressions

python3 bench/eval_result_index_ci_smoke.py
```

`eval_report_audit.py` validates existing `eval_compare.py` JSON reports for
internal consistency without rerunning either engine. It recomputes core
summary counters from rows, checks metric bounds, rejects pass-status reports
that still contain regression entries, and writes JSON, Markdown, report CSV,
findings CSV, and JUnit XML outputs:

```bash
python3 bench/eval_report_audit.py \
  bench/results/eval_compare_smoke_latest.json \
  --output bench/results/eval_report_audit_smoke_latest.json \
  --markdown bench/results/eval_report_audit_smoke_latest.md \
  --csv bench/results/eval_report_audit_smoke_latest.csv \
  --findings-csv bench/results/eval_report_audit_smoke_findings_latest.csv \
  --junit bench/results/eval_report_audit_smoke_latest_junit.xml \
  --fail-on-findings

python3 bench/eval_report_audit_ci_smoke.py
```

`eval_hash_audit.py` validates `eval_compare.py` report fingerprints for
reproducible apples-to-apples eval runs. It checks canonical SHA-256 formatting
for gold/HolyC/llama artifacts, can compare `gold_sha256` against a local gold
dataset file, reports identical HolyC/llama prediction hashes as a nonblocking
warning by default, and writes JSON, Markdown, CSV, findings CSV, and JUnit outputs:

```bash
python3 bench/eval_hash_audit.py \
  bench/results/eval_compare_smoke_latest.json \
  --gold-path bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/eval_hash_audit_smoke_latest.json \
  --markdown bench/results/eval_hash_audit_smoke_latest.md \
  --csv bench/results/eval_hash_audit_smoke_latest.csv \
  --findings-csv bench/results/eval_hash_audit_smoke_latest_findings.csv \
  --junit bench/results/eval_hash_audit_smoke_latest_junit.xml \
  --fail-on-findings

python3 bench/eval_hash_audit_ci_smoke.py
```

`eval_slice_coverage_audit.py` gates existing `eval_compare.py` JSON artifacts
for required dataset/split coverage without rerunning either engine. It writes
JSON, Markdown, slice CSV, findings CSV, and JUnit XML reports:

```bash
python3 bench/eval_slice_coverage_audit.py \
  bench/results/eval_compare_smoke_latest.json \
  --output-dir bench/results \
  --output-stem eval_slice_coverage_audit_smoke_latest \
  --require-slice arc-smoke:validation \
  --require-slice hellaswag-smoke:validation \
  --require-slice truthfulqa-smoke:validation \
  --min-slices 3 \
  --min-records-per-slice 1 \
  --min-slice-holyc-accuracy 0.95 \
  --min-slice-agreement 0.95 \
  --fail-on-failed-reports \
  --fail-on-regressions

python3 bench/eval_slice_coverage_audit_ci_smoke.py
```

`eval_disagreement_audit.py` gates existing `eval_compare.py` JSON artifacts
for aggregate and dataset/split HolyC-vs-llama disagreement rates without
rerunning either engine. It writes JSON, Markdown, scope CSV, findings CSV, and
JUnit XML reports:

```bash
python3 bench/eval_disagreement_audit.py \
  bench/results/eval_compare_smoke_latest.json \
  --output bench/results/eval_disagreement_audit_smoke_latest.json \
  --markdown bench/results/eval_disagreement_audit_smoke_latest.md \
  --csv bench/results/eval_disagreement_audit_smoke_latest.csv \
  --findings-csv bench/results/eval_disagreement_audit_smoke_findings_latest.csv \
  --junit bench/results/eval_disagreement_audit_smoke_junit_latest.xml \
  --min-records 3 \
  --max-disagreement-pct 0 \
  --max-dataset-split-disagreement-pct 0 \
  --fail-on-findings

python3 bench/eval_disagreement_audit_ci_smoke.py
```

`eval_outcome_audit.py` gates existing `eval_compare.py` JSON artifacts by
paired correctness buckets, including both-correct, HolyC-only-correct,
llama-only-correct, and both-wrong rows. It is useful for catching asymmetric
llama.cpp wins even when aggregate accuracy still passes. It writes JSON,
Markdown, scope CSV, findings CSV, and JUnit XML reports:

```bash
python3 bench/eval_outcome_audit.py \
  bench/results/eval_compare_smoke_latest.json \
  --output bench/results/eval_outcome_audit_smoke_latest.json \
  --markdown bench/results/eval_outcome_audit_smoke_latest.md \
  --csv bench/results/eval_outcome_audit_smoke_latest.csv \
  --findings-csv bench/results/eval_outcome_audit_smoke_findings_latest.csv \
  --junit bench/results/eval_outcome_audit_smoke_junit_latest.xml \
  --min-records 3 \
  --max-llama-only-correct-pct 0 \
  --max-dataset-split-llama-only-correct-pct 0 \
  --max-both-wrong-pct 0 \
  --fail-on-findings

python3 bench/eval_outcome_audit_ci_smoke.py
```

`perplexity_compare.py` also supports opt-in quality gates for full-token
logprob/perplexity comparisons. Use `--max-p95-record-nll-delta` when CI should
catch the P95 positive HolyC-minus-llama per-record NLL tail while ignoring
records where HolyC improves over llama.cpp; use
`--max-p95-abs-record-nll-delta` when any large per-record divergence should
fail. Use `--min-dataset-split-record-count` and
`--min-dataset-split-token-count` when every dataset/split breakdown row needs
its own minimum coverage gate.

## QEMU Prompt Benchmark

`prompt_audit.py` validates benchmark prompt files before a guest run. It
checks prompt ID uniqueness, duplicate prompt text, byte/line stats, optional
minimum prompt count, maximum prompt byte limits, and an optional pinned suite
hash, then writes a stable suite hash so benchmark artifacts can name the exact
prompt set they used. It can also write CSV and JUnit XML artifacts so CI can
upload prompt stats and fail directly on prompt-suite errors.

Example:

```bash
python3 bench/prompt_audit.py \
  --prompts bench/prompts/smoke.jsonl \
  --output bench/results/prompt_audit_smoke_latest.json \
  --markdown bench/results/prompt_audit_smoke_latest.md \
  --csv bench/results/prompt_audit_smoke_latest.csv \
  --junit bench/results/prompt_audit_smoke_latest_junit.xml \
  --min-prompts 2 \
  --max-prompt-bytes 1024 \
  --expect-suite-sha256 <pinned-suite-sha256>
```

The prompt audit smoke covers both the pinned-suite pass path and the drift
failure path:

```bash
python3 bench/prompt_audit_ci_smoke.py
```

`qemu_prompt_bench.py` launches an air-gapped QEMU guest once per prompt, captures
serial output, extracts token timing records, and writes normalized JSON to
`bench/results/`. The runner always injects `-nic none` and rejects conflicting
network flags such as `-netdev` or virtual NIC devices, including legacy QEMU
NIC models such as e1000, ne2k, pcnet, rtl8139, usb-net, virtio-net, and vmxnet.
Dry-run and measured JSON/Markdown reports include
`artifact_schema_version` so downstream dashboards and audits can distinguish
benchmark format changes from throughput changes.
It also rejects socket-style QEMU endpoints such as `-chardev socket`, TCP/UDP
serial or monitor transports, forwarded host/guest sockets, and remote VNC
display sockets so benchmark launches remain fully air-gapped.
Extra QEMU options can be passed one token at a time with `--qemu-arg`, after
`--`, or from local `--qemu-args-file` files. Argument files are offline-only:
`.json` files must contain a string array, while other files use shell-style
tokenization with `#` comments. Loaded file tokens go through the same air-gap
network rejection before any dry run or guest launch.

Use `--warmup N` to launch each prompt before measurement without mixing those
runs into throughput dashboards, and `--repeat N` to run every prompt multiple
times. Use `--max-launches N` to fail before booting QEMU when
`prompts * (warmup + repeat)` would exceed the expected launch budget, and
`--min-prompt-count N` to fail before booting QEMU when a benchmark suite has
too few prompts for a comparable run. Reports
include separate warmup records, raw measured per-run records, an overall suite
summary, per-phase warmup/measured/all summaries, and per-prompt medians,
OK/failed/timeout/nonzero-exit counts, min/max tok/s, P05/P95 tok/s, and
P05-to-P95 spread percentages in JSON and Markdown. Suite, phase, and prompt
summaries also expose exit-class counts for ok, timeout, launch-error, and
guest nonzero exits, making launch failures distinguishable without scanning
raw launch rows. The suite summary includes
measured prompt count, run count, total
prompt bytes launched, total tokens, total elapsed time,
OK/failed/timeout/nonzero-exit counts, P05/median/P95 tok/s, P05/median/P95
wall tok/s, tok/s standard deviation, coefficient of variation, P05-to-P95
spread percentage, interquartile tok/s spread percentage, wall tok/s
interquartile and P05-to-P95 spread percentages, and max memory.
Per-run and per-prompt reports include UTF-8 prompt byte counts so
benchmark changes can be separated from prompt-suite size drift. Per-run
records also include optional guest-reported prompt SHA256 telemetry from
`prompt_sha256`, `input_sha256`, `prompt_hash`, or `input_hash`, plus a
host-side match flag so CI can prove the guest benchmarked the exact prompt
sent by the host. They also record optional guest-reported prompt byte counts
from `prompt_bytes`, `input_bytes`, `prompt_size_bytes`, or `input_size_bytes`
with a host-side match flag, giving CI a cheap independent check that the guest
consumed the same input length. Suite, phase, and prompt summaries include
guest prompt hash/byte record, match, and mismatch counts so prompt-integrity
coverage is visible without scanning every launch row. Per-run records also
include stdout/stderr byte counts plus combined serial byte and line counts.
Suite, phase, and prompt summaries roll those up so verbose guest logging can be gated
separately from decode throughput. Optional first-token latency telemetry is
normalized from `ttft_us`, `time_to_first_token_us`, `first_token_us`, and their
`_ms` or `_s` variants into `ttft_us`; suite and per-prompt reports include
median and P95 TTFT when present. The runner also writes a deterministic
prompt-suite SHA256 matching `prompt_audit.py`, plus
`qemu_prompt_bench_latest.csv` with one row per measured run for CI artifact
upload, spreadsheets, and simple shell comparisons. It also writes
`qemu_prompt_bench_summary_latest.csv` with one suite row plus one row per
prompt aggregate so CI can publish compact latency, throughput, prompt-byte,
and memory rollups without parsing JSON. `qemu_prompt_bench_phases_latest.csv`
adds one row each for warmup, measured, and all launches so CI can compare
launch health and token totals across phases without parsing raw run rows.
Measured reports also write `qemu_prompt_bench_prompt_rank_latest.csv` for
slowest-prompt triage and `qemu_prompt_bench_prompt_variability_latest.csv`
for prompts with the highest wall tok/s IQR or P05-to-P95 spread. They also
write `qemu_prompt_bench_prompt_efficiency_latest.csv` to rank prompts by
median wall prompt-bytes/s, tokens per prompt byte, and wall tok/s so prompt
length effects are visible without parsing raw launch rows. Prompt reliability
triage is exported as `qemu_prompt_bench_prompt_failures_latest.csv`, sorted by
failed launches, timeouts, nonzero guest exits, and OK-run percentage.
JSON and Markdown reports also include
host provenance for reproducibility: platform, machine, Python version, CPU
count, QEMU binary/path, QEMU version when discoverable, and a stable SHA256
fingerprint of the QEMU command line. Dry-run, measured, per-run, launch CSV,
and launch JSONL artifacts also record command-level air-gap evidence:
whether the command includes explicit `-nic none`, whether legacy `-net none`
is present, and any detected network-device violations. Measured reports also
include the same deterministic launch plan and launch-plan SHA256 used by dry-runs, and write
`qemu_prompt_bench_launches_latest.csv` with one row per warmup or measured
launch so CI can audit actual launch order against the reviewed plan. They also
write `qemu_prompt_bench_launches_latest.jsonl` with the same ordered launch
records plus report-level hashes, status, profile, model, quantization, commit,
and prompt-suite SHA256 for direct ingestion by perf-regression jobs. The
runner separately fingerprints the realized launch sequence and fails the
report if observed launch order, prompt hashes, prompt byte counts, phases, or
iterations diverge from the planned sequence. They also
record input artifact metadata for the disk image path and any
`--qemu-args-file` sources; pass `--hash-image` when the disk image digest
should be recorded alongside size and mtime.
JSON and JSONL prompt rows can declare `expected_tokens` or
`expected_generated_tokens` to pin intended decode length for comparable
throughput runs. Use `--require-expected-tokens` when CI should fail any
measured run whose prompt omits that baseline, and use
`--require-expected-tokens-match` when CI should fail any measured run whose
emitted token count differs from the prompt-declared count.
Expected-token metadata is reported separately and does not alter the prompt
suite hash, which remains tied to prompt IDs and prompt text.

Prompt files can be JSON, JSONL, or plain text split with `---`. Guest output may
include a JSON line such as:

```text
BENCH_RESULT: {"tokens": 128, "elapsed_us": 500000, "ttft_us": 42000, "tok_per_s": 256.0, "prompt_sha256": "...", "prompt_bytes": 512}
```

Memory telemetry is optional. The runner normalizes `memory_bytes`,
`max_rss_bytes`, `rss_bytes`, `peak_memory_bytes`, plus `_kib`, `_kb`, `_mib`,
and `_mb` variants into `memory_bytes` so the perf regression dashboard can
track peak memory alongside tok/s. When token counts are present, reports also
derive `memory_bytes_per_token` for each run and include median/max rollups in
JSON, Markdown, summary CSV, and JUnit properties.

Each measured run also records `wall_tok_per_s`, derived from host wall-clock
elapsed time. Reports now include `host_overhead_us` and `host_overhead_pct`,
computed as host wall elapsed time minus guest-reported elapsed time, so QEMU
launch/serial/host orchestration overhead can be tracked separately from guest
decode telemetry. This is reported next to guest telemetry in JSON, Markdown,
and CSV so suspicious guest-side timing can be compared against the
host-observed launch duration. Per-run records also include `timeout_seconds`
and `wall_timeout_pct`, making it visible when a benchmark technically passes
but is consuming too much of the configured launch timeout. On Unix hosts, the
runner also records
`host_child_user_cpu_us`, `host_child_system_cpu_us`, `host_child_cpu_us`, and
`host_child_cpu_pct` from child-process resource usage around each QEMU launch;
suite and prompt summaries include median child CPU time, utilization, and
`host_child_tok_per_cpu_s` efficiency so CPU saturation can be distinguished
from guest decode timing drift. The runner also samples direct-child RSS while
QEMU is running and records `host_child_peak_rss_bytes`; suite, prompt, matrix,
CSV, Markdown, and JSON reports surface the maximum sampled RSS separately from
guest-reported `memory_bytes`. The same artifacts also include derived
`us_per_token` and `wall_us_per_token` latency metrics, plus median/P95 latency
rollups, so dashboards can compare either throughput or per-token decode cost
without reprocessing raw elapsed times. Per-run, suite, prompt, CSV, and
Markdown outputs also include `tokens_per_prompt_byte` telemetry so runs that
look fast because they emitted too little output for the input size can be
gated directly. Suite and prompt summaries include `ok_run_pct` next to OK,
failed, timeout, and nonzero-exit counts, making
partially successful benchmark sweeps visible in JSON, Markdown, and summary
CSV artifacts.

Use `--max-suite-cv-pct`, `--max-prompt-cv-pct`, `--max-suite-iqr-pct`, and
`--max-prompt-iqr-pct` to fail noisy benchmark runs when measured tok/s
coefficient of variation or interquartile spread exceeds a CI threshold. Gate
findings are written into the JSON and Markdown reports as
`variability_findings`. The runner also writes
`qemu_prompt_bench_junit_latest.xml` so CI can surface failed prompt launches
and variability gate failures directly from the benchmark job.
Use `--require-tokens`, `--require-tok-per-s`, `--require-memory`,
`--require-ttft-us`, `--min-tokens`, `--min-tok-per-s`,
`--min-total-tokens`, `--min-wall-tok-per-s`, `--max-memory-bytes`,
`--max-ttft-us`, `--max-host-overhead-us`, `--max-host-overhead-pct`,
`--max-wall-timeout-pct`, `--min-host-child-tok-per-cpu-s`,
`--min-tokens-per-prompt-byte`, `--require-host-child-rss`,
`--max-host-child-rss-bytes`, `--max-memory-bytes-per-token`,
`--max-serial-output-bytes`, `--max-serial-output-lines`,
`--require-expected-tokens`, `--require-guest-prompt-sha256-match`, and
`--require-guest-prompt-bytes-match` to fail measured runs or suites that omit required
telemetry, produce too little work for a trustworthy throughput sample, exceed
a host-observed latency, memory, RSS, orchestration overhead, or serial
verbosity budget, exceed a first-token latency or timeout-budget threshold, fall
below a host child-CPU or output-density floor, omit expected-token baselines,
or report a missing/mismatched guest prompt hash or byte count.
Telemetry gate failures are written as `telemetry_findings` in JSON/Markdown and as
`benchmark_telemetry` failures in the JUnit report.

The qemu prompt benchmark smoke uses the synthetic local QEMU-compatible
fixture and does not boot a guest:

```bash
python3 bench/qemu_prompt_bench_ci_smoke.py
```

`qemu_prompt_coverage_audit.py` reads existing prompt benchmark JSON artifacts
and verifies that every prompt declared in `prompt_suite.source` was measured,
that the recorded prompt-suite SHA256 still matches the suite file, and that
each expected prompt has a minimum number of successful measured runs. It is
host-side only and does not launch QEMU. Use `--include-warmups` when coverage
must include the benchmark runner's top-level `warmups` rows in addition to
measured `benchmarks` rows.

```bash
python3 bench/qemu_prompt_coverage_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_prompt_coverage_audit_latest \
  --require-suite-file \
  --require-success \
  --fail-on-unexpected-prompts \
  --min-artifacts 1 \
  --min-prompts 2 \
  --min-runs-per-prompt 2
```

The smoke gate builds a synthetic air-gapped benchmark artifact and checks the
coverage audit pass path, top-level warmup inclusion, minimum-run failure path,
and empty-input failure path:

```bash
python3 bench/qemu_prompt_coverage_audit_ci_smoke.py
```

`qemu_prompt_balance_audit.py` reads existing prompt benchmark JSON artifacts
and verifies that measured successful samples are balanced across prompts. It
can require a minimum prompt count, minimum measured rows, minimum successful
runs per prompt, zero successful-run skew, and no failed measured rows. It is
host-side only and does not launch QEMU.

```bash
python3 bench/qemu_prompt_balance_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_prompt_balance_audit_latest \
  --min-prompts 2 \
  --min-measured-runs 4 \
  --min-successful-runs-per-prompt 2 \
  --max-successful-run-delta 0 \
  --fail-on-failed-runs
```

The smoke gate checks the balanced prompt pass path against the synthetic
air-gapped benchmark artifact:

```bash
python3 bench/qemu_prompt_balance_audit_ci_smoke.py
```

`qemu_prompt_schema_audit.py` reads existing prompt benchmark JSON artifacts and
verifies the artifact schema, per-row timing/throughput telemetry, launch-count
consistency, command hashes, and saved QEMU air-gap telemetry. It is host-side
only and does not launch QEMU.

```bash
python3 bench/qemu_prompt_schema_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_prompt_schema_audit_latest \
  --min-artifacts 1 \
  --min-measured-rows 2 \
  --require-success \
  --require-ok-telemetry ttft_us \
  --require-ok-telemetry memory_bytes
```

The smoke gate builds a synthetic air-gapped benchmark artifact and checks the
schema audit pass path plus a legacy-network-command failure path:

```bash
python3 bench/qemu_prompt_schema_audit_ci_smoke.py
```

`qemu_summary_consistency_audit.py` reads existing prompt benchmark JSON
artifacts and recomputes suite and per-prompt summaries from raw benchmark rows
to catch stale or hand-edited aggregate metrics before dashboards consume them.
It is host-side only and does not launch QEMU.

```bash
python3 bench/qemu_summary_consistency_audit.py \
  bench/results/qemu_prompt_bench_latest.json \
  --output-dir bench/results \
  --output-stem qemu_summary_consistency_audit_latest \
  --min-artifacts 1 \
  --min-measured-rows 2
```

The smoke gate builds a synthetic air-gapped benchmark artifact and checks the
summary consistency pass path plus stale suite/prompt summary failure paths:

```bash
python3 bench/qemu_summary_consistency_audit_ci_smoke.py
```

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
  --min-prompt-count 3 \
  --max-prompt-cv-pct 5 \
  --max-prompt-iqr-pct 5 \
  --require-tokens \
  --require-tok-per-s \
  --require-ttft-us \
  --min-tokens 16 \
  --min-total-tokens 512 \
  --min-wall-tok-per-s 10 \
  --max-memory-bytes 536870912 \
  --max-ttft-us 1000000 \
  --max-host-overhead-pct 25 \
  --min-host-child-tok-per-cpu-s 20 \
  --require-host-child-rss \
  --max-host-child-rss-bytes 1073741824 \
  --max-memory-bytes-per-token 16777216 \
  --max-serial-output-bytes 65536 \
  --max-serial-output-lines 256 \
  --require-expected-tokens \
  --require-guest-prompt-sha256-match \
  --require-guest-prompt-bytes-match \
  --qemu-args-file bench/fixtures/local-qemu.args \
  --hash-image \
  -- -m 512M
```

Validate the final QEMU command without launching:

```bash
python3 bench/qemu_prompt_bench.py \
  --image path/to/TempleOS.img \
  --prompts bench/prompts/smoke.jsonl \
  --max-launches 10 \
  --min-prompt-count 1 \
  --dry-run
```

Dry-runs also write `qemu_prompt_bench_dry_run_latest.json`, Markdown, CSV, and
JUnit XML artifacts under the selected output directory, plus
`qemu_prompt_bench_dry_run_launches_latest.csv` with one row per planned launch.
These artifacts record the exact `-nic none` command, the command SHA256
fingerprint, prompt-suite hash, launch-plan SHA256, planned launch sequence
SHA256, warmup count, repeat count, configured launch budget, prompt-count
floor, and planned launch totals for CI review without booting a guest, plus
the same host/QEMU provenance fields used by measured benchmark reports. Dry-run JSON, Markdown, CSV, and JUnit also include disk
image metadata and SHA256 hashes for any local QEMU args files, making reviewed
launch plans reproducible before a VM is started.

`qemu_source_audit.py` statically scans host-side docs/config/shell-like files
for literal `qemu-system*` launch snippets and applies the same air-gap command
rules. This catches unsafe copied commands before they become benchmark scripts
or operator runbooks. It also checks JSON `qemu_args`/`qemu_extra_args`/
`qemu_flags` fragments, standalone JSON QEMU args arrays, YAML and TOML
`qemu_args`/`qemu_extra_args`/`qemu_flags` fragments, `qemu_args_file`/
`qemu_args_files` references resolved relative to the config file that names
them, and `.args` files for network-enabling options such as `-netdev`,
non-`none` `-nic`, legacy `-net none`, and virtual NIC devices. Fragment audits
do not require `-nic none` because the launcher injects it; they reject options
that re-enable networking, drift back to legacy network-disabling flags, or
point QEMU at URL-backed disks, kernels, initrds, or block devices. Raw QEMU
examples must keep `-nic none` explicit:

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

Exercise the audit parser and failing gates without launching QEMU:

```bash
python3 bench/qemu_source_audit_ci_smoke.py
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
`.csv`, `bench_matrix_summary_latest.csv`, and
`bench_matrix_junit_latest.xml`. QEMU launches still flow through the air-gap
guard that injects `-nic none` and rejects NIC/network arguments.
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
Host child CPU rollups include median CPU time, CPU utilization, tok/CPU-second
efficiency, and peak sampled direct-child RSS so matrix dashboards can separate
guest throughput changes from host saturation.
They also include total, minimum, and maximum prompt byte counts for each cell
plus total emitted tokens and total guest elapsed time so prompt-size and
generation-volume changes are visible next to tok/s, RSS, and latency rollups.
`bench_matrix_summary_latest.csv` adds one matrix-wide aggregate row plus one
compact row per cell for CI dashboards that only need pass/fail counts,
throughput ranges, prompt-byte and token totals, latency, RSS, and variability
findings.
Use `expect_cells` in the matrix JSON, or `--expect-cells` on the CLI, to fail
the matrix when the expanded profile/model/quantization cell count changes
unexpectedly. The gate also works with `--dry-run`, so CI can catch accidental
matrix coverage loss before launching QEMU.
Use `max_suite_cv_pct`, `max_prompt_cv_pct`, `max_suite_iqr_pct`, and
`max_prompt_iqr_pct` in the matrix JSON, or the matching CLI flags, to pass
tok/s variability gates through to every cell. A cell that fails a variability
gate still writes its prompt-benchmark report and is preserved in the matrix
summary as a failed cell with a findings count.

Example:

```bash
python3 bench/bench_matrix.py \
  --matrix bench/fixtures/bench_matrix_smoke.json \
  --output-dir bench/results \
  --max-prompt-cv-pct 5 \
  --expect-cells 2
```

Validate the expanded matrix without launching QEMU:

```bash
python3 bench/bench_matrix.py \
  --matrix bench/fixtures/bench_matrix_smoke.json \
  --dry-run
```

Audit matrix coverage and air-gap-safe QEMU argument fragments without
launching QEMU:

```bash
python3 bench/bench_matrix_audit.py bench/fixtures/bench_matrix_smoke.json \
  --output-dir bench/results \
  --output-stem bench_matrix_audit_latest \
  --expect-cells 2 \
  --require-quantization Q4_0 \
  --require-quantization Q8_0
```

The audit writes JSON, Markdown, CSV, and JUnit outputs. Its smoke gate checks
passing coverage plus rejection of legacy `-net none` drift:

```bash
python3 bench/bench_matrix_audit_ci_smoke.py
```

`qemu_benchmark_matrix.py` plans air-gapped QEMU command/launch matrices
without launching QEMU. `qemu_benchmark_matrix_audit.py` audits the saved
planner JSON for argv hash drift, explicit `-nic none` air-gap compliance,
per-build launch-count formulas, warmup/measured phase counts, contiguous
launch indexes, and summary rollup drift:

```bash
python3 bench/qemu_benchmark_matrix_audit.py \
  bench/results/qemu_benchmark_matrix_smoke_latest.json \
  --output-dir bench/results \
  --output-stem qemu_benchmark_matrix_audit_latest
```

Its smoke gate builds a synthetic local planner artifact and audits it without
starting QEMU:

```bash
python3 bench/qemu_benchmark_matrix_audit_ci_smoke.py
```

`qemu_matrix_budget_audit.py` reads saved `qemu_benchmark_matrix.py` planner
artifacts and gates the planned launch volume before a benchmark job spends VM
time. It reports global and per-build launch counts, warmup/measured counts,
prompt bytes, expected-token totals, missing expected-token coverage, and
air-gap status without launching QEMU:

```bash
python3 bench/qemu_matrix_budget_audit.py \
  bench/results/qemu_benchmark_matrix_smoke_latest.json \
  --output-dir bench/results \
  --output-stem qemu_matrix_budget_audit_latest \
  --max-launches 64 \
  --max-launches-per-build 16 \
  --max-expected-tokens 8192 \
  --max-prompt-bytes-per-build 65536 \
  --require-expected-tokens \
  --require-airgap
```

The smoke gate covers passing budget reports plus launch-budget and
global expected-token budget, launch-budget, and expected-token coverage
failures:

```bash
python3 bench/qemu_matrix_budget_audit_ci_smoke.py
```

`qemu_matrix_plan_diff.py` compares two saved `qemu_benchmark_matrix.py`
planner JSON artifacts and gates command-hash, build, launch-count, and
per-prompt launch-key drift before benchmark VM time is spent. It reads saved
artifacts only and never launches QEMU:

```bash
python3 bench/qemu_matrix_plan_diff.py \
  bench/results/qemu_benchmark_matrix_baseline.json \
  bench/results/qemu_benchmark_matrix_candidate.json \
  --output-dir bench/results \
  --output-stem qemu_matrix_plan_diff_latest
```

The smoke gate covers identical plans plus command and launch drift failures:

```bash
python3 bench/qemu_matrix_plan_diff_ci_smoke.py
```

`bench_smoke_manifest.py` scans host-side `*_ci_smoke.py` scripts and writes a
dashboard-friendly coverage manifest with paired-tool status, script metadata,
area rollups, findings CSV, and JUnit output. It never runs smoke scripts and
never launches QEMU:

```bash
python3 bench/bench_smoke_manifest.py bench \
  --output-dir bench/results \
  --output-stem bench_smoke_manifest_latest \
  --require-paired-tools \
  --min-smokes 100
```

Its own smoke gate covers paired coverage and malformed/unpaired script
failures:

```bash
python3 bench/bench_smoke_manifest_ci_smoke.py
```

`bench_result_index.py` scans existing QEMU prompt, QEMU prompt dry-run, and
matrix JSON reports, rolls their tok/s, wall-clock tok/s, TTFT, host-overhead,
per-token latency, host child CPU efficiency/RSS, memory, memory-per-token,
serial-output, prompt-byte efficiency, prompt-suite, expected-token parity,
elapsed-time, and run-count metadata into a single JSON/Markdown/CSV/JUnit XML
index, and checks each
recorded QEMU command for explicit `-nic none` air-gap compliance. Dry-run
reports are indexed as planned launch artifacts: their command hash is
recomputed, their launch-plan hash is recomputed from the embedded launch plan,
planned warmup/measured launch counts are checked against the launch plan, and they are excluded from
latest comparable throughput rollups because they have no measured tok/s. It
also carries `expected_token_prompts`, `expected_tokens_total`,
`expected_tokens_matches`, and `expected_tokens_mismatches` through the full
index and latest-comparable CSVs; `--fail-on-telemetry` rejects measured
artifacts with any expected-token mismatches so tok/s comparisons cannot hide
short or overlong decodes.
Measured artifacts also carry `measured_prompt_bytes_total`,
`prompt_bytes_min`, `prompt_bytes_max`, `prompt_bytes_per_s_median`,
`wall_prompt_bytes_per_s_median`, and `tokens_per_prompt_byte_median` through
the index and latest-comparable CSVs so dashboards can distinguish token
throughput regressions from prompt-size or prompt-efficiency drift.
also reports prompt-suite drift when comparable profile/model/quantization
artifacts carry different non-empty suite hashes, and command drift when
comparable artifacts carry different `command_sha256` values. It also
fingerprints host/QEMU environment metadata and reports environment drift when
comparable artifacts use the same prompt suite and command hash but were
produced under different host or QEMU identities.
The JSON and Markdown reports also include a latest-per-comparable-key rollup,
with the matching CSV at `bench_result_index_latest_comparable_latest.csv`, so
dashboards can ingest only the newest artifact for each stable
profile/model/quantization/prompt-suite/command/environment key without
discarding historical rows from the full index.
Use `--min-history-per-key` with `--fail-on-history-coverage` when CI should
reject comparable keys that have too few measured artifacts for trend or
regression dashboards; violations are exported to
`bench_result_index_history_coverage_latest.csv` and surfaced in JUnit.
These drift checks catch accidental prompt, launch, or host changes before
throughput numbers are compared. The index marks artifacts with
missing required telemetry, such as zero measured runs or absent median tok/s,
as failures so empty or malformed benchmark reports do not enter CI dashboards
as valid data. The drift findings are also written to
`bench_result_index_prompt_suite_drift_latest.csv` and
`bench_result_index_command_drift_latest.csv`,
`bench_result_index_environment_drift_latest.csv`, and
`bench_result_index_junit_latest.xml` exposes artifact failures, air-gap
violations, missing telemetry, inconsistent commit metadata, inconsistent
command hash metadata, prompt drift, command drift, environment drift, and
history coverage failures as CI test failures. It
also recomputes `command_sha256` from each recorded command and treats mismatches
as inconsistent command hash metadata, which catches stale or hand-edited
benchmark artifacts before they reach dashboards. It also recomputes
`launch_plan_sha256` when an artifact embeds a launch plan and treats mismatches
as inconsistent launch-plan hash metadata before throughput dashboards compare
different planned prompt sequences. It
never launches QEMU. The index also records per-artifact commit metadata and can
optionally fail when benchmark artifacts were produced from a different commit
than the current checkout. It can also enforce freshness with
`--max-artifact-age-hours`, marking artifacts stale when their `generated_at`
timestamp is too old and exporting stale rows to
`bench_result_index_freshness_failures_latest.csv`. Use `--fail-on-airgap`, `--fail-on-telemetry`,
`--fail-on-commit-metadata`, `--fail-on-command-hash-metadata`,
`--fail-on-launch-plan-hash-metadata`, `--fail-on-drift`, `--fail-on-command-drift`,
`--fail-on-environment-drift`, `--fail-on-history-coverage`, and
`--fail-on-stale-artifact` to gate those failure classes independently.

Example:

```bash
python3 bench/bench_result_index.py \
  --input bench/results \
  --output-dir bench/results \
  --fail-on-airgap \
  --fail-on-drift \
  --fail-on-command-drift \
  --fail-on-environment-drift \
  --min-history-per-key 2 \
  --fail-on-history-coverage
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

`bench_result_index_ci_smoke.py` is a stdlib-only CI gate for the indexer. It
builds synthetic QEMU prompt, dry-run, and matrix reports, verifies the JSON,
Markdown, CSV, latest-comparable CSV, launch-plan drift CSV, freshness-failure
CSV, and JUnit outputs, and checks that command-hash mismatches,
launch-plan-hash mismatches, NIC-enabled QEMU commands, stale artifacts,
environment drift, and insufficient comparable history are rejected by their
opt-in gates:

```bash
python3 bench/bench_result_index_ci_smoke.py
```

`bench_artifact_manifest.py` builds on the same indexer and writes a
deterministic latest-artifact manifest for CI upload/download consumers. It
records SHA256 and byte size for each benchmark JSON, retains compact history,
selects the newest artifact for each profile/model/quantization/prompt-suite
key, preserves prompt count, wall-clock throughput, TTFT, host overhead,
host child CPU/RSS, emitted-token totals, elapsed guest time, and guest/wall
per-token latency telemetry, writes
latest-key, full-history, missing dry-run coverage, environment drift, and
freshness-failure CSV exports plus timestamp-collision CSV exports, keeps the same recorded-command
air-gap, command SHA256, and commit
metadata checks while preserving environment fingerprints, and writes
`bench_artifact_manifest_junit_latest.xml` so CI can
surface failed artifacts, air-gap violations, missing telemetry, stale
artifacts, inconsistent command hashes, inconsistent commit metadata, sparse
per-key history, sample coverage failures, missing measured-run dry-run plans,
duplicate key/timestamp artifacts, and empty manifests directly. Empty manifests are marked failed so missing benchmark
uploads do not pass silently. For current-job manifests,
`--fail-on-stale-commit` returns non-zero when any artifact was produced from a
different commit than the current checkout. `--max-artifact-age-hours` records
artifact freshness status, and `--fail-on-stale-artifact` returns non-zero when
any artifact exceeds that age. Use `--fail-on-airgap`, `--fail-on-telemetry`,
`--fail-on-command-hash-metadata`, and `--fail-on-commit-metadata` to gate those
failure classes independently. Use `--min-history-per-key` with
`--fail-on-history-coverage` when a CI consumer needs at least N retained
artifacts for every profile/model/quantization/prompt-suite key before trend or
regression decisions are trusted. Use `--min-measured-runs`,
`--min-total-tokens`, and `--fail-on-sample-coverage` when promoted artifacts
must have enough measured samples before dashboard consumers ingest them. Use
`--fail-on-missing-dry-run` when measured QEMU prompt artifacts must have a
matching reviewed dry-run launch plan with the same profile, model,
quantization, prompt-suite hash, command hash, launch-plan hash, and
environment hash. Use `--fail-on-timestamp-collision` when CI should reject
multiple artifacts for the same profile/model/quantization/prompt-suite key
with the same `generated_at` timestamp.

```bash
python3 bench/bench_artifact_manifest.py \
  --input bench/results \
  --output-dir bench/results \
  --fail-on-stale-commit \
  --max-artifact-age-hours 6 \
  --fail-on-stale-artifact \
  --min-history-per-key 2 \
  --fail-on-history-coverage \
  --min-measured-runs 3 \
  --min-total-tokens 512 \
  --fail-on-sample-coverage \
  --fail-on-missing-dry-run \
  --fail-on-timestamp-collision \
  --fail-on-airgap \
  --fail-on-telemetry \
  --fail-on-command-hash-metadata
```

## Eval Efficiency Frontier

`eval_efficiency_frontier.py` reads saved eval perf scorecards and marks
non-dominated model/quantization rows by quality and throughput. Use
`--memory-aware` to include `max_memory_bytes` as a lower-is-better Pareto
dimension, which keeps low-memory builds visible when they trade quality or
speed for footprint.

```bash
python3 bench/eval_efficiency_frontier.py bench/results/eval_perf_scorecard_smoke_latest.json \
  --output-dir bench/results \
  --output-stem eval_efficiency_frontier_latest \
  --quality-metric holyc_accuracy \
  --speed-metric median_wall_tok_per_s \
  --memory-aware \
  --fail-on-missing-metrics
```

## Perf Regression Dashboard

`perf_regression.py` scans JSON, JSONL, and CSV benchmark artifacts, groups
results by benchmark/profile/model/quantization/prompt plus commit, and writes
guest tok/s, host wall-clock tok/s, guest/wall microseconds per token, emitted
token counts, serial output bytes, serial output bytes per emitted token, QEMU
host overhead percentage, guest memory, host child peak RSS,
host child tok/CPU-second efficiency, first-token latency, and
sample-coverage dashboards
under `bench/dashboards/`.
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
  --max-wall-tok-cv-pct 7.5 \
  --p05-tok-regression-pct 7.5 \
  --wall-tok-regression-pct 7.5 \
  --p05-wall-tok-regression-pct 7.5 \
  --us-per-token-regression-pct 7.5 \
  --wall-us-per-token-regression-pct 7.5 \
  --token-drop-regression-pct 5 \
  --min-token-drop-regression-pct 25 \
  --serial-output-regression-pct 25 \
  --serial-output-per-token-regression-pct 25 \
  --p95-ttft-regression-pct 15 \
  --host-overhead-regression-pct 25 \
  --host-child-peak-rss-regression-pct 10 \
  --host-child-tok-per-cpu-s-regression-pct 10 \
  --require-tok-per-s \
  --require-wall-tok-per-s \
  --require-us-per-token \
  --require-wall-us-per-token \
  --require-tokens \
  --require-serial-output-bytes \
  --require-serial-output-bytes-per-token \
  --require-ttft-us \
  --require-host-overhead-pct \
  --require-environment-sha256 \
  --require-host-platform \
  --require-host-machine \
  --require-qemu-version \
  --require-qemu-bin \
  --require-memory \
  --require-host-child-peak-rss \
  --require-host-child-tok-per-cpu-s \
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
`--max-wall-tok-cv-pct` applies the same variability gate to host-observed
wall-clock tok/s, catching unstable QEMU/host runs before they become baselines.
`--p05-tok-regression-pct` optionally gates low-tail guest tok/s drops, which
catches slow individual runs that median throughput can hide.
`--wall-tok-regression-pct` optionally gates host-observed wall-clock tok/s
drops, which is useful when guest-side timing looks suspicious.
`--p05-wall-tok-regression-pct` optionally gates low-tail host-observed tok/s
drops, catching wall-clock outliers that median host throughput can hide.
`--us-per-token-regression-pct` and `--wall-us-per-token-regression-pct`
optionally gate guest and host token-latency growth directly, which is easier to
reason about for short prompts where throughput deltas compress latency changes.
`--ttft-regression-pct` and `--p95-ttft-regression-pct` optionally gate median
and tail first-token latency growth. `--host-overhead-regression-pct`
optionally gates increases in QEMU host overhead.
`--host-child-peak-rss-regression-pct` optionally gates host-observed QEMU
child peak RSS growth separately from guest-reported memory.
`--host-child-tok-per-cpu-s-regression-pct` optionally gates drops in QEMU
child process CPU efficiency when host child CPU telemetry is available.
`--token-drop-regression-pct`
optionally gates drops in median emitted token count so faster runs cannot pass
only because they generated less output. `--min-token-drop-regression-pct`
applies the same check to the minimum emitted token count so one truncated
prompt cannot hide behind a healthy median. `--require-tok-per-s`,
`--require-wall-tok-per-s`, `--require-us-per-token`,
`--require-wall-us-per-token`, `--require-tokens`, `--require-ttft-us`,
`--require-serial-output-bytes`,
`--require-serial-output-bytes-per-token`, `--require-host-overhead-pct`,
`--require-memory`, `--require-host-child-peak-rss`, and
`--require-host-child-tok-per-cpu-s` fail
the dashboard when any benchmark key/commit point has zero samples for that
telemetry field. `--require-environment-sha256`, `--require-host-platform`,
`--require-host-machine`, `--require-qemu-version`, and `--require-qemu-bin`
similarly require host/QEMU provenance fields for every commit point. These
gates catch malformed or partially uploaded artifacts before CI treats missing
metrics as merely non-comparable.
Prompt-suite hashes from QEMU benchmark reports are carried into commit points;
the dashboard fails when comparable benchmark/profile/model/quantization/prompt
records contain multiple non-empty prompt-suite hashes, preventing accidental
throughput comparisons across different prompt sets.
The dashboard also writes `perf_regression_junit_latest.xml` so CI systems can
surface throughput regressions, sample-coverage failures, commit-coverage
failures, comparison-coverage failures, prompt-suite drift, and guest or wall
variability failures as test failures. When `--baseline-commit` or
`--candidate-commit` is
provided, every benchmark key must contain the requested commit before the
dashboard passes; missing explicit comparison commits are written to
`perf_regression_comparison_coverage_violations_latest.csv`.

## Build Benchmark Compare

`build_compare.py` compares multiple `qemu_prompt_bench.py` JSON reports by
prompt/profile/model/quantization and writes per-build throughput and elapsed
time deltas to `bench/results/` as JSON, Markdown, CSV, and JUnit XML. It also
compares P05 guest tok/s, median and P05 host wall-clock tok/s, first-token
latency, guest/wall us-per-token latency, direct-child CPU time/utilization,
direct-child tok/CPU-second efficiency, direct-child peak RSS, serial output
bytes, and max memory bytes when benchmark reports include that telemetry. Use
`--fail-on-regression` with `--max-tok-regression-pct` to gate median guest
tok/s drops in CI without launching QEMU, add `--max-p05-tok-regression-pct` to
gate low-tail guest tok/s drops, add `--max-wall-tok-regression-pct` to gate
host-observed tok/s drops, add `--max-p05-wall-tok-regression-pct` to gate
low-tail host-observed tok/s drops, add `--max-ttft-growth-pct` to gate
first-token latency growth, add `--max-us-per-token-growth-pct` or
`--max-wall-us-per-token-growth-pct` to gate per-token latency growth, add
`--max-host-child-cpu-growth-pct`, `--max-host-child-cpu-pct-growth-pct`,
`--max-host-child-tok-per-cpu-s-regression-pct`, or
`--max-host-child-rss-growth-pct` to gate host CPU/RSS drift, add
`--max-serial-output-growth-pct` to gate verbose serial logging growth, and
add `--max-memory-growth-pct` to gate peak memory growth. Use
`--min-ok-runs-per-build` with `--fail-on-coverage` to reject comparisons where
the baseline or candidate build has too few successful runs for a prompt key.
Coverage violations are written to
`build_compare_coverage_violations_latest.csv` and surfaced in the JUnit report.
Use `--fail-on-comparison-coverage` to reject build pairs whose prompt keys do
not match exactly; missing baseline/candidate keys are written to
`build_compare_comparison_coverage_latest.csv` and surfaced in JUnit so dropped
prompts cannot be hidden by the comparable-delta join.
Prompt-suite SHA256s are carried through from QEMU prompt benchmark reports;
`--fail-on-prompt-suite-drift` rejects comparable build pairs whose prompt-suite
hashes differ, with details written to
`build_compare_prompt_suite_drift_latest.csv` and the JUnit report.
QEMU command SHA256s are also carried through; `--fail-on-command-drift`
rejects comparable build pairs whose launch command fingerprints differ, with
details written to `build_compare_command_drift_latest.csv` and the JUnit
report.
Host/QEMU environment fingerprints are derived from benchmark `environment`
metadata or explicit `environment_sha256` fields; `--fail-on-environment-drift`
rejects comparable build pairs collected under different host/QEMU identities,
with details written to `build_compare_environment_drift_latest.csv` and the
JUnit report.

`build_compare_ci_smoke.py` is a stdlib-only CI gate for the build comparison
dashboard. It creates synthetic local benchmark reports, checks the pass path,
and verifies command-drift, environment-drift, OK-run coverage, and
comparison-coverage gates without launching QEMU:

```bash
python3 bench/build_compare_ci_smoke.py
```

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
  --max-us-per-token-growth-pct 10 \
  --max-wall-us-per-token-growth-pct 10 \
  --max-host-child-tok-per-cpu-s-regression-pct 10 \
  --max-host-child-rss-growth-pct 10 \
  --max-serial-output-growth-pct 10 \
  --max-memory-growth-pct 10 \
  --min-ok-runs-per-build 3 \
  --fail-on-coverage \
  --fail-on-comparison-coverage \
  --fail-on-command-drift \
  --fail-on-environment-drift
```

`build_pair_manifest_audit.py` gates the `build_pair_select.py` output before
perf CI consumes it. It verifies that baseline/candidate sources and commits are
distinct, candidate timestamps are not older than baselines, both sides have the
required measured-run count, and the emitted `build_compare.py` arguments still
reference both selected artifacts.

```bash
python3 bench/build_pair_manifest_audit.py bench/results/build_pair_select_latest.json \
  --output-dir bench/results \
  --output-stem build_pair_manifest_audit_latest \
  --min-measured-runs 4
python3 bench/build_pair_manifest_audit_ci_smoke.py
```

## HolyC vs llama.cpp Eval Compare

`eval_compare.py` compares offline multiple-choice predictions from HolyC and
llama.cpp against the same local gold JSONL dataset. It aligns by record id,
supports prediction indexes, labels, exact choice text, or score arrays, and
writes JSON, Markdown, per-record CSV, per-dataset/split breakdown CSV,
confusion-matrix CSV, calibration-bin CSV, score-margin CSV,
score-NLL CSV, score-rank CSV, score-tie CSV, engine-disagreement CSV, and JUnit XML reports to `bench/results/`.
Reports include accuracy, agreement, macro-F1, per-answer F1,
per-dataset/split breakdowns, and confusion matrices for each engine.
Score-vector reports include calibration, gold-rank, mean gold-answer NLL,
choice-set perplexity, and predicted-vs-runner-up margin telemetry; use
`--min-holyc-margin-coverage` and `--min-holyc-mean-margin` to fail CI when
HolyC score margins are missing or too weak. Use `--min-holyc-nll-coverage`,
`--max-holyc-mean-nll`, and `--max-holyc-nll-delta` to gate score-vector
cross-entropy. Top-score tie telemetry catches ambiguous argmax cases; use
`--max-holyc-score-tie-rate` to gate tied best-choice rows. Accuracy and
agreement summaries also include stdlib-only Wilson
confidence intervals; use `--confidence-level` to select 0.80, 0.90, 0.95,
0.98, or 0.99. Add `--gate-dataset-breakdowns` to apply the same quality gates
to each dataset/split bucket, which prevents mixed eval suites from hiding
small-subset regressions behind healthy aggregate scores.

Before comparing, `eval_input_audit.py` can gate apples-to-apples inputs. It
checks gold/prediction record coverage, duplicates, invalid prediction indexes,
dataset/split metadata, optional model/quantization metadata drift, and gold
answer distribution. Use `--max-majority-gold-answer-pct` to fail early when a
gold file is too label-skewed for a useful paired comparison. Use
`--min-choices` and `--max-choices` to enforce homogeneous normalized
multiple-choice rows before prediction files are scored. It also records
score-vector coverage and top-score ties; use `--min-score-coverage-pct` and
`--max-top-score-tie-pct` to catch missing or ambiguous score vectors before
running quality comparisons. Use `--min-top-score-margin` to gate scored rows
whose top choice barely beats the runner-up. Prediction rows may also carry `prompt_sha256`,
`choices_sha256`, and `input_sha256` either at top level or under `metadata`;
`--require-input-hashes` fails the audit unless those hashes match the normalized
gold prompt and choices used for comparison. Add `--record-csv` to export
per-engine row telemetry with normalized predictions, correctness, score-vector
coverage, top-score tie counts, score margins, and input-hash status for dashboards. The audit
writes JSON, Markdown, CSV, and JUnit XML reports and exits non-zero when it
finds errors:

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
  --min-choices 4 \
  --max-choices 4 \
  --min-top-score-margin 0 \
  --record-csv bench/results/eval_input_audit_smoke_records_latest.csv \
  --output-stem eval_input_audit_smoke_latest
```

`eval_repro_audit.py` checks existing HolyC and llama.cpp prediction artifacts
for deterministic decoding metadata parity before quality numbers are compared.
It validates seed, temperature, top-k, top-p, and max-token metadata at the top
level or under `metadata`, can require temperature-0 seeded runs, and writes
JSON, Markdown, CSV, findings CSV, and JUnit XML reports:

```bash
python3 bench/eval_repro_audit.py \
  bench/eval/samples/holyc_smoke_predictions.jsonl \
  bench/eval/samples/llama_smoke_predictions.jsonl \
  --output-dir bench/results \
  --output-stem eval_repro_audit_latest \
  --require-metadata \
  --require-deterministic \
  --expect seed=1234 \
  --expect temperature=0
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

`eval_artifact_drift_audit.py` checks existing eval_compare JSON reports for
artifact drift before suite summaries are trusted. It verifies required
gold/HolyC/llama SHA256 fields, flags multiple gold hashes for the same
dataset/split, and can reject duplicate dataset/split/model/quantization report
keys with different artifact signatures:

```bash
python3 bench/eval_artifact_drift_audit.py bench/results \
  --output-dir bench/results \
  --output-stem eval_artifact_drift_audit_latest \
  --min-reports 2 \
  --require-hashes \
  --fail-on-failed-reports \
  --fail-on-duplicate-key-drift
```

`eval_rank_audit.py` gates scored multiple-choice rank metrics from existing
`eval_compare.py` JSON reports. It checks score-vector coverage, top-k accuracy,
mean reciprocal rank, and HolyC rank loss versus llama.cpp, with optional
dataset/split breakdown gates:

```bash
python3 bench/eval_rank_audit.py bench/results/eval_compare_smoke_latest.json \
  --output-dir bench/results \
  --output-stem eval_rank_audit_smoke_latest \
  --min-score-coverage 0.3 \
  --min-top-1-accuracy 0.5 \
  --min-mean-reciprocal-rank 0.5 \
  --max-holyc-top-1-loss 0.25 \
  --max-holyc-mrr-loss 0.25 \
  --fail-on-findings
```

`eval_rank_audit_ci_smoke.py` exercises pass and failure paths without launching
QEMU:

```bash
python3 bench/eval_rank_audit_ci_smoke.py
```

`perplexity_input_audit.py` validates offline token logprob or aggregate NLL
artifacts before HolyC-vs-llama comparison. It accepts JSON, JSONL, and CSV
records, rejects duplicate ids, positive logprobs, invalid token counts,
NLL/perplexity formula drift, and optional missing dataset/split metadata, then
writes JSON, Markdown, record CSV, source summary CSV, findings CSV, and JUnit
reports without launching QEMU.

```bash
python3 bench/perplexity_input_audit.py \
  bench/eval/samples/holyc_smoke_logprobs.jsonl \
  bench/eval/samples/llama_smoke_logprobs.jsonl \
  --output-dir bench/results \
  --output-stem perplexity_input_audit_latest \
  --min-records 6 \
  --min-records-per-source 3 \
  --min-tokens 22 \
  --min-tokens-per-source 11
```

`perplexity_input_audit_ci_smoke.py` exercises passing input artifacts and
positive-logprob/missing-metadata failure paths:

```bash
python3 bench/perplexity_input_audit_ci_smoke.py
```

`perplexity_pairing_audit.py` checks HolyC and llama.cpp logprob/perplexity
artifacts before comparison to ensure record ids, token counts, dataset
metadata, and split metadata are paired one-to-one. It reads local JSON, JSONL,
or CSV inputs and writes JSON, Markdown, pair CSV, findings CSV, and JUnit
reports without launching QEMU.

```bash
python3 bench/perplexity_pairing_audit.py \
  --holyc bench/eval/samples/holyc_smoke_logprobs.jsonl \
  --llama bench/eval/samples/llama_smoke_logprobs.jsonl \
  --output-dir bench/results \
  --output-stem perplexity_pairing_audit_latest \
  --min-pairs 3
```

`perplexity_pairing_audit_ci_smoke.py` exercises passing pair coverage plus
duplicate-id, missing-record, token-count, and metadata mismatch failures:

```bash
python3 bench/perplexity_pairing_audit_ci_smoke.py
```

`perplexity_compare.py` compares offline token logprob or aggregate NLL outputs
from HolyC and llama.cpp. It aligns rows by record id, computes token-weighted
NLL/token and perplexity, writes JSON, Markdown, per-record CSV,
dataset/split breakdown CSV, regression CSV, and JUnit XML reports, and fails on token-count
mismatches unless `--allow-token-count-mismatch` is passed. If both engine
outputs include `dataset`/`split` metadata for an id, conflicting metadata is
rejected before reporting. Optional quality gates can fail CI when aggregate
NLL drift, HolyC/llama.cpp perplexity ratio, or per-record NLL delta
distribution bounds exceed configured thresholds. Use `--min-record-count` and
`--min-token-count` to prevent accidentally promoting tiny perplexity runs with
too little coverage, and use `--min-dataset-split-record-count` or
`--min-dataset-split-token-count` to gate each dataset/split breakdown row.

Example:

```bash
python3 bench/perplexity_compare.py \
  --holyc bench/eval/samples/holyc_smoke_logprobs.jsonl \
  --llama bench/eval/samples/llama_smoke_logprobs.jsonl \
  --dataset smoke-eval \
  --split validation \
  --model synthetic-smoke \
  --quantization Q4_0 \
  --min-record-count 3 \
  --min-token-count 11 \
  --min-dataset-split-record-count 3 \
  --min-dataset-split-token-count 11 \
  --max-nll-delta 0.02 \
  --max-perplexity-ratio 1.05 \
  --max-p95-abs-record-nll-delta 0.05 \
  --max-record-nll-delta 0.10 \
  --fail-on-regression \
  --output-stem perplexity_compare_smoke_latest
```

`perplexity_compare_ci_smoke.py` exercises JSON, Markdown, CSV, breakdown CSV,
regression CSV, JUnit, and coverage-gate failure paths without launching QEMU:

```bash
python3 bench/perplexity_compare_ci_smoke.py
```

## Perf Regression Dashboard

`perf_regression.py` scans host-side benchmark result files and writes dashboards
to `bench/dashboards/`. It accepts JSON, JSONL, and CSV records with `tok_per_s`
or `tok_per_s_milli`, optional `wall_tok_per_s` or `wall_tok_per_s_milli`,
optional `us_per_token` / `wall_us_per_token`, optional first-token latency
fields such as `ttft_us` or `ttft_ms`, optional `host_overhead_pct`, guest
memory fields such as `memory_bytes` or `max_rss_bytes`, optional host RSS
fields such as `host_child_peak_rss_bytes` or `qemu_peak_rss_bytes`, optional
host CPU efficiency fields such as `host_child_tok_per_cpu_s`, and
emitted-token fields such as `tokens`, `output_tokens`, `generated_tokens`, or
`completion_tokens`. It also accepts serial output byte fields such as
`serial_output_bytes`, `serial_bytes`, `output_bytes`, or
`qemu_serial_output_bytes`, and can derive serial bytes per emitted token when
both counters are present.
Regression checks compare
commit-level aggregates, so repeated runs and duplicate latest/stamped result
files are collapsed by benchmark key and commit before the latest distinct
commits are compared. Outputs include JSON, Markdown, JUnit XML, commit-point
CSV, baseline/candidate comparison CSV, regression CSV, sample-coverage CSV,
commit-coverage CSV, comparison-coverage CSV, prompt-suite drift CSV,
environment-coverage CSV, telemetry-coverage CSV, tok/s variability CSV, and
wall tok/s variability CSV artifacts.

Example:

```bash
python3 bench/perf_regression.py --input bench/results --output-dir bench/dashboards
```

CI can fail on median throughput, low-tail guest or host wall-clock throughput,
median or P95 guest/wall microseconds per token, emitted-token drops, median or P95
first-token latency, QEMU host overhead, guest memory, or host child peak RSS
or tok/CPU-second regressions with:

```bash
python3 bench/perf_regression.py \
  --max-tok-cv-pct 7.5 \
  --max-wall-tok-cv-pct 7.5 \
  --p05-tok-regression-pct 7.5 \
  --wall-tok-regression-pct 7.5 \
  --p05-wall-tok-regression-pct 7.5 \
  --us-per-token-regression-pct 7.5 \
  --p95-us-per-token-regression-pct 7.5 \
  --wall-us-per-token-regression-pct 7.5 \
  --p95-wall-us-per-token-regression-pct 7.5 \
  --token-drop-regression-pct 5 \
  --min-token-drop-regression-pct 25 \
  --serial-output-regression-pct 25 \
  --serial-output-per-token-regression-pct 25 \
  --ttft-regression-pct 15 \
  --p95-ttft-regression-pct 15 \
  --host-overhead-regression-pct 25 \
  --host-child-peak-rss-regression-pct 10 \
  --host-child-tok-per-cpu-s-regression-pct 10 \
  --require-us-per-token \
  --require-wall-us-per-token \
  --require-tokens \
  --require-serial-output-bytes \
  --require-serial-output-bytes-per-token \
  --require-ttft-us \
  --require-host-overhead-pct \
  --require-environment-sha256 \
  --require-host-platform \
  --require-host-machine \
  --require-qemu-version \
  --require-qemu-bin \
  --require-host-child-peak-rss \
  --require-host-child-tok-per-cpu-s \
  --fail-on-regression
```

Pin an explicit baseline/candidate pair when CI provides known SHAs:

```bash
python3 bench/perf_regression.py \
  --baseline-commit "$BASE_SHA" \
  --candidate-commit "$GITHUB_SHA" \
  --fail-on-regression
```

`qemu_serial_endpoint_audit.py` checks saved QEMU benchmark command telemetry
for stdio-only serial routing. It rejects socket/TCP/UDP serial, monitor, QMP,
gdb, and chardev endpoints and, by default, requires `-serial stdio`,
`-serial mon:stdio`, or `-nographic`:

```bash
python3 bench/qemu_serial_endpoint_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_serial_endpoint_audit_latest \
  --require-top-command
```

`qemu_display_policy_audit.py` checks saved QEMU benchmark command telemetry
for repeatable headless display routing. It requires `-display none` or
`-nographic`, and rejects GUI/remote display backends such as GTK, SDL, Cocoa,
VNC, and SPICE:

```bash
python3 bench/qemu_display_policy_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_display_policy_audit_latest \
  --require-top-command
```

`qemu_stdio_hygiene_audit.py` checks saved QEMU benchmark artifacts for noisy
OK rows, silent failures, byte-counter drift, and bounded stdout/stderr/tail
payloads before dashboards ingest them:

```bash
python3 bench/qemu_stdio_hygiene_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_stdio_hygiene_audit_latest \
  --max-stdout-bytes 1048576 \
  --max-stderr-bytes 65536 \
  --max-stdout-tail-bytes 8192 \
  --max-stderr-tail-bytes 8192
```

`qemu_artifact_secret_audit.py` scans saved QEMU benchmark artifacts for
high-confidence secret-like text in commands, captured stdio tails, failure
reasons, and environment maps before result bundles are retained or uploaded.
It emits JSON, Markdown, CSV, findings CSV, and JUnit sidecars:

```bash
python3 bench/qemu_artifact_secret_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_artifact_secret_audit_latest
```

`qemu_artifact_secret_audit_ci_smoke.py` exercises clean artifacts plus
OpenAI/GitHub token, credential-bearing URL, and sensitive-field failure paths:

```bash
python3 bench/qemu_artifact_secret_audit_ci_smoke.py
```

`qemu_artifact_network_text_audit.py` scans saved QEMU benchmark artifacts for
network URL, QEMU endpoint, IP:port, and network-keyword text in retained
commands, stdio tails, failure reasons, metadata, and retained CSV/Markdown/XML
sidecars. Endpoint matches fail; standalone network keywords are warnings unless
`--fail-on-keywords` is set:

```bash
python3 bench/qemu_artifact_network_text_audit.py bench/results \
  --output-dir bench/results \
  --output-stem qemu_artifact_network_text_audit_latest
```

Its smoke gate exercises clean artifacts, clean text sidecars, keyword-only
warnings, JSON endpoint failures, and text-sidecar endpoint failures:

```bash
python3 bench/qemu_artifact_network_text_audit_ci_smoke.py
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
