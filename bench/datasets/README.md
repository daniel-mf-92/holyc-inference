# Offline Eval Dataset Packing

`bench/dataset_pack.py` converts local JSONL evaluation records into a deterministic
HolyC-loadable multiple-choice binary. It is intentionally offline: datasets must
be placed on disk with provenance notes before packing.

## Supported Input Shapes

- Normalized: `id`, `dataset`, `split`, `prompt`, `choices`, `answer_index`.
- HellaSwag-style: `ctx`, `endings`, `label`.
- ARC-style: `question`, `choices` objects with `label`/`text`, `answerKey`.
- TruthfulQA-style: `question`, `mc1_targets.choices`, `mc1_targets.labels`.

## Curation

Run `bench/dataset_schema_audit.py` before curation when staging new local
sources. It verifies that every row can normalize into the packer schema, emits
dataset/split counts, answer and choice histograms, duplicate ID warnings, UTF-8
byte telemetry, and can enforce provenance, answer-skew, and fixed loader-size
budgets:

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
  --max-majority-answer-pct 100 \
  --max-dataset-split-majority-answer-pct 100 \
  --fail-on-duplicate-ids \
  --fail-on-duplicate-payloads \
  --fail-on-conflicting-payload-answers \
  --fail-on-findings
```

The optional `--record-csv` export writes one row per normalized eval record
with prompt bytes, choice bytes, record payload bytes, answer index, and a
stable prompt+choices payload hash for review before packing.
Use `--min-answer-labels` and `--min-dataset-split-answer-labels` when a
promoted subset must exercise at least N answer indexes overall or within every
dataset/split bucket.

Run `bench/dataset_choice_audit.py` after schema validation when reviewing a
local multiple-choice subset for option-quality issues. It flags duplicate
choice text inside a row, choices where one normalized option contains another,
choices that still carry label prefixes such as `A.`/`B)`, prompts that contain
the correct choice text or any candidate choice text, and large choice-length
skew:

```bash
python3 bench/dataset_choice_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_choice_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_choice_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_choice_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_choice_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_choice_audit_smoke_latest_junit.xml \
  --fail-on-duplicate-choices \
  --fail-on-choice-overlap \
  --fail-on-label-prefixes \
  --fail-on-prompt-answer-leak \
  --fail-on-prompt-choice-leak \
  --max-choice-length-ratio 100 \
  --fail-on-length-skew \
  --fail-on-findings
```

The optional `--record-csv` export writes one row per normalized eval record
with choice byte ranges, correct-answer byte share, duplicate/overlap counts,
label-prefix counts, and prompt leak telemetry for quick review before packing.

Run `bench/dataset_text_audit.py` before packing promoted subsets when prompt
and choice strings need loader-safe byte telemetry. It flags blank normalized
text, disallowed C0 controls, Unicode replacement markers, and prompt/choice or
line byte-budget overruns:

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
  --fail-on-findings
```

Run `bench/dataset_provenance_balance_audit.py` before packing promoted subsets
that combine multiple staged local sources. It checks non-empty provenance,
required source coverage, minimum source cardinality, and dominant provenance
share overall or within each dataset/split bucket:

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

Run `bench/dataset_split_overlap_audit.py` before packing promoted subsets that
mix train/dev/test style splits. It hashes normalized prompts and prompt+choice
payloads, then flags rows reused across splits within the same dataset, with an
optional global scope for cross-dataset suites:

```bash
python3 bench/dataset_split_overlap_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_split_overlap_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_split_overlap_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_split_overlap_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_split_overlap_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_split_overlap_audit_smoke_latest_junit.xml \
  --fail-on-prompt-overlap \
  --fail-on-payload-overlap \
  --fail-on-findings
```

Run `bench/dataset_duplicate_audit.py` before packing promoted subsets when
exact repeated examples should be reviewed within a split, dataset, or full
suite. It hashes normalized prompts and prompt+choice payloads, emits
per-record duplicate counts, and can fail duplicate groups whose answer indexes
conflict:

```bash
python3 bench/dataset_duplicate_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_duplicate_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_duplicate_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_duplicate_audit_smoke_latest_findings.csv \
  --record-csv bench/results/datasets/dataset_duplicate_audit_smoke_records_latest.csv \
  --junit bench/results/datasets/dataset_duplicate_audit_smoke_latest_junit.xml \
  --scope dataset_split \
  --fail-on-duplicate-prompts \
  --fail-on-duplicate-payloads \
  --fail-on-conflicting-answers \
  --fail-on-findings
```

Run `bench/dataset_split_balance_audit.py` before packing mixed-split subsets
when CI should enforce split coverage and prevent one split from dominating a
dataset. It emits per-dataset and per-dataset/split CSV sidecars plus JSON,
Markdown, and JUnit reports:

```bash
python3 bench/dataset_split_balance_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output-dir bench/results/datasets \
  --output-stem dataset_split_balance_audit_smoke_latest \
  --require-split validation \
  --require-dataset-split arc-smoke:validation \
  --min-records 3 \
  --max-largest-split-pct 100
```

Run `bench/dataset_answer_bias_audit.py` after choice-quality review to catch
multiple-choice subsets where the correct option is systematically the longest
or shortest answer. It emits aggregate and dataset/split answer-position
histograms plus per-record answer/distractor byte telemetry:

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

Use the longest/shortest percentage gates and mean answer/distractor byte-ratio
gates before promoting a local subset whose answer text length could leak the
label independent of model quality.

Run `bench/dataset_id_audit.py` before packing promoted subsets when stable
record IDs matter for reproducible manifests and HolyC eval artifacts. It
emits duplicate-ID rollups plus per-record ID telemetry, and can require
explicit lowercase slug IDs scoped by dataset and split:

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

Run `bench/dataset_order_audit.py` before packing promoted subsets when source
row order could bias evaluation. It records answer-index sequences, transition
counts, leading/trailing runs, and longest same-answer runs by overall dataset,
dataset, or dataset/split scope. The gates catch long blocks of identical answer
positions before they become HCEval binaries:

```bash
python3 bench/dataset_order_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_order_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_order_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_order_audit_smoke_latest.csv \
  --record-csv bench/results/datasets/dataset_order_audit_smoke_records_latest.csv \
  --findings-csv bench/results/datasets/dataset_order_audit_smoke_latest_findings.csv \
  --junit bench/results/datasets/dataset_order_audit_smoke_latest_junit.xml \
  --group-by overall \
  --max-longest-answer-run 3 \
  --max-longest-answer-run-pct 100 \
  --max-edge-answer-run 3 \
  --min-answer-switches 0 \
  --fail-on-findings
```

Use `--record-csv` to emit the source-order answer sequence with previous/next
answer indexes and run membership for diagnosing long-block or low-transition
gate failures.

Run `bench/dataset_contamination_audit.py` on mixed eval suites to catch local
cross-dataset contamination before packing. It detects normalized prompt reuse,
prompt+choice payload reuse, and conflicting answer labels across dataset
families without fetching remote data:

```bash
python3 bench/dataset_contamination_audit.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/dataset_contamination_audit_smoke_latest.json \
  --markdown bench/results/datasets/dataset_contamination_audit_smoke_latest.md \
  --csv bench/results/datasets/dataset_contamination_audit_smoke_latest.csv \
  --junit bench/results/datasets/dataset_contamination_audit_smoke_latest_junit.xml \
  --fail-on-contamination
```

Run `bench/dataset_fingerprint.py` when a curated JSONL needs stable prompt,
choice, and prompt+choice input hashes before packing or before collecting
HolyC/llama.cpp predictions. The hashes match the metadata expected by
`bench/eval_input_audit.py`:

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
```

Compare two fingerprint reports with `bench/dataset_fingerprint_diff.py` and
archive both row-level changes and structured gate findings:

```bash
python3 bench/dataset_fingerprint_diff.py \
  --baseline bench/results/datasets/dataset_fingerprint_smoke_latest.json \
  --candidate bench/results/datasets/dataset_fingerprint_smoke_latest.json \
  --output bench/results/datasets/dataset_fingerprint_diff_smoke_latest.json \
  --csv bench/results/datasets/dataset_fingerprint_diff_smoke_latest.csv \
  --findings-csv bench/results/datasets/dataset_fingerprint_diff_smoke_findings_latest.csv \
  --fail-on-any-change \
  --fail-on-findings
```

Use `bench/dataset_curate.py` to make a reproducible subset from locally staged
source JSONL before packing. The curator normalizes rows through the packer, can
filter by dataset or split, samples with a stable SHA-256 key, rejects duplicate
record ids, and writes a manifest with source hashes and provenance fields.

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

Add `--balance-answer-index` when capping a larger local source to round-robin
the deterministic sample across answer labels. The manifest records the flag and
the resulting answer histogram for downstream eval reproducibility checks.
Use `--max-records-per-dataset-split` when combining multiple datasets and
splits so one large dataset/split pair cannot dominate the curated subset.
Use `--max-prompt-bytes`, `--max-choice-bytes`, and
`--max-record-payload-bytes` during curation when you want oversized rows
dropped before deterministic sampling instead of rejected later by the packer.

## Binary Format

- Header: `<8sHHII32s`
- Magic: `HCEVAL1\0`
- Version: `1`
- Metadata: UTF-8 JSON with `format`, `dataset`, `split`, `record_count`, `version`
- Records: `<IIIIII>` header with byte lengths, choice count, answer index, flags,
  followed by UTF-8 `id`, `prompt`, `provenance`, and each u32-prefixed choice.

All integers are little-endian. The manifest records the source digest, binary
digest, answer histogram, UTF-8 prompt/choice/record byte statistics, aggregate
binary layout byte counts, record binary spans, binary-recoverable record
fingerprints, and cleaned records.

## Sample Build

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

## Binary Inspection

Validate packed binaries and their manifests before loading them in TempleOS:

```bash
python3 bench/hceval_inspect.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --output bench/results/datasets/smoke_eval.inspect.json \
  --markdown bench/results/datasets/smoke_eval.inspect.md \
  --csv bench/results/datasets/smoke_eval.inspect.csv \
  --fingerprints-csv bench/results/datasets/smoke_eval.inspect.fingerprints.csv \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024
```

Use the optional byte gates to reject local subsets that exceed the fixed buffer
budgets chosen for a TempleOS loader before the binary reaches the guest. The
inspection report also emits parsed `binary_layout`, `record_spans`, and
`record_fingerprints`, can write spans as CSV for offset diffs, and checks them
against the manifest when present.

## Artifact Index

Summarize curated JSONL manifests, packed binary manifests, and inspection
reports into one local provenance/hash rollup:

```bash
python3 bench/dataset_index.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --require-artifact-type curated_manifest \
  --require-artifact-type pack_manifest \
  --require-artifact-type inspect_report \
  --fail-on-coverage \
  --fail-on-findings
```

Use the artifact-type coverage gate when promoting packed eval bundles so CI
fails if the curated JSONL manifest, packed `.hceval` manifest, or binary
inspection report is absent from the local artifact set.
Run `python3 bench/dataset_index_ci_smoke.py` for a focused stdlib-only smoke
gate that checks the passing rollup plus missing artifact-type and dataset/split
coverage failures.

## Provenance Audit

Run the focused provenance audit on curated manifests before promoting an eval
subset. It verifies source/license metadata, source and normalized hashes,
selected record IDs, count rollups, pack outputs, non-empty row provenance, and
overall/per-dataset/per-split/per-dataset-split answer histograms. Curated
manifests also record per-provenance contribution counts, and
`--max-provenance-pct` can fail CI when one staged local source dominates a
compact subset.
The audit also writes
`bench/results/datasets/dataset_provenance_audit_records_latest.csv` with one
row per selected eval record: record ID, dataset, split, provenance contribution
percentage, answer index, byte budgets, and a stable prompt+choices input hash.
Synthetic smoke manifests are allowed to omit `source_url`; real dataset
manifests should record one, and `--require-source-url` turns that policy into a
hard gate. Repeat `--allow-license` or `--deny-license` to enforce exact
case-folded license/usage-note policy values for promoted local subsets. Repeat
`--allow-source-url-scheme`, `--deny-source-url-scheme`,
`--allow-source-url-host`, and `--deny-source-url-host` when promoted manifests
must use reviewed URL schemes or hosts. Use
`--allow-source-url-path-prefix` or `--deny-source-url-path-prefix` to pin
reviewed source URL paths; no URL is fetched.

```bash
python3 bench/dataset_provenance_audit.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --max-provenance-pct 80 \
  --max-dataset-split-majority-answer-pct 90 \
  --allow-license synthetic-smoke \
  --allow-source-url-scheme https \
  --allow-source-url-path-prefix /datasets/ \
  --fail-on-findings
```

For a complete local smoke gate of the dataset toolchain, run:

```bash
python3 bench/dataset_ci_smoke.py
```

## Split Leakage Audit

Before freezing a train/dev/test subset, run the local leakage audit over the
curated JSONL files. It checks duplicate record IDs, normalized prompt reuse
across splits, repeated prompt+choice payloads across splits, and answer
conflicts for identical payloads:

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

## Provenance

The committed sample is synthetic and hand-written for packer validation only. For
real HellaSwag, ARC, or TruthfulQA subsets, store the original dataset name,
version or release, split, filtering criteria, and license note in each row's
`provenance` field or in a companion README beside the JSONL source.
