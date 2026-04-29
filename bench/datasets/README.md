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
  --junit bench/results/datasets/dataset_schema_audit_smoke_latest_junit.xml \
  --require-provenance \
  --min-choices 4 \
  --max-choices 4 \
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024 \
  --max-record-payload-bytes 8192 \
  --max-majority-answer-pct 80 \
  --max-dataset-split-majority-answer-pct 90 \
  --fail-on-duplicate-ids \
  --fail-on-duplicate-payloads \
  --fail-on-conflicting-payload-answers \
  --fail-on-findings
```

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
with choice byte ranges, duplicate/overlap counts, label-prefix counts, and
prompt leak telemetry for quick review before packing.

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
  --findings-csv bench/results/datasets/dataset_order_audit_smoke_latest_findings.csv \
  --junit bench/results/datasets/dataset_order_audit_smoke_latest_junit.xml \
  --group-by overall \
  --max-longest-answer-run 3 \
  --max-longest-answer-run-pct 100 \
  --max-edge-answer-run 3 \
  --min-answer-switches 0 \
  --fail-on-findings
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

## Provenance Audit

Run the focused provenance audit on curated manifests before promoting an eval
subset. It verifies source/license metadata, source and normalized hashes,
selected record IDs, count rollups, pack outputs, non-empty row provenance, and
overall/per-dataset/per-split/per-dataset-split answer histograms. Curated
manifests also record per-provenance contribution counts, and
`--max-provenance-pct` can fail CI when one staged local source dominates a
compact subset.
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
```

## Provenance

The committed sample is synthetic and hand-written for packer validation only. For
real HellaSwag, ARC, or TruthfulQA subsets, store the original dataset name,
version or release, split, filtering criteria, and license note in each row's
`provenance` field or in a companion README beside the JSONL source.
