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

## Binary Format

- Header: `<8sHHII32s`
- Magic: `HCEVAL1\0`
- Version: `1`
- Metadata: UTF-8 JSON with `format`, `dataset`, `split`, `record_count`, `version`
- Records: `<IIIIII>` header with byte lengths, choice count, answer index, flags,
  followed by UTF-8 `id`, `prompt`, `provenance`, and each u32-prefixed choice.

All integers are little-endian. The manifest records the source digest, binary
digest, answer histogram, UTF-8 prompt/choice/record byte statistics, and cleaned
records.

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
  --max-prompt-bytes 4096 \
  --max-choice-bytes 1024
```

Use the optional byte gates to reject local subsets that exceed the fixed buffer
budgets chosen for a TempleOS loader before the binary reaches the guest.

## Artifact Index

Summarize curated JSONL manifests, packed binary manifests, and inspection
reports into one local provenance/hash rollup:

```bash
python3 bench/dataset_index.py \
  --input bench/results/datasets \
  --output-dir bench/results/datasets \
  --fail-on-findings
```

## Provenance

The committed sample is synthetic and hand-written for packer validation only. For
real HellaSwag, ARC, or TruthfulQA subsets, store the original dataset name,
version or release, split, filtering criteria, and license note in each row's
`provenance` field or in a companion README beside the JSONL source.
