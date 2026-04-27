# Offline Eval Dataset Packing

`bench/dataset_pack.py` converts local JSONL evaluation records into a deterministic
HolyC-loadable multiple-choice binary. It is intentionally offline: datasets must
be placed on disk with provenance notes before packing.

## Supported Input Shapes

- Normalized: `id`, `dataset`, `split`, `prompt`, `choices`, `answer_index`.
- HellaSwag-style: `ctx`, `endings`, `label`.
- ARC-style: `question`, `choices` objects with `label`/`text`, `answerKey`.
- TruthfulQA-style: `question`, `mc1_targets.choices`, `mc1_targets.labels`.

## Binary Format

- Header: `<8sHHII32s`
- Magic: `HCEVAL1\0`
- Version: `1`
- Metadata: UTF-8 JSON with `format`, `dataset`, `split`, `record_count`, `version`
- Records: `<IIIIII>` header with byte lengths, choice count, answer index, flags,
  followed by UTF-8 `id`, `prompt`, `provenance`, and each u32-prefixed choice.

All integers are little-endian. The manifest records the source digest, binary
digest, answer histogram, and cleaned records.

## Sample Build

```bash
python3 bench/dataset_pack.py \
  --input bench/datasets/samples/smoke_eval.jsonl \
  --output bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --dataset smoke-eval \
  --split validation
```

## Binary Inspection

Validate packed binaries and their manifests before loading them in TempleOS:

```bash
python3 bench/hceval_inspect.py \
  --input bench/results/datasets/smoke_eval.hceval \
  --manifest bench/results/datasets/smoke_eval.manifest.json \
  --output bench/results/datasets/smoke_eval.inspect.json \
  --markdown bench/results/datasets/smoke_eval.inspect.md
```

## Provenance

The committed sample is synthetic and hand-written for packer validation only. For
real HellaSwag, ARC, or TruthfulQA subsets, store the original dataset name,
version or release, split, filtering criteria, and license note in each row's
`provenance` field or in a companion README beside the JSONL source.
