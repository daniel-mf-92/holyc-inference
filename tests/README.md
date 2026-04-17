# Host-Side GGUF Parity Fixture Plan

This directory contains host-side validation harnesses only. The TempleOS inference runtime remains HolyC-only.

## Scope

- Validate GGUF parser behavior against llama.cpp-compatible expectations.
- Produce deterministic binary fixtures for header, metadata, tensor table, and tensor payload offsets.
- Keep fixture generation and checks fully offline and reproducible.

## Parity Targets

- Header parse parity: magic (`GGUF`), version acceptance window, tensor count, metadata count.
- Tensor table parity: name bytes, dimension array ordering, tensor type code, relative offset semantics.
- Alignment parity: tensor-data base computed from `general.alignment` (or default 32), then `align_up`.
- Range parity: each tensor absolute `[start,end)` span is in-bounds, non-overlapping, and overflow-safe.

## Canonical Binary Fixture Shapes

- `fixtures/gguf/header_valid_v2_minimal.gguf`
- `fixtures/gguf/header_bad_magic.gguf`
- `fixtures/gguf/header_bad_version_unsupported.gguf`
- `fixtures/gguf/header_truncated_u64_counts.gguf`
- `fixtures/gguf/tensorinfo_two_tensors_q4_q8.gguf`
- `fixtures/gguf/tensorinfo_alignment_32_64.gguf`
- `fixtures/gguf/tensorinfo_overflow_offsets.gguf`
- `fixtures/gguf/tensorinfo_overlap_ranges.gguf`

Each fixture intentionally targets exactly one primary invariant so failure diagnostics stay sharp.

## Fixture Schema (documented bytes)

For every fixture, add a sidecar manifest:

- Path: `fixtures/gguf/<name>.json`
- Fields:
  - `description`
  - `gguf_version`
  - `metadata_count`
  - `tensor_count`
  - `general_alignment`
  - `expected_header` (`magic_ok`, `version_ok`, `tensor_count`, `metadata_count`)
  - `expected_tensors` (ordered list of `name`, `dims`, `type`, `rel_offset`, `nbytes`)
  - `expected_ranges_abs` (ordered `[start,end)` pairs)
  - `expected_error` (null on success fixtures)

The sidecar is the single source of truth for host harness assertions.

## Validation Harness Contract

Harnesses in `tests/` must follow this structure:

- Read binary fixture into bytes.
- Parse with tiny, explicit host reference logic matching GGUF spec byte-for-byte.
- Compare parsed values to sidecar manifest.
- Mirror HolyC parser contracts:
  - BAD_MAGIC
  - UNSUPPORTED_VERSION
  - TRUNCATED
  - OVERFLOW
  - OUT_OF_RANGE
  - OVERLAP

No fuzzy checks: every expected integer must match exactly.

## Determinism Rules

- All generated fixture bytes are little-endian.
- No wall-clock timestamps in fixture artifacts.
- If random data is used, seed must be fixed and recorded in sidecar.
- Commit fixture bytes and sidecars together.

## Execution Matrix

Primary fixture validations:

- `python3 tests/test_gguf_header_parse.py`
- `python3 tests/test_gguf_tensorinfo_parse.py`
- `python3 tests/test_gguf_tensor_data_base.py`

Extended parity sweep:

- `python3 tests/test_gguf_metadata_parse.py`

A fixture addition is complete only when all four commands pass.

## Implementation Checklist

- Add or update fixture bytes under `tests/fixtures/gguf/`.
- Add matching sidecar manifest for each fixture.
- Add focused test case with explicit expected contract.
- Run primary matrix commands.
- Record failure surface and fix location in test name/docstring.

## Reference Behavior Notes

- GGUF header uses fixed-width little-endian fields.
- Tensor payload starts at aligned tensor-data base, not immediately after tensor table.
- For quantized tensors (Q4_0/Q8_0), byte-size depends on block geometry and element count; invalid block divisibility must fail fast.

## Non-Goals

- No model downloading.
- No runtime inference implementation in Python.
- No networking or remote fixture fetches.
