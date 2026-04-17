# GGUF Format Reference (HolyC Parser Contract)

This document is the implementation contract for TempleOS HolyC GGUF parsing.
Every byte rule here maps to explicit integer code paths in `src/gguf/*.HC`.

Runtime philosophy:
- deterministic integer-only parsing
- checked arithmetic for every offset/length/count operation
- no hidden control flow, no external dependencies, no networking

## 1) File Skeleton

GGUF is parsed with one monotonic cursor:
1. header
2. metadata table
3. tensor info table
4. aligned tensor payload region

No random seeks are required for the structural pass.

## 2) Header Section

Binary layout (little-endian, fixed 24 bytes):
- `magic: U32`
- `version: U32`
- `tensor_count: U64`
- `metadata_kv_count: U64`

HolyC constants:
- `GGUF_MAGIC_U32 = 0x46554747`
- supported versions: `2`, `3`

Header invariants:
- reject bad magic
- reject unsupported version
- reject `tensor_count == 0`
- reject truncated 24-byte header
- reject counts above parser caps

## 3) Primitive Encoding Rules

All scalar values are little-endian.

String encoding:
- `len: U64`
- `len` raw bytes (not null-terminated on disk)

HolyC parser behavior:
- allocate `len + 1`
- copy payload and append `0`
- reject `len` beyond configured caps

Checked helpers expected in parser paths:
- `CheckedAddU64`
- `CheckedMulU64`
- `CheckedAlignUpU64`
- `ValidateRange(offset, length, file_size)`

## 4) Metadata Section

Each metadata KV entry encodes:
1. key string
2. `value_type: U32` (`gguf_type`)
3. value payload (shape depends on type)

Supported scalar type IDs:
- `0 UINT8`
- `1 INT8`
- `2 UINT16`
- `3 INT16`
- `4 UINT32`
- `5 INT32`
- `6 FLOAT32` (stored/parity-checked as raw bits)
- `7 BOOL`
- `8 STRING`
- `9 ARRAY`
- `10 UINT64`
- `11 INT64`
- `12 FLOAT64` (stored/parity-checked as raw bits)

Array payload layout:
- `elem_type: U32`
- `array_len: U64`
- contiguous `array_len` elements

Metadata guard rules:
- reject unknown type IDs
- reject nested arrays
- reject key collisions only when callers require uniqueness (parser itself may preserve order)
- reject truncation at any element boundary
- keep cursor monotonic and range-checked

Recommended parser caps (hostile-file hardening):
- `MAX_METADATA_COUNT`
- `MAX_STRING_BYTES`
- `MAX_ARRAY_ELEMS`

## 5) Tensor Info Section

Each tensor descriptor encodes:
1. name string
2. `n_dims: U32`
3. `dims[n_dims]: U64`
4. `ggml_type: U32`
5. `offset: U64` (relative to aligned tensor payload base)

Tensor info invariants:
- `n_dims > 0`
- each dimension is non-zero
- checked product for total element count
- type must be recognized by byte-size resolver
- relative offset arithmetic must not overflow

## 6) Quant Block Math (Sizing Contract)

Supported type sizing for current runtime:
- `F32`: block size `1`, block bytes `4`
- `F16`: block size `1`, block bytes `2`
- `Q4_0`: block size `32`, block bytes `18` (`fp16 scale + 16 packed nibbles`)
- `Q8_0`: block size `32`, block bytes `34` (`fp16 scale + 32 signed i8`)

Tensor byte-size algorithm:
1. require `elem_count % block_size == 0`
2. `block_count = elem_count / block_size`
3. `tensor_nbytes = block_count * block_bytes` (checked multiply)

Failure if any step overflows or divisibility fails.

## 7) Alignment Section

After metadata + tensor descriptors, compute tensor payload base:
- `tensor_data_base = AlignUp(cursor_after_tensor_info, alignment)`

Current default alignment:
- `32`

Alignment invariants:
- alignment is non-zero power-of-two
- aligned base is representable in `U64`
- tensor relative offsets satisfy required alignment

## 8) Absolute Tensor Range Resolution

Per tensor:
- `abs_start = tensor_data_base + rel_offset`
- `abs_end = abs_start + tensor_nbytes`
- range is half-open `[abs_start, abs_end)`

Guard rules:
- checked add on both additions
- `abs_end <= file_size`
- no overlap in validated tensor tables unless caller explicitly allows it

Overlap predicate:
- overlap exists iff `(start_a < end_b) && (start_b < end_a)`
- touching boundaries are valid (`end_a == start_b`)

## 9) Deterministic Error Surface

Expose integer status codes only.

Expected classes:
- `BAD_PARAM` (null pointers, illegal args)
- `BAD_MAGIC` / `BAD_VERSION`
- `BAD_COUNT` / `BAD_LEN`
- `TRUNCATED`
- `UNKNOWN_TYPE`
- `MISALIGNED`
- `OVERFLOW`
- `OUT_OF_BOUNDS`
- `OVERLAP`

Design requirement: scalar and any optimized paths return identical error classes for identical invalid inputs.

## 10) Parser Walkthrough (Reference Order)

1. parse header
2. preflight metadata/tensor counts against caps
3. parse metadata table with strict cursor checks
4. parse tensor info records + precompute `elem_count` and `nbytes`
5. compute aligned tensor payload base
6. resolve absolute tensor ranges
7. validate range bounds/overlap invariants
8. expose lookup helpers by tensor name and absolute offset

## 11) Host-Side Validation Matrix

Validation scripts in `tests/` should cover:
- header: valid, bad magic, unsupported version, truncation
- metadata: scalar/string/array decoding and nested-array rejection
- tensor info: dim overflow, zero dim rejection, unknown type rejection
- sizing: block-divisibility and block-bytes parity for F32/F16/Q4_0/Q8_0
- alignment: non-power-of-two and overflowed align-up rejection
- ranges: out-of-bounds and overlap adversarial cases near `U64_MAX`

Host validation is parity tooling only; runtime remains HolyC-only.

## 12) Air-Gap Constraint

GGUF parser and inference runtime are disk-only.
No sockets, HTTP, model downloaders, or VM guest networking.
TempleOS guest runs must keep NIC disabled (`-nic none`; fallback `-net none`).
