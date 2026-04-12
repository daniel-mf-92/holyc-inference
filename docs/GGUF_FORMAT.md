# GGUF Format Reference (HolyC Implementation Contract)

This is the parser contract for the TempleOS HolyC runtime in `src/gguf/*.HC`.
It is intentionally concrete: every field, offset rule, and bounds check here maps
straight to explicit integer code.

Scope of this reference (current runtime):
- Parse GGUF header (`magic`, `version`, counts).
- Parse metadata key/value table with scalar, string, and array payloads.
- Parse tensor directory records (name, dims, type, relative offset).
- Resolve aligned tensor data base and absolute tensor byte ranges.
- Support tensor storage sizing for `F32`, `F16`, `Q4_0`, `Q8_0`.
- Keep parser integer-only and fully auditable (no float math required).

## 1) Container Layout

GGUF file sections are read strictly in order:
1. Header
2. Metadata key/value table
3. Tensor info table
4. Tensor data region (aligned)

No out-of-band index is required. A single monotonic cursor can parse the file.

## 2) Header (Fixed 24 Bytes)

Binary layout (little-endian):
- `magic: U32`
- `version: U32`
- `tensor_count: U64`
- `metadata_kv_count: U64`

TempleOS runtime constants in `src/gguf/header.HC`:
- `GGUF_MAGIC_U32 = 0x46554747`
- supported versions: `2`, `3`

Header parse invariants:
- Reject non-GGUF magic.
- Reject unsupported version.
- Reject `tensor_count == 0`.
- Keep cursor bounds-checked before every field read.

## 3) Primitive Encodings

All scalar reads use explicit LE helpers:
- `GGUFReadU32LE`, `GGUFReadU64LE` for header.
- Metadata/tensor readers have analogous per-width helpers.

String encoding:
- `len: U64`
- `bytes[len]` (not null-terminated on disk)

HolyC runtime behavior:
- Allocate `len + 1` bytes, copy payload, append `\0`.
- Enforce hard caps for malformed files.

Current parser caps (`src/gguf/metadata.HC`):
- `GGUF_MAX_METADATA_COUNT = 1 << 20`
- `GGUF_MAX_STRING_BYTES = 1 << 20`
- `GGUF_MAX_ARRAY_ELEMS = 1 << 24`

## 4) Metadata Table

Each metadata entry:
1. key string
2. `value_type: U32` (`gguf_type`)
3. value payload (type-dependent)

Supported `gguf_type` IDs:
- `0 UINT8`
- `1 INT8`
- `2 UINT16`
- `3 INT16`
- `4 UINT32`
- `5 INT32`
- `6 FLOAT32` (stored as raw bits, not float math)
- `7 BOOL`
- `8 STRING`
- `9 ARRAY`
- `10 UINT64`
- `11 INT64`
- `12 FLOAT64` (stored as raw bits)

Array payload format:
- `elem_type: U32`
- `array_len: U64`
- `elem_payload[array_len]`

Safety policy:
- Reject nested arrays.
- Reject unknown value types.
- Reject truncation at any point.
- Preserve float payload bits as integers; runtime stays integer-only.

Runtime representation:
- `GGUFMetadataTable` owns an array of `GGUFMetadataKV`.
- Each value stores scalar slots or lazy array payload offsets/byte spans.
- Lookup helpers (`GGUFMetaFindByKey`, typed extractors) are exact-key and type-strict.

## 5) Tensor Info Table

Each tensor record:
1. tensor name string
2. `n_dims: U32`
3. `dims[n_dims]: U64`
4. `ggml_type: U32`
5. `offset: U64` (relative to aligned tensor data base)

Parser invariants:
- `n_dims` must be positive and within parser max.
- each dimension must be non-zero.
- element-count multiplication must be overflow-checked.
- `ggml_type` must be known for byte-size resolution.

Runtime currently requires sizing rules for:
- `GGML_TYPE_F32` (`0`)
- `GGML_TYPE_F16` (`1`)
- `GGML_TYPE_Q4_0` (`2`)
- `GGML_TYPE_Q8_0` (`8`)

## 6) Tensor Type Block Math

Type sizing is block-based for quantized tensors.

Per-type block elements:
- `F32`: block size `1`
- `F16`: block size `1`
- `Q4_0`: block size `32`
- `Q8_0`: block size `32`

Per-type block bytes:
- `F32`: `4`
- `F16`: `2`
- `Q4_0`: `18` (`fp16 scale + 16 packed nibbles`)
- `Q8_0`: `34` (`fp16 scale + 32 signed bytes`)

Tensor payload sizing contract:
1. `elem_count % block_size == 0` (else error)
2. `block_count = elem_count / block_size`
3. `nbytes = block_count * block_bytes` with overflow checks

This is implemented in `src/gguf/tensor_data_base.HC` via:
- `GGUFTensorTypeBlockSize`
- `GGUFTensorTypeBlockBytes`
- `GGUFTensorBytesForType`

## 7) Alignment Rules

After metadata + tensor info parsing, compute aligned data base:
- `tensor_data_base = AlignUp(cursor_after_tensor_info, alignment)`

Default alignment constant:
- `GGUF_DEFAULT_ALIGNMENT = 32`

Alignment validity:
- alignment must be a non-zero power-of-two.
- base and every tensor relative offset must be alignment-aligned.

Absolute tensor payload start:
- `tensor_abs = tensor_data_base + tensor_rel_offset`

All additions are overflow-checked on `U64`.

## 8) Absolute Range Resolution

Tensor byte span is half-open interval:
- `[abs_start, abs_end)`
- `abs_start = tensor_abs`
- `abs_end = abs_start + tensor_nbytes`

Range validity checks:
- no arithmetic overflow while building `abs_end`
- `abs_end <= gguf_file_nbytes`

Overlap validity for tensor table:
- two ranges overlap iff:
  - `start_a < end_b` and `start_b < end_a`
- touching boundaries (`end_a == start_b`) are legal.

Implemented helpers in `src/gguf/tensor_data_base.HC`:
- `GGUFTensorResolveRange`
- `GGUFValidateTensorRanges`
- `GGUFValidateTensorRangesSorted`
- lookup/index helpers over sorted `[start,end)` spans

## 9) Error Model (Deterministic, Integer)

Parser and tensor-range helpers return explicit integer error codes.
Representative failures:
- null pointer arguments
- truncated read
- bad magic/version
- bad counts/string lengths/array lengths
- unknown metadata/tensor type
- misaligned base/offset
- add/mul overflow
- out-of-bounds tensor payload
- overlapping tensor ranges

No exceptions, no hidden control flow. Every failure path is a branch you can audit.

## 10) HolyC API Shape

Current API surface (trimmed):
- Header: `GGUFParseHeader(...)`
- Metadata: `GGUFParseMetadataTable(...)`, key/value extractors
- Tensor info: tensor directory parse helpers in `src/gguf/tensorinfo.HC`
- Data base/ranges: `GGUFTensorDataBaseAlign*`, `GGUFTensorResolveRange*`, validators, lookup/index helpers

Design rule: direct loops + explicit arithmetic > abstract indirection.

## 11) Validation Contract (Host-Side Only)

Validation scripts in `tests/` mirror HolyC parser math and error rules.
They verify:
- header field parsing and rejection cases
- metadata scalar/string/array decoding and type checks
- tensor byte-size calculations for supported types
- alignment, range, and overlap behavior parity
- offset lookup/index invariants

Host tools are for parity checking only; inference runtime remains pure HolyC.

## 12) Air-Gap Policy Reminder

GGUF parser/runtime is disk-only.
- No HTTP/client stack
- No model downloading
- No socket usage
- No network dependency for parsing or inference

TempleOS VM runs must keep NIC disabled (`-nic none`, fallback `-net none`).
