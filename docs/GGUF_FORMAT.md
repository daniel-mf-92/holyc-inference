# GGUF Format Reference (for HolyC Parser)

This document defines the subset of GGUF required by the TempleOS HolyC
inference runtime. It is written as a direct implementation guide for
`src/gguf/*.HC`.

Scope for first implementation pass:
- Parse container header and version.
- Parse metadata key/value table.
- Parse tensor directory (name, shape, type, offset).
- Compute absolute tensor data addresses with alignment rules.
- Load quantized tensor bytes for Q4_0 and Q8_0 paths.

## 1) File Layout (High Level)

GGUF file layout is:
1. Header
2. Metadata key/value entries
3. Tensor info entries
4. Tensor data region

The parser must read sections in-order; no external index is required.

## 2) Header

The header contains:
- `magic`: 4-byte ASCII signature identifying GGUF.
- `version`: unsigned integer format version.
- `tensor_count`: number of tensor info records.
- `metadata_kv_count`: number of metadata entries.

HolyC parser requirements:
- Reject file if magic is not GGUF signature.
- Reject unsupported versions explicitly.
- Reject zero/negative-equivalent counts after bounds checks.
- Use 64-bit counters/offset math to avoid overflow on large models.

## 3) Core Primitive Types

Use fixed-width decoding helpers in HolyC:
- `U8`, `I8`
- `U16`, `I16`
- `U32`, `I32`
- `U64`, `I64`

Strings are stored as:
- `len: U64`
- `bytes[len]` (not null-terminated in file)

Parser behavior:
- Allocate `len + 1` bytes in memory and append `\0` for HolyC string use.
- Reject unreasonably large lengths with a hard cap.

## 4) Metadata Table

Each metadata record is:
1. Key string
2. Value type tag (`gguf_type`)
3. Value payload (layout depends on type)

Common value types to support early:
- Unsigned/signed integers
- Boolean
- String
- Array (including array of strings and numerics)

TempleOS runtime only needs a practical subset for LLaMA-family inference,
but parser should safely skip/consume unknown metadata value kinds if
encountered in future files.

Recommended metadata keys used by inference setup (model-dependent):
- Architecture identifier
- Context length
- Embedding size
- Layer count
- Attention head counts
- Rope parameters
- Tokenizer and vocab-related keys

Implementation rule:
- Parse all entries into an internal table.
- Allow key lookup by exact string for later subsystems.

## 5) Tensor Info Table

Each tensor info entry contains:
1. Tensor name (string)
2. `n_dims` (dimension count)
3. `dims[n_dims]` (typically U64 extents)
4. `ggml_type` (tensor storage / quantization type)
5. `offset` (relative offset into tensor data region)

HolyC-side checks:
- `n_dims` must be within parser maximum.
- Dimensions must be non-zero.
- Product of dims must not overflow 64-bit during size checks.
- Type must be known or explicitly marked unsupported.

Store tensor info in a simple linear array for predictable traversal.

## 6) Tensor Data Region and Alignment

After metadata and tensor info tables, the file position is rounded up to a
tensor alignment boundary. Tensor byte payloads start at:
- `tensor_data_base = AlignUp(current_pos, general_alignment)`

Each tensor info `offset` is relative to `tensor_data_base`.
Absolute tensor address:
- `tensor_abs = tensor_data_base + tensor.offset`

Parser must:
- Use 64-bit math for all offset calculations.
- Validate `tensor_abs + tensor_size <= file_size` before reads.
- Reject overlapping/out-of-range tensor payloads.

## 7) Quantization Type Handling

For this project’s first runtime target:
- Required: `Q4_0`, `Q8_0`
- Optional/unsupported in first pass: other GGML quantization kinds

Design rule:
- Keep type enum mapping explicit in HolyC (switch by numeric type ID).
- Fail fast with clear error when model requires unsupported tensor type.

## 8) Endianness

Assume little-endian host/guest for TempleOS x86_64.

Still implement reads through byte-wise decode helpers rather than blind
pointer casting. This keeps parsing auditable and guards against alignment
fault assumptions.

## 9) Safety Rules for HolyC Parser

- No floating-point dependence in parser logic.
- No external libraries.
- Bounds-check every read before consuming bytes.
- Centralize read cursor updates in helper functions.
- Return explicit error code + message for each failure class.

Suggested failure classes:
- Bad magic
- Unsupported version
- Truncated file
- Invalid metadata type
- Invalid tensor dimensions
- Tensor out of bounds
- Unsupported tensor quantization type

## 10) Minimal HolyC Parser API Shape

Suggested API (naming may change):
- `Bool GGUF_ParseHeader(U8 *buf, I64 size, GGUFHeader *out, I64 *cursor)`
- `Bool GGUF_ParseMetadata(...)`
- `Bool GGUF_ParseTensorInfo(...)`
- `Bool GGUF_FinalizeTensorBase(...)`
- `Bool GGUF_LoadTensorBytes(...)`

Keep each function direct and readable: explicit loops, explicit offset math,
no abstraction layers that hide binary layout details.

## 11) Validation Targets

Host-side validation scripts in `tests/` should verify:
- Header fields match llama.cpp dump for same model.
- Tensor count and selected tensor names match reference.
- Absolute offsets computed by HolyC logic match reference parser.
- Q4_0/Q8_0 tensor byte spans are valid and non-overlapping.

These scripts are validation-only and not part of the TempleOS runtime.
