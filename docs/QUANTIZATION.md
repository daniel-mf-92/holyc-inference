# Quantization Reference (Q4_0 / Q8_0, Integer-Only HolyC)

This document defines the first-pass quantization math for the TempleOS
HolyC inference runtime. Runtime computation is integer-only: no floating
point ops in tensor kernels, matmul, attention, or sampling.

Scope:
- Q4_0 block decode rules
- Q8_0 block decode rules
- Integer fixed-point representation for scales
- Integer dot-product accumulation strategy

## 1) Constants and Notation

- `QK4_0 = 32` values per Q4_0 block.
- `QK8_0 = 32` values per Q8_0 block.
- `d_fp16` = per-block scale stored as fp16 in GGUF.
- `d_q16` = integer fixed-point scale with 16 fractional bits.
- `SCALE_SHIFT = 16`.

HolyC runtime rule:
- Convert `d_fp16` to fixed-point integer once per block load.
- Keep inner loops as integer multiply/add/shift only.

## 2) Q4_0 Block Layout

GGUF/ggml Q4_0 block for 32 values:
- `d_fp16` (2 bytes)
- `qs[16]` packed nibbles (`2 x 4-bit` values per byte)

Per value extraction:
1. Read byte `b = qs[i >> 1]`
2. If even index: `q = b & 0x0F`
3. If odd index:  `q = (b >> 4) & 0x0F`
4. Zero-point shift: `q_signed = q - 8`  (range `[-8, 7]`)

Reference real-value equation:
- `x = d * (q - 8)`

HolyC integer form (Q16):
- `x_q16 = d_q16 * q_signed`

`x_q16` is kept as widened integer (at least 32-bit, usually 64-bit).

## 3) Q8_0 Block Layout

GGUF/ggml Q8_0 block for 32 values:
- `d_fp16` (2 bytes)
- `qs[32]` signed int8 values

Per value extraction:
- `q_signed = (I8)qs[i]`  (range `[-128, 127]`)

Reference real-value equation:
- `x = d * q`

HolyC integer form (Q16):
- `x_q16 = d_q16 * q_signed`

## 4) Fixed-Point Scale Conversion

TempleOS kernels must not depend on float arithmetic. Use one of these paths:
- Preferred: fp16 -> integer conversion routine in HolyC (bit decode, integer math)
- Validation-only fallback: host tools compute expected `d_q16`

Scale conversion definition:
- `d_q16 = Round(d_fp16 * (1 << SCALE_SHIFT))`

All subsequent kernel math uses `d_q16` only.

## 5) Integer Dot Product Formulas

For a blockwise dot product, accumulate in 64-bit integer.

### Q8_0 · Q8_0

For each element `i` in block:
- `term = (qA_i * qB_i)`

Block sum:
- `sum_i32 = Σ term`

Scaled block contribution in Q32:
- `block_q32 = (dA_q16 * dB_q16) * sum_i32`

Final rescale back to Q16 (or integer logits domain) is a right shift:
- `block_q16 = block_q32 >> SCALE_SHIFT`

### Q4_0 · Q8_0

For each element `i` in block:
- `q4_i = unpack_nibble(i) - 8`
- `term = q4_i * q8_i`

Block sum and scaling match above:
- `sum_i32 = Σ term`
- `block_q32 = (d4_q16 * d8_q16) * sum_i32`

### Q4_0 · Q4_0

For each element `i` in block:
- `qA_i = unpack_nibble_A(i) - 8`
- `qB_i = unpack_nibble_B(i) - 8`
- `term = qA_i * qB_i`

Block sum and scaling:
- `sum_i32 = Σ term`
- `block_q32 = (dA_q16 * dB_q16) * sum_i32`

## 6) Overflow and Accumulator Rules

- Per-term multiplies are small (`I8 * I8`), but long vectors require 64-bit sum.
- Accumulate block contributions in `I64`.
- Apply deterministic shift/round policy at layer boundaries only.
- Saturate only at explicit tensor writeback points (not every multiply).

## 7) Kernel Implementation Rules (HolyC Purity)

- `.HC` source only for inference runtime.
- No libc/libm calls.
- No hidden abstraction layers in hot math paths.
- Keep decode + dot loops explicit and auditable.
- Optional AVX2 asm may optimize loops, but must preserve integer equations above.

## 8) Host-Side Validation Expectations

Host-side scripts in `tests/` may use Python/C only for parity checks against
llama.cpp reference outputs.

Validation checks should confirm:
- Q4_0 nibble unpack and `-8` offset are exact.
- Q8_0 signed-byte interpretation is exact.
- Fixed-point scaled dot products match reference within declared integer rounding policy.

