#!/usr/bin/env python3
"""Parity harness for Q4_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride (IQ-988)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref

Q4_0_OK = 0
Q4_0_ERR_NULL_PTR = 1
Q4_0_ERR_BAD_DST_LEN = 2
Q4_0_ERR_OVERFLOW = 3
Q4_0_VALUES_PER_BLOCK = 32
Q4_0_PACKED_BYTES = 16
Q4_0_I64_MAX = (1 << 63) - 1
Q4_0_I64_MIN = -(1 << 63)


def try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs == 0 or rhs == 0:
        return True, 0
    if (lhs == Q4_0_I64_MIN and rhs == -1) or (rhs == Q4_0_I64_MIN and lhs == -1):
        return False, 0

    abs_lhs = -lhs if lhs < 0 else lhs
    abs_rhs = -rhs if rhs < 0 else rhs
    if abs_lhs > Q4_0_I64_MAX // abs_rhs:
        return False, 0
    return True, lhs * rhs


def q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
    src_blocks,
    src_block_capacity: int,
    src_block_stride: int,
    block_count: int,
    dst_q16,
    dst_q16_capacity: int,
) -> int:
    if src_blocks is None or dst_q16 is None:
        return Q4_0_ERR_NULL_PTR

    if src_block_capacity < 0 or dst_q16_capacity < 0:
        return Q4_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return Q4_0_ERR_BAD_DST_LEN
    if src_block_stride <= 0:
        return Q4_0_ERR_BAD_DST_LEN
    if src_blocks is dst_q16:
        return Q4_0_ERR_BAD_DST_LEN

    dst_stride = Q4_0_VALUES_PER_BLOCK

    if block_count == 0:
        return Q4_0_OK

    src_last_offset = (block_count - 1) * src_block_stride
    dst_last_base = (block_count - 1) * dst_stride
    src_required_blocks = src_last_offset + 1
    dst_required_values = dst_last_base + Q4_0_VALUES_PER_BLOCK

    if src_required_blocks > Q4_0_I64_MAX or dst_required_values > Q4_0_I64_MAX:
        return Q4_0_ERR_OVERFLOW

    if src_required_blocks > src_block_capacity:
        return Q4_0_ERR_BAD_DST_LEN
    if dst_required_values > dst_q16_capacity:
        return Q4_0_ERR_BAD_DST_LEN

    # Overflow preflight pass (zero-write).
    for block_idx in range(block_count):
        src_index = block_idx * src_block_stride
        d_fp16, q4_bytes = src_blocks[src_index]
        scale_q16 = ref.f16_to_q16(d_fp16)

        q4_signed = ref.unpack_q4_signed(q4_bytes)
        for lane in range(Q4_0_VALUES_PER_BLOCK):
            ok, _ = try_mul_i64(scale_q16, q4_signed[lane])
            if not ok:
                return Q4_0_ERR_OVERFLOW

    staged = dst_q16[:]
    for block_idx in range(block_count):
        src_index = block_idx * src_block_stride
        dst_base = block_idx * dst_stride

        d_fp16, q4_bytes = src_blocks[src_index]
        scale_q16 = ref.f16_to_q16(d_fp16)
        q4_signed = ref.unpack_q4_signed(q4_bytes)

        for lane in range(Q4_0_VALUES_PER_BLOCK):
            ok, decode_q16 = try_mul_i64(scale_q16, q4_signed[lane])
            if not ok:
                return Q4_0_ERR_OVERFLOW
            staged[dst_base + lane] = decode_q16

    for idx, value in enumerate(staged):
        dst_q16[idx] = value

    return Q4_0_OK


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-4.0, 4.0)
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q4_from_signed(vals)


def expected_blocks(src_blocks, src_stride: int, block_count: int) -> list[list[int]]:
    out = []
    for block_idx in range(block_count):
        src_index = block_idx * src_stride
        d_fp16, q4_bytes = src_blocks[src_index]
        scale_q16 = ref.f16_to_q16(d_fp16)
        q4_signed = ref.unpack_q4_signed(q4_bytes)
        out.append([scale_q16 * q for q in q4_signed])
    return out


def test_source_contains_iq988_default_stride_contract() -> None:
    source = Path("src/quant/q4_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "dst_block_stride_values = Q4_0_VALUES_PER_BLOCK;" in body
    assert "if ((I64 *)src_blocks == dst_q16)" in body
    assert "if (!Q4_0TryMulI64NonNeg(block_count - 1, src_block_stride, &src_last_offset))" in body
    assert "if (!Q4_0TryAddI64(dst_last_base, Q4_0_VALUES_PER_BLOCK, &dst_required_values))" in body
    assert "// Overflow preflight pass (zero-write): every lane decode must be safe" in body
    assert "if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))" in body
    assert "// Commit pass: decode each block only after the full preflight succeeds." in body
    assert "dst_q16[dst_base + lane + 1] = decode_q16;" in body


def test_null_shape_and_alias_guards() -> None:
    rng = random.Random(9880)
    blocks = [make_q4_block(rng) for _ in range(5)]
    out = [123] * 96

    assert (
        q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            None,
            5,
            1,
            1,
            out,
            len(out),
        )
        == Q4_0_ERR_NULL_PTR
    )

    assert (
        q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            blocks,
            5,
            0,
            1,
            out,
            len(out),
        )
        == Q4_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            blocks,
            5,
            1,
            -1,
            out,
            len(out),
        )
        == Q4_0_ERR_BAD_DST_LEN
    )


def test_preflight_capacity_failure_keeps_output_unchanged() -> None:
    rng = random.Random(9881)
    blocks = [make_q4_block(rng) for _ in range(4)]
    out = [7777] * 128
    before = out[:]

    err = q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        blocks,
        src_block_capacity=4,
        src_block_stride=2,
        block_count=3,
        dst_q16=out,
        dst_q16_capacity=len(out),
    )

    assert err == Q4_0_ERR_BAD_DST_LEN
    assert out == before


def test_overflow_preflight_keeps_output_unchanged() -> None:
    # +INF scale saturates to huge Q16. q=-8 lane must overflow I64 multiply.
    overflow_q4 = bytes([0x00] * Q4_0_PACKED_BYTES)
    blocks = [(0x7C00, overflow_q4)]

    out = [2468] * 64
    before = out[:]

    err = q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        blocks,
        src_block_capacity=1,
        src_block_stride=1,
        block_count=1,
        dst_q16=out,
        dst_q16_capacity=len(out),
    )

    assert err == Q4_0_ERR_OVERFLOW
    assert out == before


def test_known_vector_signed_nibble_extremes() -> None:
    scale_one = ref.half_bits(1.0)
    scale_neg_half = ref.half_bits(-0.5)

    block0_vals = [-8, -7, -4, -1, 0, 1, 6, 7] * 4
    block1_vals = [7, 6, 2, 1, -1, -2, -6, -7] * 4

    blocks = [
        (scale_one, ref.pack_q4_from_signed(block0_vals)),
        (scale_neg_half, ref.pack_q4_from_signed(block1_vals)),
    ]

    out = [0] * 80
    err = q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        blocks,
        src_block_capacity=2,
        src_block_stride=1,
        block_count=2,
        dst_q16=out,
        dst_q16_capacity=len(out),
    )
    assert err == Q4_0_OK

    s0 = ref.f16_to_q16(scale_one)
    s1 = ref.f16_to_q16(scale_neg_half)
    assert out[0:32] == [s0 * v for v in block0_vals]
    assert out[32:64] == [s1 * v for v in block1_vals]


def test_random_strided_source_parity() -> None:
    rng = random.Random(9882)

    for _ in range(160):
        block_count = rng.randint(1, 6)
        src_stride = rng.randint(1, 3)

        src_capacity = (block_count - 1) * src_stride + 1 + rng.randint(0, 2)
        blocks = [make_q4_block(rng) for _ in range(src_capacity)]

        dst_capacity = block_count * Q4_0_VALUES_PER_BLOCK + rng.randint(0, 8)
        out = [rng.randrange(-1000, 1001) for _ in range(dst_capacity)]
        out_before = out[:]

        err = q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            blocks,
            src_capacity,
            src_stride,
            block_count,
            out,
            dst_capacity,
        )
        assert err == Q4_0_OK

        exp = expected_blocks(blocks, src_stride, block_count)
        for block_idx in range(block_count):
            start = block_idx * Q4_0_VALUES_PER_BLOCK
            assert out[start : start + Q4_0_VALUES_PER_BLOCK] == exp[block_idx]

        untouched_start = block_count * Q4_0_VALUES_PER_BLOCK
        assert out[untouched_start:] == out_before[untouched_start:]


if __name__ == "__main__":
    test_source_contains_iq988_default_stride_contract()
    test_null_shape_and_alias_guards()
    test_preflight_capacity_failure_keeps_output_unchanged()
    test_overflow_preflight_keeps_output_unchanged()
    test_known_vector_signed_nibble_extremes()
    test_random_strided_source_parity()
    print("ok")
