#!/usr/bin/env python3
"""Parity harness for Q8_0DequantizeBlockQ16CheckedNoPartialArray (IQ-928)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref

Q8_0_OK = 0
Q8_0_ERR_NULL_PTR = 1
Q8_0_ERR_BAD_DST_LEN = 2
Q8_0_ERR_OVERFLOW = 3
Q8_0_VALUES_PER_BLOCK = 32
Q8_0_I64_MAX = (1 << 63) - 1
Q8_0_I64_MIN = -(1 << 63)


def try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs == 0 or rhs == 0:
        return True, 0
    if (lhs == Q8_0_I64_MIN and rhs == -1) or (rhs == Q8_0_I64_MIN and lhs == -1):
        return False, 0

    abs_lhs = -lhs if lhs < 0 else lhs
    abs_rhs = -rhs if rhs < 0 else rhs
    if abs_lhs > Q8_0_I64_MAX // abs_rhs:
        return False, 0
    return True, lhs * rhs


def q8_0_dequantize_block_q16_checked_no_partial_array(
    src_blocks,
    src_block_capacity: int,
    src_block_stride: int,
    block_count: int,
    dst_q16,
    dst_q16_capacity: int,
    dst_block_stride_values: int,
) -> int:
    if src_blocks is None or dst_q16 is None:
        return Q8_0_ERR_NULL_PTR

    if src_block_capacity < 0 or dst_q16_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if src_block_stride <= 0 or dst_block_stride_values <= 0:
        return Q8_0_ERR_BAD_DST_LEN
    if dst_block_stride_values < Q8_0_VALUES_PER_BLOCK:
        return Q8_0_ERR_BAD_DST_LEN
    if src_blocks is dst_q16:
        return Q8_0_ERR_BAD_DST_LEN

    if block_count == 0:
        return Q8_0_OK

    src_last_offset = (block_count - 1) * src_block_stride
    dst_last_base = (block_count - 1) * dst_block_stride_values

    src_required_blocks = src_last_offset + 1
    dst_required_values = dst_last_base + Q8_0_VALUES_PER_BLOCK

    if src_required_blocks > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW
    if dst_required_values > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW

    if src_required_blocks > src_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if dst_required_values > dst_q16_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    # Overflow preflight pass (zero-write).
    for block_idx in range(block_count):
        src_index = block_idx * src_block_stride
        scale_q16 = ref.f16_to_q16(src_blocks[src_index][0])
        q8_signed = ref.unpack_q8_signed(src_blocks[src_index][1])
        for lane in range(Q8_0_VALUES_PER_BLOCK):
            ok, _ = try_mul_i64(scale_q16, q8_signed[lane])
            if not ok:
                return Q8_0_ERR_OVERFLOW

    staged = dst_q16[:]
    for block_idx in range(block_count):
        src_index = block_idx * src_block_stride
        dst_base = block_idx * dst_block_stride_values

        scale_q16 = ref.f16_to_q16(src_blocks[src_index][0])
        q8_signed = ref.unpack_q8_signed(src_blocks[src_index][1])
        for lane in range(Q8_0_VALUES_PER_BLOCK):
            ok, decode_q16 = try_mul_i64(scale_q16, q8_signed[lane])
            if not ok:
                return Q8_0_ERR_OVERFLOW
            staged[dst_base + lane] = decode_q16

    for idx, value in enumerate(staged):
        dst_q16[idx] = value

    return Q8_0_OK


def explicit_expected(src_blocks, src_block_stride: int, block_count: int) -> list[list[int]]:
    out: list[list[int]] = []
    for block_idx in range(block_count):
        src_index = block_idx * src_block_stride
        d_fp16, q8_bytes = src_blocks[src_index]
        scale_q16 = ref.f16_to_q16(d_fp16)
        q8_signed = ref.unpack_q8_signed(q8_bytes)
        out.append([scale_q16 * v for v in q8_signed])
    return out


def make_q8_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.25, 2.25)
    vals = [rng.randrange(-128, 128) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q8_signed(vals)


def test_source_contains_iq928_function() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArray("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "if (dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)" in body
    assert "if (!Q8_0TryMulI64NonNeg(block_count - 1, src_block_stride, &src_last_offset))" in body
    assert "if (!Q8_0TryAddI64(dst_last_base, Q8_0_VALUES_PER_BLOCK, &dst_required_values))" in body
    assert "// Overflow preflight pass (zero-write): validate every per-lane decode" in body
    assert "if (!Q8_0TryMulI64(scale_q16, q_signed, &decode_q16))" in body
    assert "// Commit pass: decode each block only after full preflight success." in body
    assert "dst_q16[dst_base + lane] = decode_q16;" in body


def test_null_and_shape_guards() -> None:
    rng = random.Random(928)
    blocks = [make_q8_block(rng) for _ in range(8)]
    out = [111] * 96

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array(
            None,
            len(blocks),
            1,
            2,
            out,
            len(out),
            32,
        )
        == Q8_0_ERR_NULL_PTR
    )

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array(
            blocks,
            len(blocks),
            0,
            2,
            out,
            len(out),
            32,
        )
        == Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array(
            blocks,
            len(blocks),
            1,
            2,
            out,
            len(out),
            31,
        )
        == Q8_0_ERR_BAD_DST_LEN
    )


def test_preflight_failure_keeps_output_unchanged() -> None:
    rng = random.Random(20260421)
    blocks = [make_q8_block(rng) for _ in range(6)]
    out = [7777] * 128
    before = out[:]

    err = q8_0_dequantize_block_q16_checked_no_partial_array(
        blocks,
        len(blocks),
        3,
        3,
        out,
        len(out),
        48,
    )

    assert err == Q8_0_ERR_BAD_DST_LEN
    assert out == before


def test_overflow_preflight_keeps_output_unchanged() -> None:
    # d_fp16=+INF saturates to huge Q16; times 127 must overflow I64.
    overflow_block = (0x7C00, ref.pack_q8_signed([127] * 32))
    blocks = [overflow_block]
    out = [4321] * 64
    before = out[:]

    err = q8_0_dequantize_block_q16_checked_no_partial_array(
        blocks,
        src_block_capacity=1,
        src_block_stride=1,
        block_count=1,
        dst_q16=out,
        dst_q16_capacity=len(out),
        dst_block_stride_values=32,
    )

    assert err == Q8_0_ERR_OVERFLOW
    assert out == before


def test_known_edge_vectors_signed_byte_extremes() -> None:
    scale_one = ref.half_bits(1.0)
    scale_neg_half = ref.half_bits(-0.5)

    block0_vals = [-128, -127, -64, -1, 0, 1, 63, 64] * 4
    block1_vals = [127, 126, 32, 2, -2, -32, -126, -127] * 4
    blocks = [
        (scale_one, ref.pack_q8_signed(block0_vals)),
        (scale_neg_half, ref.pack_q8_signed(block1_vals)),
    ]

    out = [0] * 80
    err = q8_0_dequantize_block_q16_checked_no_partial_array(
        blocks,
        src_block_capacity=2,
        src_block_stride=1,
        block_count=2,
        dst_q16=out,
        dst_q16_capacity=len(out),
        dst_block_stride_values=40,
    )
    assert err == Q8_0_OK

    s0 = ref.f16_to_q16(scale_one)
    s1 = ref.f16_to_q16(scale_neg_half)
    assert out[0:32] == [s0 * v for v in block0_vals]
    assert out[40:72] == [s1 * v for v in block1_vals]


def test_random_strided_parity() -> None:
    rng = random.Random(381928)

    for _ in range(120):
        block_count = rng.randint(1, 5)
        src_stride = rng.randint(1, 3)
        dst_stride = rng.randint(32, 56)

        src_capacity = block_count * src_stride + rng.randint(0, 2)
        blocks = [make_q8_block(rng) for _ in range(src_capacity)]

        dst_capacity = (block_count - 1) * dst_stride + 32 + rng.randint(0, 5)
        out = [rng.randrange(-5000, 5001) for _ in range(dst_capacity)]
        out_before = out[:]

        err = q8_0_dequantize_block_q16_checked_no_partial_array(
            blocks,
            src_capacity,
            src_stride,
            block_count,
            out,
            dst_capacity,
            dst_stride,
        )
        assert err == Q8_0_OK

        exp = explicit_expected(blocks, src_stride, block_count)
        for block_idx in range(block_count):
            start = block_idx * dst_stride
            assert out[start : start + 32] == exp[block_idx]

        for idx in range(dst_capacity):
            lane_owner = None
            for block_idx in range(block_count):
                start = block_idx * dst_stride
                end = start + 32
                if start <= idx < end:
                    lane_owner = block_idx
                    break
            if lane_owner is None:
                assert out[idx] == out_before[idx]


if __name__ == "__main__":
    test_source_contains_iq928_function()
    test_null_and_shape_guards()
    test_preflight_failure_keeps_output_unchanged()
    test_overflow_preflight_keeps_output_unchanged()
    test_known_edge_vectors_signed_byte_extremes()
    test_random_strided_parity()
    print("ok")
