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


def nibble_to_signed(packed: int, upper_nibble: bool) -> int:
    q_unsigned = (packed >> 4) & 0x0F if upper_nibble else packed & 0x0F
    return q_unsigned - 8


def unpack_q4_signed(qs_packed: bytes) -> list[int]:
    out: list[int] = []
    for packed in qs_packed:
        out.append(nibble_to_signed(packed, False))
        out.append(nibble_to_signed(packed, True))
    return out


def q4_0_dequantize_block_q16_checked_no_partial_array(
    src_blocks,
    src_block_capacity: int,
    src_block_stride: int,
    block_count: int,
    dst_q16,
    dst_q16_capacity: int,
    dst_block_stride_values: int,
) -> int:
    if src_blocks is None or dst_q16 is None:
        return Q4_0_ERR_NULL_PTR

    if src_block_capacity < 0 or dst_q16_capacity < 0:
        return Q4_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return Q4_0_ERR_BAD_DST_LEN
    if src_block_stride <= 0 or dst_block_stride_values <= 0:
        return Q4_0_ERR_BAD_DST_LEN
    if dst_block_stride_values < Q4_0_VALUES_PER_BLOCK:
        return Q4_0_ERR_BAD_DST_LEN
    if src_blocks is dst_q16:
        return Q4_0_ERR_BAD_DST_LEN

    if block_count == 0:
        return Q4_0_OK

    src_last_offset = (block_count - 1) * src_block_stride
    dst_last_base = (block_count - 1) * dst_block_stride_values
    src_required_blocks = src_last_offset + 1
    dst_required_values = dst_last_base + Q4_0_VALUES_PER_BLOCK

    if src_required_blocks > Q4_0_I64_MAX:
        return Q4_0_ERR_OVERFLOW
    if dst_required_values > Q4_0_I64_MAX:
        return Q4_0_ERR_OVERFLOW

    if src_required_blocks > src_block_capacity:
        return Q4_0_ERR_BAD_DST_LEN
    if dst_required_values > dst_q16_capacity:
        return Q4_0_ERR_BAD_DST_LEN

    for block_idx in range(block_count):
        src_index = block_idx * src_block_stride
        scale_q16 = ref.f16_to_q16(src_blocks[src_index][0])
        q4_signed = unpack_q4_signed(src_blocks[src_index][1])

        for lane in range(Q4_0_VALUES_PER_BLOCK):
            ok, _ = try_mul_i64(scale_q16, q4_signed[lane])
            if not ok:
                return Q4_0_ERR_OVERFLOW

    staged = dst_q16[:]
    for block_idx in range(block_count):
        src_index = block_idx * src_block_stride
        dst_base = block_idx * dst_block_stride_values

        scale_q16 = ref.f16_to_q16(src_blocks[src_index][0])
        q4_signed = unpack_q4_signed(src_blocks[src_index][1])

        for lane in range(Q4_0_VALUES_PER_BLOCK):
            ok, decode_q16 = try_mul_i64(scale_q16, q4_signed[lane])
            if not ok:
                return Q4_0_ERR_OVERFLOW
            staged[dst_base + lane] = decode_q16

    for index, value in enumerate(staged):
        dst_q16[index] = value

    return Q4_0_OK


def q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
    src_blocks,
    src_block_capacity: int,
    src_block_stride: int,
    block_count: int,
    dst_q16,
    dst_q16_capacity: int,
) -> int:
    return q4_0_dequantize_block_q16_checked_no_partial_array(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        Q4_0_VALUES_PER_BLOCK,
    )


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-4.0, 4.0)
    scale_fp16 = ref.half_bits(scale)
    values = [rng.randint(-8, 7) for _ in range(Q4_0_VALUES_PER_BLOCK)]
    return scale_fp16, ref.pack_q4_from_signed(values)


def test_source_contains_iq988_default_stride_wrapper_and_checked_core() -> None:
    source = Path("src/quant/q4_0.HC").read_text(encoding="utf-8")

    assert "#define Q4_0_BLOCK_SIZE" in source
    assert "#define Q4_0_ERR_OVERFLOW" in source

    checked_sig = "I32 Q4_0DequantizeBlockQ16CheckedNoPartialArray("
    assert checked_sig in source
    checked_body = source.split(checked_sig, 1)[1]

    assert "// Preflight (zero-write): prove decode math is safe for every output lane." in checked_body
    assert "if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))" in checked_body
    assert "// Commit pass: preflight already proved all writes/ops are valid." in checked_body

    wrapper_sig = "I32 Q4_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride("
    assert wrapper_sig in source
    wrapper_body = source.split(wrapper_sig, 1)[1]

    assert "out_stride_values = Q4_0_BLOCK_SIZE;" in wrapper_body
    assert "return Q4_0DequantizeBlockQ16CheckedNoPartialArray(" in wrapper_body


def test_null_and_shape_guards() -> None:
    rng = random.Random(988)
    blocks = [make_q4_block(rng) for _ in range(8)]
    out = [111] * 96

    assert (
        q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            None,
            len(blocks),
            1,
            2,
            out,
            len(out),
        )
        == Q4_0_ERR_NULL_PTR
    )

    assert (
        q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            blocks,
            len(blocks),
            0,
            2,
            out,
            len(out),
        )
        == Q4_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            blocks,
            len(blocks),
            1,
            2,
            out,
            63,
        )
        == Q4_0_ERR_BAD_DST_LEN
    )


def test_preflight_failure_keeps_output_unchanged() -> None:
    rng = random.Random(20260422)
    blocks = [make_q4_block(rng) for _ in range(7)]
    out = [7777] * 256
    before = out[:]

    err = q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        blocks,
        len(blocks),
        3,
        3,
        out,
        len(out),
    )

    assert err == Q4_0_ERR_BAD_DST_LEN
    assert out == before


def test_overflow_preflight_keeps_output_unchanged() -> None:
    overflow_block = (0x7C00, ref.pack_q4_from_signed([7] * Q4_0_VALUES_PER_BLOCK))
    out = [4321] * 64
    before = out[:]

    err = q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
        [overflow_block],
        src_block_capacity=1,
        src_block_stride=1,
        block_count=1,
        dst_q16=out,
        dst_q16_capacity=len(out),
    )

    assert err == Q4_0_ERR_OVERFLOW
    assert out == before


def test_known_edge_vectors_nibble_extremes() -> None:
    scale_one = ref.half_bits(1.0)
    scale_neg_half = ref.half_bits(-0.5)

    block0_vals = [-8, -7, -6, -5, -4, -3, -2, -1] * 4
    block1_vals = [7, 6, 5, 4, 3, 2, 1, 0] * 4
    blocks = [
        (scale_one, ref.pack_q4_from_signed(block0_vals)),
        (scale_neg_half, ref.pack_q4_from_signed(block1_vals)),
    ]

    out = [0] * 64
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
    assert out[0:32] == [s0 * value for value in block0_vals]
    assert out[32:64] == [s1 * value for value in block1_vals]


def test_random_default_stride_parity() -> None:
    rng = random.Random(908988)

    for _ in range(140):
        block_count = rng.randint(0, 6)
        src_stride = rng.randint(1, 4)

        src_capacity = 0 if block_count == 0 else 1 + (block_count - 1) * src_stride + rng.randint(0, 2)
        blocks = [make_q4_block(rng) for _ in range(max(src_capacity, 1))]

        dst_capacity = 0 if block_count == 0 else block_count * Q4_0_VALUES_PER_BLOCK + rng.randint(0, 3)
        out = [rng.randrange(-5000, 5001) for _ in range(max(dst_capacity, 1))]
        out_before = out[:]

        err = q4_0_dequantize_block_q16_checked_no_partial_array_default_stride(
            blocks,
            src_capacity,
            src_stride,
            block_count,
            out,
            dst_capacity,
        )

        if block_count == 0:
            assert err == Q4_0_OK
            assert out == out_before
            continue

        if dst_capacity < block_count * Q4_0_VALUES_PER_BLOCK:
            assert err == Q4_0_ERR_BAD_DST_LEN
            assert out == out_before
            continue

        assert err == Q4_0_OK

        for block_idx in range(block_count):
            src_index = block_idx * src_stride
            scale_q16 = ref.f16_to_q16(blocks[src_index][0])
            q4_signed = unpack_q4_signed(blocks[src_index][1])
            expected = [scale_q16 * q for q in q4_signed]
            start = block_idx * Q4_0_VALUES_PER_BLOCK
            assert out[start : start + Q4_0_VALUES_PER_BLOCK] == expected


def main() -> None:
    test_source_contains_iq988_default_stride_wrapper_and_checked_core()
    test_null_and_shape_guards()
    test_preflight_failure_keeps_output_unchanged()
    test_overflow_preflight_keeps_output_unchanged()
    test_known_edge_vectors_nibble_extremes()
    test_random_default_stride_parity()
    print("q4_0_dequantize_block_q16_checked_no_partial_array_default_stride=ok")


if __name__ == "__main__":
    main()
