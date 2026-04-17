#!/usr/bin/env python3
"""Reference checks for Q4_0DotProductBlocksQ32AVX2 semantics."""

from __future__ import annotations

import random
import struct

Q4_0_AVX2_VALUES_PER_BLOCK = 32
Q4_0_AVX2_PACKED_BYTES = 16

Q4_0_AVX2_OK = 0
Q4_0_AVX2_ERR_NULL_PTR = 1
Q4_0_AVX2_ERR_BAD_LEN = 2


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    round_bias = 1 << (shift - 1)
    return (value + round_bias) >> shift


def f16_to_q16(fp16_bits: int) -> int:
    sign_bit = (fp16_bits >> 15) & 1
    exponent_bits = (fp16_bits >> 10) & 0x1F
    fraction_bits = fp16_bits & 0x03FF

    if exponent_bits == 0:
        if fraction_bits == 0:
            return 0
        magnitude_q16 = round_shift_right_unsigned(fraction_bits, 8)
        return -magnitude_q16 if sign_bit else magnitude_q16

    if exponent_bits == 0x1F:
        return -0x3FFFFFFFFFFFFFFF if sign_bit else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + fraction_bits
    shift_amount = exponent_bits - 9

    if shift_amount >= 0:
        magnitude_q16 = mantissa << shift_amount
    else:
        magnitude_q16 = round_shift_right_unsigned(mantissa, -shift_amount)

    return -magnitude_q16 if sign_bit else magnitude_q16


def nibble_to_signed(packed: int, upper_nibble: bool) -> int:
    q_unsigned = ((packed >> 4) & 0x0F) if upper_nibble else (packed & 0x0F)
    return q_unsigned - 8


def unpack32_to_i16_lanes_avx2(packed_q4: bytes) -> tuple[int, list[int]]:
    if packed_q4 is None:
        return Q4_0_AVX2_ERR_NULL_PTR, []
    if len(packed_q4) < Q4_0_AVX2_PACKED_BYTES:
        return Q4_0_AVX2_ERR_BAD_LEN, []

    lanes = [0] * Q4_0_AVX2_VALUES_PER_BLOCK
    for byte_index in range(Q4_0_AVX2_PACKED_BYTES):
        packed = packed_q4[byte_index]
        lane_base = byte_index * 2
        lanes[lane_base] = nibble_to_signed(packed, False)
        lanes[lane_base + 1] = nibble_to_signed(packed, True)

    return Q4_0_AVX2_OK, lanes


def dot_i16_lanes_avx2(lhs_i16: list[int], rhs_i16: list[int]) -> tuple[int, int]:
    if lhs_i16 is None or rhs_i16 is None:
        return Q4_0_AVX2_ERR_NULL_PTR, 0
    if len(lhs_i16) < Q4_0_AVX2_VALUES_PER_BLOCK or len(rhs_i16) < Q4_0_AVX2_VALUES_PER_BLOCK:
        return Q4_0_AVX2_ERR_BAD_LEN, 0

    q_dot_q0 = 0
    for lane in range(Q4_0_AVX2_VALUES_PER_BLOCK):
        q_dot_q0 += int(lhs_i16[lane]) * int(rhs_i16[lane])

    return Q4_0_AVX2_OK, q_dot_q0


def dot_product_block_q32_avx2(lhs_block, rhs_block) -> tuple[int, int]:
    lhs_scale_fp16, lhs_packed = lhs_block
    rhs_scale_fp16, rhs_packed = rhs_block

    err, lhs_lanes = unpack32_to_i16_lanes_avx2(lhs_packed)
    if err != Q4_0_AVX2_OK:
        return err, 0

    err, rhs_lanes = unpack32_to_i16_lanes_avx2(rhs_packed)
    if err != Q4_0_AVX2_OK:
        return err, 0

    err, q_dot_q0 = dot_i16_lanes_avx2(lhs_lanes, rhs_lanes)
    if err != Q4_0_AVX2_OK:
        return err, 0

    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16
    return Q4_0_AVX2_OK, scale_prod_q32 * q_dot_q0


def dot_product_blocks_q32_avx2(lhs_blocks, rhs_blocks, block_count: int) -> tuple[int, int]:
    if lhs_blocks is None or rhs_blocks is None:
        return Q4_0_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, 0
    if len(lhs_blocks) < block_count or len(rhs_blocks) < block_count:
        return Q4_0_AVX2_ERR_BAD_LEN, 0

    total_q32 = 0
    for i in range(block_count):
        err, block_q32 = dot_product_block_q32_avx2(lhs_blocks[i], rhs_blocks[i])
        if err != Q4_0_AVX2_OK:
            return err, 0
        total_q32 += block_q32

    return Q4_0_AVX2_OK, total_q32


def dot_product_blocks_q32_scalar(lhs_blocks, rhs_blocks, block_count: int) -> tuple[int, int]:
    if lhs_blocks is None or rhs_blocks is None:
        return Q4_0_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q4_0_AVX2_ERR_BAD_LEN, 0
    if len(lhs_blocks) < block_count or len(rhs_blocks) < block_count:
        return Q4_0_AVX2_ERR_BAD_LEN, 0

    total_q32 = 0
    for i in range(block_count):
        lhs_scale_fp16, lhs_packed = lhs_blocks[i]
        rhs_scale_fp16, rhs_packed = rhs_blocks[i]

        if len(lhs_packed) != Q4_0_AVX2_PACKED_BYTES or len(rhs_packed) != Q4_0_AVX2_PACKED_BYTES:
            return Q4_0_AVX2_ERR_BAD_LEN, 0

        lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
        rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
        scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16

        q_dot_q0 = 0
        for packed_l, packed_r in zip(lhs_packed, rhs_packed):
            q_dot_q0 += nibble_to_signed(packed_l, False) * nibble_to_signed(packed_r, False)
            q_dot_q0 += nibble_to_signed(packed_l, True) * nibble_to_signed(packed_r, True)

        total_q32 += scale_prod_q32 * q_dot_q0

    return Q4_0_AVX2_OK, total_q32


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def pack_q4_signed(vals: list[int]) -> bytes:
    assert len(vals) == 32
    out = bytearray()
    for i in range(0, 32, 2):
        lo = vals[i] + 8
        hi = vals[i + 1] + 8
        assert 0 <= lo <= 15 and 0 <= hi <= 15
        out.append((lo & 0x0F) | ((hi & 0x0F) << 4))
    return bytes(out)


def test_known_block_matches_scalar() -> None:
    lhs = (half_bits(1.0), pack_q4_signed([i - 8 for i in range(16)] * 2))
    rhs = (half_bits(-0.5), pack_q4_signed([7 - (i % 16) for i in range(32)]))

    err_a, got = dot_product_blocks_q32_avx2([lhs], [rhs], 1)
    err_b, expected = dot_product_blocks_q32_scalar([lhs], [rhs], 1)
    assert err_a == Q4_0_AVX2_OK
    assert err_b == Q4_0_AVX2_OK
    assert got == expected


def test_randomized_multi_block_matches_scalar() -> None:
    rng = random.Random(2026041801)
    scales = [half_bits(v) for v in [0.0, 0.125, 0.25, 0.5, 1.0, 1.5, -0.25, -0.5, -1.0, -2.0]]

    for _ in range(500):
        block_count = rng.randint(1, 41)
        lhs_blocks = []
        rhs_blocks = []

        for _ in range(block_count):
            lhs_blocks.append((rng.choice(scales), pack_q4_signed([rng.randrange(-8, 8) for _ in range(32)])))
            rhs_blocks.append((rng.choice(scales), pack_q4_signed([rng.randrange(-8, 8) for _ in range(32)])))

        err_a, got = dot_product_blocks_q32_avx2(lhs_blocks, rhs_blocks, block_count)
        err_b, expected = dot_product_blocks_q32_scalar(lhs_blocks, rhs_blocks, block_count)
        assert err_a == Q4_0_AVX2_OK
        assert err_b == Q4_0_AVX2_OK
        assert got == expected


def test_nibble_lane_order_contract() -> None:
    # Byte = 0xE1 -> low nibble 1 (signed -7), high nibble 14 (signed +6).
    packed = bytes([0xE1] + [0x88] * 15)
    err, lanes = unpack32_to_i16_lanes_avx2(packed)
    assert err == Q4_0_AVX2_OK
    assert lanes[0] == -7
    assert lanes[1] == 6


def test_error_paths() -> None:
    err, _ = dot_product_blocks_q32_avx2(None, [], 0)
    assert err == Q4_0_AVX2_ERR_NULL_PTR

    err, _ = dot_product_blocks_q32_avx2([], None, 0)
    assert err == Q4_0_AVX2_ERR_NULL_PTR

    err, _ = dot_product_blocks_q32_avx2([], [], -1)
    assert err == Q4_0_AVX2_ERR_BAD_LEN

    err, _ = dot_product_blocks_q32_avx2([], [], 1)
    assert err == Q4_0_AVX2_ERR_BAD_LEN


def test_zero_blocks_returns_zero() -> None:
    err, got = dot_product_blocks_q32_avx2([], [], 0)
    assert err == Q4_0_AVX2_OK
    assert got == 0


def run() -> None:
    test_known_block_matches_scalar()
    test_randomized_multi_block_matches_scalar()
    test_nibble_lane_order_contract()
    test_error_paths()
    test_zero_blocks_returns_zero()
    print("q4_0_avx2_dot_q32_reference_checks=ok")


if __name__ == "__main__":
    run()
