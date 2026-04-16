#!/usr/bin/env python3
"""Reference checks for Q4_0Q8_0DotBlocksAVX2Q32Checked semantics."""

from __future__ import annotations

import random
import struct

Q4_0_PACKED_BYTES = 16
Q8_0_PACKED_BYTES = 32
Q4_0_Q8_0_AVX2_LANES = 32
Q4_0_Q8_0_AVX2_PAIR_COUNT = 16

Q4_0_Q8_0_OK = 0
Q4_0_Q8_0_ERR_NULL_PTR = 1
Q4_0_Q8_0_ERR_BAD_DST_LEN = 2
Q4_0_Q8_0_ERR_OVERFLOW = 3

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


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
        return -(0x3FFFFFFFFFFFFFFF) if sign_bit else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + fraction_bits
    shift_amount = exponent_bits - 9
    if shift_amount >= 0:
        magnitude_q16 = mantissa << shift_amount
    else:
        magnitude_q16 = round_shift_right_unsigned(mantissa, -shift_amount)

    return -magnitude_q16 if sign_bit else magnitude_q16


def try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


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


def pack_q8_signed(vals: list[int]) -> list[int]:
    assert len(vals) == 32
    return [int(v) for v in vals]


def nibble_to_signed(packed: int, upper: bool) -> int:
    q_unsigned = ((packed >> 4) & 0x0F) if upper else (packed & 0x0F)
    return q_unsigned - 8


def pack_q4_block_to_i16_lanes_avx2(q4_packed: bytes) -> tuple[int, list[int]]:
    if q4_packed is None:
        return Q4_0_Q8_0_ERR_NULL_PTR, []
    if len(q4_packed) < Q4_0_PACKED_BYTES:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    lanes = [0] * Q4_0_Q8_0_AVX2_LANES
    for byte_index in range(Q4_0_PACKED_BYTES):
        packed = q4_packed[byte_index]
        lanes[2 * byte_index] = nibble_to_signed(packed, False)
        lanes[2 * byte_index + 1] = nibble_to_signed(packed, True)

    return Q4_0_Q8_0_OK, lanes


def pack_q8_block_to_i16_lanes_avx2(q8_vals: list[int]) -> tuple[int, list[int]]:
    if q8_vals is None:
        return Q4_0_Q8_0_ERR_NULL_PTR, []
    if len(q8_vals) < Q8_0_PACKED_BYTES:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    return Q4_0_Q8_0_OK, [int(v) for v in q8_vals[:Q4_0_Q8_0_AVX2_LANES]]


def mul_i16_lanes_to_i32_pairs_avx2(lhs_i16: list[int], rhs_i16: list[int]) -> tuple[int, list[int]]:
    if lhs_i16 is None or rhs_i16 is None:
        return Q4_0_Q8_0_ERR_NULL_PTR, []
    if len(lhs_i16) < Q4_0_Q8_0_AVX2_LANES or len(rhs_i16) < Q4_0_Q8_0_AVX2_LANES:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    pairs = [0] * Q4_0_Q8_0_AVX2_PAIR_COUNT
    for pair_index in range(Q4_0_Q8_0_AVX2_PAIR_COUNT):
        lane = pair_index * 2
        pairs[pair_index] = lhs_i16[lane] * rhs_i16[lane] + lhs_i16[lane + 1] * rhs_i16[lane + 1]

    return Q4_0_Q8_0_OK, pairs


def hsum_i32_pairs_avx2(pairs_i32: list[int]) -> tuple[int, int]:
    if pairs_i32 is None:
        return Q4_0_Q8_0_ERR_NULL_PTR, 0
    if len(pairs_i32) < Q4_0_Q8_0_AVX2_PAIR_COUNT:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    total = 0
    for term in pairs_i32[:Q4_0_Q8_0_AVX2_PAIR_COUNT]:
        ok, total = try_add_i64(total, int(term))
        if not ok:
            return Q4_0_Q8_0_ERR_OVERFLOW, 0

    return Q4_0_Q8_0_OK, total


def dot_i16_lanes_avx2(lhs_i16: list[int], rhs_i16: list[int]) -> tuple[int, int]:
    err, pairs = mul_i16_lanes_to_i32_pairs_avx2(lhs_i16, rhs_i16)
    if err != Q4_0_Q8_0_OK:
        return err, 0
    return hsum_i32_pairs_avx2(pairs)


def dot_block_avx2_q32_checked(lhs_block, rhs_block) -> tuple[int, int]:
    l_scale_fp16, l_q4_packed = lhs_block
    r_scale_fp16, r_q8_vals = rhs_block

    err, lhs_lanes = pack_q4_block_to_i16_lanes_avx2(l_q4_packed)
    if err != Q4_0_Q8_0_OK:
        return err, 0

    err, rhs_lanes = pack_q8_block_to_i16_lanes_avx2(r_q8_vals)
    if err != Q4_0_Q8_0_OK:
        return err, 0

    err, q_dot_q0 = dot_i16_lanes_avx2(lhs_lanes, rhs_lanes)
    if err != Q4_0_Q8_0_OK:
        return err, 0

    lhs_scale_q16 = f16_to_q16(l_scale_fp16)
    rhs_scale_q16 = f16_to_q16(r_scale_fp16)

    ok, scale_prod_q32 = try_mul_i64(lhs_scale_q16, rhs_scale_q16)
    if not ok:
        return Q4_0_Q8_0_ERR_OVERFLOW, 0

    ok, block_dot_q32 = try_mul_i64(scale_prod_q32, q_dot_q0)
    if not ok:
        return Q4_0_Q8_0_ERR_OVERFLOW, 0

    return Q4_0_Q8_0_OK, block_dot_q32


def dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count: int) -> tuple[int, int]:
    if lhs_blocks is None or rhs_blocks is None:
        return Q4_0_Q8_0_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0
    if len(lhs_blocks) < block_count or len(rhs_blocks) < block_count:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    total = 0
    for index in range(block_count):
        err, block_dot_q32 = dot_block_avx2_q32_checked(lhs_blocks[index], rhs_blocks[index])
        if err != Q4_0_Q8_0_OK:
            return err, 0

        ok, total = try_add_i64(total, block_dot_q32)
        if not ok:
            return Q4_0_Q8_0_ERR_OVERFLOW, 0

    return Q4_0_Q8_0_OK, total


def dot_block_scalar_q32(lhs_block, rhs_block) -> int:
    l_scale_fp16, l_q4_packed = lhs_block
    r_scale_fp16, r_q8_vals = rhs_block
    l_scale_q16 = f16_to_q16(l_scale_fp16)
    r_scale_q16 = f16_to_q16(r_scale_fp16)

    q4_vals: list[int] = []
    for packed in l_q4_packed:
        q4_vals.append(nibble_to_signed(packed, False))
        q4_vals.append(nibble_to_signed(packed, True))

    q_dot_q0 = sum(a * b for a, b in zip(q4_vals, r_q8_vals))
    return l_scale_q16 * r_scale_q16 * q_dot_q0


def test_known_block_matches_scalar_reference() -> None:
    lhs_scale_fp16 = half_bits(1.0)
    rhs_scale_fp16 = half_bits(0.5)
    q4_vals = [((i % 16) - 8) for i in range(32)]
    q8_vals = [((i % 13) - 6) * 7 for i in range(32)]

    lhs_block = (lhs_scale_fp16, pack_q4_signed(q4_vals))
    rhs_block = (rhs_scale_fp16, pack_q8_signed(q8_vals))

    err, got_q32 = dot_blocks_avx2_q32_checked([lhs_block], [rhs_block], 1)
    assert err == Q4_0_Q8_0_OK
    assert got_q32 == dot_block_scalar_q32(lhs_block, rhs_block)


def test_randomized_multi_block_matches_scalar_reference() -> None:
    rng = random.Random(20260416)

    scales = [
        half_bits(v)
        for v in [
            0.0,
            0.125,
            0.25,
            0.5,
            1.0,
            1.75,
            -0.125,
            -0.5,
            -1.0,
            -2.0,
        ]
    ]

    for _ in range(400):
        block_count = rng.randint(1, 33)
        lhs_blocks = []
        rhs_blocks = []
        for _ in range(block_count):
            q4_vals = [rng.randrange(-8, 8) for _ in range(32)]
            q8_vals = [rng.randrange(-128, 128) for _ in range(32)]

            lhs_blocks.append((rng.choice(scales), pack_q4_signed(q4_vals)))
            rhs_blocks.append((rng.choice(scales), pack_q8_signed(q8_vals)))

        err, got_q32 = dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count)
        assert err == Q4_0_Q8_0_OK

        expected_q32 = sum(dot_block_scalar_q32(lhs, rhs) for lhs, rhs in zip(lhs_blocks, rhs_blocks))
        assert got_q32 == expected_q32


def test_zero_block_count_returns_zero() -> None:
    err, dot_q32 = dot_blocks_avx2_q32_checked([], [], 0)
    assert err == Q4_0_Q8_0_OK
    assert dot_q32 == 0


def test_lane_pack_nibble_order_contract() -> None:
    # Nibbles: low=1 (signed -7), high=14 (signed +6)
    q4_packed = bytes([0xE1] + [0x88] * 15)
    err, lanes = pack_q4_block_to_i16_lanes_avx2(q4_packed)
    assert err == Q4_0_Q8_0_OK
    assert lanes[0] == -7
    assert lanes[1] == 6


def test_error_paths() -> None:
    err, _ = dot_blocks_avx2_q32_checked(None, [], 0)
    assert err == Q4_0_Q8_0_ERR_NULL_PTR

    err, _ = dot_blocks_avx2_q32_checked([], None, 0)
    assert err == Q4_0_Q8_0_ERR_NULL_PTR

    err, _ = dot_blocks_avx2_q32_checked([], [], -1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_blocks_avx2_q32_checked([], [], 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN


def test_overflow_guard_on_scale_product() -> None:
    # Exponent=all ones gives saturated sentinel in HolyC converter.
    huge_scale = 0x7C00
    q4_vals = [7] * 32
    q8_vals = [127] * 32

    lhs_block = (huge_scale, pack_q4_signed(q4_vals))
    rhs_block = (huge_scale, pack_q8_signed(q8_vals))

    err, _ = dot_blocks_avx2_q32_checked([lhs_block], [rhs_block], 1)
    assert err == Q4_0_Q8_0_ERR_OVERFLOW


if __name__ == "__main__":
    test_known_block_matches_scalar_reference()
    test_randomized_multi_block_matches_scalar_reference()
    test_zero_block_count_returns_zero()
    test_lane_pack_nibble_order_contract()
    test_error_paths()
    test_overflow_guard_on_scale_product()
    print("q4_0_q8_0_avx2_blocks_q32_reference_checks=ok")
