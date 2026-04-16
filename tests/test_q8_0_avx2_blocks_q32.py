#!/usr/bin/env python3
"""Reference checks for Q8_0DotBlocksAVX2Q32Checked semantics."""

from __future__ import annotations

import random

Q8_0_AVX2_VALUES_PER_BLOCK = 32
Q8_0_AVX2_OK = 0
Q8_0_AVX2_ERR_NULL_PTR = 1
Q8_0_AVX2_ERR_BAD_LEN = 2
Q8_0_AVX2_ERR_OVERFLOW = 3

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def q8_0_try_add_i64(lhs: int, rhs: int):
    if rhs > 0 and lhs > I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def q8_0_try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


def q8_0_f16_to_q16(fp16_bits: int) -> int:
    sign_bit = (fp16_bits >> 15) & 1
    exponent_bits = (fp16_bits >> 10) & 0x1F
    fraction_bits = fp16_bits & 0x03FF

    if exponent_bits == 0:
        if fraction_bits == 0:
            return 0
        magnitude_q16 = (fraction_bits + (1 << 7)) >> 8
        return -magnitude_q16 if sign_bit else magnitude_q16

    if exponent_bits == 0x1F:
        return -0x3FFFFFFFFFFFFFFF if sign_bit else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + fraction_bits
    shift_amount = exponent_bits - 9
    if shift_amount >= 0:
        magnitude_q16 = mantissa << shift_amount
    else:
        rounding = 1 << ((-shift_amount) - 1)
        magnitude_q16 = (mantissa + rounding) >> (-shift_amount)

    return -magnitude_q16 if sign_bit else magnitude_q16


def q8_0_pack32_to_i16_lanes_avx2(src_q8):
    if src_q8 is None:
        return Q8_0_AVX2_ERR_NULL_PTR, []
    if len(src_q8) < Q8_0_AVX2_VALUES_PER_BLOCK:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    return Q8_0_AVX2_OK, [int(src_q8[i]) for i in range(32)]


def q8_0_dot_i16_lanes_avx2(lhs_i16, rhs_i16):
    if lhs_i16 is None or rhs_i16 is None:
        return Q8_0_AVX2_ERR_NULL_PTR, 0
    if len(lhs_i16) < 32 or len(rhs_i16) < 32:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    return Q8_0_AVX2_OK, sum(int(a) * int(b) for a, b in zip(lhs_i16[:32], rhs_i16[:32]))


def q8_0_dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count):
    if lhs_blocks is None or rhs_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    if len(lhs_blocks) < block_count or len(rhs_blocks) < block_count:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    total = 0
    for i in range(block_count):
        err, lhs_lanes = q8_0_pack32_to_i16_lanes_avx2(lhs_blocks[i]["qs"])
        if err != Q8_0_AVX2_OK:
            return err, 0
        err, rhs_lanes = q8_0_pack32_to_i16_lanes_avx2(rhs_blocks[i]["qs"])
        if err != Q8_0_AVX2_OK:
            return err, 0

        err, q_dot = q8_0_dot_i16_lanes_avx2(lhs_lanes, rhs_lanes)
        if err != Q8_0_AVX2_OK:
            return err, 0

        lhs_scale_q16 = q8_0_f16_to_q16(lhs_blocks[i]["d_fp16"])
        rhs_scale_q16 = q8_0_f16_to_q16(rhs_blocks[i]["d_fp16"])

        ok, scale_prod_q32 = q8_0_try_mul_i64(lhs_scale_q16, rhs_scale_q16)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

        ok, block_dot_q32 = q8_0_try_mul_i64(scale_prod_q32, q_dot)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

        ok, total = q8_0_try_add_i64(total, block_dot_q32)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q8_0_AVX2_OK, total


def make_block(d_fp16: int, qs):
    assert len(qs) == 32
    return {"d_fp16": d_fp16 & 0xFFFF, "qs": [int(x) for x in qs]}


def test_known_multi_block_matches_scalar_formula() -> None:
    lhs = [
        make_block(0x3C00, [i - 16 for i in range(32)]),
        make_block(0x4000, [16 - i for i in range(32)]),
        make_block(0x3800, [(-1) ** i * (i % 17) for i in range(32)]),
    ]
    rhs = [
        make_block(0x3C00, [2 * (i - 16) for i in range(32)]),
        make_block(0x3555, [i - 8 for i in range(32)]),
        make_block(0x3A00, [(-1) ** (i + 1) * (i % 11) for i in range(32)]),
    ]

    err, got = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, 3)
    assert err == Q8_0_AVX2_OK

    expected = 0
    for a, b in zip(lhs, rhs):
        scale = q8_0_f16_to_q16(a["d_fp16"]) * q8_0_f16_to_q16(b["d_fp16"])
        qdot = sum(int(x) * int(y) for x, y in zip(a["qs"], b["qs"]))
        expected += scale * qdot
    assert got == expected


def test_randomized_blocks_match_scalar_reference() -> None:
    rng = random.Random(2026041601)
    # Keep scale samples in finite fp16 range used by quantized blocks.
    fp16_scales = [0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0x4400, 0xB800, 0xBC00]

    for _ in range(300):
        block_count = rng.randint(1, 17)
        lhs = []
        rhs = []
        for _ in range(block_count):
            lhs.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))
            rhs.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))

        err, got = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, block_count)
        assert err == Q8_0_AVX2_OK

        expected = 0
        for a, b in zip(lhs, rhs):
            scale = q8_0_f16_to_q16(a["d_fp16"]) * q8_0_f16_to_q16(b["d_fp16"])
            qdot = sum(int(x) * int(y) for x, y in zip(a["qs"], b["qs"]))
            expected += scale * qdot
        assert got == expected


def test_error_paths() -> None:
    err, _ = q8_0_dot_blocks_avx2_q32_checked(None, [], 0)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_blocks_avx2_q32_checked([], None, 0)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_blocks_avx2_q32_checked([], [], -1)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_dot_blocks_avx2_q32_checked([], [], 1)
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def test_zero_block_count_returns_zero() -> None:
    lhs = [make_block(0x3C00, [1] * 32)]
    rhs = [make_block(0x3C00, [2] * 32)]
    err, got = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, 0)
    assert err == Q8_0_AVX2_OK
    assert got == 0


def test_fp16_scale_conversion_known_values() -> None:
    # Spot-check exact integer conversion behavior of HolyC-style fp16->Q16 path.
    # Values chosen to cover zero, unit, powers of two, sign bit, and subnormal.
    cases = {
        0x0000: 0,      # +0
        0x8000: 0,      # -0
        0x3C00: 65536,  # +1.0
        0xBC00: -65536, # -1.0
        0x4000: 131072, # +2.0
        0xC000: -131072,# -2.0
        0x0400: 4,      # smallest normal (2^-14)
        0x0001: 0,      # minimum subnormal rounds to 0 in Q16
        0x03FF: 4,      # max subnormal rounds to 4
    }
    for fp16, expected in cases.items():
        assert q8_0_f16_to_q16(fp16) == expected


def test_overflow_scale_product_reports_error() -> None:
    # fp16 inf in this helper saturates to a very large signed sentinel.
    # Multiplying two such sentinels must trip checked I64 multiply.
    lhs = [make_block(0x7C00, [1] * 32)]
    rhs = [make_block(0x7C00, [1] * 32)]
    err, got = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, 1)
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert got == 0


def test_overflow_total_accumulator_reports_error() -> None:
    # Build a per-block dot large enough that summing three blocks overflows I64
    # while each individual block multiply remains in range.
    # qdot(32x127x127)=516128 and scale(0x5BFF)->4292870144 in Q16.
    # block_dot ~= 9.506e18 (< I64_MAX), but 3x block_dot > I64_MAX.
    qs = [127] * 32
    lhs = [make_block(0x5BFF, qs) for _ in range(3)]
    rhs = [make_block(0x5BFF, qs) for _ in range(3)]
    err, got = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, 3)
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert got == 0


def test_sign_symmetry_with_negated_rhs() -> None:
    rng = random.Random(2026041602)
    block_count = 9
    fp16_scales = [0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00]

    lhs = []
    rhs = []
    rhs_neg = []
    for _ in range(block_count):
        lq = [rng.randint(-128, 127) for _ in range(32)]
        rq = [rng.randint(-128, 127) for _ in range(32)]
        d_l = rng.choice(fp16_scales)
        d_r = rng.choice(fp16_scales)

        lhs.append(make_block(d_l, lq))
        rhs.append(make_block(d_r, rq))
        rhs_neg.append(make_block(d_r, [-v for v in rq]))

    err_pos, dot_pos = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, block_count)
    err_neg, dot_neg = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs_neg, block_count)

    assert err_pos == Q8_0_AVX2_OK
    assert err_neg == Q8_0_AVX2_OK
    assert dot_neg == -dot_pos


def test_lane_permutation_changes_result() -> None:
    # AVX2 lane mapping contract is order-sensitive; swapping lane order should
    # change results for asymmetric inputs.
    base = [i - 16 for i in range(32)]
    rev = list(reversed(base))
    lhs = [make_block(0x3C00, base)]
    rhs = [make_block(0x3C00, base)]
    rhs_perm = [make_block(0x3C00, rev)]

    err_a, dot_a = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs, 1)
    err_b, dot_b = q8_0_dot_blocks_avx2_q32_checked(lhs, rhs_perm, 1)
    assert err_a == Q8_0_AVX2_OK
    assert err_b == Q8_0_AVX2_OK
    assert dot_a != dot_b


def run() -> None:
    test_known_multi_block_matches_scalar_formula()
    test_randomized_blocks_match_scalar_reference()
    test_error_paths()
    test_zero_block_count_returns_zero()
    test_fp16_scale_conversion_known_values()
    test_overflow_scale_product_reports_error()
    test_overflow_total_accumulator_reports_error()
    test_sign_symmetry_with_negated_rhs()
    test_lane_permutation_changes_result()
    print("q8_0_avx2_blocks_q32_reference_checks=ok")


if __name__ == "__main__":
    run()
